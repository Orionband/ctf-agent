"""Click CLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console

from backend.config import Settings
from backend.models import DEFAULT_MODELS, model_id_from_spec

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiodocker").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%X"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


@click.command()
@click.argument(
    "challenge_dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--watch",
    "watch_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Watch a directory of challenge subfolders and run the coordinator (advanced).",
)
@click.option(
    "--model",
    "single_model",
    default=None,
    help="Run only one model (e.g. openrouter/qwen/qwen3.6-plus-preview:free).",
)
@click.option(
    "--gemini",
    "include_gemini",
    is_flag=True,
    help="Include Gemini direct API model (gemini/gemini-flash-latest) in the solver lineup.",
)
@click.option(
    "--gemini-rotate",
    "gemini_rotate",
    is_flag=True,
    help="With default 3 OpenRouter models, adds Gemini (3-flash↔2.5-flash rotation). With --model gemini/..., Gemini-only with rotation.",
)
@click.option(
    "--check-keys",
    is_flag=True,
    help="Validate each configured OpenRouter key and print per-key status, then exit.",
)
@click.option(
    "--no-submit",
    is_flag=True,
    help="Dry run — solvers will not lock in a flag via submit_flag.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(
    challenge_dir: Path | None,
    watch_dir: Path | None,
    single_model: str | None,
    include_gemini: bool,
    gemini_rotate: bool,
    check_keys: bool,
    no_submit: bool,
    verbose: bool,
) -> None:
    """Solve CTF challenges from a folder (OpenRouter, three models). Set OPENROUTER_API_KEY.

    Put challenge text in challenge.txt or README.md, optional hints.txt or hints/, drop other files anywhere in the folder.

    Examples:

        ctf-solve ./my-challenge

        ctf-solve

    If you omit the path, a folder named "challenge" in the current directory is used.

    Watch mode (multiple challenge subfolders):

        ctf-solve --watch ./challenges
    """
    _setup_logging(verbose)

    settings = Settings()
    gemini_rotate_with_defaults = False
    if gemini_rotate:
        if single_model and not single_model.strip().startswith("gemini/"):
            console.print("[red]--gemini-rotate with --model requires gemini/... (or omit --model to add Gemini to the default 3 OpenRouter models).[/red]")
            sys.exit(1)
        if single_model and single_model.strip().startswith("gemini/"):
            mid = model_id_from_spec(single_model.strip())
            if mid == "gemini-2.5-flash":
                chain = "gemini-2.5-flash,gemini-3-flash-preview"
            else:
                chain = "gemini-3-flash-preview,gemini-2.5-flash"
        else:
            chain = "gemini-3-flash-preview,gemini-2.5-flash"
            gemini_rotate_with_defaults = True
        settings = settings.model_copy(update={"gemini_rotate_chain": chain})
    model_specs = _select_models(
        single_model,
        include_gemini,
        gemini_rotate_with_defaults=gemini_rotate_with_defaults,
    )
    openrouter_keys = settings.get_openrouter_keys()
    gemini_keys = settings.get_gemini_keys()

    needs_openrouter = any(m.startswith("openrouter/") for m in model_specs)
    needs_gemini = any(m.startswith("gemini/") for m in model_specs)

    if needs_openrouter and not openrouter_keys:
        console.print("[red]Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS in .env or the environment.[/red]")
        sys.exit(1)
    if needs_gemini and not gemini_keys:
        console.print("[red]Set GEMINI_API_KEY or GEMINI_API_KEYS in .env or the environment.[/red]")
        sys.exit(1)

    if check_keys:
        if needs_openrouter:
            asyncio.run(_check_keys(openrouter_keys, model_specs[0]))
        else:
            console.print("[yellow]--check-keys currently checks OpenRouter keys. Pick an openrouter model.[/yellow]")
        return

    if watch_dir is not None:
        console.print("[bold]CTF Agent[/bold] — watch mode (coordinator)")
        console.print(f"  Watching: {watch_dir}")
        console.print(f"  Models: {', '.join(model_specs)}")
        if needs_openrouter:
            console.print(f"  OpenRouter keys: {len(openrouter_keys)} configured")
        if needs_gemini:
            console.print(f"  Gemini keys: {len(gemini_keys)} configured")
        if getattr(settings, "gemini_rotate_chain", "").strip():
            console.print(f"  Gemini rotate: {settings.gemini_rotate_chain}")
        console.print()
        asyncio.run(_run_coordinator(settings, str(watch_dir), no_submit, model_specs))
        return

    if challenge_dir is None:
        default = Path("challenge")
        if default.is_dir():
            challenge_dir = default
        else:
            console.print(
                "[red]Pass a challenge folder, e.g. [bold]ctf-solve ./my-challenge[/bold], "
                "or create a [bold]challenge[/bold] directory here.[/red]"
            )
            sys.exit(1)

    console.print("[bold]CTF Agent[/bold]")
    console.print(f"  Folder: {challenge_dir.resolve()}")
    console.print(f"  Models: {', '.join(model_specs)}")
    if needs_openrouter:
        console.print(f"  OpenRouter keys: {len(openrouter_keys)} configured")
    if needs_gemini:
        console.print(f"  Gemini keys: {len(gemini_keys)} configured")
    if getattr(settings, "gemini_rotate_chain", "").strip():
        console.print(f"  Gemini rotate: {settings.gemini_rotate_chain}")
    console.print()
    asyncio.run(_run_single(settings, str(challenge_dir), no_submit, model_specs))


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


async def _check_keys(keys: list[str], model_spec: str) -> None:
    """Run lightweight per-key diagnostics against OpenRouter."""
    model_id = model_spec.split("/", 1)[1] if model_spec.startswith("openrouter/") else model_spec
    console.print("[bold]OpenRouter key diagnostics[/bold]")
    console.print(f"  Model probe: {model_id}")
    console.print(f"  Keys found: {len(keys)}")
    console.print()

    async with httpx.AsyncClient(timeout=25.0) as client:
        for i, key in enumerate(keys, 1):
            masked = _mask_key(key)
            headers = {"Authorization": f"Bearer {key}"}

            # Check auth first
            auth_ok = False
            auth_msg = ""
            try:
                r = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
                if r.status_code == 200:
                    auth_ok = True
                    auth_msg = "auth ok"
                else:
                    try:
                        body = json.dumps(r.json(), ensure_ascii=False)[:180]
                    except Exception:
                        body = r.text[:180]
                    auth_msg = f"auth fail HTTP {r.status_code}: {body}"
            except Exception as e:
                auth_msg = f"auth request error: {e}"

            if not auth_ok:
                console.print(f"  [{i}] {masked} -> [red]{auth_msg}[/red]")
                continue

            # Tiny model probe
            probe_msg = ""
            try:
                probe = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Reply with OK only."}],
                        "max_tokens": 4,
                    },
                )
                if probe.status_code == 200:
                    probe_msg = "probe ok"
                else:
                    try:
                        body = json.dumps(probe.json(), ensure_ascii=False)[:220]
                    except Exception:
                        body = probe.text[:220]
                    probe_msg = f"probe HTTP {probe.status_code}: {body}"
            except Exception as e:
                probe_msg = f"probe error: {e}"

            color = "green" if probe_msg == "probe ok" else "yellow"
            console.print(f"  [{i}] {masked} -> [green]{auth_msg}[/green], [{color}]{probe_msg}[/{color}]")


def _select_models(
    single_model: str | None,
    include_gemini: bool = False,
    *,
    gemini_rotate_with_defaults: bool = False,
) -> list[str]:
    if not single_model:
        models = list(DEFAULT_MODELS)
        if gemini_rotate_with_defaults:
            if "gemini/gemini-3-flash-preview" not in models:
                models.append("gemini/gemini-3-flash-preview")
        elif include_gemini and "gemini/gemini-flash-latest" not in models:
            models.append("gemini/gemini-flash-latest")
        return models
    spec = single_model.strip()
    if not spec:
        models = list(DEFAULT_MODELS)
        if gemini_rotate_with_defaults:
            if "gemini/gemini-3-flash-preview" not in models:
                models.append("gemini/gemini-3-flash-preview")
        elif include_gemini and "gemini/gemini-flash-latest" not in models:
            models.append("gemini/gemini-flash-latest")
        return models
    if not spec.startswith("openrouter/") and not spec.startswith("gemini/"):
        if spec.startswith("gemini-") or spec.startswith("models/gemini"):
            spec = f"gemini/{spec.replace('models/', '')}"
        else:
            spec = f"openrouter/{spec}"
    return [spec]


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    no_submit: bool,
    model_specs: list[str],
) -> None:
    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.prompts import ChallengeMeta
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    # In one-model mode, always show model/tool debug output by default.
    if len(model_specs) == 1:
        settings = settings.model_copy(update={"always_debug_single_model": True})

    max_concurrent = settings.max_concurrent_challenges
    configure_semaphore(max_concurrent * len(model_specs))
    await cleanup_orphan_containers()

    try:
        meta = ChallengeMeta.from_directory(challenge_dir)
    except Exception as e:
        console.print(f"[red]Could not load challenge folder: {e}[/red]")
        sys.exit(1)

    console.print(f"[bold]Challenge:[/bold] {meta.name}")

    cost_tracker = CostTracker()
    swarm = ChallengeSwarm(
        challenge_dir=challenge_dir,
        meta=meta,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=model_specs,
        no_submit=no_submit,
    )

    try:
        result = await swarm.run()
        from backend.solver_base import FLAG_FOUND

        if result and result.status == FLAG_FOUND:
            console.print(f"\n[bold green]FLAG:[/bold green] {result.flag}")
            console.print(
                "[dim]Submit this flag to the competition yourself if the event uses a website.[/dim]"
            )
        else:
            console.print("\n[bold red]No flag found.[/bold red]")

        console.print("\n[bold]Cost Summary:[/bold]")
        for agent_name in cost_tracker.by_agent:
            console.print(f"  {agent_name}: {cost_tracker.format_usage(agent_name)}")
        console.print(f"  [bold]Total: ${cost_tracker.total_cost_usd:.2f}[/bold]")
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Stopping... cleaning up containers.[/yellow]")
        swarm.kill()
        await asyncio.sleep(0.2)
    finally:
        swarm.kill()


async def _run_coordinator(
    settings: Settings,
    challenges_root: str,
    no_submit: bool,
    model_specs: list[str],
) -> None:
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    configure_semaphore(settings.max_concurrent_challenges * len(model_specs))
    await cleanup_orphan_containers()
    console.print("[bold]Coordinator[/bold] (Ctrl+C to stop)...\n")

    from backend.agents.openrouter_coordinator import run_openrouter_coordinator

    results = await run_openrouter_coordinator(
        settings=settings,
        model_specs=model_specs,
        challenges_root=challenges_root,
        no_submit=no_submit,
        coordinator_model=settings.coordinator_model or None,
        msg_port=0,
    )

    console.print("\n[bold]Final Results:[/bold]")
    for challenge, data in results.get("results", {}).items():
        console.print(f"  {challenge}: {data.get('flag', 'no flag')}")
    console.print(f"\n[bold]Total cost: ${results.get('total_cost_usd', 0):.2f}[/bold]")


@click.command()
@click.argument("message")
@click.option("--port", default=9400, type=int, help="Coordinator message port")
@click.option("--host", default="127.0.0.1", help="Coordinator host")
def msg(message: str, port: int, host: str) -> None:
    """Send a message to the running coordinator."""
    import json
    import urllib.request

    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/msg",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            console.print(f"[green]Sent:[/green] {data.get('queued', message[:200])}")
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        console.print("Is the coordinator running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
