"""Click CLI entry point."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from backend.config import Settings
from backend.models import DEFAULT_MODELS

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
    "--no-submit",
    is_flag=True,
    help="Dry run — solvers will not lock in a flag via submit_flag.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(
    challenge_dir: Path | None,
    watch_dir: Path | None,
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

    if not settings.openrouter_api_key:
        console.print("[red]Set OPENROUTER_API_KEY in .env or the environment.[/red]")
        sys.exit(1)

    if watch_dir is not None:
        console.print("[bold]CTF Agent[/bold] — watch mode (coordinator)")
        console.print(f"  Watching: {watch_dir}")
        console.print(f"  Models: {', '.join(DEFAULT_MODELS)}")
        console.print()
        asyncio.run(_run_coordinator(settings, str(watch_dir), no_submit))
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
    console.print(f"  Models: {', '.join(DEFAULT_MODELS)}")
    console.print()
    asyncio.run(_run_single(settings, str(challenge_dir), no_submit))


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    no_submit: bool,
) -> None:
    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.prompts import ChallengeMeta
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    max_concurrent = settings.max_concurrent_challenges
    configure_semaphore(max_concurrent * len(DEFAULT_MODELS))
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
        model_specs=list(DEFAULT_MODELS),
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
    finally:
        pass


async def _run_coordinator(
    settings: Settings,
    challenges_root: str,
    no_submit: bool,
) -> None:
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    configure_semaphore(settings.max_concurrent_challenges * len(DEFAULT_MODELS))
    await cleanup_orphan_containers()
    console.print("[bold]Coordinator[/bold] (Ctrl+C to stop)...\n")

    from backend.agents.openrouter_coordinator import run_openrouter_coordinator

    results = await run_openrouter_coordinator(
        settings=settings,
        model_specs=list(DEFAULT_MODELS),
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
