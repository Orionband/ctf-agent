"""Shared coordinator event loop — local challenges folder, no platform polling."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from backend.agents.coordinator_core import _scan_challenges_root, do_spawn_swarm
from backend.config import Settings
from backend.cost_tracker import CostTracker
from backend.deps import CoordinatorDeps
from backend.models import DEFAULT_MODELS
from backend.prompts import ChallengeMeta

logger = logging.getLogger(__name__)

TurnFn = Callable[[str], Coroutine[Any, Any, None]]


def build_deps(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    challenge_dirs: dict[str, str] | None = None,
    challenge_metas: dict[str, ChallengeMeta] | None = None,
) -> tuple[CostTracker, CoordinatorDeps]:
    """Create cost tracker and coordinator deps."""
    cost_tracker = CostTracker()
    specs = model_specs or list(DEFAULT_MODELS)
    Path(challenges_root).mkdir(parents=True, exist_ok=True)

    deps = CoordinatorDeps(
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=specs,
        challenges_root=challenges_root,
        no_submit=no_submit,
        max_concurrent_challenges=getattr(settings, "max_concurrent_challenges", 10),
        challenge_dirs=challenge_dirs or {},
        challenge_metas=challenge_metas or {},
    )

    for d in Path(challenges_root).iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        try:
            meta = ChallengeMeta.from_directory(d)
        except Exception:
            continue
        key = d.name
        if key not in deps.challenge_dirs:
            deps.challenge_dirs[key] = str(d)
            deps.challenge_metas[key] = meta

    return cost_tracker, deps


async def run_event_loop(
    deps: CoordinatorDeps,
    cost_tracker: CostTracker,
    turn_fn: TurnFn,
    status_interval: int = 60,
) -> dict[str, Any]:
    """Run the shared coordinator event loop."""
    msg_server = await _start_msg_server(deps.operator_inbox, deps.msg_port)

    _scan_challenges_root(deps)
    known = set(deps.challenge_metas.keys())
    unsolved = known - deps.solved_challenges
    initial_msg = (
        f"Challenges directory: {deps.challenges_root}. "
        f"{len(known)} challenge(s) on disk. "
        f"{len(deps.solved_challenges)} solved this session.\n"
        f"Unsolved: {sorted(unsolved) if unsolved else 'NONE'}\n"
        "Spawn swarms for unsolved challenges. New subfolders appear when you add them."
    )

    logger.info(
        "Coordinator starting: %d models, %d local challenge(s)",
        len(deps.model_specs),
        len(known),
    )

    try:
        await turn_fn(initial_msg)
        await _auto_spawn_unsolved(deps)
        deps.announced_disk_challenges.update(deps.challenge_metas.keys())

        last_status = asyncio.get_event_loop().time()

        while True:
            parts: list[str] = []

            _scan_challenges_root(deps)
            for name in list(deps.challenge_metas.keys()):
                if name in deps.announced_disk_challenges:
                    continue
                deps.announced_disk_challenges.add(name)
                if name not in deps.solved_challenges:
                    parts.append(f"NEW CHALLENGE ON DISK: '{name}'. Spawning swarm.")
                    await _auto_spawn_one(deps, name)

            for name, task in list(deps.swarm_tasks.items()):
                if task.done():
                    parts.append(f"SOLVER FINISHED: Swarm for '{name}' completed. Check results or retry.")
                    deps.swarm_tasks.pop(name, None)

            while True:
                try:
                    solver_msg = deps.coordinator_inbox.get_nowait()
                    parts.append(f"SOLVER MESSAGE: {solver_msg}")
                except asyncio.QueueEmpty:
                    break

            while True:
                try:
                    op_msg = deps.operator_inbox.get_nowait()
                    parts.append(f"OPERATOR MESSAGE: {op_msg}")
                    logger.info("Operator message: %s", op_msg[:200])
                except asyncio.QueueEmpty:
                    break

            now = asyncio.get_event_loop().time()
            if now - last_status >= status_interval:
                last_status = now
                active = [n for n, t in deps.swarm_tasks.items() if not t.done()]
                known_n = len(deps.challenge_metas)
                status_line = (
                    f"STATUS: {len(deps.solved_challenges)}/{known_n} solved this session, "
                    f"{len(active)} active swarms. Cost: ${cost_tracker.total_cost_usd:.2f}"
                )
                if active or parts:
                    parts.append(status_line)
                else:
                    logger.info("Heartbeat: %s", status_line)

            if parts:
                msg = "\n\n".join(parts)
                logger.info("Event -> coordinator: %s", msg[:200])
                await turn_fn(msg)

            await asyncio.sleep(2.0)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Coordinator shutting down...")
    except Exception as e:
        logger.error("Coordinator fatal: %s", e, exc_info=True)
    finally:
        if msg_server:
            msg_server.close()
            await msg_server.wait_closed()
        for swarm in deps.swarms.values():
            swarm.kill()
        for task in deps.swarm_tasks.values():
            task.cancel()
        if deps.swarm_tasks:
            await asyncio.gather(*deps.swarm_tasks.values(), return_exceptions=True)
        cost_tracker.log_summary()

    return {
        "results": deps.results,
        "total_cost_usd": cost_tracker.total_cost_usd,
        "total_tokens": cost_tracker.total_tokens,
    }


async def _auto_spawn_one(deps: CoordinatorDeps, challenge_name: str) -> None:
    if challenge_name in deps.swarms:
        return
    active = sum(1 for t in deps.swarm_tasks.values() if not t.done())
    if active >= deps.max_concurrent_challenges:
        return
    try:
        result = await do_spawn_swarm(deps, challenge_name)
        logger.info("Auto-spawn %s: %s", challenge_name, result[:100])
    except Exception as e:
        logger.warning("Auto-spawn failed for %s: %s", challenge_name, e)


async def _auto_spawn_unsolved(deps: CoordinatorDeps) -> None:
    _scan_challenges_root(deps)
    for name in sorted(deps.challenge_metas.keys()):
        if name in deps.solved_challenges:
            continue
        await _auto_spawn_one(deps, name)


async def _start_msg_server(inbox: asyncio.Queue, port: int = 0) -> asyncio.Server | None:
    """Start a tiny HTTP server that accepts operator messages via POST."""

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            headers: dict[str, str] = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    k, v = line.decode().split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            method = request_line.decode().split()[0] if request_line else ""
            content_length = int(headers.get("content-length", 0))

            if method == "POST" and content_length > 0:
                body = await asyncio.wait_for(reader.read(content_length), timeout=5)
                try:
                    data = json.loads(body)
                    message = data.get("message", body.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    message = body.decode("utf-8", errors="replace")

                inbox.put_nowait(message)
                resp = json.dumps({"ok": True, "queued": message[:200]})
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode()
                )
            else:
                resp = json.dumps({"error": "POST with JSON body required", "usage": 'POST {"message": "..."}'})
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {len(resp)}\r\n\r\n{resp}".encode()
                )

            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(_handle, "127.0.0.1", port)
        actual_port = server.sockets[0].getsockname()[1]
        logger.info("Operator message endpoint listening on http://127.0.0.1:%s", actual_port)
        return server
    except OSError as e:
        logger.warning("Could not start operator message endpoint: %s", e)
        return None
