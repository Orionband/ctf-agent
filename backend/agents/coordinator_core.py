"""Shared coordinator tool logic — local challenge folders only."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from backend.deps import CoordinatorDeps
from backend.prompts import ChallengeMeta
from backend.solver_base import FLAG_FOUND
from backend.tools.core import do_submit_flag as record_local_flag

logger = logging.getLogger(__name__)


def _scan_challenges_root(deps: CoordinatorDeps) -> None:
    """Load any new challenge subfolders from disk into deps.challenge_dirs / challenge_metas."""
    root = Path(deps.challenges_root)
    if not root.is_dir():
        return
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        try:
            meta = ChallengeMeta.from_directory(d)
        except Exception as e:
            logger.warning("Skip %s: %s", d, e)
            continue
        key = d.name
        if key not in deps.challenge_dirs:
            deps.challenge_dirs[key] = str(d)
            deps.challenge_metas[key] = meta


async def do_fetch_challenges(deps: CoordinatorDeps) -> str:
    _scan_challenges_root(deps)
    result = [
        {
            "folder": key,
            "name": meta.name,
            "category": meta.category,
            "value": meta.value,
            "solves": meta.solves,
            "status": "SOLVED" if key in deps.solved_challenges else "unsolved",
            "description": (meta.description or "")[:200],
        }
        for key, meta in deps.challenge_metas.items()
    ]
    return json.dumps(result, indent=2)


async def do_get_solve_status(deps: CoordinatorDeps) -> str:
    swarm_status = {name: swarm.get_status() for name, swarm in deps.swarms.items()}
    return json.dumps(
        {
            "solved_this_session": sorted(deps.solved_challenges),
            "active_swarms": swarm_status,
        },
        indent=2,
    )


async def do_spawn_swarm(deps: CoordinatorDeps, challenge_name: str) -> str:
    _scan_challenges_root(deps)

    finished = [
        name
        for name, swarm in deps.swarms.items()
        if swarm.cancel_event.is_set() or (name in deps.swarm_tasks and deps.swarm_tasks[name].done())
    ]
    for name in finished:
        del deps.swarms[name]
        deps.swarm_tasks.pop(name, None)

    active_count = len(deps.swarms)
    if active_count >= deps.max_concurrent_challenges:
        return f"At capacity ({active_count}/{deps.max_concurrent_challenges} challenges running). Wait for one to finish."

    if challenge_name in deps.swarms:
        return f"Swarm still running for {challenge_name}"

    if challenge_name not in deps.challenge_dirs:
        return (
            f"Challenge folder '{challenge_name}' not found under {deps.challenges_root}. "
            "Add a subfolder with your challenge text and files, then try again."
        )

    from backend.agents.swarm import ChallengeSwarm

    swarm = ChallengeSwarm(
        challenge_dir=deps.challenge_dirs[challenge_name],
        meta=deps.challenge_metas[challenge_name],
        cost_tracker=deps.cost_tracker,
        settings=deps.settings,
        model_specs=deps.model_specs,
        no_submit=deps.no_submit,
        coordinator_inbox=deps.coordinator_inbox,
    )
    deps.swarms[challenge_name] = swarm

    async def _run_and_cleanup() -> None:
        result = await swarm.run()
        if result and result.status == FLAG_FOUND:
            deps.results[challenge_name] = {
                "flag": result.flag,
                "note": "local session — submit flag to the competition manually if required",
            }
            deps.solved_challenges.add(challenge_name)

    task = asyncio.create_task(_run_and_cleanup(), name=f"swarm-{challenge_name}")
    deps.swarm_tasks[challenge_name] = task
    return f"Swarm spawned for {challenge_name} with {len(deps.model_specs)} models"


async def do_check_swarm_status(deps: CoordinatorDeps, challenge_name: str) -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    return json.dumps(swarm.get_status(), indent=2)


async def do_submit_flag(deps: CoordinatorDeps, challenge_name: str, flag: str) -> str:
    """Coordinator-only: record intent (solvers use the submit_flag tool)."""
    if deps.no_submit:
        return f'DRY RUN — would accept "{flag.strip()}" for {challenge_name}'
    _, ok = await record_local_flag(flag)
    if ok:
        deps.solved_challenges.add(challenge_name.strip())
        return f'CORRECT — recorded for {challenge_name}. Flag: {flag.strip()}'
    return "Could not record flag."


async def do_kill_swarm(deps: CoordinatorDeps, challenge_name: str) -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    swarm.kill()
    return f"Swarm for {challenge_name} cancelled"


async def do_bump_agent(deps: CoordinatorDeps, challenge_name: str, model_spec: str, insights: str) -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    solver = swarm.solvers.get(model_spec)
    if not solver:
        return f"No solver for {model_spec} in {challenge_name}"
    solver.bump(insights)
    return f"Bumped {model_spec} on {challenge_name}"


async def do_read_solver_trace(deps: CoordinatorDeps, challenge_name: str, model_spec: str, last_n: int = 20) -> str:
    """Read the last N trace events from a solver's JSONL log."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm for {challenge_name}"
    solver = swarm.solvers.get(model_spec)
    if not solver:
        return f"No solver for {model_spec}"
    trace_path = getattr(solver, "tracer", None)
    if not trace_path:
        return "No tracer on solver"
    path = trace_path.path if hasattr(trace_path, "path") else str(trace_path)
    try:
        lines = Path(path).read_text().strip().split("\n")
        recent = lines[-last_n:]
        summary = []
        for line in recent:
            try:
                d = json.loads(line)
                t = d.get("type", "?")
                if t == "tool_call":
                    args_str = str(d.get("args", ""))[:100]
                    summary.append(f"step {d.get('step','?')} CALL {d.get('tool','?')}: {args_str}")
                elif t == "tool_result":
                    result_str = str(d.get("result", ""))[:100]
                    summary.append(f"step {d.get('step','?')} RESULT {d.get('tool','?')}: {result_str}")
                elif t in ("finish", "error", "bump", "turn_failed"):
                    summary.append(f"** {t}: {json.dumps({k:v for k,v in d.items() if k != 'ts'})}")
                elif t == "usage":
                    summary.append(
                        f"usage: in={d.get('input_tokens',0)} out={d.get('output_tokens',0)} cost=${d.get('cost_usd',0):.4f}"
                    )
                else:
                    summary.append(f"{t}: {str(d)[:80]}")
            except Exception:
                summary.append(line[:100])
        return "\n".join(summary)
    except FileNotFoundError:
        return f"Trace file not found: {path}"
    except Exception as e:
        return f"Error reading trace: {e}"


async def do_broadcast(deps: CoordinatorDeps, challenge_name: str, message: str) -> str:
    """Broadcast a message to all solvers working on a challenge."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    await swarm.message_bus.broadcast(message)
    return f"Broadcast to all solvers on {challenge_name}"
