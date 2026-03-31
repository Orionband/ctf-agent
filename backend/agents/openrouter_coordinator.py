"""Coordinator LLM via OpenRouter (pydantic-ai) + shared event loop."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

from backend.agents.coordinator_core import (
    do_broadcast,
    do_bump_agent,
    do_check_swarm_status,
    do_fetch_challenges,
    do_get_solve_status,
    do_kill_swarm,
    do_read_solver_trace,
    do_spawn_swarm,
    do_submit_flag,
)
from backend.agents.coordinator_loop import build_deps, run_event_loop
from backend.config import Settings
from backend.deps import CoordinatorDeps
from backend.models import DEFAULT_MODELS, model_id_from_spec, provider_from_spec, resolve_model, resolve_model_settings

logger = logging.getLogger(__name__)

COORDINATOR_PROMPT = """\
You are a CTF competition coordinator. Each challenge is a folder on disk (challenge text, \
hints, attachments — any layout). There is no CTF platform integration — when solvers find \
a flag it is printed locally; operators submit flags to the competition manually if required.

Strategy:
- Spawn swarms for unsolved challenges, prioritizing easier-looking challenges when obvious
- Use read_solver_trace to monitor stuck solvers and bump_agent with concrete next steps
- Use broadcast to share insights across models on the same challenge

Rules:
- Prefer not to kill swarms; help them with targeted bumps when stuck
- Use spawn_swarm to start solving; use check_swarm_status for progress
- submit_flag from the coordinator is optional — solvers normally report flags via their own tool
"""


async def tool_fetch_challenges(ctx: RunContext[CoordinatorDeps]) -> str:
    return await do_fetch_challenges(ctx.deps)


async def tool_get_solve_status(ctx: RunContext[CoordinatorDeps]) -> str:
    return await do_get_solve_status(ctx.deps)


async def tool_spawn_swarm(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
    return await do_spawn_swarm(ctx.deps, challenge_name)


async def tool_check_swarm_status(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
    return await do_check_swarm_status(ctx.deps, challenge_name)


async def tool_submit_flag(ctx: RunContext[CoordinatorDeps], challenge_name: str, flag: str) -> str:
    return await do_submit_flag(ctx.deps, challenge_name, flag)


async def tool_kill_swarm(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
    return await do_kill_swarm(ctx.deps, challenge_name)


async def tool_bump_agent(
    ctx: RunContext[CoordinatorDeps], challenge_name: str, model_spec: str, insights: str
) -> str:
    return await do_bump_agent(ctx.deps, challenge_name, model_spec, insights)


async def tool_broadcast(ctx: RunContext[CoordinatorDeps], challenge_name: str, message: str) -> str:
    return await do_broadcast(ctx.deps, challenge_name, message)


async def tool_read_solver_trace(
    ctx: RunContext[CoordinatorDeps], challenge_name: str, model_spec: str, last_n: int = 20
) -> str:
    return await do_read_solver_trace(ctx.deps, challenge_name, model_spec, last_n)


_CO_TOOLS = FunctionToolset(
    tools=[
        tool_fetch_challenges,
        tool_get_solve_status,
        tool_spawn_swarm,
        tool_check_swarm_status,
        tool_submit_flag,
        tool_kill_swarm,
        tool_bump_agent,
        tool_broadcast,
        tool_read_solver_trace,
    ],
    max_retries=2,
)


async def run_openrouter_coordinator(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    coordinator_model: str | None = None,
    msg_port: int = 0,
) -> dict[str, Any]:
    specs = model_specs or list(DEFAULT_MODELS)
    cost_tracker, deps = build_deps(settings, specs, challenges_root, no_submit)
    deps.msg_port = msg_port

    co_spec = coordinator_model or settings.coordinator_model or specs[0]
    model = resolve_model(co_spec, settings)
    model_settings = resolve_model_settings(co_spec)

    agent = Agent(
        model,
        deps_type=CoordinatorDeps,
        system_prompt=COORDINATOR_PROMPT,
        model_settings=model_settings,
        toolsets=[_CO_TOOLS],
    )

    async def turn_fn(message: str) -> None:
        logger.debug("Coordinator prompt: %s", message[:200])
        from pydantic_ai.usage import UsageLimits

        result = await agent.run(message, deps=deps, usage_limits=UsageLimits(request_limit=64))
        usage = result.usage()
        cost_tracker.record(
            "coordinator",
            usage,
            model_id_from_spec(co_spec),
            provider_spec=provider_from_spec(co_spec),
            duration_seconds=0.0,
        )

    return await run_event_loop(deps, cost_tracker, turn_fn)
