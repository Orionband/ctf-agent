"""ChallengeSwarm — parallel OpenRouter solvers racing on one challenge."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.agents.gemini_solver import GeminiSolver
from backend.agents.openrouter_solver import OpenRouterSolver
from backend.cost_tracker import CostTracker
from backend.message_bus import ChallengeMessageBus
from backend.models import DEFAULT_MODELS
from backend.prompts import ChallengeMeta
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    QUOTA_ERROR,
    SolverProtocol,
    SolverResult,
)

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ChallengeSwarm:
    """Parallel solvers racing on one challenge."""

    challenge_dir: str
    meta: ChallengeMeta
    cost_tracker: CostTracker
    settings: Settings
    model_specs: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    no_submit: bool = False
    coordinator_inbox: asyncio.Queue | None = None

    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    solvers: dict[str, SolverProtocol] = field(default_factory=dict)
    findings: dict[str, str] = field(default_factory=dict)
    winner: SolverResult | None = None
    confirmed_flag: str | None = None
    _flag_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _submit_count: dict[str, int] = field(default_factory=dict)
    _submitted_flags: set[str] = field(default_factory=set)
    _last_submit_time: dict[str, float] = field(default_factory=dict)
    message_bus: ChallengeMessageBus = field(default_factory=ChallengeMessageBus)

    def _create_solver(self, model_spec: str):
        if model_spec.startswith("gemini/"):
            solver = GeminiSolver(
                model_spec=model_spec,
                challenge_dir=self.challenge_dir,
                meta=self.meta,
                cost_tracker=self.cost_tracker,
                settings=self.settings,
                cancel_event=self.cancel_event,
            )
        else:
            solver = OpenRouterSolver(
                model_spec=model_spec,
                challenge_dir=self.challenge_dir,
                meta=self.meta,
                cost_tracker=self.cost_tracker,
                settings=self.settings,
                cancel_event=self.cancel_event,
            )
        solver.deps.message_bus = self.message_bus
        solver.deps.model_spec = model_spec
        solver.deps.no_submit = self.no_submit
        solver.deps.submit_fn = lambda flag: self.try_submit_flag(flag, model_spec)
        solver.deps.notify_coordinator = self._make_notify_fn(model_spec)
        return solver

    def _make_notify_fn(self, model_spec: str):
        async def _notify(message: str) -> None:
            if self.coordinator_inbox:
                self.coordinator_inbox.put_nowait(f"[{self.meta.name}/{model_spec}] {message}")

        return _notify

    def _gather_sibling_insights(self, exclude_model: str) -> str:
        parts: list[str] = []
        for model, finding in self.findings.items():
            if model != exclude_model and finding:
                parts.append(f"[{model}]: {finding}")
        return "\n\n".join(parts) if parts else "No sibling insights available yet."

    SUBMISSION_COOLDOWNS = [0, 30, 120, 300, 600]

    async def try_submit_flag(self, flag: str, model_spec: str) -> tuple[str, bool]:
        """Cooldown-gated, deduplicated flag acceptance (local verification)."""
        async with self._flag_lock:
            if self.confirmed_flag:
                return f"ALREADY SOLVED — flag already confirmed: {self.confirmed_flag}", True

            normalized = flag.strip()
            if not normalized:
                return "Empty flag — nothing to submit.", False

            if normalized in self._submitted_flags:
                return "INCORRECT — already tried this exact flag.", False

            wrong_count = self._submit_count.get(model_spec, 0)
            cooldown_idx = min(wrong_count, len(self.SUBMISSION_COOLDOWNS) - 1)
            cooldown = self.SUBMISSION_COOLDOWNS[cooldown_idx]
            if cooldown > 0:
                last_time = self._last_submit_time.get(model_spec, 0)
                elapsed = time.monotonic() - last_time
                if elapsed < cooldown:
                    remaining = int(cooldown - elapsed)
                    return (
                        f"COOLDOWN — wait {remaining}s before submitting again. "
                        f"You have {wrong_count} duplicate or rejected attempts. "
                        "Use this time to verify your flag.",
                        False,
                    )

            self._submitted_flags.add(normalized)

            from backend.tools.core import do_submit_flag

            display, is_confirmed = await do_submit_flag(flag)
            if is_confirmed:
                self.confirmed_flag = normalized
            else:
                self._submit_count[model_spec] = wrong_count + 1
                self._last_submit_time[model_spec] = time.monotonic()
            return display, is_confirmed

    async def _run_solver(self, model_spec: str) -> SolverResult | None:
        solver = self._create_solver(model_spec)
        self.solvers[model_spec] = solver

        try:
            result, final_solver = await self._run_solver_loop(solver, model_spec)
            solver = final_solver
            return result
        except Exception as e:
            logger.error("[%s/%s] Fatal: %s", self.meta.name, model_spec, e, exc_info=True)
            return None
        finally:
            await solver.stop()

    async def _run_solver_loop(self, solver, model_spec: str) -> tuple[SolverResult, SolverProtocol]:
        bump_count = 0
        consecutive_errors = 0
        result = SolverResult(
            flag=None,
            status=CANCELLED,
            findings_summary="",
            step_count=0,
            cost_usd=0.0,
            log_path="",
        )
        await solver.start()

        while not self.cancel_event.is_set():
            result = await solver.run_until_done_or_gave_up()

            if (
                result.status not in (ERROR,)
                and not (result.step_count == 0 and result.cost_usd == 0)
                and result.findings_summary
                and not result.findings_summary.startswith(("Error:", "Turn failed:"))
            ):
                self.findings[model_spec] = result.findings_summary
                await self.message_bus.post(model_spec, result.findings_summary[:500])

            if result.status == FLAG_FOUND:
                self.cancel_event.set()
                self.winner = result
                logger.info("[%s] Flag found by %s: %s", self.meta.name, model_spec, result.flag)
                return result, solver

            if result.status == CANCELLED:
                break

            if result.status == QUOTA_ERROR:
                logger.warning(
                    "[%s/%s] Quota/rate-limited — stopping this model for now",
                    self.meta.name,
                    model_spec,
                )
                break

            if result.status in (GAVE_UP, ERROR):
                if result.step_count == 0 and result.cost_usd == 0:
                    detail = (result.findings_summary or "").strip()
                    if detail:
                        logger.warning(
                            "[%s/%s] Broken (0 steps, $0) — not bumping — %s",
                            self.meta.name,
                            model_spec,
                            detail[:500],
                        )
                    else:
                        logger.warning(
                            "[%s/%s] Broken (0 steps, $0) — not bumping",
                            self.meta.name,
                            model_spec,
                        )
                    break

                if result.status == ERROR:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        logger.warning(
                            "[%s/%s] %d consecutive errors — giving up",
                            self.meta.name,
                            model_spec,
                            consecutive_errors,
                        )
                        break
                else:
                    consecutive_errors = 0

                bump_count += 1
                try:
                    await asyncio.wait_for(self.cancel_event.wait(), timeout=min(bump_count * 30, 300))
                    break
                except TimeoutError:
                    pass
                insights = self._gather_sibling_insights(model_spec)
                solver.bump(insights)
                logger.info("[%s/%s] Bumped (%d), resuming", self.meta.name, model_spec, bump_count)
                continue

        return result, solver

    async def run(self) -> SolverResult | None:
        tasks = [
            asyncio.create_task(self._run_solver(spec), name=f"solver-{spec}") for spec in self.model_specs
        ]

        try:
            while tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        result = task.result()
                    except Exception:
                        continue
                    if result and result.status == FLAG_FOUND:
                        self.cancel_event.set()
                        for p in pending:
                            p.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        return result

                tasks = list(pending)

            self.cancel_event.set()
            return self.winner
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.cancel_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return self.winner
        except Exception as e:
            logger.error("[%s] Swarm error: %s", self.meta.name, e, exc_info=True)
            self.cancel_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return None

    def kill(self) -> None:
        self.cancel_event.set()

    def get_status(self) -> dict:
        return {
            "challenge": self.meta.name,
            "cancelled": self.cancel_event.is_set(),
            "winner": self.winner.flag if self.winner else None,
            "agents": {
                spec: {
                    "findings": self.findings.get(spec, ""),
                    "status": (
                        "running"
                        if spec in self.solvers and not self.cancel_event.is_set()
                        else ("won" if self.winner and self.winner.flag else "finished")
                    ),
                }
                for spec in self.model_specs
            },
        }
