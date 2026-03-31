"""Per-model solver agent using OpenRouter function calling.

Problem:
Pydantic AI (OpenAI-compatible backend) sends a `tool_choice` field that some
OpenRouter endpoints reject for these models, causing 404s before any tools
can be used.

This solver implements a minimal OpenRouter chat loop that:
- Enables OpenRouter reasoning
- Provides our tools via the OpenAI-compatible `tools` parameter
- Omits `tool_choice` entirely
- Executes tool calls locally inside the existing Docker sandbox
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from backend.cost_tracker import CostTracker
from backend.deps import SolverDeps
from backend.loop_detect import LOOP_WARNING_MESSAGE, LoopDetector
from backend.models import model_id_from_spec, provider_from_spec
from backend.prompts import ChallengeMeta, build_prompt
from backend.sandbox import DockerSandbox
from backend.solver_base import CANCELLED, CORRECT_MARKERS, ERROR, FLAG_FOUND, GAVE_UP, QUOTA_ERROR, SolverResult
from backend.tracing import SolverTracer
from backend.openrouter_key_pool import next_openrouter_key
from backend.tools.core import (
    do_bash,
    do_check_findings,
    do_list_files,
    do_read_file,
    do_submit_flag,
    do_web_fetch,
    do_webhook_create,
    do_webhook_get_requests,
    do_view_image,
    do_write_file,
)

logger = logging.getLogger(__name__)

FlagPattern = re.compile(r"FLAG\s*:\s*(.+)", re.IGNORECASE)


ToolHandler = Callable[..., Awaitable[str]]


@dataclass
class _ToolDef:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: ToolHandler


class OpenRouterSolver:
    """A single solver: one model, one container, one challenge."""

    def __init__(
        self,
        model_spec: str,
        challenge_dir: str,
        meta: ChallengeMeta,
        cost_tracker: CostTracker,
        settings: Any,
        cancel_event: asyncio.Event | None = None,
        sandbox: DockerSandbox | None = None,
        owns_sandbox: bool | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.model_id = model_id_from_spec(model_spec)
        self.challenge_dir = challenge_dir
        self.meta = meta
        self.cost_tracker = cost_tracker
        self.settings = settings
        self.cancel_event = cancel_event or asyncio.Event()
        self._owns_sandbox = owns_sandbox if owns_sandbox is not None else (sandbox is None)

        self.sandbox = sandbox or DockerSandbox(
            image=getattr(settings, "sandbox_image", "ctf-sandbox"),
            challenge_dir=challenge_dir,
            memory_limit=getattr(settings, "container_memory_limit", "4g"),
        )
        # Note: we keep this, but our function-calling view_image tool returns
        # only a short textual summary (not actual image bytes).
        self.use_vision = False

        self.deps = SolverDeps(
            sandbox=self.sandbox,
            challenge_dir=challenge_dir,
            challenge_name=meta.name,
            workspace_dir="",
            use_vision=self.use_vision,
            cost_tracker=cost_tracker,
        )

        self.loop_detector = LoopDetector()
        self.tracer = SolverTracer(meta.name, self.model_id)
        self.agent_name = f"{meta.name}/{self.model_id}"

        self._tool_defs: dict[str, _ToolDef] = {}
        self._messages: list[dict[str, Any]] = []
        self._step_count = 0

        self._confirmed = False
        self._flag: str | None = None
        self._findings: str = ""

        self._tools_enabled = True

    async def start(self) -> None:
        if not self.sandbox._container:
            await self.sandbox.start()
        self.deps.workspace_dir = self.sandbox.workspace_dir

        arch_result = await self.sandbox.exec("uname -m", timeout_s=10)
        container_arch = arch_result.stdout.strip() or "unknown"

        attachments = self._list_challenge_attachments()
        system_prompt = build_prompt(self.meta, attachments, container_arch=container_arch)

        self._messages = [{"role": "system", "content": system_prompt}]
        self._build_tools()

        self.tracer.event("start", challenge=self.meta.name, model=self.model_id)
        logger.info("[%s] Solver started", self.agent_name)

    def _list_challenge_attachments(self) -> list[str]:
        # Reuse the same helper used by prompts.py (kept local to avoid circular deps).
        from backend.prompts import list_challenge_attachments

        return list_challenge_attachments(self.challenge_dir)

    def _build_tools(self) -> None:
        # Tool schemas are OpenAI-compatible JSON Schema for function calling.
        # We omit `tool_choice` at the HTTP layer to avoid OpenRouter routing 404s.

        async def _bash(command: str, timeout_seconds: int = 60) -> str:
            result = await do_bash(self.sandbox, command, timeout_seconds=timeout_seconds)
            return result

        async def _read_file(path: str) -> str:
            return await do_read_file(self.sandbox, path)

        async def _write_file(path: str, content: str) -> str:
            return await do_write_file(self.sandbox, path, content)

        async def _list_files(path: str = "/challenge/challenge") -> str:
            return await do_list_files(self.sandbox, path=path)

        async def _web_fetch(url: str, method: str = "GET", body: str = "") -> str:
            return await do_web_fetch(url=url, method=method, body=body)

        async def _webhook_create() -> str:
            return await do_webhook_create()

        async def _webhook_get_requests(uuid: str) -> str:
            return await do_webhook_get_requests(uuid)

        async def _check_findings() -> str:
            if not self.deps.message_bus:
                return "No message bus available."
            return await do_check_findings(self.deps.message_bus, self.deps.model_spec)

        async def _notify_coordinator(message: str) -> str:
            if not self.deps.notify_coordinator:
                return "No coordinator connected."
            return await self.deps.notify_coordinator(message)

        async def _submit_flag(flag: str) -> str:
            flag = flag.strip()
            if self.deps.no_submit:
                return f'DRY RUN — would accept "{flag}" but --no-submit is set.'

            if self.deps.submit_fn:
                display, is_confirmed = await self.deps.submit_fn(flag)
            else:
                display, is_confirmed = await do_submit_flag(flag)

            if is_confirmed:
                self._confirmed = True
                self._flag = flag
            return display

        async def _view_image(filename: str) -> str:
            # Our custom loop does not support passing raw image bytes to the model.
            # Provide a useful text summary of what the image tool can read.
            result = await do_view_image(self.sandbox, filename, use_vision=True)
            if isinstance(result, tuple):
                data, media_type = result
                return f"IMAGE SUMMARY: {filename} (mime={media_type}, bytes={len(data)})"
            return str(result)

        tools: list[_ToolDef] = [
            _ToolDef(
                name="bash",
                description="Execute a shell command inside the sandbox.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout_seconds": {"type": "integer", "description": "Command timeout"},
                    },
                    "required": ["command"],
                },
                handler=_bash,
            ),
            _ToolDef(
                name="read_file",
                description="Read a file from the sandbox.",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                handler=_read_file,
            ),
            _ToolDef(
                name="write_file",
                description="Write a text file into the sandbox.",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
                handler=_write_file,
            ),
            _ToolDef(
                name="list_files",
                description="List files inside the sandbox.",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                handler=_list_files,
            ),
            _ToolDef(
                name="web_fetch",
                description="Fetch a URL from the host (web challenges).",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["url"],
                },
                handler=_web_fetch,
            ),
            _ToolDef(
                name="webhook_create",
                description="Create a webhook.site token for out-of-band callbacks.",
                parameters_schema={"type": "object", "properties": {}},
                handler=_webhook_create,
            ),
            _ToolDef(
                name="webhook_get_requests",
                description="Fetch any requests received for a webhook token.",
                parameters_schema={
                    "type": "object",
                    "properties": {"uuid": {"type": "string"}},
                    "required": ["uuid"],
                },
                handler=_webhook_get_requests,
            ),
            _ToolDef(
                name="check_findings",
                description="Read unread findings from sibling agents (if available).",
                parameters_schema={"type": "object", "properties": {}},
                handler=_check_findings,
            ),
            _ToolDef(
                name="notify_coordinator",
                description="Send a message to the coordinator (if in coordinator mode).",
                parameters_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=_notify_coordinator,
            ),
            _ToolDef(
                name="submit_flag",
                description="Submit the discovered flag for local confirmation.",
                parameters_schema={
                    "type": "object",
                    "properties": {"flag": {"type": "string"}},
                    "required": ["flag"],
                },
                handler=_submit_flag,
            ),
            _ToolDef(
                name="view_image",
                description="Load an image file and return a short text summary.",
                parameters_schema={
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
                handler=_view_image,
            ),
        ]

        self._tool_defs = {t.name: t for t in tools}

    def bump(self, insights: str) -> None:
        self.loop_detector.reset()
        self._messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous attempt did not find the flag. "
                    "Here are insights from other agents:\n\n"
                    f"{insights}\n\n"
                    "Use these insights to try a different approach. "
                    "Do NOT repeat what has already been tried."
                ),
            }
        )
        self.tracer.event("bump", insights=insights[:500])
        logger.info("[%s] Bumped with sibling insights", self.agent_name)

    async def run_until_done_or_gave_up(self) -> SolverResult:
        if not self._messages:
            await self.start()

        # OpenRouter request loop
        t0 = time.monotonic()
        total_tool_calls = 0
        appended_initial_prompt = any(m.get("role") == "user" for m in self._messages)

        # Reduce load and the chance of repeatedly hitting rate limits.
        max_tool_calls = 12

        debug_model_substr = os.getenv("CTF_AGENT_DEBUG_MODEL", "").strip()
        debug_enabled = bool(debug_model_substr) and (debug_model_substr in self.model_spec)
        debug_enabled = debug_enabled or bool(getattr(self.settings, "always_debug_single_model", False))
        while not self.cancel_event.is_set():
            # Add an initial user prompt if we don't already have one (bump() may add one).
            if not appended_initial_prompt:
                self._messages.append({"role": "user", "content": "Solve this CTF challenge."})
                appended_initial_prompt = True

            request_body: dict[str, Any] = {
                "model": self.model_id,
                "messages": self._messages[-200:],
                "reasoning": {"enabled": True},
            }

            if self._tools_enabled:
                request_body["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters_schema,
                        },
                    }
                    for t in self._tool_defs.values()
                ]

            keys = self.settings.get_openrouter_keys()
            url = "https://openrouter.ai/api/v1/chat/completions"

            # Retry for rate limits and transient network errors.
            retries = 0
            max_retries = 6
            backoff_s = 3.0
            auth_failures: set[str] = set()
            rate_limited_keys: set[str] = set()
            while True:
                try:
                    key = next_openrouter_key(keys)
                    headers = {"Authorization": f"Bearer {key}"}
                    if debug_enabled:
                        masked = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
                        print(f"[DEBUG {self.model_id}] using key: {masked}")
                    async with httpx.AsyncClient(timeout=180.0) as client:
                        resp = await client.post(url, headers=headers, json=request_body)
                    resp.raise_for_status()
                    data = resp.json()
                    break  # success

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else None
                    body_msg = ""
                    try:
                        body_msg = (
                            json.dumps(e.response.json(), ensure_ascii=False)[:500]
                            if e.response is not None
                            else ""
                        )
                    except Exception:
                        body_msg = str(e)[:500]

                    # Auth failure: try other keys first (if available), then fail clearly.
                    if status in (401, 403):
                        auth_failures.add(key)
                        if len(auth_failures) < len(keys):
                            logger.warning(
                                "[%s] OpenRouter auth failed for one key (%s); trying next key",
                                self.agent_name,
                                status,
                            )
                            retries += 1
                            await asyncio.sleep(0.2)
                            continue
                        self._findings = (
                            f"OpenRouter authentication failed (HTTP {status}). "
                            "All configured API keys were rejected. "
                            "Check OPENROUTER_API_KEY / OPENROUTER_API_KEYS."
                        )
                        self.tracer.event("error", error=self._findings)
                        return self._result(QUOTA_ERROR, run_cost=None, run_steps=self._step_count)

                    if status == 429:
                        rate_limited_keys.add(key)
                        # If we still have untried keys, switch keys immediately
                        # instead of sleeping on the current limited key.
                        if len(rate_limited_keys) < len(keys):
                            logger.warning(
                                "[%s] OpenRouter 429 on one key; trying next key immediately",
                                self.agent_name,
                            )
                            retries += 1
                            continue

                        # All keys are currently rate limited; only now back off.
                        if retries < max_retries:
                            retries += 1
                            retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                            wait_s: float | None = None
                            if retry_after:
                                try:
                                    wait_s = float(retry_after)
                                except ValueError:
                                    wait_s = None
                            if wait_s is None:
                                wait_s = backoff_s
                            wait_s = min(90.0, max(1.0, wait_s))
                            logger.warning(
                                "[%s] OpenRouter 429 on all keys — retry %d/%d in %.1fs: %s",
                                self.agent_name,
                                retries,
                                max_retries,
                                wait_s,
                                body_msg[:120],
                            )
                            await asyncio.sleep(wait_s)
                            backoff_s *= 1.7
                            rate_limited_keys.clear()
                            continue

                    # Retries exhausted or non-429
                    if status == 429:
                        if "free-models-per-day" in body_msg or "per day" in body_msg.lower():
                            logger.warning(
                                "[%s] OpenRouter free-tier daily quota exhausted for available keys/accounts",
                                self.agent_name,
                            )
                        logger.warning(
                            "[%s] Final 429 body: %s",
                            self.agent_name,
                            body_msg[:300],
                        )
                        self._findings = f"OpenRouter 429 after retries: {body_msg}"
                        self.tracer.event("error", error=self._findings)
                        return self._result(QUOTA_ERROR, run_cost=None, run_steps=self._step_count)

                    logger.error(
                        "[%s] OpenRouter HTTP error (%s): %s",
                        self.agent_name,
                        status,
                        body_msg[:120],
                        exc_info=True,
                    )
                    self._findings = f"HTTP {status}: {body_msg}"
                    self.tracer.event("error", error=self._findings)
                    return self._result(ERROR, run_cost=None, run_steps=self._step_count)

                except (httpx.ReadTimeout, httpx.ConnectError) as e:
                    if retries < max_retries:
                        retries += 1
                        wait_s = min(90.0, backoff_s)
                        logger.warning(
                            "[%s] OpenRouter network error — retry %d/%d in %.1fs: %s",
                            self.agent_name,
                            retries,
                            max_retries,
                            wait_s,
                            str(e)[:120],
                        )
                        await asyncio.sleep(wait_s)
                        backoff_s *= 1.7
                        continue

                    self._findings = f"Network error after retries: {e}"
                    self.tracer.event("error", error=str(e))
                    return self._result(QUOTA_ERROR, run_cost=None, run_steps=self._step_count)

                except Exception as e:
                    logger.error("[%s] OpenRouter error: %s", self.agent_name, e, exc_info=True)
                    self._findings = f"Error: {e}"
                    self.tracer.event("error", error=str(e))
                    return self._result(ERROR, run_cost=None, run_steps=self._step_count)

            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            cache_read_tokens = int(usage.get("cache_read_tokens") or 0)

            self.cost_tracker.record_tokens(
                agent_name=self.agent_name,
                model_name=self.model_id,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cache_read_tokens=cache_read_tokens,
                provider_spec=provider_from_spec(self.model_spec),
                duration_seconds=max(0.0, time.monotonic() - t0),
            )

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}

            # Remove the appended user_prompt before re-adding assistant reply
            # (we want the loop prompt to stay, but tool loop will rely on assistant message).
            assistant_content = msg.get("content") or ""
            reasoning_details = msg.get("reasoning_details")
            tool_calls = msg.get("tool_calls") or []

            if debug_enabled:
                print("\n" + "=" * 80)
                print(f"[DEBUG {self.model_id}] assistant_content:\n{assistant_content[:1500]}")
                if tool_calls:
                    print(
                        f"[DEBUG {self.model_id}] tool_calls:\n"
                        f"{json.dumps(tool_calls, ensure_ascii=False)[:2000]}"
                    )
                else:
                    print(f"[DEBUG {self.model_id}] tool_calls: none")
                print("=" * 80)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if reasoning_details is not None:
                assistant_msg["reasoning_details"] = reasoning_details
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            self._messages.append(assistant_msg)
            self.tracer.model_response(assistant_content[:500], self._step_count)

            # Final content path (no tool calls)
            if not tool_calls:
                if assistant_content:
                    m = FlagPattern.search(assistant_content)
                    if m:
                        self._flag = m.group(1).strip().splitlines()[0].strip()
                        self._findings = f"Flag found via model output: {self._flag}"
                if self._confirmed and self._flag:
                    return self._result(FLAG_FOUND)
                return self._result(GAVE_UP)

            # Tool execution path
            for tc in tool_calls:
                if self.cancel_event.is_set():
                    return self._result(CANCELLED)

                total_tool_calls += 1
                self._step_count += 1
                tool_name = tc.get("function", {}).get("name")
                tool_call_id = tc.get("id") or ""
                raw_args = tc.get("function", {}).get("arguments") or "{}"
                try:
                    tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    tool_args = {}

                if tool_name not in self._tool_defs:
                    tool_result = f"Unknown tool requested: {tool_name}"
                else:
                    self.tracer.tool_call(tool_name, tool_args, self._step_count)

                    loop_status = self.loop_detector.check(tool_name, tool_args)
                    if loop_status == "break":
                        self.tracer.event("loop_break", tool=tool_name, step=self._step_count)
                        tool_result = LOOP_WARNING_MESSAGE
                    else:
                        try:
                            tool_result = await self._tool_defs[tool_name].handler(**tool_args)
                        except Exception as e:
                            tool_result = f"Tool error ({tool_name}): {e}"

                    tool_result = str(tool_result)
                    if loop_status == "warn":
                        tool_result = f"{tool_result}\n\n{LOOP_WARNING_MESSAGE}"

                    # Auto-inject sibling findings every 5 tool calls
                    if (
                        total_tool_calls % 5 == 0
                        and self.deps.message_bus
                        and isinstance(tool_result, str)
                        and tool_result.strip()
                    ):
                        findings_text = await do_check_findings(self.deps.message_bus, self.model_spec)
                        if findings_text and "No new findings" not in findings_text:
                            tool_result = f"{tool_result}\n\n---\n{findings_text}"
                            self.tracer.event("findings_injected", step=self._step_count)

                    self.tracer.tool_result(tool_name, tool_result, self._step_count)

                    if debug_enabled:
                        print(f"[DEBUG {self.model_id}] tool_result {tool_name}: {tool_result[:1200]}")

                    # Flag confirmation detection
                    if tool_name == "submit_flag" and any(m in tool_result for m in CORRECT_MARKERS):
                        # The tool itself sets _confirmed/_flag.
                        pass

                # Append tool result message
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result,
                    }
                )

                if self._confirmed and self._flag:
                    return self._result(FLAG_FOUND)

                if total_tool_calls >= max_tool_calls:
                    return self._result(GAVE_UP)

            # Continue outer while loop; next OpenRouter request will see tool results.

        return self._result(CANCELLED)

    def _result(self, status: str, run_cost: float | None = None, run_steps: int | None = None) -> SolverResult:
        agent_cost = self.cost_tracker.by_agent.get(self.agent_name)
        cost = agent_cost.cost_usd if agent_cost else 0.0
        self.tracer.event(
            "finish",
            status=status,
            flag=self._flag,
            confirmed=self._confirmed,
            cost_usd=round(run_cost if run_cost is not None else cost, 4),
        )
        return SolverResult(
            flag=self._flag,
            status=status,
            findings_summary=self._findings[:2000],
            step_count=run_steps if run_steps is not None else self._step_count,
            cost_usd=run_cost if run_cost is not None else cost,
            log_path=self.tracer.path,
        )

    async def stop(self) -> None:
        self.tracer.event("stop", step_count=self._step_count)
        self.tracer.close()
        if self._owns_sandbox and self.sandbox:
            await self.sandbox.stop()

