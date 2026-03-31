"""Per-model solver agent using direct Gemini API."""

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
from backend.gemini_key_pool import next_gemini_key
from backend.loop_detect import LOOP_WARNING_MESSAGE, LoopDetector
from backend.models import model_id_from_spec, provider_from_spec
from backend.prompts import ChallengeMeta, build_prompt, list_challenge_attachments
from backend.sandbox import DockerSandbox
from backend.solver_base import CANCELLED, CORRECT_MARKERS, ERROR, FLAG_FOUND, GAVE_UP, QUOTA_ERROR, SolverResult
from backend.tools.core import (
    do_bash,
    do_check_findings,
    do_list_files,
    do_read_file,
    do_submit_flag,
    do_view_image,
    do_web_fetch,
    do_webhook_create,
    do_webhook_get_requests,
    do_write_file,
)
from backend.tracing import SolverTracer

logger = logging.getLogger(__name__)
FlagPattern = re.compile(r"FLAG\s*:\s*(.+)", re.IGNORECASE)

ToolHandler = Callable[..., Awaitable[str]]

_MAX_TEXT_ONLY_NUDGES = 5
_GEMINI_TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})
_GEMINI_MAX_HTTP_ATTEMPTS = 28


def _gemini_quota_cooldown_seconds(payload: dict[str, Any] | str | None) -> float | None:
    """If error body is the 'exceeded quota / wait longer' shape, return seconds to sleep (default 20)."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        err = payload.get("error")
        if not isinstance(err, dict):
            return None
        text = str(err.get("message", "") or "")
    else:
        text = str(payload)
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if not t:
        return None
    if "exceeded your current quota" in t or ("exceeded" in t and "quota" in t):
        m = re.search(r"(\d+)\s*(?:seconds?|secs?\b)", t)
        if m:
            return float(min(max(int(m.group(1)), 5), 120))
        return 20.0
    if "wait longer" in t:
        return 20.0
    return None


def _collect_function_calls(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        fc = p.get("functionCall") or p.get("function_call")
        if isinstance(fc, dict):
            out.append(fc)
    return out


def _coerce_function_args(fc: dict[str, Any]) -> dict[str, Any]:
    raw = fc.get("args")
    if raw is None:
        raw = fc.get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            parsed = {}
        raw = parsed
    if not isinstance(raw, dict):
        return {}
    return raw


@dataclass
class _ToolDef:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: ToolHandler


def _gemini_function_decl(t: _ToolDef) -> dict[str, Any]:
    """Omit empty `parameters` — Gemini rejects some zero-arg schemas."""
    decl: dict[str, Any] = {"name": t.name, "description": t.description}
    ps = t.parameters_schema
    if not ps:
        return decl
    if ps.get("type") == "object" and isinstance(ps.get("properties"), dict) and len(ps["properties"]) == 0:
        return decl
    decl["parameters"] = ps
    return decl


class GeminiSolver:
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
        chain_raw = (getattr(settings, "gemini_rotate_chain", None) or "").strip()
        if chain_raw:
            ids = [p.strip() for p in chain_raw.split(",") if p.strip()]
            self._gemini_rotate_ids: tuple[str, ...] = tuple(ids)
            self._gemini_rotate_index = 0
            if len(self._gemini_rotate_ids) >= 1:
                self.model_id = self._gemini_rotate_ids[0]
        else:
            self._gemini_rotate_ids = ()
            self._gemini_rotate_index = 0
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
        self._contents: list[dict[str, Any]] = []
        self._step_count = 0
        self._confirmed = False
        self._flag: str | None = None
        self._findings: str = ""
        self._system_prompt = ""
        self._gemini_last_error_status: str = ERROR

    def _rotate_gemini_model(self) -> bool:
        if len(self._gemini_rotate_ids) < 2:
            return False
        self._gemini_rotate_index = (self._gemini_rotate_index + 1) % len(self._gemini_rotate_ids)
        self.model_id = self._gemini_rotate_ids[self._gemini_rotate_index]
        return True

    async def _gemini_post_with_retries(
        self,
        request_body: dict[str, Any],
        keys: list[str],
        debug_enabled: bool,
    ) -> dict[str, Any] | None:
        """POST generateContent; rotate models on transient HTTP errors and network timeouts."""
        self._gemini_last_error_status = ERROR
        last_detail = ""
        backoff_s = 1.2
        for attempt in range(_GEMINI_MAX_HTTP_ATTEMPTS):
            key = next_gemini_key(keys)
            if debug_enabled:
                masked = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
                print(f"[DEBUG {self.model_id}] using Gemini key: {masked}")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent"
            headers = {"Content-Type": "application/json", "X-goog-api-key": key}

            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(url, headers=headers, json=request_body)
            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                last_detail = f"network: {e}"
                logger.warning("[%s] Gemini %s — retry %s", self.agent_name, last_detail, attempt + 1)
                if self._rotate_gemini_model():
                    logger.warning("[%s] Switched model to %s after network error", self.agent_name, self.model_id)
                await asyncio.sleep(min(backoff_s * (1.35**min(attempt, 12)), 45.0))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception as e:
                    self._findings = f"Gemini: invalid JSON in 200 response: {e}"
                    return None

            parsed_err: dict[str, Any] | None = None
            try:
                parsed_err = resp.json()
                body = json.dumps(parsed_err, ensure_ascii=False)[:900]
            except Exception:
                body = resp.text[:900]
                parsed_err = None
            code = resp.status_code
            last_detail = f"HTTP {code}: {body}"
            self._findings = f"Gemini {last_detail}"

            if code in (401,):
                self._gemini_last_error_status = QUOTA_ERROR
                logger.warning("[%s] %s", self.agent_name, self._findings[:600])
                return None

            if code in (403,):
                self._gemini_last_error_status = QUOTA_ERROR
                logger.warning("[%s] %s", self.agent_name, self._findings[:600])
                return None

            if code in _GEMINI_TRANSIENT_HTTP:
                self._gemini_last_error_status = QUOTA_ERROR
                quota_wait: float | None = None
                if code == 429:
                    quota_wait = _gemini_quota_cooldown_seconds(parsed_err)
                    if quota_wait is None:
                        quota_wait = _gemini_quota_cooldown_seconds(body)
                    if quota_wait is not None:
                        logger.warning(
                            "[%s] Gemini quota message — sleeping %.0fs before retry/rotate (attempt %s/%s)",
                            self.agent_name,
                            quota_wait,
                            attempt + 1,
                            _GEMINI_MAX_HTTP_ATTEMPTS,
                        )
                        await asyncio.sleep(quota_wait)
                rotated = self._rotate_gemini_model()
                if rotated:
                    logger.warning(
                        "[%s] Gemini HTTP %s — switching to model %s (attempt %s/%s)",
                        self.agent_name,
                        code,
                        self.model_id,
                        attempt + 1,
                        _GEMINI_MAX_HTTP_ATTEMPTS,
                    )
                else:
                    logger.warning(
                        "[%s] Gemini HTTP %s — retry %s/%s (no rotate chain)",
                        self.agent_name,
                        code,
                        attempt + 1,
                        _GEMINI_MAX_HTTP_ATTEMPTS,
                    )
                if quota_wait is not None:
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(min(backoff_s * (1.35**min(attempt, 12)), 45.0))
                continue

            logger.warning("[%s] %s", self.agent_name, self._findings[:600])
            self._gemini_last_error_status = ERROR
            return None

        self._findings = f"Gemini: exhausted retries. Last: {last_detail[:800]}"
        self._gemini_last_error_status = ERROR
        logger.warning("[%s] %s", self.agent_name, self._findings[:500])
        return None

    async def start(self) -> None:
        if not self.sandbox._container:
            await self.sandbox.start()
        self.deps.workspace_dir = self.sandbox.workspace_dir
        arch_result = await self.sandbox.exec("uname -m", timeout_s=10)
        container_arch = arch_result.stdout.strip() or "unknown"
        attachments = list_challenge_attachments(self.challenge_dir)
        self._system_prompt = build_prompt(self.meta, attachments, container_arch=container_arch)
        self._contents = [{"role": "user", "parts": [{"text": "Solve this CTF challenge."}]}]
        self._build_tools()
        self.tracer.event("start", challenge=self.meta.name, model=self.model_id)
        if self._gemini_rotate_ids:
            logger.info("[%s] Solver started (Gemini rotate: %s)", self.agent_name, ", ".join(self._gemini_rotate_ids))
        else:
            logger.info("[%s] Solver started", self.agent_name)

    def _build_tools(self) -> None:
        async def _bash(command: str, timeout_seconds: int = 60) -> str:
            return await do_bash(self.sandbox, command, timeout_seconds=timeout_seconds)

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
            await self.deps.notify_coordinator(message)
            return "Message sent."

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
            result = await do_view_image(self.sandbox, filename, use_vision=True)
            if isinstance(result, tuple):
                data, media_type = result
                return f"IMAGE SUMMARY: {filename} (mime={media_type}, bytes={len(data)})"
            return str(result)

        tools = [
            _ToolDef("bash", "Execute a shell command inside sandbox.", {"type": "object", "properties": {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, "required": ["command"]}, _bash),
            _ToolDef("read_file", "Read file from sandbox.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, _read_file),
            _ToolDef("write_file", "Write text file.", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, _write_file),
            _ToolDef("list_files", "List files in directory.", {"type": "object", "properties": {"path": {"type": "string"}}}, _list_files),
            _ToolDef("web_fetch", "Fetch URL from host.", {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}, "body": {"type": "string"}}, "required": ["url"]}, _web_fetch),
            _ToolDef("webhook_create", "Create webhook token.", {"type": "object", "properties": {}}, _webhook_create),
            _ToolDef("webhook_get_requests", "Get webhook requests.", {"type": "object", "properties": {"uuid": {"type": "string"}}, "required": ["uuid"]}, _webhook_get_requests),
            _ToolDef("check_findings", "Read sibling findings.", {"type": "object", "properties": {}}, _check_findings),
            _ToolDef("notify_coordinator", "Notify coordinator.", {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}, _notify_coordinator),
            _ToolDef("submit_flag", "Submit candidate flag.", {"type": "object", "properties": {"flag": {"type": "string"}}, "required": ["flag"]}, _submit_flag),
            _ToolDef("view_image", "View image and summarize.", {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}, _view_image),
        ]
        self._tool_defs = {t.name: t for t in tools}

    def bump(self, insights: str) -> None:
        self.loop_detector.reset()
        self._contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Your previous attempt did not find the flag. Here are insights:\n\n"
                            f"{insights}\n\n"
                            "Use them and try a different approach."
                        )
                    }
                ],
            }
        )

    async def run_until_done_or_gave_up(self) -> SolverResult:
        if not self._contents:
            await self.start()

        max_tool_calls = 12
        total_tool_calls = 0
        debug_model_substr = os.getenv("CTF_AGENT_DEBUG_MODEL", "").strip()
        debug_enabled = bool(debug_model_substr) and (debug_model_substr in self.model_spec)
        debug_enabled = debug_enabled or bool(getattr(self.settings, "always_debug_single_model", False))

        text_only_rounds = 0
        while not self.cancel_event.is_set():
            keys = self.settings.get_gemini_keys()
            if not keys:
                self._findings = "No Gemini API key configured (GEMINI_API_KEY / GEMINI_API_KEYS)."
                return self._result(QUOTA_ERROR)

            request_body = {
                "systemInstruction": {"parts": [{"text": self._system_prompt}]},
                "contents": self._contents[-200:],
                "tools": [{"functionDeclarations": [_gemini_function_decl(t) for t in self._tool_defs.values()]}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            }

            data = await self._gemini_post_with_retries(request_body, keys, debug_enabled)
            if data is None:
                return self._result(self._gemini_last_error_status)

            usage = data.get("usageMetadata") or {}
            self.cost_tracker.record_tokens(
                agent_name=self.agent_name,
                model_name=self.model_id,
                input_tokens=int(usage.get("promptTokenCount") or 0),
                output_tokens=int(usage.get("candidatesTokenCount") or 0),
                cache_read_tokens=0,
                provider_spec=provider_from_spec(self.model_spec),
                duration_seconds=0.0,
            )

            candidates = data.get("candidates") or []
            if not candidates:
                pf = data.get("promptFeedback") or {}
                self._findings = f"Gemini returned no candidates (blocked or empty). promptFeedback={json.dumps(pf, ensure_ascii=False)[:800]}"
                logger.warning("[%s] %s", self.agent_name, self._findings)
                return self._result(ERROR)

            cand = candidates[0]
            finish = cand.get("finishReason") or ""
            if finish in (
                "SAFETY",
                "RECITATION",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "MALFORMED_FUNCTION_CALL",
            ):
                self._findings = f"Gemini stopped ({finish}). First candidate: {json.dumps(cand, ensure_ascii=False)[:1200]}"
                logger.warning("[%s] %s", self.agent_name, self._findings[:500])
                return self._result(ERROR)

            content = cand.get("content") or {}
            parts = content.get("parts") or []
            text_parts: list[str] = []
            function_calls = _collect_function_calls(parts)

            for p in parts:
                if "text" in p:
                    text_parts.append(p.get("text") or "")

            assistant_text = "\n".join([t for t in text_parts if t]).strip()
            if debug_enabled:
                print("\n" + "=" * 80)
                print(f"[DEBUG {self.model_id}] assistant_content:\n{assistant_text[:1500]}")
                print(f"[DEBUG {self.model_id}] function_calls: {json.dumps(function_calls, ensure_ascii=False)[:2000]}")
                print("=" * 80)
            self.tracer.model_response(assistant_text[:500], self._step_count)

            # Record model response in contents (preserve thought_signature / all part fields).
            self._contents.append({"role": "model", "parts": parts})

            if not function_calls:
                if assistant_text:
                    m = FlagPattern.search(assistant_text)
                    if m:
                        self._flag = m.group(1).strip().splitlines()[0].strip()
                        self._findings = f"Flag found via model output: {self._flag}"
                if self._confirmed and self._flag:
                    return self._result(FLAG_FOUND)
                if text_only_rounds < _MAX_TEXT_ONLY_NUDGES:
                    text_only_rounds += 1
                    self._contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        "You must call the provided tools to solve this challenge. "
                                        "Do not answer with only plain text. Start with `list_files` "
                                        "or `bash` to inspect `/challenge/challenge` and `/challenge/workspace`."
                                    )
                                }
                            ],
                        }
                    )
                    continue
                self._findings = (
                    "Gemini replied without calling tools after several nudges "
                    "(or only returned a flag pattern in text without submit_flag)."
                )
                return self._result(GAVE_UP)

            text_only_rounds = 0

            # Execute tool calls; send one user turn with all functionResponse parts (Gemini expects this).
            response_parts: list[dict[str, Any]] = []
            for fc in function_calls:
                total_tool_calls += 1
                self._step_count += 1
                name = fc.get("name")
                args = _coerce_function_args(fc)
                fc_id = fc.get("id")

                if name not in self._tool_defs:
                    result = f"Unknown tool requested: {name}"
                else:
                    loop_status = self.loop_detector.check(name, args)
                    if loop_status == "break":
                        result = LOOP_WARNING_MESSAGE
                    else:
                        try:
                            result = await self._tool_defs[name].handler(**args)
                        except Exception as e:
                            result = f"Tool error ({name}): {e}"
                        result = str(result)
                        if loop_status == "warn":
                            result = f"{result}\n\n{LOOP_WARNING_MESSAGE}"
                    if debug_enabled:
                        print(f"[DEBUG {self.model_id}] tool_result {name}: {result[:1200]}")
                    self.tracer.tool_result(name, result, self._step_count)
                    if name == "submit_flag" and any(m in result for m in CORRECT_MARKERS):
                        pass

                fr_body: dict[str, Any] = {
                    "name": name,
                    "response": {"result": result},
                }
                if fc_id:
                    fr_body["id"] = fc_id
                response_parts.append({"functionResponse": fr_body})

                if self._confirmed and self._flag:
                    self._contents.append({"role": "user", "parts": response_parts})
                    return self._result(FLAG_FOUND)
                if total_tool_calls >= max_tool_calls:
                    self._contents.append({"role": "user", "parts": response_parts})
                    return self._result(GAVE_UP)

            if response_parts:
                self._contents.append({"role": "user", "parts": response_parts})

        return self._result(CANCELLED)

    def _result(self, status: str) -> SolverResult:
        agent_usage = self.cost_tracker.by_agent.get(self.agent_name)
        cost = agent_usage.cost_usd if agent_usage else 0.0
        self.tracer.event("finish", status=status, flag=self._flag, confirmed=self._confirmed, cost_usd=round(cost, 4))
        return SolverResult(
            flag=self._flag,
            status=status,
            findings_summary=self._findings[:2000],
            step_count=self._step_count,
            cost_usd=cost,
            log_path=self.tracer.path,
        )

    async def stop(self) -> None:
        self.tracer.event("stop", step_count=self._step_count)
        self.tracer.close()
        if self._owns_sandbox and self.sandbox:
            await self.sandbox.stop()

