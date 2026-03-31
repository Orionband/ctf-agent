"""System prompt builder + ChallengeMeta — load from a folder (no YAML required)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.tools.core import IMAGE_EXTS_FOR_VISION as IMAGE_EXTS

# Text files we read for the challenge statement (first match wins)
_DESCRIPTION_FILES = (
    "challenge.txt",
    "description.txt",
    "challenge.md",
    "README.md",
    "readme.md",
    "readme.txt",
)

# Optional hint files (in addition to metadata.yml hints)
_HINT_FILES = ("hints.txt", "hint.txt", "hints.md", "hint.md")

_IGNORE_DIR_NAMES = frozenset({".git", "__pycache__", ".venv", "node_modules"})


@dataclass
class ChallengeMeta:
    name: str = "Unknown"
    category: str = ""
    value: int = 0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    connection_info: str = ""
    hints: list[dict[str, Any]] = field(default_factory=list)
    solves: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChallengeMeta:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            name=data.get("name", "Unknown"),
            category=data.get("category", ""),
            value=data.get("value", 0),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            connection_info=data.get("connection_info", ""),
            hints=data.get("hints", []),
            solves=data.get("solves", 0),
        )

    @classmethod
    def from_directory(cls, path: str | Path) -> ChallengeMeta:
        """Load challenge from a folder: optional metadata.yml + text files + hints + attachments."""
        root = Path(path).resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        folder_label = root.name.replace("_", " ").strip() or "challenge"
        data: dict[str, Any] = {}
        yml = root / "metadata.yml"
        if yml.is_file():
            with open(yml, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        name = str(data.get("name") or folder_label).strip() or folder_label
        category = str(data.get("category", "") or "")
        value = int(data.get("value", 0) or 0)
        tags = list(data.get("tags", []) or [])
        connection_info = str(data.get("connection_info", "") or "").strip()
        solves = int(data.get("solves", 0) or 0)

        description = str(data.get("description", "") or "").strip()
        if not description:
            for fname in _DESCRIPTION_FILES:
                p = root / fname
                if p.is_file():
                    description = p.read_text(encoding="utf-8", errors="replace").strip()
                    break

        if not description:
            description = (
                "_Add your challenge write-up to **challenge.txt** or **README.md** in this folder._"
            )

        hints: list[dict[str, Any]] = []
        raw_hints = data.get("hints", [])
        if isinstance(raw_hints, list):
            for h in raw_hints:
                if isinstance(h, dict) and h.get("content"):
                    hints.append({"content": str(h["content"])})
                elif isinstance(h, str) and h.strip():
                    hints.append({"content": h.strip()})

        hints_dir = root / "hints"
        if hints_dir.is_dir():
            for hp in sorted(hints_dir.glob("*")):
                if hp.is_file() and not hp.name.startswith("."):
                    text = hp.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        hints.append({"content": f"[{hp.name}] {text}"})

        for hf in _HINT_FILES:
            p = root / hf
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            if "\n\n" in text:
                for block in text.split("\n\n"):
                    b = block.strip()
                    if b:
                        hints.append({"content": b})
            else:
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        hints.append({"content": line})
            break

        conn_file = root / "connection.txt"
        if conn_file.is_file() and not connection_info:
            connection_info = conn_file.read_text(encoding="utf-8", errors="replace").strip()

        # If no explicit connection.txt, try to extract a usable connection string
        # from the statement / hints text (common for CTFs).
        if not connection_info:
            combined = description + "\n" + "\n".join(h.get("content", "") for h in hints[:30])
            url_match = re.search(r"https?://[^\\s)]+", combined, flags=re.IGNORECASE)
            if url_match:
                connection_info = url_match.group(0).strip()
            else:
                # Capture lines like: "nc host port"
                nc_match = re.search(r"(?m)^\\s*(nc\\s+[^\\n\\r]+)$", combined)
                if nc_match:
                    connection_info = nc_match.group(1).strip()

        return cls(
            name=name,
            category=category,
            value=value,
            description=description,
            tags=tags,
            connection_info=connection_info,
            hints=hints,
            solves=solves,
        )


def list_challenge_attachments(challenge_dir: str) -> list[str]:
    """Relative paths of all non-hidden files under the challenge folder (for prompts and tooling)."""
    root = Path(challenge_dir).resolve()
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if any(part in _IGNORE_DIR_NAMES for part in p.parts):
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        out.append(str(rel).replace("\\", "/"))
        if len(out) >= 250:
            break
    return out


def list_distfiles(challenge_dir: str) -> list[str]:
    """Backward-compatible name: list all attachment paths under the challenge directory."""
    return list_challenge_attachments(challenge_dir)


def _rewrite_connection_info(conn: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal for bridge networking."""
    if not conn:
        return conn
    conn = re.sub(r"\blocalhost\b", "host.docker.internal", conn)
    conn = re.sub(r"\b127\.0\.0\.1\b", "host.docker.internal", conn)
    return conn


def build_prompt(
    meta: ChallengeMeta,
    attachment_paths: list[str],
    container_arch: str = "unknown",
    has_named_tools: bool = True,
) -> str:
    """Build the system prompt.

    attachment_paths: relative paths under the challenge folder (mounted at /challenge/challenge/).
    """
    conn_info = _rewrite_connection_info(meta.connection_info.strip())

    lines: list[str] = [
        "You are an expert CTF solver. Find the real flag for the challenge below.",
        "",
    ]

    if conn_info:
        lines += [
            "> **FIRST ACTION REQUIRED**: Your very first tool call MUST connect to the service.",
            f"> Run: `{conn_info}` (use a heredoc or pwntools script as shown below).",
            "> Do NOT explore the sandbox filesystem first. The flag is on the service, not in the container.",
            "",
        ]

    lines += [
        "## Challenge",
        f"**Name**    : {meta.name}",
        f"**Category**: {meta.category or 'Unknown'}",
        f"**Points**  : {meta.value or '?'}",
        f"**Arch**    : {container_arch}",
    ]
    if meta.tags:
        lines.append(f"**Tags**    : {', '.join(meta.tags)}")
    lines += ["", "## Description", meta.description or "_No description provided._", ""]

    if conn_info:
        if re.match(r"^https?://", conn_info):
            hint = "This is a **web service**. Use `bash` with `curl`/`python3 requests`, or use `web_fetch`."
        elif conn_info.startswith("nc "):
            hint = (
                "This is a **TCP service**. Each `bash` call is a fresh process — "
                "use a heredoc to send multiple lines in one shot:\n"
                "```\n"
                f"{conn_info} <<'EOF'\ncommand1\ncommand2\nEOF\n"
                "```\n"
                "Or write a Python `socket` / `pwntools` script for stateful interaction."
            )
        else:
            hint = "Connect using the details above."
        lines += ["## Service Connection", "```", conn_info, "```", hint, ""]

    if attachment_paths:
        lines.append("## Files (read-only under `/challenge/challenge/`)")
        for rel in attachment_paths:
            name = Path(rel).name
            ext = Path(name).suffix.lower()
            is_img = ext in IMAGE_EXTS
            if is_img and has_named_tools:
                suffix = "  <- **IMAGE: call `view_image` immediately** (fix magic bytes first if corrupt)"
            elif is_img:
                suffix = "  <- **IMAGE: use `exiftool`, `steghide`, `zsteg`, `strings` via bash**"
            else:
                suffix = ""
            lines.append(f"- `/challenge/challenge/{rel}`{suffix}")
        lines.append("")

    visible_hints = [h for h in meta.hints if h.get("content")]
    if visible_hints:
        lines.append("## Hints")
        for h in visible_hints:
            lines.append(f"- {h['content']}")
        lines.append("")

    cat_lower = (meta.category or "").lower()
    if cat_lower in ("reverse", "reversing", "re", "pwn", "binary", "misc", "") or attachment_paths:
        lines += [
            "## Binary Analysis",
            "**pyghidra** is installed for decompilation. Use it via bash:",
            "```python",
            "import pyghidra",
            "with pyghidra.open_program('/challenge/challenge/binary') as flat_api:",
            "    listing = flat_api.currentProgram.getListing()",
            "    # Iterate functions, decompile, etc.",
            "```",
            "Also available: radare2 (`r2`), gdb, angr, capstone.",
            "",
        ]

    if has_named_tools:
        image_hint = "**Images: call `view_image` FIRST, before any other analysis.**"
        web_hint = "Web: fuzz params, check JS source, cookies, robots.txt. For XSS/SSRF: use `webhook_create`."
        submit_hint = "**Verify every candidate with `submit_flag`** before reporting."
    else:
        image_hint = "**Images: use `exiftool`, `steghide`, `zsteg`, `strings`, `xxd` via bash.**"
        web_hint = "Web: fuzz params, check JS source, cookies, robots.txt. For XSS/SSRF: use `curl` to webhook.site."
        submit_hint = "**Verify every candidate with `submit_flag '<flag>'`** (bash command) before reporting."

    lines += [
        "",
        "## Instructions",
        "**Use tools immediately. Do not describe — execute.**",
        "",
        "1. " + ("Connect to the service now." if conn_info else "Inspect challenge files under `/challenge/challenge/`."),
        "2. Keep using tools until you have the flag.",
        "   - If you need to run a binary/script from `/challenge/challenge/`, copy it to the writable workspace first:",
        "     - `cp /challenge/challenge/<file> /challenge/workspace/ && chmod +x /challenge/workspace/<file> && /challenge/workspace/<file>`",
        "3. **Be creative and thorough** — try the obvious path, then explore further:",
        "   - Hidden files, env vars, backup files, HTTP headers, error messages, timing, encoding tricks.",
        f"   - {image_hint}",
        f"   - {web_hint}",
        (
            "   - Crypto: identify algorithm, weak keys, nonce reuse, padding oracles. "
            "For RSA: use `RsaCtfTool`, sage ECM, or `cado-nfs`."
        ),
        "   - Pwn: `stty raw -echo` before launching vulnerable binaries over nc.",
        '4. **Ignore placeholder flags** — `CTF{flag}`, `CTF{placeholder}` are not real flags.',
        f"5. {submit_hint}",
        "6. Once CORRECT: output `FLAG: <value>` on its own line.",
        "7. Do not guess. Do not ask. Cover maximum surface area.",
    ]

    return "\n".join(lines)
