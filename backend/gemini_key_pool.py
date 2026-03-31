"""Round-robin Gemini API key selection."""

from __future__ import annotations

import threading
from itertools import count

_LOCK = threading.Lock()
_COUNTER = count(0)
_LAST_KEYS: tuple[str, ...] = ()


def next_gemini_key(keys: list[str]) -> str:
    if not keys:
        raise RuntimeError("No Gemini API keys configured")
    key_tuple = tuple(keys)
    with _LOCK:
        global _LAST_KEYS
        if _LAST_KEYS != key_tuple:
            _LAST_KEYS = key_tuple
        idx = next(_COUNTER) % len(key_tuple)
        return key_tuple[idx]

