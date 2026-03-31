"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

import re

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter (required for all LLM calls)
    openrouter_api_key: str = ""
    # Optional multi-key support. Comma or newline separated.
    # Example:
    # OPENROUTER_API_KEYS=sk-or-v1-key1,sk-or-v1-key2
    openrouter_api_keys: str = ""

    # Optional: coordinator uses this model spec (openrouter/...); empty = first DEFAULT_MODELS entry
    coordinator_model: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    max_attempts_per_challenge: int = 3
    container_memory_limit: str = "16g"

    # Debug
    always_debug_single_model: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_openrouter_keys(self) -> list[str]:
        """Get all configured OpenRouter keys (multi-key first, then single-key fallback)."""
        keys: list[str] = []
        if self.openrouter_api_keys:
            raw = self.openrouter_api_keys
            # Accept comma/newline/semicolon/space separated values.
            parts = re.split(r"[\s,;]+", raw)
            for p in parts:
                k = p.strip().strip('"').strip("'")
                if not k:
                    continue
                if k.startswith("sk-or-v1-"):
                    keys.append(k)
        if self.openrouter_api_key and self.openrouter_api_key.strip():
            key = self.openrouter_api_key.strip().strip('"').strip("'")
            if key not in keys:
                keys.append(key)
        # Preserve order, drop duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            deduped.append(k)
        return deduped
