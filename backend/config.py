"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter (required for all LLM calls)
    openrouter_api_key: str = ""

    # Optional: coordinator uses this model spec (openrouter/...); empty = first DEFAULT_MODELS entry
    coordinator_model: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    max_attempts_per_challenge: int = 3
    container_memory_limit: str = "16g"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
