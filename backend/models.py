"""Model resolution — OpenRouter via pydantic-ai OpenRouterModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from backend.config import Settings

# Default model specs (provider is always openrouter)
DEFAULT_MODELS: list[str] = [
    "openrouter/qwen/qwen3.6-plus-preview:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/stepfun/step-3.5-flash:free",
]

CONTEXT_WINDOWS: dict[str, int] = {
    "qwen/qwen3.6-plus-preview:free": 200_000,
    "nvidia/nemotron-3-super-120b-a12b:free": 200_000,
    "stepfun/step-3.5-flash:free": 200_000,
}

VISION_MODELS: set[str] = {
    "qwen/qwen3.6-plus-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
}


def resolve_model(spec: str, settings: Settings) -> Model:
    """Resolve an 'openrouter/model_id' spec to a Pydantic AI Model."""
    provider = provider_from_spec(spec)
    if provider != "openrouter":
        raise ValueError(f"Only openrouter is supported; got {spec!r}")
    model_id = model_id_from_spec(spec)
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenRouterModel(
        model_id,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


def resolve_model_settings(spec: str) -> ModelSettings:
    """OpenRouter settings with reasoning enabled (reasoning_details preserved across turns)."""
    provider = spec.split("/", 1)[0]
    if provider != "openrouter":
        return ModelSettings(max_tokens=128_000)
    return OpenRouterModelSettings(
        max_tokens=128_000,
        openrouter_reasoning={"enabled": True},
    )


def model_id_from_spec(spec: str) -> str:
    """Model id as used by OpenRouter (everything after the provider prefix)."""
    parts = spec.split("/")
    if len(parts) >= 2 and parts[0] == "openrouter":
        return "/".join(parts[1:])
    if len(parts) >= 2:
        return "/".join(parts[1:])
    return spec


def provider_from_spec(spec: str) -> str:
    """Extract the provider from a spec (always openrouter for valid specs)."""
    return spec.split("/", 1)[0]


def effort_from_spec(spec: str) -> str | None:
    """Unused for OpenRouter defaults; kept for API compatibility."""
    return None


def supports_vision(spec: str) -> bool:
    return model_id_from_spec(spec) in VISION_MODELS


def context_window(spec: str) -> int:
    return CONTEXT_WINDOWS.get(model_id_from_spec(spec), 200_000)
