"""Provider factory for instantiating concrete provider adapters.

This module is the ONLY place that knows about concrete adapters.
The Agent Runtime and all business logic depend on interfaces only.
"""

from app.core.config import settings
from app.providers.llm.interface import LLMInterface
from app.providers.stt.interface import STTInterface
from app.providers.tts.interface import TTSInterface

# Supported provider names per type
SUPPORTED_STT_PROVIDERS = {"deepgram"}
SUPPORTED_LLM_PROVIDERS = {"openrouter"}
SUPPORTED_TTS_PROVIDERS = {"elevenlabs"}

# Curated OpenRouter LLM model catalogue for the Chat and Voice UIs.
# NOTE: Benchmark configurations are separate (app/benchmarks/configurations.py)
# and are intentionally NOT coupled to this list.

# Free / experimental models: zero cost, rate-limited, non-production.
LLM_FREE_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "google/gemini-2.0-flash-exp:free",
)

# Paid / PAYG models: current IDs for paid-account testing through the
# Chat and Voice flows. The stale "anthropic/claude-3.5-sonnet" entry was
# removed (provider errors; no longer a valid selectable model here).
LLM_PAID_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-5-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-chat-v3.1",
)


class ProviderError(Exception):
    """Raised when a provider cannot be instantiated."""


def get_stt_provider(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> STTInterface:
    """Instantiate an STT provider adapter by name.

    Args:
        provider: Provider name (e.g. 'deepgram'). Defaults to configured default.
        model: Model override.
        api_key: API key override (for testing).

    Returns:
        An STTInterface implementation.

    Raises:
        ProviderError: If the provider is unknown or not configured.
    """
    name = provider or settings.default_stt_provider

    if name == "deepgram":
        from app.providers.stt.deepgram import DeepgramAdapter

        return DeepgramAdapter(api_key=api_key, default_model=model)

    raise ProviderError(
        f"Unsupported STT provider: '{name}'. Supported: {sorted(SUPPORTED_STT_PROVIDERS)}"
    )


def get_llm_provider(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMInterface:
    """Instantiate an LLM provider adapter by name.

    Args:
        provider: Provider name (e.g. 'openrouter'). Defaults to configured default.
        model: Model override.
        api_key: API key override (for testing).

    Returns:
        An LLMInterface implementation.

    Raises:
        ProviderError: If the provider is unknown or not configured.
    """
    name = provider or settings.default_llm_provider

    if name == "openrouter":
        from app.providers.llm.openrouter import OpenRouterAdapter

        return OpenRouterAdapter(api_key=api_key, default_model=model)

    raise ProviderError(
        f"Unsupported LLM provider: '{name}'. Supported: {sorted(SUPPORTED_LLM_PROVIDERS)}"
    )


def get_tts_provider(
    provider: str | None = None,
    *,
    voice: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> TTSInterface:
    """Instantiate a TTS provider adapter by name.

    Args:
        provider: Provider name (e.g. 'elevenlabs'). Defaults to configured default.
        voice: Voice override.
        model: Model override.
        api_key: API key override (for testing).

    Returns:
        A TTSInterface implementation.

    Raises:
        ProviderError: If the provider is unknown or not configured.
    """
    name = provider or settings.default_tts_provider

    if name == "elevenlabs":
        from app.providers.tts.elevenlabs import ElevenLabsAdapter

        return ElevenLabsAdapter(
            api_key=api_key,
            default_voice=voice,
            default_model=model,
        )

    raise ProviderError(
        f"Unsupported TTS provider: '{name}'. Supported: {sorted(SUPPORTED_TTS_PROVIDERS)}"
    )


def get_available_providers() -> dict:
    """Return metadata about all available providers and their configuration status.

    This is safe to expose via API — no secrets are included.
    """
    return {
        "stt": [
            {
                "provider": "deepgram",
                "models": [settings.default_stt_model or "nova-3"],
                "configured": settings.is_provider_configured("deepgram"),
            },
        ],
        "llm": [
            {
                "provider": "openrouter",
                "default_model": settings.default_llm_model or "nvidia/nemotron-3.5-lightning:free",
                "models": [*LLM_FREE_MODELS, *LLM_PAID_MODELS],
                "paid_models": list(LLM_PAID_MODELS),
                "configured": settings.is_provider_configured("openrouter"),
            },
        ],
        "tts": [
            {
                "provider": "elevenlabs",
                "default_model": settings.default_tts_model or "eleven_flash_v2_5",
                "default_voice": settings.default_tts_voice or "JBFqnCBsd6RMkjVDRZzb",
                "models": ["eleven_flash_v2_5", "eleven_multilingual_v2", "eleven_v3"],
                "configured": settings.is_provider_configured("elevenlabs"),
            },
        ],
    }
