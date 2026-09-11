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
                "models": [
                    "nvidia/nemotron-3.5-lightning:free",
                    "google/gemma-4-31b-it:free",
                    "meta-llama/llama-3.1-8b-instruct:free",
                    "microsoft/phi-3-medium-128k-instruct:free",
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.0-flash-exp:free",
                ],
                "configured": settings.is_provider_configured("openrouter"),
            },
        ],
        "tts": [
            {
                "provider": "elevenlabs",
                "models": ["eleven_monolingual_v1", "eleven_multilingual_v2"],
                "configured": settings.is_provider_configured("elevenlabs"),
            },
        ],
    }
