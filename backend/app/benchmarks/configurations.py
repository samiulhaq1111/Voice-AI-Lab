"""Benchmark configuration definitions (Phase 5E).

Each configuration specifies a provider/model combination that can be
compared against other configurations running the same scenario.

Configurations are static and code-backed (no database table).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """A provider/model combination for benchmark comparison."""

    configuration_id: str
    name: str
    description: str
    llm_provider: str
    llm_model: str
    tts_provider: str = "elevenlabs"
    tts_model: str = "eleven_flash_v2_5"
    tts_voice: str | None = None
    # Metadata for fair comparison display
    pricing_type: str = "payg"  # "payg" or "free"
    production_eligible: bool = True


# ---------------------------------------------------------------------------
# Configuration catalogue
# ---------------------------------------------------------------------------
# Uses models already supported by the provider adapters.
# Free models are clearly labelled and marked non-production.

CONFIG_DEFAULT = BenchmarkConfiguration(
    configuration_id="default",
    name="Default (Nemotron Free)",
    description="Default configuration using free Nemotron model. Good for quick tests.",
    llm_provider="openrouter",
    llm_model="nvidia/nemotron-3.5-lightning:free",
    tts_provider="elevenlabs",
    tts_model="eleven_flash_v2_5",
    pricing_type="free",
    production_eligible=False,
)

CONFIG_GPT4O_MINI = BenchmarkConfiguration(
    configuration_id="gpt4o_mini",
    name="GPT-4o-mini + ElevenLabs",
    description="OpenAI GPT-4o-mini via OpenRouter with ElevenLabs Flash TTS. Production PAYG.",
    llm_provider="openrouter",
    llm_model="openai/gpt-4o-mini",
    tts_provider="elevenlabs",
    tts_model="eleven_flash_v2_5",
    pricing_type="payg",
    production_eligible=True,
)

CONFIG_GEMMA_FREE = BenchmarkConfiguration(
    configuration_id="gemma_free",
    name="Gemma Free + ElevenLabs",
    description="Google Gemma 4 31B IT (free) via OpenRouter with ElevenLabs Flash TTS.",
    llm_provider="openrouter",
    llm_model="google/gemma-4-31b-it:free",
    tts_provider="elevenlabs",
    tts_model="eleven_flash_v2_5",
    pricing_type="free",
    production_eligible=False,
)

CONFIG_LLAMA_FREE = BenchmarkConfiguration(
    configuration_id="llama_free",
    name="Llama Free + ElevenLabs",
    description="Meta Llama 3.1 8B Instruct (free) via OpenRouter with ElevenLabs Flash TTS.",
    llm_provider="openrouter",
    llm_model="meta-llama/llama-3.1-8b-instruct:free",
    tts_provider="elevenlabs",
    tts_model="eleven_flash_v2_5",
    pricing_type="free",
    production_eligible=False,
)

CONFIG_CLAUDE_SONNET = BenchmarkConfiguration(
    configuration_id="claude_sonnet",
    name="Claude 3.5 Sonnet + ElevenLabs",
    description=(
        "Anthropic Claude 3.5 Sonnet via OpenRouter with ElevenLabs Flash TTS. "
        "Production PAYG."
    ),
    llm_provider="openrouter",
    llm_model="anthropic/claude-3.5-sonnet",
    tts_provider="elevenlabs",
    tts_model="eleven_flash_v2_5",
    pricing_type="payg",
    production_eligible=True,
)

CONFIG_MULTILINGUAL_TTS = BenchmarkConfiguration(
    configuration_id="multilingual_tts",
    name="Default LLM + Multilingual TTS",
    description="Default LLM with ElevenLabs Multilingual v2 TTS for higher quality.",
    llm_provider="openrouter",
    llm_model="nvidia/nemotron-3.5-lightning:free",
    tts_provider="elevenlabs",
    tts_model="eleven_multilingual_v2",
    pricing_type="free",
    production_eligible=False,
)


# Ordered catalogue for API listing
ALL_CONFIGURATIONS: list[BenchmarkConfiguration] = [
    CONFIG_DEFAULT,
    CONFIG_GPT4O_MINI,
    CONFIG_GEMMA_FREE,
    CONFIG_LLAMA_FREE,
    CONFIG_CLAUDE_SONNET,
    CONFIG_MULTILINGUAL_TTS,
]

_CONFIG_MAP: dict[str, BenchmarkConfiguration] = {
    c.configuration_id: c for c in ALL_CONFIGURATIONS
}


def get_configuration(configuration_id: str) -> BenchmarkConfiguration | None:
    """Look up a configuration by ID."""
    return _CONFIG_MAP.get(configuration_id)


def list_configurations() -> list[BenchmarkConfiguration]:
    """Return all available configurations."""
    return list(ALL_CONFIGURATIONS)


def validate_configuration(config: BenchmarkConfiguration) -> list[str]:
    """Validate that a configuration's provider/model combinations are supported.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Check LLM provider
    from app.providers.factory import SUPPORTED_LLM_PROVIDERS, SUPPORTED_TTS_PROVIDERS

    if config.llm_provider not in SUPPORTED_LLM_PROVIDERS:
        errors.append(
            f"Unsupported LLM provider: '{config.llm_provider}'. "
            f"Supported: {sorted(SUPPORTED_LLM_PROVIDERS)}"
        )

    if config.tts_provider not in SUPPORTED_TTS_PROVIDERS:
        errors.append(
            f"Unsupported TTS provider: '{config.tts_provider}'. "
            f"Supported: {sorted(SUPPORTED_TTS_PROVIDERS)}"
        )

    return errors
