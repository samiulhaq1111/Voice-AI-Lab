"""Provider pricing catalog for cost calculation (Phase 5D).

Pricing is based on official provider pricing pages as of September 2026.
All prices are in USD.

Sources:
- Deepgram: https://deepgram.com/pricing (checked 2026-09-15)
- ElevenLabs: https://elevenlabs.io/pricing/api (checked 2026-09-15)
- OpenRouter: https://openrouter.ai/models (checked 2026-09-15)

IMPORTANT:
- Free-tier models have $0 pricing
- Paid models use pay-as-you-go production pricing
- If pricing is ambiguous or unavailable, use None
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model."""

    provider: str
    model: str
    service: str  # "stt", "llm_input", "llm_output", "tts"
    unit: str  # "minute", "1k_tokens", "1k_characters"
    price_per_unit: Decimal | None  # None = unavailable
    pricing_type: str = "payg"  # "payg" or "free"
    production_eligible: bool = True  # False for free/rate-limited endpoints


# ---------------------------------------------------------------------------
# STT Pricing (Deepgram)
# ---------------------------------------------------------------------------
# Deepgram Nova-3 pre-recorded pricing (pay-as-you-go):
# - Nova-3 Monolingual: $0.0043/min
# - Nova-3 Multilingual: $0.0052/min
# Source: https://deepgram.com/pricing

STT_PRICING: dict[str, ModelPricing] = {
    # Deepgram Nova-3 (default model)
    "deepgram:nova-3": ModelPricing(
        provider="deepgram",
        model="nova-3",
        service="stt",
        unit="minute",
        price_per_unit=Decimal("0.0043"),
    ),
    "deepgram:nova-3-general": ModelPricing(
        provider="deepgram",
        model="nova-3-general",
        service="stt",
        unit="minute",
        price_per_unit=Decimal("0.0043"),
    ),
}


# ---------------------------------------------------------------------------
# LLM Pricing (OpenRouter)
# ---------------------------------------------------------------------------
# OpenRouter pricing varies by underlying model.
# Free models have $0 pricing.
# Paid models: input and output tokens priced separately per 1M tokens.
# Source: https://openrouter.ai/models

LLM_PRICING: dict[str, dict[str, ModelPricing]] = {
    # Free models
    "openrouter:nvidia/nemotron-3.5-lightning:free": {
        "input": ModelPricing(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="nvidia/nemotron-3.5-lightning:free",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
    },
    "openrouter:google/gemma-4-31b-it:free": {
        "input": ModelPricing(
            provider="openrouter",
            model="google/gemma-4-31b-it:free",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="google/gemma-4-31b-it:free",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
    },
    "openrouter:meta-llama/llama-3.1-8b-instruct:free": {
        "input": ModelPricing(
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct:free",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct:free",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
    },
    "openrouter:microsoft/phi-3-medium-128k-instruct:free": {
        "input": ModelPricing(
            provider="openrouter",
            model="microsoft/phi-3-medium-128k-instruct:free",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="microsoft/phi-3-medium-128k-instruct:free",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
    },
    "openrouter:google/gemini-2.0-flash-exp:free": {
        "input": ModelPricing(
            provider="openrouter",
            model="google/gemini-2.0-flash-exp:free",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="google/gemini-2.0-flash-exp:free",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0"),
            pricing_type="free",
            production_eligible=False,
        ),
    },
    # Paid models (OpenAI GPT-4o-mini via OpenRouter)
    # Pricing: $0.15/1M input tokens, $0.60/1M output tokens
    # Source: https://openrouter.ai/openai/gpt-4o-mini
    "openrouter:openai/gpt-4o-mini": {
        "input": ModelPricing(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0.15"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0.60"),
        ),
    },
    # --- Validated paid models (Benchmark catalogue update) ---
    # Pricing fetched from OpenRouter's public models API on 2026-09-11.
    # NOTE: the anthropic/claude-3.5-sonnet entry below is intentionally
    # RETAINED so historical benchmark records keep their computed costs,
    # even though it is no longer a selectable configuration.

    # OpenAI GPT-4.1 Mini via OpenRouter
    # Pricing: $0.40/1M input tokens, $1.60/1M output tokens
    # Source: https://openrouter.ai/openai/gpt-4.1-mini
    "openrouter:openai/gpt-4.1-mini": {
        "input": ModelPricing(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0.40"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("1.60"),
        ),
    },
    # OpenAI GPT-5 Mini via OpenRouter
    # Pricing: $0.25/1M input tokens, $2.00/1M output tokens
    # Source: https://openrouter.ai/openai/gpt-5-mini
    "openrouter:openai/gpt-5-mini": {
        "input": ModelPricing(
            provider="openrouter",
            model="openai/gpt-5-mini",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0.25"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="openai/gpt-5-mini",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("2.00"),
        ),
    },
    # Google Gemini 2.5 Flash via OpenRouter
    # Pricing: $0.30/1M input tokens, $2.50/1M output tokens
    # Source: https://openrouter.ai/google/gemini-2.5-flash
    "openrouter:google/gemini-2.5-flash": {
        "input": ModelPricing(
            provider="openrouter",
            model="google/gemini-2.5-flash",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0.30"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="google/gemini-2.5-flash",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("2.50"),
        ),
    },
    # Anthropic Claude Haiku 4.5 via OpenRouter
    # Pricing: $1.00/1M input tokens, $5.00/1M output tokens
    # Source: https://openrouter.ai/anthropic/claude-haiku-4.5
    "openrouter:anthropic/claude-haiku-4.5": {
        "input": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("1.00"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("5.00"),
        ),
    },
    # Anthropic Claude Sonnet 4.6 via OpenRouter
    # Pricing: $3.00/1M input tokens, $15.00/1M output tokens
    # Source: https://openrouter.ai/anthropic/claude-sonnet-4.6
    "openrouter:anthropic/claude-sonnet-4.6": {
        "input": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("3.00"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("15.00"),
        ),
    },
    # DeepSeek V3.1 via OpenRouter
    # Pricing: $0.25/1M input tokens, $0.95/1M output tokens
    # Source: https://openrouter.ai/deepseek/deepseek-chat-v3.1
    "openrouter:deepseek/deepseek-chat-v3.1": {
        "input": ModelPricing(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3.1",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("0.25"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3.1",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("0.95"),
        ),
    },
    # Anthropic Claude 3.5 Sonnet via OpenRouter
    # Pricing: $3.00/1M input tokens, $15.00/1M output tokens
    # Source: https://openrouter.ai/anthropic/claude-3.5-sonnet
    # RETAINED for historical benchmark cost re-computation only — no longer
    # a selectable Chat/Voice model or Benchmark configuration.
    "openrouter:anthropic/claude-3.5-sonnet": {
        "input": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            service="llm_input",
            unit="1m_tokens",
            price_per_unit=Decimal("3.00"),
        ),
        "output": ModelPricing(
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            service="llm_output",
            unit="1m_tokens",
            price_per_unit=Decimal("15.00"),
        ),
    },
}


# ---------------------------------------------------------------------------
# TTS Pricing (ElevenLabs)
# ---------------------------------------------------------------------------
# ElevenLabs pricing (pay-as-you-go):
# - Flash/Turbo models: $0.05 per 1K characters
# - v3/Multilingual v2 models: $0.10 per 1K characters
# Source: https://elevenlabs.io/pricing/api

TTS_PRICING: dict[str, ModelPricing] = {
    # Flash models ($0.05/1K chars)
    "elevenlabs:eleven_flash_v2_5": ModelPricing(
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        service="tts",
        unit="1k_characters",
        price_per_unit=Decimal("0.05"),
    ),
    "elevenlabs:eleven_flash_v2": ModelPricing(
        provider="elevenlabs",
        model="eleven_flash_v2",
        service="tts",
        unit="1k_characters",
        price_per_unit=Decimal("0.05"),
    ),
    "elevenlabs:eleven_turbo_v2_5": ModelPricing(
        provider="elevenlabs",
        model="eleven_turbo_v2_5",
        service="tts",
        unit="1k_characters",
        price_per_unit=Decimal("0.05"),
    ),
    # Multilingual v2 and v3 models ($0.10/1K chars)
    "elevenlabs:eleven_multilingual_v2": ModelPricing(
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        service="tts",
        unit="1k_characters",
        price_per_unit=Decimal("0.10"),
    ),
    "elevenlabs:eleven_v3": ModelPricing(
        provider="elevenlabs",
        model="eleven_v3",
        service="tts",
        unit="1k_characters",
        price_per_unit=Decimal("0.10"),
    ),
}


# ---------------------------------------------------------------------------
# Pricing version
# ---------------------------------------------------------------------------

PRICING_VERSION = "2026-09-15"
PRICING_CURRENCY = "USD"


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_stt_pricing(provider: str, model: str | None) -> ModelPricing | None:
    """Get STT pricing for a provider/model combination."""
    key = f"{provider}:{model}" if model else provider
    return STT_PRICING.get(key)


def get_llm_pricing(provider: str, model: str | None) -> dict[str, ModelPricing] | None:
    """Get LLM pricing (input + output) for a provider/model combination."""
    key = f"{provider}:{model}" if model else provider
    return LLM_PRICING.get(key)


def get_tts_pricing(provider: str, model: str | None) -> ModelPricing | None:
    """Get TTS pricing for a provider/model combination."""
    key = f"{provider}:{model}" if model else provider
    return TTS_PRICING.get(key)


def get_all_pricing() -> dict[str, Any]:
    """Return all pricing data for inspection."""
    stt_dict = {
        k: {"price_per_unit": str(v.price_per_unit), "unit": v.unit}
        for k, v in STT_PRICING.items()
    }
    llm_dict = {
        k: {
            "input": str(v["input"].price_per_unit),
            "output": str(v["output"].price_per_unit),
            "unit": v["input"].unit,
        }
        for k, v in LLM_PRICING.items()
    }
    tts_dict = {
        k: {"price_per_unit": str(v.price_per_unit), "unit": v.unit}
        for k, v in TTS_PRICING.items()
    }
    return {
        "version": PRICING_VERSION,
        "currency": PRICING_CURRENCY,
        "stt": stt_dict,
        "llm": llm_dict,
        "tts": tts_dict,
    }
