"""Benchmark cost calculation service (Phase 5D).

Calculates estimated provider costs from measured usage data.
Uses the pricing catalog in app.services.pricing.

IMPORTANT:
- NULL usage → NULL cost (not $0)
- Unknown provider/model → pricing unavailable
- All monetary calculations use Decimal for precision
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.core.logging import logger
from app.models.benchmark_result import BenchmarkResult
from app.services.pricing import (
    PRICING_CURRENCY,
    PRICING_VERSION,
    get_llm_pricing,
    get_stt_pricing,
    get_tts_pricing,
)


@dataclass
class CostBreakdown:
    """Cost breakdown for a single benchmark run."""

    run_id: str | None = None
    scenario_id: str | None = None
    benchmark_mode: str | None = None

    # Individual costs (None = unavailable/not measured)
    stt_cost: Decimal | None = None
    llm_input_cost: Decimal | None = None
    llm_output_cost: Decimal | None = None
    llm_total_cost: Decimal | None = None
    tts_cost: Decimal | None = None
    total_cost: Decimal | None = None

    # Metadata
    currency: str = PRICING_CURRENCY
    pricing_version: str = PRICING_VERSION
    pricing_available: bool = True

    # Pricing source info
    stt_pricing_source: str | None = None
    llm_pricing_source: str | None = None
    tts_pricing_source: str | None = None

    # Usage info (for debugging)
    stt_audio_duration_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tts_characters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response dict."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "benchmark_mode": self.benchmark_mode,
            "stt_cost": _decimal_to_float(self.stt_cost),
            "llm_input_cost": _decimal_to_float(self.llm_input_cost),
            "llm_output_cost": _decimal_to_float(self.llm_output_cost),
            "llm_total_cost": _decimal_to_float(self.llm_total_cost),
            "tts_cost": _decimal_to_float(self.tts_cost),
            "total_cost": _decimal_to_float(self.total_cost),
            "currency": self.currency,
            "pricing_version": self.pricing_version,
            "pricing_available": self.pricing_available,
            "stt_pricing_source": self.stt_pricing_source,
            "llm_pricing_source": self.llm_pricing_source,
            "tts_pricing_source": self.tts_pricing_source,
            "stt_audio_duration_seconds": self.stt_audio_duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tts_characters": self.tts_characters,
        }


def _decimal_to_float(d: Decimal | None) -> float | None:
    """Convert Decimal to float for JSON serialization."""
    if d is None:
        return None
    return float(d)


def _calculate_stt_cost(
    audio_duration_seconds: float | None,
    provider: str | None,
    model: str | None,
) -> tuple[Decimal | None, str | None]:
    """Calculate STT cost from audio duration.

    Returns:
        Tuple of (cost, pricing_source) or (None, None) if unavailable.
    """
    if audio_duration_seconds is None or provider is None:
        return None, None

    pricing = get_stt_pricing(provider, model)
    if pricing is None or pricing.price_per_unit is None:
        return None, None

    # Convert seconds to minutes
    audio_minutes = Decimal(str(audio_duration_seconds)) / Decimal("60")
    cost = audio_minutes * pricing.price_per_unit
    source = f"{provider}:{model or 'default'}"
    return cost, source


def _calculate_llm_costs(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    provider: str | None,
    model: str | None,
) -> tuple[Decimal | None, Decimal | None, Decimal | None, str | None]:
    """Calculate LLM input/output costs from token counts.

    Returns:
        Tuple of (input_cost, output_cost, total_cost, pricing_source) or (None, None, None, None).
    """
    if provider is None:
        return None, None, None, None

    pricing = get_llm_pricing(provider, model)
    if pricing is None:
        return None, None, None, None

    input_pricing = pricing.get("input")
    output_pricing = pricing.get("output")

    input_cost = None
    output_cost = None

    # Calculate input cost
    if prompt_tokens is not None and input_pricing and input_pricing.price_per_unit is not None:
        # Price is per 1M tokens
        tokens = Decimal(str(prompt_tokens))
        input_cost = (tokens / Decimal("1000000")) * input_pricing.price_per_unit

    # Calculate output cost
    has_output = completion_tokens is not None
    if has_output and output_pricing and output_pricing.price_per_unit is not None:
        tokens = Decimal(str(completion_tokens))
        output_cost = (tokens / Decimal("1000000")) * output_pricing.price_per_unit

    # Calculate total
    total_cost = None
    if input_cost is not None and output_cost is not None:
        total_cost = input_cost + output_cost
    elif input_cost is not None:
        total_cost = input_cost
    elif output_cost is not None:
        total_cost = output_cost

    source = f"{provider}:{model or 'default'}"
    return input_cost, output_cost, total_cost, source


def _calculate_tts_cost(
    characters: int | None,
    provider: str | None,
    model: str | None,
) -> tuple[Decimal | None, str | None]:
    """Calculate TTS cost from character count.

    Returns:
        Tuple of (cost, pricing_source) or (None, None) if unavailable.
    """
    if characters is None or provider is None:
        return None, None

    pricing = get_tts_pricing(provider, model)
    if pricing is None or pricing.price_per_unit is None:
        return None, None

    # Price is per 1K characters
    cost = (Decimal(str(characters)) / Decimal("1000")) * pricing.price_per_unit
    source = f"{provider}:{model or 'default'}"
    return cost, source


def calculate_benchmark_cost(result: BenchmarkResult) -> CostBreakdown:
    """Calculate cost breakdown for a single benchmark result.

    Args:
        result: BenchmarkResult with usage data.

    Returns:
        CostBreakdown with calculated costs.
    """
    breakdown = CostBreakdown(
        run_id=result.run_id,
        scenario_id=result.scenario_id,
        benchmark_mode=result.benchmark_mode,
        stt_audio_duration_seconds=result.stt_audio_duration_seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tts_characters=result.tts_characters,
    )

    pricing_available = True

    # STT cost
    stt_cost, stt_source = _calculate_stt_cost(
        result.stt_audio_duration_seconds,
        result.stt_provider,
        result.stt_model,
    )
    breakdown.stt_cost = stt_cost
    breakdown.stt_pricing_source = stt_source

    # Check if STT pricing was expected but unavailable
    if result.stt_provider and result.stt_audio_duration_seconds is not None and stt_cost is None:
        pricing_available = False
        logger.warning(
            "[COST] STT pricing unavailable provider=%s model=%s",
            result.stt_provider,
            result.stt_model,
        )

    # LLM costs
    llm_input, llm_output, llm_total, llm_source = _calculate_llm_costs(
        result.prompt_tokens,
        result.completion_tokens,
        result.llm_provider,
        result.llm_model,
    )
    breakdown.llm_input_cost = llm_input
    breakdown.llm_output_cost = llm_output
    breakdown.llm_total_cost = llm_total
    breakdown.llm_pricing_source = llm_source

    # Check if LLM pricing was expected but unavailable
    if result.llm_provider and (
        result.prompt_tokens is not None or result.completion_tokens is not None
    ):
        if llm_input is None and llm_output is None:
            pricing_available = False
            logger.warning(
                "[COST] LLM pricing unavailable provider=%s model=%s",
                result.llm_provider,
                result.llm_model,
            )

    # TTS cost
    tts_cost, tts_source = _calculate_tts_cost(
        result.tts_characters,
        result.tts_provider,
        result.tts_model,
    )
    breakdown.tts_cost = tts_cost
    breakdown.tts_pricing_source = tts_source

    # Check if TTS pricing was expected but unavailable
    if result.tts_provider and result.tts_characters is not None and tts_cost is None:
        pricing_available = False
        logger.warning(
            "[COST] TTS pricing unavailable provider=%s model=%s",
            result.tts_provider,
            result.tts_model,
        )

    # Calculate total cost (only if all measured components have costs)
    total = Decimal("0")
    has_any_cost = False

    for cost in [stt_cost, llm_total, tts_cost]:
        if cost is not None:
            total += cost
            has_any_cost = True
        # Note: NULL cost means not measured (e.g., no STT in text_input)
        # This is different from $0 cost

    if has_any_cost:
        breakdown.total_cost = total

    breakdown.pricing_available = pricing_available

    return breakdown


def calculate_cost_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Calculate aggregate cost summary across multiple benchmark results.

    Returns:
        Dictionary with aggregate cost statistics.
    """
    if not results:
        return {
            "total_runs": 0,
            "runs_with_cost": 0,
            "total_cost": None,
            "avg_cost": None,
            "min_cost": None,
            "max_cost": None,
            "currency": PRICING_CURRENCY,
            "pricing_version": PRICING_VERSION,
        }

    costs: list[Decimal] = []
    for r in results:
        breakdown = calculate_benchmark_cost(r)
        if breakdown.total_cost is not None:
            costs.append(breakdown.total_cost)

    if not costs:
        return {
            "total_runs": len(results),
            "runs_with_cost": 0,
            "total_cost": None,
            "avg_cost": None,
            "min_cost": None,
            "max_cost": None,
            "currency": PRICING_CURRENCY,
            "pricing_version": PRICING_VERSION,
        }

    total = sum(costs)
    avg = total / len(costs)
    min_cost = min(costs)
    max_cost = max(costs)

    return {
        "total_runs": len(results),
        "runs_with_cost": len(costs),
        "total_cost": float(total),
        "avg_cost": float(avg),
        "min_cost": float(min_cost),
        "max_cost": float(max_cost),
        "currency": PRICING_CURRENCY,
        "pricing_version": PRICING_VERSION,
    }
