"""Benchmark comparison runner (Phase 5E).

Runs the same scenario against different provider/model configurations
and aggregates results for comparison.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.benchmarks.configurations import (
    get_configuration,
    validate_configuration,
)
from app.benchmarks.runner import (
    BenchmarkRunResult,
    run_scenario,
)
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Comparison result types
# ---------------------------------------------------------------------------


@dataclass
class ConfigurationResult:
    """Results for a single configuration within a comparison."""

    configuration_id: str
    configuration_name: str
    pricing_type: str
    production_eligible: bool
    runs: list[BenchmarkRunResult] = field(default_factory=list)

    # Aggregated stats
    run_count: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate: float = 0.0

    # Latency stats
    avg_total_latency_ms: float | None = None
    median_total_latency_ms: float | None = None
    min_total_latency_ms: float | None = None
    max_total_latency_ms: float | None = None
    avg_llm_latency_ms: float | None = None
    avg_tts_latency_ms: float | None = None

    # Usage stats
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None

    # Cost stats
    avg_cost: float | None = None
    total_cost: float | None = None
    cost_available: bool = True
    cost_note: str | None = None  # e.g. "free", "unavailable"

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "configuration_name": self.configuration_name,
            "pricing_type": self.pricing_type,
            "production_eligible": self.production_eligible,
            "run_count": self.run_count,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "avg_total_latency_ms": self.avg_total_latency_ms,
            "median_total_latency_ms": self.median_total_latency_ms,
            "min_total_latency_ms": self.min_total_latency_ms,
            "max_total_latency_ms": self.max_total_latency_ms,
            "avg_llm_latency_ms": self.avg_llm_latency_ms,
            "avg_tts_latency_ms": self.avg_tts_latency_ms,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_tts_characters": self.avg_tts_characters,
            "avg_cost": self.avg_cost,
            "total_cost": self.total_cost,
            "cost_available": self.cost_available,
            "cost_note": self.cost_note,
            "runs": [r.to_api_dict() for r in self.runs],
        }


@dataclass
class ComparisonResult:
    """Result of a comparison across multiple configurations."""

    comparison_id: str
    scenario_id: str
    repetitions: int
    configurations: list[ConfigurationResult] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "scenario_id": self.scenario_id,
            "repetitions": self.repetitions,
            "configurations": [c.to_api_dict() for c in self.configurations],
            "validation_errors": self.validation_errors,
        }


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


async def run_comparison(
    db: Session,
    scenario_id: str,
    configuration_ids: list[str],
    repetitions: int = 1,
) -> ComparisonResult:
    """Run a scenario against multiple configurations and aggregate results.

    Each configuration runs the same scenario with identical inputs.
    Only the provider/model configuration changes between runs.
    """
    comparison_id = str(uuid.uuid4())

    logger.info(
        "[COMPARE] started comparison_id=%s scenario=%s configs=%s reps=%d",
        comparison_id,
        scenario_id,
        configuration_ids,
        repetitions,
    )

    comparison = ComparisonResult(
        comparison_id=comparison_id,
        scenario_id=scenario_id,
        repetitions=repetitions,
    )

    # Validate configurations
    for config_id in configuration_ids:
        config = get_configuration(config_id)
        if config is None:
            comparison.validation_errors.append(f"Unknown configuration: '{config_id}'")
            continue

        errors = validate_configuration(config)
        if errors:
            comparison.validation_errors.extend(
                [f"Configuration '{config_id}': {e}" for e in errors]
            )

    if comparison.validation_errors:
        logger.warning(
            "[COMPARE] validation errors: %s",
            comparison.validation_errors,
        )
        return comparison

    # Run each configuration
    for config_id in configuration_ids:
        config = get_configuration(config_id)
        if config is None:
            continue

        config_result = ConfigurationResult(
            configuration_id=config.configuration_id,
            configuration_name=config.name,
            pricing_type=config.pricing_type,
            production_eligible=config.production_eligible,
        )

        # Run scenario N times with this configuration
        for i in range(repetitions):
            logger.info(
                "[COMPARE] run config=%s rep=%d/%d",
                config_id,
                i + 1,
                repetitions,
            )

            run_result = await run_scenario(
                db,
                scenario_id,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                tts_provider=config.tts_provider,
                tts_model=config.tts_model,
                tts_voice=config.tts_voice,
            )

            # Tag with comparison and configuration IDs
            _tag_run_with_comparison(db, run_result, comparison_id, config.configuration_id)

            config_result.runs.append(run_result)

        # Aggregate results for this configuration
        _aggregate_config_result(config_result)

        comparison.configurations.append(config_result)

    logger.info(
        "[COMPARE] completed comparison_id=%s configs=%d",
        comparison_id,
        len(comparison.configurations),
    )

    return comparison


def _tag_run_with_comparison(
    db: Session,
    run_result: BenchmarkRunResult,
    comparison_id: str,
    configuration_id: str,
) -> None:
    """Update the persisted BenchmarkResult with comparison metadata."""
    from app.models.benchmark_result import BenchmarkResult

    try:
        br = (
            db.query(BenchmarkResult)
            .filter(BenchmarkResult.run_id == run_result.run_id)
            .first()
        )
        if br:
            br.comparison_id = comparison_id
            br.configuration_id = configuration_id
            db.commit()
    except Exception as e:
        logger.error(
            "[COMPARE] failed to tag run comparison_id=%s run_id=%s error=%s",
            comparison_id,
            run_result.run_id,
            e,
        )
        try:
            db.rollback()
        except Exception:
            pass


def _aggregate_config_result(config_result: ConfigurationResult) -> None:
    """Compute aggregated statistics for a configuration's runs."""
    runs = config_result.runs
    if not runs:
        return

    config_result.run_count = len(runs)
    config_result.successful_runs = sum(1 for r in runs if r.success)
    config_result.failed_runs = len(runs) - config_result.successful_runs
    config_result.success_rate = config_result.successful_runs / len(runs)

    # Latency stats from SUCCESSFUL runs only (failed don't reflect AI perf)
    successful_runs = [r for r in runs if r.success]

    import statistics

    total_latencies = [
        r.total_processing_ms
        for r in successful_runs
        if r.total_processing_ms is not None
    ]
    if total_latencies:
        config_result.avg_total_latency_ms = round(statistics.mean(total_latencies), 2)
        config_result.median_total_latency_ms = round(statistics.median(total_latencies), 2)
        config_result.min_total_latency_ms = round(min(total_latencies), 2)
        config_result.max_total_latency_ms = round(max(total_latencies), 2)

    llm_latencies = [r.llm_latency_ms for r in successful_runs if r.llm_latency_ms is not None]
    if llm_latencies:
        config_result.avg_llm_latency_ms = round(statistics.mean(llm_latencies), 2)

    tts_latencies = [r.tts_latency_ms for r in successful_runs if r.tts_latency_ms is not None]
    if tts_latencies:
        config_result.avg_tts_latency_ms = round(statistics.mean(tts_latencies), 2)

    # Token usage stats from successful runs only
    prompt_tokens = [
        r.usage.get("prompt_tokens")
        for r in successful_runs
        if r.usage.get("prompt_tokens") is not None
    ]
    if prompt_tokens:
        config_result.avg_prompt_tokens = round(statistics.mean(prompt_tokens), 1)

    completion_tokens = [
        r.usage.get("completion_tokens")
        for r in successful_runs
        if r.usage.get("completion_tokens") is not None
    ]
    if completion_tokens:
        config_result.avg_completion_tokens = round(statistics.mean(completion_tokens), 1)

    total_tokens = [
        r.usage.get("total_tokens")
        for r in successful_runs
        if r.usage.get("total_tokens") is not None
    ]
    if total_tokens:
        config_result.avg_total_tokens = round(statistics.mean(total_tokens), 1)

    # TTS stats from successful runs only
    tts_chars = [r.tts_characters for r in successful_runs if r.tts_characters is not None]
    if tts_chars:
        config_result.avg_tts_characters = round(statistics.mean(tts_chars), 1)

    # Calculate cost from run data
    _calculate_cost_from_runs(config_result)


def _calculate_cost_from_runs(config_result: ConfigurationResult) -> None:
    """Calculate cost statistics for a configuration's runs.

    This uses the pricing catalog and the run's usage data.
    """
    from decimal import Decimal

    from app.services.pricing import get_llm_pricing, get_tts_pricing

    # Determine pricing from the first run's provider/model info
    if not config_result.runs:
        return

    first_run = config_result.runs[0]
    llm_provider = first_run.llm_provider
    llm_model = first_run.llm_model
    tts_provider = first_run.tts_provider
    tts_model = first_run.tts_model

    # Check pricing type
    if config_result.pricing_type == "free":
        # Free models: only show $0 if at least one run succeeded
        if config_result.successful_runs > 0:
            config_result.cost_available = True
            config_result.cost_note = "free"
            config_result.avg_cost = 0.0
            config_result.total_cost = 0.0
        else:
            # All runs failed — no actual usage occurred
            config_result.cost_available = False
            config_result.cost_note = "no_usage"
        return

    # Get LLM pricing
    llm_pricing = get_llm_pricing(llm_provider, llm_model) if llm_provider else None
    tts_pricing = get_tts_pricing(tts_provider, tts_model) if tts_provider else None

    if llm_pricing is None and tts_pricing is None:
        config_result.cost_available = False
        config_result.cost_note = "unavailable"
        return

    # Calculate cost per run
    costs: list[float] = []
    for run in config_result.runs:
        run_cost = Decimal("0")
        has_cost = False

        # LLM cost
        if llm_pricing and run.usage:
            prompt = run.usage.get("prompt_tokens")
            completion = run.usage.get("completion_tokens")
            if prompt is not None and llm_pricing.get("input"):
                price = llm_pricing["input"].price_per_unit
                if price is not None:
                    run_cost += (Decimal(str(prompt)) / Decimal("1000000")) * price
                    has_cost = True
            if completion is not None and llm_pricing.get("output"):
                price = llm_pricing["output"].price_per_unit
                if price is not None:
                    run_cost += (Decimal(str(completion)) / Decimal("1000000")) * price
                    has_cost = True

        # TTS cost
        if tts_pricing and run.tts_characters is not None:
            price = tts_pricing.price_per_unit
            if price is not None:
                run_cost += (Decimal(str(run.tts_characters)) / Decimal("1000")) * price
                has_cost = True

        if has_cost:
            costs.append(float(run_cost))

    if costs:
        import statistics

        config_result.avg_cost = round(statistics.mean(costs), 6)
        config_result.total_cost = round(sum(costs), 6)
        config_result.cost_available = True
    else:
        config_result.cost_available = False
        config_result.cost_note = "unavailable"
