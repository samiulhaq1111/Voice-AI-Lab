"""Benchmark analytics service (Phase 5C).

Provides read-only analytics over existing BenchmarkResult data:
- Overall summary
- Per-scenario summary
- Per-provider/model summary
- Recent results listing

NULL handling: metrics that are NULL (e.g. STT in text_input mode)
are excluded from averages rather than treated as zero.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import logger
from app.models.benchmark_result import BenchmarkResult

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    avg_ms: float | None = None
    median_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_ms": self.avg_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class OverallSummary:
    """Summary across all benchmark runs."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    stt_latency: LatencyStats = field(default_factory=LatencyStats)
    llm_latency: LatencyStats = field(default_factory=LatencyStats)
    tts_latency: LatencyStats = field(default_factory=LatencyStats)
    tool_execution: LatencyStats = field(default_factory=LatencyStats)
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None
    avg_tts_audio_bytes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "latency": self.latency.to_dict(),
            "stt_latency": self.stt_latency.to_dict(),
            "llm_latency": self.llm_latency.to_dict(),
            "tts_latency": self.tts_latency.to_dict(),
            "tool_execution": self.tool_execution.to_dict(),
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_tts_characters": self.avg_tts_characters,
            "avg_tts_audio_bytes": self.avg_tts_audio_bytes,
        }


@dataclass
class ScenarioSummary:
    """Summary for a single benchmark scenario."""

    scenario_id: str
    run_count: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    stt_latency: LatencyStats = field(default_factory=LatencyStats)
    llm_latency: LatencyStats = field(default_factory=LatencyStats)
    tts_latency: LatencyStats = field(default_factory=LatencyStats)
    tool_execution: LatencyStats = field(default_factory=LatencyStats)
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_count": self.run_count,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "latency": self.latency.to_dict(),
            "stt_latency": self.stt_latency.to_dict(),
            "llm_latency": self.llm_latency.to_dict(),
            "tts_latency": self.tts_latency.to_dict(),
            "tool_execution": self.tool_execution.to_dict(),
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_tts_characters": self.avg_tts_characters,
        }


@dataclass
class ProviderSummary:
    """Summary grouped by provider + model for a given stage."""

    provider: str | None
    model: str | None
    stage: str  # "stt", "llm", or "tts"
    run_count: int = 0
    successful_runs: int = 0
    success_rate: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None
    avg_tts_audio_bytes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "stage": self.stage,
            "run_count": self.run_count,
            "successful_runs": self.successful_runs,
            "success_rate": round(self.success_rate, 4),
            "latency": self.latency.to_dict(),
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_tts_characters": self.avg_tts_characters,
            "avg_tts_audio_bytes": self.avg_tts_audio_bytes,
        }


@dataclass
class RecentResult:
    """Lightweight representation of a recent benchmark run."""

    id: str
    run_id: str | None
    scenario_id: str | None
    benchmark_mode: str | None
    success: bool
    created_at: datetime
    total_processing_ms: float | None
    stt_latency_ms: float | None
    llm_latency_ms: float | None
    tts_latency_ms: float | None
    tool_execution_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    token_usage: int | None
    tts_characters: int | None
    tts_audio_bytes: int | None
    llm_provider: str | None
    llm_model: str | None
    tts_provider: str | None
    tts_model: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "benchmark_mode": self.benchmark_mode,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
            "total_processing_ms": self.total_processing_ms,
            "stt_latency_ms": self.stt_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "tool_execution_ms": self.tool_execution_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "token_usage": self.token_usage,
            "tts_characters": self.tts_characters,
            "tts_audio_bytes": self.tts_audio_bytes,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_latency_stats(values: list[float | None]) -> LatencyStats:
    """Compute latency statistics from a list of values, excluding NULLs."""
    clean = [v for v in values if v is not None]
    if not clean:
        return LatencyStats()
    return LatencyStats(
        avg_ms=round(statistics.mean(clean), 2),
        median_ms=round(statistics.median(clean), 2),
        min_ms=round(min(clean), 2),
        max_ms=round(max(clean), 2),
    )


def _avg_int(values: list[int | None]) -> float | None:
    """Compute average of integer values, excluding NULLs."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 2)


def _avg_float(values: list[float | None]) -> float | None:
    """Compute average of float values, excluding NULLs."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 2)


def _is_success(record: BenchmarkResult) -> bool:
    """Determine if a benchmark result is successful."""
    if record.conversation_success == "true":
        return True
    return False


# ---------------------------------------------------------------------------
# Analytics service
# ---------------------------------------------------------------------------

class BenchmarkAnalyticsService:
    """Read-only analytics over BenchmarkResult data."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_overall_summary(
        self,
        *,
        scenario_id: str | None = None,
        benchmark_mode: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> OverallSummary:
        """Compute overall summary across all benchmark runs."""
        query = self.db.query(BenchmarkResult)
        query = self._apply_filters(
            query,
            scenario_id=scenario_id,
            benchmark_mode=benchmark_mode,
            date_from=date_from,
            date_to=date_to,
        )
        records = query.all()

        if not records:
            return OverallSummary()

        total = len(records)
        successful = sum(1 for r in records if _is_success(r))
        failed = total - successful

        summary = OverallSummary(
            total_runs=total,
            successful_runs=successful,
            failed_runs=failed,
            success_rate=successful / total if total > 0 else 0.0,
            latency=_compute_latency_stats([r.total_processing_ms for r in records]),
            stt_latency=_compute_latency_stats([r.stt_latency_ms for r in records]),
            llm_latency=_compute_latency_stats([r.llm_latency_ms for r in records]),
            tts_latency=_compute_latency_stats([r.tts_latency_ms for r in records]),
            tool_execution=_compute_latency_stats([r.tool_execution_ms for r in records]),
            avg_prompt_tokens=_avg_int([r.prompt_tokens for r in records]),
            avg_completion_tokens=_avg_int([r.completion_tokens for r in records]),
            avg_total_tokens=_avg_int([r.token_usage for r in records]),
            avg_tts_characters=_avg_int([r.tts_characters for r in records]),
            avg_tts_audio_bytes=_avg_int([r.tts_audio_bytes for r in records]),
        )

        logger.info(
            "[ANALYTICS] overall_summary total=%d success=%d rate=%.2f",
            total,
            successful,
            summary.success_rate,
        )
        return summary

    def get_scenario_summaries(
        self,
        *,
        benchmark_mode: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ScenarioSummary]:
        """Compute summary for each scenario."""
        query = self.db.query(BenchmarkResult).filter(
            BenchmarkResult.scenario_id.isnot(None),
        )
        query = self._apply_filters(
            query,
            scenario_id=None,
            benchmark_mode=benchmark_mode,
            date_from=date_from,
            date_to=date_to,
        )
        records = query.all()

        # Group by scenario_id
        grouped: dict[str, list[BenchmarkResult]] = {}
        for r in records:
            sid = r.scenario_id
            if sid:
                grouped.setdefault(sid, []).append(r)

        summaries = []
        for scenario_id, group in sorted(grouped.items()):
            total = len(group)
            successful = sum(1 for r in group if _is_success(r))
            summaries.append(
                ScenarioSummary(
                    scenario_id=scenario_id,
                    run_count=total,
                    successful_runs=successful,
                    failed_runs=total - successful,
                    success_rate=successful / total if total > 0 else 0.0,
                    latency=_compute_latency_stats([r.total_processing_ms for r in group]),
                    stt_latency=_compute_latency_stats([r.stt_latency_ms for r in group]),
                    llm_latency=_compute_latency_stats([r.llm_latency_ms for r in group]),
                    tts_latency=_compute_latency_stats([r.tts_latency_ms for r in group]),
                    tool_execution=_compute_latency_stats([r.tool_execution_ms for r in group]),
                    avg_prompt_tokens=_avg_int([r.prompt_tokens for r in group]),
                    avg_completion_tokens=_avg_int([r.completion_tokens for r in group]),
                    avg_total_tokens=_avg_int([r.token_usage for r in group]),
                    avg_tts_characters=_avg_int([r.tts_characters for r in group]),
                )
            )

        logger.info("[ANALYTICS] scenario_summaries count=%d", len(summaries))
        return summaries

    def get_provider_summaries(
        self,
        *,
        benchmark_mode: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ProviderSummary]:
        """Compute summaries grouped by provider + model for each stage."""
        query = self.db.query(BenchmarkResult)
        query = self._apply_filters(
            query,
            scenario_id=None,
            benchmark_mode=benchmark_mode,
            date_from=date_from,
            date_to=date_to,
        )
        records = query.all()

        if not records:
            return []

        # Group by (stage, provider, model)
        stt_groups: dict[tuple[str | None, str | None], list[BenchmarkResult]] = {}
        llm_groups: dict[tuple[str | None, str | None], list[BenchmarkResult]] = {}
        tts_groups: dict[tuple[str | None, str | None], list[BenchmarkResult]] = {}

        for r in records:
            if r.stt_provider:
                key = (r.stt_provider, r.stt_model)
                stt_groups.setdefault(key, []).append(r)
            if r.llm_provider:
                key = (r.llm_provider, r.llm_model)
                llm_groups.setdefault(key, []).append(r)
            if r.tts_provider:
                key = (r.tts_provider, r.tts_model)
                tts_groups.setdefault(key, []).append(r)

        summaries: list[ProviderSummary] = []

        for (provider, model), group in sorted(stt_groups.items()):
            total = len(group)
            successful = sum(1 for r in group if _is_success(r))
            summaries.append(
                ProviderSummary(
                    provider=provider,
                    model=model,
                    stage="stt",
                    run_count=total,
                    successful_runs=successful,
                    success_rate=successful / total if total > 0 else 0.0,
                    latency=_compute_latency_stats([r.stt_latency_ms for r in group]),
                )
            )

        for (provider, model), group in sorted(llm_groups.items()):
            total = len(group)
            successful = sum(1 for r in group if _is_success(r))
            summaries.append(
                ProviderSummary(
                    provider=provider,
                    model=model,
                    stage="llm",
                    run_count=total,
                    successful_runs=successful,
                    success_rate=successful / total if total > 0 else 0.0,
                    latency=_compute_latency_stats([r.llm_latency_ms for r in group]),
                    avg_prompt_tokens=_avg_int([r.prompt_tokens for r in group]),
                    avg_completion_tokens=_avg_int([r.completion_tokens for r in group]),
                    avg_total_tokens=_avg_int([r.token_usage for r in group]),
                )
            )

        for (provider, model), group in sorted(tts_groups.items()):
            total = len(group)
            successful = sum(1 for r in group if _is_success(r))
            summaries.append(
                ProviderSummary(
                    provider=provider,
                    model=model,
                    stage="tts",
                    run_count=total,
                    successful_runs=successful,
                    success_rate=successful / total if total > 0 else 0.0,
                    latency=_compute_latency_stats([r.tts_latency_ms for r in group]),
                    avg_tts_characters=_avg_int([r.tts_characters for r in group]),
                    avg_tts_audio_bytes=_avg_int([r.tts_audio_bytes for r in group]),
                )
            )

        logger.info("[ANALYTICS] provider_summaries count=%d", len(summaries))
        return summaries

    def get_recent_results(
        self,
        *,
        limit: int = 20,
        scenario_id: str | None = None,
        benchmark_mode: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[RecentResult]:
        """Return recent benchmark results."""
        query = self.db.query(BenchmarkResult)
        query = self._apply_filters(
            query,
            scenario_id=scenario_id,
            benchmark_mode=benchmark_mode,
            date_from=date_from,
            date_to=date_to,
        )
        query = query.order_by(BenchmarkResult.created_at.desc()).limit(limit)
        records = query.all()

        results = []
        for r in records:
            results.append(
                RecentResult(
                    id=r.id,
                    run_id=r.run_id,
                    scenario_id=r.scenario_id,
                    benchmark_mode=r.benchmark_mode,
                    success=_is_success(r),
                    created_at=r.created_at,
                    total_processing_ms=r.total_processing_ms,
                    stt_latency_ms=r.stt_latency_ms,
                    llm_latency_ms=r.llm_latency_ms,
                    tts_latency_ms=r.tts_latency_ms,
                    tool_execution_ms=r.tool_execution_ms,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    token_usage=r.token_usage,
                    tts_characters=r.tts_characters,
                    tts_audio_bytes=r.tts_audio_bytes,
                    llm_provider=r.llm_provider,
                    llm_model=r.llm_model,
                    tts_provider=r.tts_provider,
                    tts_model=r.tts_model,
                )
            )

        logger.info("[ANALYTICS] recent_results count=%d", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        query: Any,
        *,
        scenario_id: str | None = None,
        benchmark_mode: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Any:
        """Apply common filters to a query."""
        if scenario_id:
            query = query.filter(BenchmarkResult.scenario_id == scenario_id)
        if benchmark_mode:
            query = query.filter(BenchmarkResult.benchmark_mode == benchmark_mode)
        if date_from:
            query = query.filter(BenchmarkResult.created_at >= date_from)
        if date_to:
            query = query.filter(BenchmarkResult.created_at <= date_to)
        return query
