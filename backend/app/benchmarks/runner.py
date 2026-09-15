"""Benchmark runner — executes scenarios through the agent pipeline.

The runner reuses the existing AgentRuntime, provider factories,
and VoiceTurnMetrics infrastructure.  It creates a *benchmark-specific*
tool registry with deterministic handlers so that results are
repeatable.

Benchmark mode: ``text_input`` — no STT is measured.
"""

import statistics
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.benchmarks.deterministic_tools import create_benchmark_tools
from app.benchmarks.scenarios import BenchmarkScenario, get_scenario
from app.core.config import settings
from app.core.logging import logger
from app.providers.factory import ProviderError, get_llm_provider, get_tts_provider
from app.services.metrics_service import VoiceTurnMetrics, persist_metrics
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

BENCHMARK_MODE = "text_input"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRunResult:
    """Result of a single benchmark scenario execution."""

    run_id: str
    scenario_id: str
    success: bool
    benchmark_mode: str = BENCHMARK_MODE

    # Agent output
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    iterations: int = 0

    # Timing (ms)
    llm_latency_ms: float | None = None
    tts_latency_ms: float | None = None
    total_processing_ms: float | None = None
    tool_execution_ms: float | None = None

    # TTS output
    tts_audio_bytes: int | None = None
    tts_characters: int | None = None

    # Validation
    expected_tool_calls: int = 0
    actual_tool_calls: int = 0
    tool_call_match: bool = True
    validation_errors: list[str] = field(default_factory=list)

    # Provider info
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None

    # Persistence
    benchmark_result_id: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize for the API response."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "success": self.success,
            "benchmark_mode": self.benchmark_mode,
            "response_text": self.response_text,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "iterations": self.iterations,
            "llm_latency_ms": self.llm_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "total_processing_ms": self.total_processing_ms,
            "tool_execution_ms": self.tool_execution_ms,
            "tts_audio_bytes": self.tts_audio_bytes,
            "tts_characters": self.tts_characters,
            "expected_tool_calls": self.expected_tool_calls,
            "actual_tool_calls": self.actual_tool_calls,
            "tool_call_match": self.tool_call_match,
            "validation_errors": self.validation_errors,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
            "benchmark_result_id": self.benchmark_result_id,
        }


@dataclass
class BenchmarkAggregation:
    """Basic statistics across multiple runs of the same scenario."""

    scenario_id: str
    count: int
    success_count: int
    failure_count: int
    success_rate: float
    llm_latency_ms: dict[str, float | None]
    tts_latency_ms: dict[str, float | None]
    total_processing_ms: dict[str, float | None]

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "count": self.count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "llm_latency_ms": self.llm_latency_ms,
            "tts_latency_ms": self.tts_latency_ms,
            "total_processing_ms": self.total_processing_ms,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_benchmark_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with deterministic handlers."""
    registry = ToolRegistry()
    for tool in create_benchmark_tools():
        registry.register(tool)
    return registry


async def run_scenario(
    db: Session,
    scenario_id: str,
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    tts_provider: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
) -> BenchmarkRunResult:
    """Execute a single benchmark scenario.

    Returns a BenchmarkRunResult with metrics and validation.
    """
    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise ValueError(f"Unknown benchmark scenario: '{scenario_id}'")

    run_id = str(uuid.uuid4())
    resolved_llm_provider = llm_provider or settings.default_llm_provider
    resolved_llm_model = llm_model or settings.default_llm_model or None
    resolved_tts_provider = tts_provider or settings.default_tts_provider
    resolved_tts_model = tts_model or settings.default_tts_model or None
    resolved_tts_voice = tts_voice or settings.default_tts_voice or None

    logger.info(
        "[BENCH] run_started run_id=%s scenario=%s mode=%s",
        run_id,
        scenario_id,
        BENCHMARK_MODE,
    )

    result = BenchmarkRunResult(
        run_id=run_id,
        scenario_id=scenario_id,
        success=False,
        expected_tool_calls=scenario.expected_tool_calls,
        llm_provider=resolved_llm_provider,
        llm_model=resolved_llm_model,
        tts_provider=resolved_tts_provider,
        tts_model=resolved_tts_model,
    )

    # Build metrics collector (no STT stage)
    metrics = VoiceTurnMetrics()
    metrics.llm_provider = resolved_llm_provider
    metrics.llm_model = resolved_llm_model

    # ----- LLM / Agent stage -----
    try:
        llm = get_llm_provider(resolved_llm_provider, model=resolved_llm_model)
    except (ProviderError, ValueError) as e:
        result.validation_errors.append(f"LLM provider error: {e}")
        metrics.fail("llm", str(e))
        persist_metrics(db, metrics)
        _persist_benchmark(db, metrics, result, scenario)
        return result

    # Benchmark-specific tool registry with deterministic handlers
    bench_registry = _make_benchmark_registry()

    # If scenario restricts tools, disable the rest
    if scenario.allowed_tools:
        for tool in bench_registry.list_tools():
            if tool.name not in scenario.allowed_tools:
                bench_registry.disable(tool.name)

    bench_executor = ToolExecutor(bench_registry)
    config = AgentConfig(
        llm_model=resolved_llm_model,
        max_tool_rounds=settings.max_agent_iterations,
    )
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=bench_registry,
        tool_executor=bench_executor,
        config=config,
    )

    metrics.start_llm(provider=resolved_llm_provider, model=resolved_llm_model or "")

    # Callback to capture per-tool timing from the ToolExecutor
    async def _on_tool_call(tc: dict[str, Any], result: dict[str, Any]) -> None:
        metrics.record_tool_call(
            success=bool(result.get("success")),
            duration_ms=result.get("duration_ms"),
        )

    try:
        agent_result: AgentResult = await runtime.run(
            scenario.input_text,
            on_tool_call=_on_tool_call,
        )
    except Exception as e:
        logger.error("[BENCH] agent failed run_id=%s error=%s", run_id, e)
        result.validation_errors.append(f"Agent error: {e}")
        metrics.fail("llm", str(e))
        persist_metrics(db, metrics)
        _persist_benchmark(db, metrics, result, scenario)
        return result

    metrics.finish_llm(usage=agent_result.usage, iterations=agent_result.iterations)
    result.response_text = agent_result.response
    result.tool_calls = agent_result.tool_calls
    result.usage = agent_result.usage
    result.iterations = agent_result.iterations
    result.llm_latency_ms = metrics.llm_latency_ms
    result.actual_tool_calls = len(agent_result.tool_calls)

    # Tool metrics already recorded via _on_tool_call callback
    metrics.log_tools()
    result.tool_execution_ms = metrics.tool_execution_ms if metrics.tool_count else None

    await runtime.close()

    # ----- TTS stage (optional) -----
    if scenario.include_tts and result.response_text:
        metrics.start_tts(
            provider=resolved_tts_provider,
            model=resolved_tts_model or "",
            voice=resolved_tts_voice,
            characters=len(result.response_text),
        )
        try:
            tts = get_tts_provider(
                resolved_tts_provider,
                voice=resolved_tts_voice,
                model=resolved_tts_model,
            )
            tts_result = await tts.synthesize(
                result.response_text,
                voice=resolved_tts_voice,
                model=resolved_tts_model,
            )
            metrics.finish_tts(audio_bytes=len(tts_result.audio_data))
            result.tts_audio_bytes = len(tts_result.audio_data)
            result.tts_characters = len(result.response_text)
            result.tts_latency_ms = metrics.tts_latency_ms
            result.tts_provider = resolved_tts_provider
            result.tts_model = resolved_tts_model
            await tts.close()
        except Exception as e:
            logger.warning("[BENCH] TTS failed run_id=%s error=%s", run_id, e)
            result.validation_errors.append(f"TTS error: {e}")
            metrics.fail("tts", str(e))
    elif not scenario.include_tts:
        metrics.tts_provider = None
        metrics.tts_model = None

    # ----- Validation -----
    _validate(result, scenario)

    # ----- Finalize -----
    if not result.validation_errors:
        metrics.complete()
        result.success = True
    else:
        # If only TTS failed but agent succeeded, still mark agent as ok
        # but overall success = False
        if not metrics.error_stage:
            metrics.complete()
        result.success = len(result.validation_errors) == 0

    metrics.total_tokens = result.usage.get("total_tokens")
    metrics.prompt_tokens = result.usage.get("prompt_tokens")
    metrics.completion_tokens = result.usage.get("completion_tokens")
    result.total_processing_ms = metrics.total_processing_ms

    # Persist
    persist_metrics(db, metrics)
    _persist_benchmark(db, metrics, result, scenario)

    logger.info(
        "[BENCH] run_completed run_id=%s scenario=%s success=%s "
        "llm_ms=%s tts_ms=%s total_ms=%s tools=%d",
        run_id,
        scenario_id,
        result.success,
        result.llm_latency_ms,
        result.tts_latency_ms,
        result.total_processing_ms,
        result.actual_tool_calls,
    )

    return result


async def run_scenario_n_times(
    db: Session,
    scenario_id: str,
    repetitions: int = 1,
    **kwargs: Any,
) -> list[BenchmarkRunResult]:
    """Run a scenario sequentially N times and return all results."""
    results: list[BenchmarkRunResult] = []
    for i in range(repetitions):
        logger.info("[BENCH] repetition %d/%d scenario=%s", i + 1, repetitions, scenario_id)
        r = await run_scenario(db, scenario_id, **kwargs)
        results.append(r)
    return results


def aggregate_results(results: list[BenchmarkRunResult]) -> BenchmarkAggregation:
    """Compute basic statistics across multiple runs."""
    if not results:
        return BenchmarkAggregation(
            scenario_id="",
            count=0,
            success_count=0,
            failure_count=0,
            success_rate=0.0,
            llm_latency_ms={},
            tts_latency_ms={},
            total_processing_ms={},
        )

    scenario_id = results[0].scenario_id
    success_count = sum(1 for r in results if r.success)
    failure_count = len(results) - success_count

    def _stats(values: list[float | None]) -> dict[str, float | None]:
        clean = [v for v in values if v is not None]
        if not clean:
            return {"avg": None, "median": None, "min": None, "max": None}
        return {
            "avg": round(statistics.mean(clean), 2),
            "median": round(statistics.median(clean), 2),
            "min": round(min(clean), 2),
            "max": round(max(clean), 2),
        }

    return BenchmarkAggregation(
        scenario_id=scenario_id,
        count=len(results),
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_count / len(results),
        llm_latency_ms=_stats([r.llm_latency_ms for r in results]),
        tts_latency_ms=_stats([r.tts_latency_ms for r in results]),
        total_processing_ms=_stats([r.total_processing_ms for r in results]),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(result: BenchmarkRunResult, scenario: BenchmarkScenario) -> None:
    """Validate run result against scenario expectations."""
    if result.actual_tool_calls != scenario.expected_tool_calls:
        result.tool_call_match = False
        result.validation_errors.append(
            f"Expected {scenario.expected_tool_calls} tool call(s), "
            f"got {result.actual_tool_calls}"
        )
    else:
        result.tool_call_match = True


def _persist_benchmark(
    db: Session,
    metrics: VoiceTurnMetrics,
    result: BenchmarkRunResult,
    scenario: BenchmarkScenario,
) -> None:
    """Add benchmark-specific fields to the persisted BenchmarkResult."""
    try:
        br = metrics.to_benchmark_result()
        # Benchmark-specific metadata
        br.session_id = None  # benchmarks don't use voice sessions
        br.scenario_id = scenario.scenario_id
        br.run_id = result.run_id
        br.benchmark_mode = BENCHMARK_MODE
        db.add(br)
        db.commit()
        db.refresh(br)
        result.benchmark_result_id = br.id
        logger.info(
            "[BENCH] persisted run_id=%s benchmark_id=%s scenario=%s",
            result.run_id,
            br.id,
            scenario.scenario_id,
        )
    except Exception as e:
        logger.error(
            "[BENCH] persistence failed run_id=%s error=%s",
            result.run_id,
            e,
        )
        try:
            db.rollback()
        except Exception:
            pass
