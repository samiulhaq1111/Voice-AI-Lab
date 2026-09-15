"""Tests for Phase 5C benchmark analytics service."""


import pytest
from sqlalchemy.orm import Session

from app.models.benchmark_result import BenchmarkResult
from app.services.benchmark_analytics import (
    BenchmarkAnalyticsService,
    LatencyStats,
    _avg_float,
    _avg_int,
    _compute_latency_stats,
    _is_success,
)

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_compute_latency_stats_empty(self) -> None:
        stats = _compute_latency_stats([])
        assert stats.avg_ms is None
        assert stats.median_ms is None

    def test_compute_latency_stats_all_null(self) -> None:
        stats = _compute_latency_stats([None, None, None])
        assert stats.avg_ms is None

    def test_compute_latency_stats_odd(self) -> None:
        stats = _compute_latency_stats([100.0, 200.0, 300.0])
        assert stats.avg_ms == 200.0
        assert stats.median_ms == 200.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 300.0

    def test_compute_latency_stats_even(self) -> None:
        stats = _compute_latency_stats([100.0, 200.0, 300.0, 400.0])
        assert stats.avg_ms == 250.0
        assert stats.median_ms == 250.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 400.0

    def test_compute_latency_stats_with_nulls(self) -> None:
        stats = _compute_latency_stats([100.0, None, 300.0])
        assert stats.avg_ms == 200.0
        assert stats.median_ms == 200.0

    def test_avg_int_empty(self) -> None:
        assert _avg_int([]) is None

    def test_avg_int_all_null(self) -> None:
        assert _avg_int([None, None]) is None

    def test_avg_int_values(self) -> None:
        assert _avg_int([10, 20, 30]) == 20.0

    def test_avg_int_with_nulls(self) -> None:
        assert _avg_int([10, None, 30]) == 20.0

    def test_avg_float_empty(self) -> None:
        assert _avg_float([]) is None

    def test_avg_float_values(self) -> None:
        assert _avg_float([10.0, 20.0]) == 15.0

    def test_is_success_true(self) -> None:
        r = BenchmarkResult(conversation_success="true")
        assert _is_success(r) is True

    def test_is_success_false(self) -> None:
        r = BenchmarkResult(conversation_success="false")
        assert _is_success(r) is False

    def test_is_success_none(self) -> None:
        r = BenchmarkResult(conversation_success=None)
        assert _is_success(r) is False


# ---------------------------------------------------------------------------
# Service tests with in-memory database
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session() -> Session:
    """Create an in-memory SQLite database session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def _make_result(
    *,
    scenario_id: str = "test_scenario",
    success: bool = True,
    total_processing_ms: float | None = 1000.0,
    stt_latency_ms: float | None = None,
    llm_latency_ms: float | None = 500.0,
    tts_latency_ms: float | None = 300.0,
    tool_execution_ms: float | None = None,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 200,
    token_usage: int | None = 300,
    tts_characters: int | None = 50,
    tts_audio_bytes: int | None = 1000,
    llm_provider: str = "openrouter",
    llm_model: str = "test-model",
    tts_provider: str = "elevenlabs",
    tts_model: str = "test-tts",
    benchmark_mode: str = "text_input",
) -> BenchmarkResult:
    """Helper to create a BenchmarkResult for testing."""
    return BenchmarkResult(
        scenario_id=scenario_id,
        run_id="test-run-id",
        benchmark_mode=benchmark_mode,
        conversation_success="true" if success else "false",
        total_processing_ms=total_processing_ms,
        stt_latency_ms=stt_latency_ms,
        llm_latency_ms=llm_latency_ms,
        tts_latency_ms=tts_latency_ms,
        tool_execution_ms=tool_execution_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_usage=token_usage,
        tts_characters=tts_characters,
        tts_audio_bytes=tts_audio_bytes,
        llm_provider=llm_provider,
        llm_model=llm_model,
        tts_provider=tts_provider,
        tts_model=tts_model,
    )


class TestOverallSummary:
    def test_empty_database(self, db_session: Session) -> None:
        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary()
        assert summary.total_runs == 0
        assert summary.successful_runs == 0
        assert summary.success_rate == 0.0
        assert summary.latency.avg_ms is None

    def test_successful_runs(self, db_session: Session) -> None:
        db_session.add(_make_result(total_processing_ms=1000.0))
        db_session.add(_make_result(total_processing_ms=2000.0))
        db_session.add(_make_result(total_processing_ms=3000.0))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary()
        assert summary.total_runs == 3
        assert summary.successful_runs == 3
        assert summary.success_rate == 1.0
        assert summary.latency.avg_ms == 2000.0
        assert summary.latency.median_ms == 2000.0
        assert summary.latency.min_ms == 1000.0
        assert summary.latency.max_ms == 3000.0

    def test_failed_runs(self, db_session: Session) -> None:
        db_session.add(_make_result(success=True))
        db_session.add(_make_result(success=False, total_processing_ms=None))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary()
        assert summary.total_runs == 2
        assert summary.successful_runs == 1
        assert summary.failed_runs == 1
        assert summary.success_rate == 0.5

    def test_null_metrics_excluded(self, db_session: Session) -> None:
        # text_input benchmarks have no STT
        db_session.add(_make_result(stt_latency_ms=None, llm_latency_ms=500.0))
        db_session.add(_make_result(stt_latency_ms=None, llm_latency_ms=700.0))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary()
        # STT should be None (not 0)
        assert summary.stt_latency.avg_ms is None
        # LLM should be computed
        assert summary.llm_latency.avg_ms == 600.0

    def test_token_averages(self, db_session: Session) -> None:
        db_session.add(_make_result(prompt_tokens=100, completion_tokens=200, token_usage=300))
        db_session.add(_make_result(prompt_tokens=200, completion_tokens=400, token_usage=600))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary()
        assert summary.avg_prompt_tokens == 150.0
        assert summary.avg_completion_tokens == 300.0
        assert summary.avg_total_tokens == 450.0


class TestScenarioSummaries:
    def test_grouping(self, db_session: Session) -> None:
        db_session.add(_make_result(scenario_id="scenario_a", total_processing_ms=1000.0))
        db_session.add(_make_result(scenario_id="scenario_a", total_processing_ms=2000.0))
        db_session.add(_make_result(scenario_id="scenario_b", total_processing_ms=5000.0))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summaries = svc.get_scenario_summaries()
        assert len(summaries) == 2

        a = next(s for s in summaries if s.scenario_id == "scenario_a")
        assert a.run_count == 2
        assert a.latency.avg_ms == 1500.0

        b = next(s for s in summaries if s.scenario_id == "scenario_b")
        assert b.run_count == 1
        assert b.latency.avg_ms == 5000.0


class TestProviderSummaries:
    def test_llm_grouping(self, db_session: Session) -> None:
        db_session.add(_make_result(llm_provider="openrouter", llm_model="model-a"))
        db_session.add(_make_result(llm_provider="openrouter", llm_model="model-a"))
        db_session.add(_make_result(llm_provider="openrouter", llm_model="model-b"))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summaries = svc.get_provider_summaries()
        llm_summaries = [s for s in summaries if s.stage == "llm"]
        assert len(llm_summaries) == 2

        model_a = next(s for s in llm_summaries if s.model == "model-a")
        assert model_a.run_count == 2

    def test_tts_grouping(self, db_session: Session) -> None:
        db_session.add(_make_result(tts_provider="elevenlabs", tts_model="flash"))
        db_session.add(_make_result(tts_provider="elevenlabs", tts_model="flash"))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summaries = svc.get_provider_summaries()
        tts_summaries = [s for s in summaries if s.stage == "tts"]
        assert len(tts_summaries) == 1
        assert tts_summaries[0].run_count == 2


class TestFilters:
    def test_filter_by_scenario(self, db_session: Session) -> None:
        db_session.add(_make_result(scenario_id="a"))
        db_session.add(_make_result(scenario_id="b"))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary(scenario_id="a")
        assert summary.total_runs == 1

    def test_filter_by_benchmark_mode(self, db_session: Session) -> None:
        db_session.add(_make_result(benchmark_mode="text_input"))
        db_session.add(_make_result(benchmark_mode="voice"))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        summary = svc.get_overall_summary(benchmark_mode="text_input")
        assert summary.total_runs == 1


class TestRecentResults:
    def test_limit(self, db_session: Session) -> None:
        for i in range(10):
            db_session.add(_make_result(total_processing_ms=float(i * 100)))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        results = svc.get_recent_results(limit=5)
        assert len(results) == 5

    def test_ordering(self, db_session: Session) -> None:
        for i in range(3):
            db_session.add(_make_result(total_processing_ms=float(i * 100)))
        db_session.commit()

        svc = BenchmarkAnalyticsService(db_session)
        results = svc.get_recent_results(limit=10)
        # Most recent first (created_at desc)
        assert len(results) == 3


class TestAPISchemas:
    def test_latency_stats_to_dict(self) -> None:
        stats = LatencyStats(avg_ms=100.0, median_ms=90.0, min_ms=50.0, max_ms=200.0)
        d = stats.to_dict()
        assert d["avg_ms"] == 100.0
        assert d["median_ms"] == 90.0

    def test_latency_stats_none(self) -> None:
        stats = LatencyStats()
        d = stats.to_dict()
        assert d["avg_ms"] is None
