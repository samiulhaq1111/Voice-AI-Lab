"""Tests for Phase 5E UX — benchmark data lifecycle, reset, and edge cases."""

from fastapi.testclient import TestClient

from app.main import app
from app.models.benchmark_result import BenchmarkResult


def _get_client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Reset endpoint
# ---------------------------------------------------------------------------


class TestResetEndpoint:
    def test_reset_empty_database(self, db_session) -> None:
        """Reset on empty DB should return 0 deleted."""
        client = _get_client()
        resp = client.delete("/api/v1/benchmarks/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 0
        assert data["status"] == "ok"

    def test_reset_deletes_benchmark_results(self, db_session) -> None:
        """Reset should delete all BenchmarkResult rows."""
        # Insert some records
        for i in range(3):
            br = BenchmarkResult(
                scenario_id="test",
                run_id=f"run-{i}",
                benchmark_mode="text_input",
                llm_provider="openrouter",
                llm_model="test-model",
            )
            db_session.add(br)
        db_session.commit()

        # Verify records exist
        count_before = db_session.query(BenchmarkResult).count()
        assert count_before == 3

        # Reset
        client = _get_client()
        resp = client.delete("/api/v1/benchmarks/results")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3

        # Verify all deleted
        count_after = db_session.query(BenchmarkResult).count()
        assert count_after == 0

    def test_reset_preserves_nothing_else(self, db_session) -> None:
        """Reset only deletes BenchmarkResult rows, nothing else."""
        client = _get_client()
        resp = client.delete("/api/v1/benchmarks/results")
        assert resp.status_code == 200

        # Scenarios endpoint should still work
        resp2 = client.get("/api/v1/benchmarks/scenarios")
        assert resp2.status_code == 200
        scenarios = resp2.json()
        assert len(scenarios) > 0

        # Configurations endpoint should still work
        resp3 = client.get("/api/v1/benchmarks/configurations")
        assert resp3.status_code == 200
        configs = resp3.json()
        assert len(configs) > 0


# ---------------------------------------------------------------------------
# Analytics with zero/failed records
# ---------------------------------------------------------------------------


class TestAnalyticsEdgeCases:
    def test_overall_summary_zero_records(self, db_session) -> None:
        """Overall summary with zero records should return zeros."""
        client = _get_client()
        resp = client.get("/api/v1/benchmarks/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0
        assert data["successful_runs"] == 0
        assert data["failed_runs"] == 0
        assert data["success_rate"] == 0.0
        assert data["latency"]["avg_ms"] is None

    def test_overall_summary_failed_only(self, db_session) -> None:
        """Overall summary with only failed records."""
        # Insert a failed record
        br = BenchmarkResult(
            scenario_id="test",
            run_id="failed-run",
            benchmark_mode="text_input",
            llm_provider="openrouter",
            llm_model="test-model",
            conversation_success="false",
            error_stage="llm",
            error_message="HTTP 429 rate limit exceeded",
        )
        db_session.add(br)
        db_session.commit()

        client = _get_client()
        resp = client.get("/api/v1/benchmarks/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 1
        assert data["successful_runs"] == 0
        assert data["failed_runs"] == 1
        assert data["success_rate"] == 0.0
        # Latency should be null since no successful measurement
        assert data["latency"]["avg_ms"] is None

    def test_recent_results_empty(self, db_session) -> None:
        """Recent results with zero records should return empty list."""
        client = _get_client()
        resp = client.get("/api/v1/benchmarks/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    def test_recent_results_include_error_message(self, db_session) -> None:
        """Recent results should include error_message for error classification."""
        br = BenchmarkResult(
            scenario_id="test",
            run_id="err-run",
            benchmark_mode="text_input",
            conversation_success="false",
            error_message="HTTP 429 rate limit",
        )
        db_session.add(br)
        db_session.commit()

        client = _get_client()
        resp = client.get("/api/v1/benchmarks/results?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["error_message"] == "HTTP 429 rate limit"

    def test_recent_results_limit_10(self, db_session) -> None:
        """Recent results should respect limit parameter."""
        for i in range(15):
            br = BenchmarkResult(
                scenario_id="test",
                run_id=f"run-{i}",
                benchmark_mode="text_input",
            )
            db_session.add(br)
        db_session.commit()

        client = _get_client()
        resp = client.get("/api/v1/benchmarks/results?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10

    def test_cost_summary_zero_records(self, db_session) -> None:
        """Cost summary with zero records."""
        client = _get_client()
        resp = client.get("/api/v1/benchmarks/cost/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0
        assert data["runs_with_cost"] == 0
        assert data["total_cost"] is None


# ---------------------------------------------------------------------------
# Reset then verify analytics is clean
# ---------------------------------------------------------------------------


class TestResetThenAnalytics:
    def test_reset_then_summary_is_zero(self, db_session) -> None:
        """After reset, summary should show zero runs."""
        # Insert records
        br = BenchmarkResult(
            scenario_id="test",
            run_id="run-1",
            benchmark_mode="text_input",
        )
        db_session.add(br)
        db_session.commit()

        client = _get_client()

        # Verify record exists
        resp = client.get("/api/v1/benchmarks/summary")
        assert resp.json()["total_runs"] == 1

        # Reset
        resp = client.delete("/api/v1/benchmarks/results")
        assert resp.json()["deleted"] == 1

        # Verify summary is now zero
        resp = client.get("/api/v1/benchmarks/summary")
        assert resp.json()["total_runs"] == 0

    def test_reset_then_recent_is_empty(self, db_session) -> None:
        """After reset, recent results should be empty."""
        br = BenchmarkResult(
            scenario_id="test",
            run_id="run-1",
            benchmark_mode="text_input",
        )
        db_session.add(br)
        db_session.commit()

        client = _get_client()

        # Reset
        client.delete("/api/v1/benchmarks/results")

        # Verify recent is empty
        resp = client.get("/api/v1/benchmarks/results")
        assert resp.json() == []
