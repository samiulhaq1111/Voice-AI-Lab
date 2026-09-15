"""Tests for individual benchmark run configuration selection.

Verifies:
1. Individual run accepts configuration_id.
2. Valid configuration_id is resolved from catalog.
3. Configuration provider/model values are passed to run_scenario().
4. BenchmarkResult stores configuration_id.
5. Invalid configuration_id returns validation error.
6. Existing scenario validation still works.
7. Run without configuration_id still works (backward compatible).
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.benchmarks.configurations import (
    get_configuration,
    list_configurations,
)
from app.benchmarks.runner import BenchmarkRunResult
from app.main import app
from app.models.benchmark_result import BenchmarkResult


def _get_client() -> TestClient:
    return TestClient(app)


def _mock_run_result(
    *,
    configuration_id: str | None = None,
    success: bool = True,
) -> BenchmarkRunResult:
    """Create a minimal BenchmarkRunResult for mocking."""
    return BenchmarkRunResult(
        run_id="test-run-id",
        scenario_id="simple_conversation",
        success=success,
        llm_provider="openrouter",
        llm_model="openai/gpt-4o-mini",
        tts_provider="elevenlabs",
        tts_model="eleven_flash_v2_5",
        configuration_id=configuration_id,
        benchmark_result_id="fake-id",
    )


# ---------------------------------------------------------------------------
# Configuration catalog sanity
# ---------------------------------------------------------------------------


class TestConfigurationCatalog:
    def test_catalog_has_configurations(self) -> None:
        """Catalog must have at least one configuration."""
        configs = list_configurations()
        assert len(configs) > 0

    def test_catalog_has_production_eligible_payg(self) -> None:
        """Catalog must have at least one production-eligible PAYG config."""
        configs = list_configurations()
        payg_prod = [
            c for c in configs
            if c.production_eligible and c.pricing_type == "payg"
        ]
        assert len(payg_prod) > 0

    def test_get_configuration_valid(self) -> None:
        """Looking up a valid configuration returns it."""
        config = get_configuration("gpt4o_mini")
        assert config is not None
        assert config.llm_provider == "openrouter"
        assert config.llm_model == "openai/gpt-4o-mini"

    def test_get_configuration_invalid(self) -> None:
        """Looking up an invalid configuration returns None."""
        config = get_configuration("nonexistent_config_xyz")
        assert config is None


# ---------------------------------------------------------------------------
# API: individual run with configuration_id
# ---------------------------------------------------------------------------


class TestRunEndpointConfiguration:
    @patch("app.api.benchmarks.run_scenario", new_callable=AsyncMock)
    def test_run_with_valid_configuration_id(
        self, mock_run: AsyncMock, db_session
    ) -> None:
        """POST /run with valid configuration_id resolves config
        and passes provider/model to run_scenario."""
        mock_run.return_value = _mock_run_result(
            configuration_id="gpt4o_mini",
        )

        client = _get_client()
        resp = client.post(
            "/api/v1/benchmarks/run",
            json={
                "scenario_id": "simple_conversation",
                "configuration_id": "gpt4o_mini",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["configuration_id"] == "gpt4o_mini"

        # Verify run_scenario was called with config's provider/model
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_provider"] == "openrouter"
        assert call_kwargs.kwargs["llm_model"] == "openai/gpt-4o-mini"
        assert call_kwargs.kwargs["tts_provider"] == "elevenlabs"
        assert call_kwargs.kwargs["tts_model"] == "eleven_flash_v2_5"

    @patch("app.api.benchmarks.run_scenario", new_callable=AsyncMock)
    def test_run_with_invalid_configuration_id(
        self, mock_run: AsyncMock, db_session
    ) -> None:
        """POST /run with invalid configuration_id returns 400."""
        client = _get_client()
        resp = client.post(
            "/api/v1/benchmarks/run",
            json={
                "scenario_id": "simple_conversation",
                "configuration_id": "nonexistent_xyz",
            },
        )
        assert resp.status_code == 400
        assert "nonexistent_xyz" in resp.json()["detail"]
        # run_scenario should NOT have been called
        mock_run.assert_not_called()

    @patch("app.api.benchmarks.run_scenario", new_callable=AsyncMock)
    def test_run_without_configuration_id_backward_compatible(
        self, mock_run: AsyncMock, db_session
    ) -> None:
        """POST /run without configuration_id still works (uses defaults)."""
        mock_run.return_value = _mock_run_result()

        client = _get_client()
        resp = client.post(
            "/api/v1/benchmarks/run",
            json={"scenario_id": "simple_conversation"},
        )
        assert resp.status_code == 200

        # run_scenario should be called with None providers (defaults)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_provider"] is None
        assert call_kwargs.kwargs["llm_model"] is None

    @patch("app.api.benchmarks.run_scenario", new_callable=AsyncMock)
    def test_run_invalid_scenario_still_rejected(
        self, mock_run: AsyncMock, db_session
    ) -> None:
        """Invalid scenario_id returns 404 even with valid configuration_id."""
        client = _get_client()
        resp = client.post(
            "/api/v1/benchmarks/run",
            json={
                "scenario_id": "nonexistent_scenario",
                "configuration_id": "gpt4o_mini",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Persistence: configuration_id stored on BenchmarkResult
# ---------------------------------------------------------------------------


class TestConfigurationPersistence:
    def test_benchmark_result_stores_configuration_id(
        self, db_session
    ) -> None:
        """BenchmarkResult model has configuration_id column."""
        br = BenchmarkResult(
            scenario_id="test",
            run_id="run-config-test",
            benchmark_mode="text_input",
            configuration_id="gpt4o_mini",
            llm_provider="openrouter",
            llm_model="openai/gpt-4o-mini",
        )
        db_session.add(br)
        db_session.commit()

        fetched = (
            db_session.query(BenchmarkResult)
            .filter(BenchmarkResult.run_id == "run-config-test")
            .first()
        )
        assert fetched is not None
        assert fetched.configuration_id == "gpt4o_mini"

    def test_benchmark_result_configuration_id_nullable(
        self, db_session
    ) -> None:
        """BenchmarkResult.configuration_id can be NULL (backward compat)."""
        br = BenchmarkResult(
            scenario_id="test",
            run_id="run-no-config",
            benchmark_mode="text_input",
            llm_provider="openrouter",
            llm_model="test-model",
        )
        db_session.add(br)
        db_session.commit()

        fetched = (
            db_session.query(BenchmarkResult)
            .filter(BenchmarkResult.run_id == "run-no-config")
            .first()
        )
        assert fetched is not None
        assert fetched.configuration_id is None
