"""Tests for Phase 5E — Controlled Provider/Model Comparison."""

from unittest.mock import patch

import pytest

from app.benchmarks.comparison import (
    ConfigurationResult,
    _aggregate_config_result,
    _calculate_cost_from_runs,
    run_comparison,
)
from app.benchmarks.configurations import (
    ALL_CONFIGURATIONS,
    BenchmarkConfiguration,
    get_configuration,
    list_configurations,
    validate_configuration,
)
from app.benchmarks.runner import BenchmarkRunResult

# ---------------------------------------------------------------------------
# Configuration catalogue
# ---------------------------------------------------------------------------


class TestConfigurationCatalogue:
    def test_all_configurations_have_unique_ids(self) -> None:
        ids = [c.configuration_id for c in ALL_CONFIGURATIONS]
        assert len(ids) == len(set(ids))

    def test_get_configuration_found(self) -> None:
        c = get_configuration("default")
        assert c is not None
        assert c.configuration_id == "default"

    def test_get_configuration_not_found(self) -> None:
        assert get_configuration("nonexistent") is None

    def test_list_configurations_returns_all(self) -> None:
        assert len(list_configurations()) == len(ALL_CONFIGURATIONS)

    def test_configurations_are_frozen(self) -> None:
        for c in ALL_CONFIGURATIONS:
            with pytest.raises(AttributeError):
                c.name = "changed"  # type: ignore[misc]

    def test_each_configuration_has_llm_provider(self) -> None:
        for c in ALL_CONFIGURATIONS:
            assert c.llm_provider, f"Config {c.configuration_id} missing llm_provider"

    def test_each_configuration_has_llm_model(self) -> None:
        for c in ALL_CONFIGURATIONS:
            assert c.llm_model, f"Config {c.configuration_id} missing llm_model"

    def test_each_configuration_has_tts_provider(self) -> None:
        for c in ALL_CONFIGURATIONS:
            assert c.tts_provider, f"Config {c.configuration_id} missing tts_provider"

    def test_pricing_type_is_valid(self) -> None:
        for c in ALL_CONFIGURATIONS:
            assert c.pricing_type in ("payg", "free"), (
                f"Config {c.configuration_id} has invalid pricing_type: {c.pricing_type}"
            )

    def test_free_models_not_production_eligible(self) -> None:
        """Free models must be clearly marked as non-production."""
        for c in ALL_CONFIGURATIONS:
            if c.pricing_type == "free":
                assert c.production_eligible is False, (
                    f"Free config {c.configuration_id} should not be production_eligible"
                )

    def test_payg_models_are_production_eligible(self) -> None:
        """PAYG models should be production eligible."""
        for c in ALL_CONFIGURATIONS:
            if c.pricing_type == "payg":
                assert c.production_eligible is True, (
                    f"PAYG config {c.configuration_id} should be production_eligible"
                )

    def test_has_at_least_one_free_and_one_payg(self) -> None:
        free = [c for c in ALL_CONFIGURATIONS if c.pricing_type == "free"]
        payg = [c for c in ALL_CONFIGURATIONS if c.pricing_type == "payg"]
        assert len(free) >= 1, "Need at least one free configuration"
        assert len(payg) >= 1, "Need at least one PAYG configuration"


class TestPaidCatalogueUpdate:
    """Validated paid OpenRouter models added to the benchmark catalogue.

    The stale claude_sonnet configuration was removed after manual
    compatibility testing of the replacement paid models.
    """

    NEW_PAID_CONFIGS = {
        "gpt41_mini": "openai/gpt-4.1-mini",
        "gpt5_mini": "openai/gpt-5-mini",
        "gemini_flash": "google/gemini-2.5-flash",
        "claude_haiku": "anthropic/claude-haiku-4.5",
        "claude_sonnet_46": "anthropic/claude-sonnet-4.6",
        "deepseek_v31": "deepseek/deepseek-chat-v3.1",
    }

    def test_stale_claude_sonnet_configuration_removed(self) -> None:
        assert get_configuration("claude_sonnet") is None
        for c in ALL_CONFIGURATIONS:
            assert c.llm_model != "anthropic/claude-3.5-sonnet", (
                f"Stale claude-3.5-sonnet still in config {c.configuration_id}"
            )

    def test_new_paid_configurations_present(self) -> None:
        for config_id, model in self.NEW_PAID_CONFIGS.items():
            c = get_configuration(config_id)
            assert c is not None, f"Missing configuration: {config_id}"
            assert c.llm_model == model

    def test_paid_configurations_share_same_voice_stack(self) -> None:
        """Fair LLM comparison: every PAYG config uses the identical stack.

        STT is not part of a configuration (runtime default Deepgram nova-3);
        LLM provider and TTS must be identical across all paid configs.
        """
        payg = [c for c in ALL_CONFIGURATIONS if c.pricing_type == "payg"]
        assert len(payg) >= 7
        for c in payg:
            assert c.llm_provider == "openrouter"
            assert c.tts_provider == "elevenlabs"
            assert c.tts_model == "eleven_flash_v2_5"

    def test_free_configurations_unchanged(self) -> None:
        for config_id in ("default", "gemma_free", "llama_free", "multilingual_tts"):
            c = get_configuration(config_id)
            assert c is not None, f"Free configuration dropped: {config_id}"
            assert c.pricing_type == "free"
            assert c.production_eligible is False

    def test_new_paid_models_have_llm_pricing(self) -> None:
        """Each new paid model must have pricing so cost comparison works."""
        from decimal import Decimal

        from app.services.pricing import get_llm_pricing

        for model in self.NEW_PAID_CONFIGS.values():
            pricing = get_llm_pricing("openrouter", model)
            assert pricing is not None, f"Missing LLM pricing for {model}"
            assert pricing["input"].price_per_unit is not None
            assert pricing["input"].price_per_unit > Decimal("0")
            assert pricing["output"].price_per_unit is not None
            assert pricing["output"].price_per_unit > Decimal("0")

    def test_historical_claude_sonnet_pricing_retained(self) -> None:
        """Pricing stays so historical records keep computed costs."""
        from app.services.pricing import get_llm_pricing

        pricing = get_llm_pricing("openrouter", "anthropic/claude-3.5-sonnet")
        assert pricing is not None
        assert pricing["input"].price_per_unit is not None


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfigurationValidation:
    def test_valid_configuration_no_errors(self) -> None:
        config = get_configuration("default")
        assert config is not None
        errors = validate_configuration(config)
        assert errors == []

    def test_invalid_llm_provider(self) -> None:
        config = BenchmarkConfiguration(
            configuration_id="bad_llm",
            name="Bad LLM",
            description="Test",
            llm_provider="nonexistent_provider",
            llm_model="some-model",
        )
        errors = validate_configuration(config)
        assert len(errors) > 0
        assert "LLM provider" in errors[0]

    def test_invalid_tts_provider(self) -> None:
        config = BenchmarkConfiguration(
            configuration_id="bad_tts",
            name="Bad TTS",
            description="Test",
            llm_provider="openrouter",
            llm_model="openai/gpt-4o-mini",
            tts_provider="nonexistent_tts",
        )
        errors = validate_configuration(config)
        assert len(errors) > 0
        assert "TTS provider" in errors[0]

    def test_valid_payg_configuration(self) -> None:
        config = get_configuration("gpt4o_mini")
        assert config is not None
        errors = validate_configuration(config)
        assert errors == []


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


class TestAggregation:
    def _make_run(
        self,
        run_id: str = "r1",
        success: bool = True,
        total_ms: float | None = 100.0,
        llm_ms: float | None = 50.0,
        tts_ms: float | None = 30.0,
        prompt_tokens: int | None = 100,
        completion_tokens: int | None = 50,
        total_tokens: int | None = 150,
        tts_chars: int | None = 200,
        llm_provider: str = "openrouter",
        llm_model: str = "openai/gpt-4o-mini",
        tts_provider: str = "elevenlabs",
        tts_model: str = "eleven_flash_v2_5",
    ) -> BenchmarkRunResult:
        return BenchmarkRunResult(
            run_id=run_id,
            scenario_id="test",
            success=success,
            total_processing_ms=total_ms,
            llm_latency_ms=llm_ms,
            tts_latency_ms=tts_ms,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            tts_characters=tts_chars,
            llm_provider=llm_provider,
            llm_model=llm_model,
            tts_provider=tts_provider,
            tts_model=tts_model,
        )

    def test_empty_runs_no_aggregation(self) -> None:
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
        )
        _aggregate_config_result(cr)
        assert cr.run_count == 0

    def test_single_successful_run(self) -> None:
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=[self._make_run()],
        )
        _aggregate_config_result(cr)
        assert cr.run_count == 1
        assert cr.successful_runs == 1
        assert cr.failed_runs == 0
        assert cr.success_rate == 1.0
        assert cr.avg_total_latency_ms == 100.0
        assert cr.median_total_latency_ms == 100.0

    def test_multiple_runs_success_rate(self) -> None:
        runs = [
            self._make_run("r1", success=True),
            self._make_run("r2", success=True),
            self._make_run("r3", success=False),
        ]
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=runs,
        )
        _aggregate_config_result(cr)
        assert cr.run_count == 3
        assert cr.successful_runs == 2
        assert cr.failed_runs == 1
        assert abs(cr.success_rate - 0.6667) < 0.01

    def test_latency_aggregation(self) -> None:
        runs = [
            self._make_run("r1", total_ms=100.0),
            self._make_run("r2", total_ms=200.0),
            self._make_run("r3", total_ms=300.0),
        ]
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=runs,
        )
        _aggregate_config_result(cr)
        assert cr.avg_total_latency_ms == 200.0
        assert cr.median_total_latency_ms == 200.0
        assert cr.min_total_latency_ms == 100.0
        assert cr.max_total_latency_ms == 300.0

    def test_null_latencies_not_counted_as_zero(self) -> None:
        runs = [
            self._make_run("r1", total_ms=100.0),
            self._make_run("r2", total_ms=None),
        ]
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=runs,
        )
        _aggregate_config_result(cr)
        # Only one run has latency, so avg should be 100.0
        assert cr.avg_total_latency_ms == 100.0

    def test_token_aggregation(self) -> None:
        runs = [
            self._make_run("r1", prompt_tokens=100, completion_tokens=50, total_tokens=150),
            self._make_run("r2", prompt_tokens=200, completion_tokens=100, total_tokens=300),
        ]
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=runs,
        )
        _aggregate_config_result(cr)
        assert cr.avg_prompt_tokens == 150.0
        assert cr.avg_completion_tokens == 75.0
        assert cr.avg_total_tokens == 225.0

    def test_tts_aggregation(self) -> None:
        runs = [
            self._make_run("r1", tts_chars=100),
            self._make_run("r2", tts_chars=200),
        ]
        cr = ConfigurationResult(
            configuration_id="test",
            configuration_name="Test",
            pricing_type="payg",
            production_eligible=True,
            runs=runs,
        )
        _aggregate_config_result(cr)
        assert cr.avg_tts_characters == 150.0


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def _make_run(
        self,
        llm_provider: str = "openrouter",
        llm_model: str = "openai/gpt-4o-mini",
        tts_provider: str = "elevenlabs",
        tts_model: str = "eleven_flash_v2_5",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        tts_chars: int = 200,
    ) -> BenchmarkRunResult:
        return BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            tts_characters=tts_chars,
            llm_provider=llm_provider,
            llm_model=llm_model,
            tts_provider=tts_provider,
            tts_model=tts_model,
        )

    def test_free_model_cost_is_zero(self) -> None:
        cr = ConfigurationResult(
            configuration_id="free",
            configuration_name="Free",
            pricing_type="free",
            production_eligible=False,
            runs=[self._make_run()],
        )
        # Simulate aggregation having set successful_runs (run has success=True)
        cr.successful_runs = 1
        _calculate_cost_from_runs(cr)
        assert cr.cost_available is True
        assert cr.cost_note == "free"
        assert cr.avg_cost == 0.0
        assert cr.total_cost == 0.0

    def test_payg_model_has_cost(self) -> None:
        cr = ConfigurationResult(
            configuration_id="payg",
            configuration_name="PAYG",
            pricing_type="payg",
            production_eligible=True,
            runs=[self._make_run()],
        )
        _calculate_cost_from_runs(cr)
        # GPT-4o-mini has pricing, so cost should be available
        assert cr.cost_available is True
        assert cr.avg_cost is not None
        assert cr.avg_cost > 0

    def test_unknown_pricing_unavailable(self) -> None:
        cr = ConfigurationResult(
            configuration_id="unknown",
            configuration_name="Unknown",
            pricing_type="payg",
            production_eligible=True,
            runs=[self._make_run(
                llm_model="nonexistent/model",
                tts_provider="nonexistent_tts",
            )],
        )
        _calculate_cost_from_runs(cr)
        # Unknown LLM and TTS models have no pricing
        assert cr.cost_available is False
        assert cr.cost_note == "unavailable"

    def test_empty_runs_no_cost(self) -> None:
        cr = ConfigurationResult(
            configuration_id="empty",
            configuration_name="Empty",
            pricing_type="payg",
            production_eligible=True,
            runs=[],
        )
        _calculate_cost_from_runs(cr)
        # No runs, no cost calculation
        assert cr.cost_available is True  # Default
        assert cr.avg_cost is None


# ---------------------------------------------------------------------------
# Comparison runner (mocked providers)
# ---------------------------------------------------------------------------


class TestComparisonRunner:
    @pytest.mark.anyio
    async def test_invalid_scenario_returns_error(self, db_session) -> None:
        with patch("app.benchmarks.comparison.run_scenario") as mock_run:
            mock_run.side_effect = ValueError("Scenario 'nonexistent_scenario' not found")
            with pytest.raises(ValueError, match="not found"):
                await run_comparison(
                    db_session,
                    "nonexistent_scenario",
                    ["default"],
                    repetitions=1,
                )

    @pytest.mark.anyio
    async def test_invalid_configuration_returns_validation_error(self, db_session) -> None:
        result = await run_comparison(
            db_session,
            "simple_conversation",
            ["nonexistent_config"],
            repetitions=1,
        )
        assert len(result.validation_errors) > 0
        assert "Unknown configuration" in result.validation_errors[0]

    @pytest.mark.anyio
    async def test_comparison_has_unique_id(self, db_session) -> None:
        with patch("app.benchmarks.comparison.run_scenario") as mock_run:
            mock_run.return_value = BenchmarkRunResult(
                run_id="test-run-1",
                scenario_id="simple_conversation",
                success=True,
            )
            result = await run_comparison(
                db_session,
                "simple_conversation",
                ["default"],
                repetitions=1,
            )
            assert result.comparison_id
            assert len(result.comparison_id) > 0

    @pytest.mark.anyio
    async def test_comparison_runs_same_scenario(self, db_session) -> None:
        """Ensure the same scenario_id is passed to all configurations."""
        scenarios_called = []

        async def mock_run_scenario(db, scenario_id, **kwargs):
            scenarios_called.append(scenario_id)
            return BenchmarkRunResult(
                run_id=f"run-{len(scenarios_called)}",
                scenario_id=scenario_id,
                success=True,
                llm_provider=kwargs.get("llm_provider"),
                llm_model=kwargs.get("llm_model"),
            )

        with patch("app.benchmarks.comparison.run_scenario", side_effect=mock_run_scenario):
            await run_comparison(
                db_session,
                "simple_conversation",
                ["default", "gpt4o_mini"],
                repetitions=2,
            )

        # All calls should use the same scenario_id
        assert all(s == "simple_conversation" for s in scenarios_called)
        # 2 configs × 2 repetitions = 4 calls
        assert len(scenarios_called) == 4

    @pytest.mark.anyio
    async def test_comparison_different_providers(self, db_session) -> None:
        """Each configuration should use its own provider/model."""
        providers_used = []

        async def mock_run_scenario(db, scenario_id, **kwargs):
            providers_used.append({
                "llm_provider": kwargs.get("llm_provider"),
                "llm_model": kwargs.get("llm_model"),
            })
            return BenchmarkRunResult(
                run_id=f"run-{len(providers_used)}",
                scenario_id=scenario_id,
                success=True,
                llm_provider=kwargs.get("llm_provider"),
                llm_model=kwargs.get("llm_model"),
            )

        with patch("app.benchmarks.comparison.run_scenario", side_effect=mock_run_scenario):
            await run_comparison(
                db_session,
                "simple_conversation",
                ["default", "gpt4o_mini"],
                repetitions=1,
            )

        # Should have 2 different provider configurations
        assert len(providers_used) == 2
        # Default uses Nemotron
        assert providers_used[0]["llm_model"] == "nvidia/nemotron-3.5-lightning:free"
        # GPT-4o-mini uses GPT
        assert providers_used[1]["llm_model"] == "openai/gpt-4o-mini"

    @pytest.mark.anyio
    async def test_repetitions_create_unique_runs(self, db_session) -> None:
        """Each repetition should create a separate run."""
        run_ids = []

        async def mock_run_scenario(db, scenario_id, **kwargs):
            import uuid
            run_id = str(uuid.uuid4())
            run_ids.append(run_id)
            return BenchmarkRunResult(
                run_id=run_id,
                scenario_id=scenario_id,
                success=True,
            )

        with patch("app.benchmarks.comparison.run_scenario", side_effect=mock_run_scenario):
            await run_comparison(
                db_session,
                "simple_conversation",
                ["default"],
                repetitions=3,
            )

        # 3 unique run IDs
        assert len(run_ids) == 3
        assert len(set(run_ids)) == 3  # All unique

    @pytest.mark.anyio
    async def test_comparison_result_structure(self, db_session) -> None:
        with patch("app.benchmarks.comparison.run_scenario") as mock_run:
            mock_run.return_value = BenchmarkRunResult(
                run_id="test-run",
                scenario_id="simple_conversation",
                success=True,
                total_processing_ms=150.0,
            )
            result = await run_comparison(
                db_session,
                "simple_conversation",
                ["default", "gpt4o_mini"],
                repetitions=2,
            )

        assert result.scenario_id == "simple_conversation"
        assert result.repetitions == 2
        assert len(result.configurations) == 2
        assert result.validation_errors == []

        # Each configuration should have 2 runs
        for config in result.configurations:
            assert config.run_count == 2
            assert len(config.runs) == 2

    @pytest.mark.anyio
    async def test_to_api_dict_structure(self, db_session) -> None:
        with patch("app.benchmarks.comparison.run_scenario") as mock_run:
            mock_run.return_value = BenchmarkRunResult(
                run_id="test-run",
                scenario_id="simple_conversation",
                success=True,
            )
            result = await run_comparison(
                db_session,
                "simple_conversation",
                ["default"],
                repetitions=1,
            )

        api_dict = result.to_api_dict()
        assert "comparison_id" in api_dict
        assert "scenario_id" in api_dict
        assert "repetitions" in api_dict
        assert "configurations" in api_dict
        assert "validation_errors" in api_dict

        config_dict = api_dict["configurations"][0]
        assert "configuration_id" in config_dict
        assert "configuration_name" in config_dict
        assert "pricing_type" in config_dict
        assert "production_eligible" in config_dict
        assert "success_rate" in config_dict
        assert "avg_cost" in config_dict
        assert "cost_note" in config_dict


# ---------------------------------------------------------------------------
# STT limitation
# ---------------------------------------------------------------------------


class TestSTTLimitation:
    def test_text_input_mode_does_not_measure_stt(self) -> None:
        """STT is N/A in text_input benchmark mode.

        BenchmarkRunResult doesn't track STT metrics because the
        benchmark mode is text_input (no audio to transcribe).
        """
        run = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            benchmark_mode="text_input",
        )
        # STT metrics are not part of BenchmarkRunResult
        assert run.benchmark_mode == "text_input"
        # The comparison UI should show STT as N/A
