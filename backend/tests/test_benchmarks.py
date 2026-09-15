"""Tests for Phase 5B benchmark system."""

import pytest

from app.benchmarks.deterministic_tools import (
    bench_get_employee_handler,
    bench_get_leave_balance_handler,
    bench_get_weather_handler,
    create_benchmark_tools,
)
from app.benchmarks.runner import (
    BenchmarkRunResult,
    _validate,
    aggregate_results,
)
from app.benchmarks.scenarios import (
    ALL_SCENARIOS,
    BenchmarkScenario,
    get_scenario,
    list_scenarios,
)
from app.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

class TestScenarioCatalogue:
    def test_all_scenarios_have_unique_ids(self) -> None:
        ids = [s.scenario_id for s in ALL_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_get_scenario_found(self) -> None:
        s = get_scenario("simple_conversation")
        assert s is not None
        assert s.scenario_id == "simple_conversation"

    def test_get_scenario_not_found(self) -> None:
        assert get_scenario("nonexistent") is None

    def test_list_scenarios_returns_all(self) -> None:
        assert len(list_scenarios()) == len(ALL_SCENARIOS)

    def test_scenarios_are_frozen(self) -> None:
        for s in ALL_SCENARIOS:
            with pytest.raises(AttributeError):
                s.name = "changed"  # type: ignore[misc]

    def test_each_scenario_has_input_text(self) -> None:
        for s in ALL_SCENARIOS:
            assert s.input_text, f"Scenario {s.scenario_id} has empty input_text"

    def test_each_scenario_has_category(self) -> None:
        for s in ALL_SCENARIOS:
            assert s.category in ("baseline", "tool", "stress")

    def test_context_conversation_expects_zero_tools(self) -> None:
        s = get_scenario("context_conversation")
        assert s is not None
        assert s.expected_tool_calls == 0

    def test_context_conversation_discourages_tool_use(self) -> None:
        """Input text must explicitly tell the agent not to look things up."""
        s = get_scenario("context_conversation")
        assert s is not None
        text_lower = s.input_text.lower()
        assert "do not look" in text_lower or "don't look" in text_lower


# ---------------------------------------------------------------------------
# Deterministic tools
# ---------------------------------------------------------------------------

class TestDeterministicTools:
    @pytest.mark.anyio
    async def test_weather_is_deterministic(self) -> None:
        r1 = await bench_get_weather_handler(location="Austin")
        r2 = await bench_get_weather_handler(location="Austin")
        assert r1 == r2
        assert r1["temperature_f"] == "72"
        assert r1["condition"] == "sunny"

    @pytest.mark.anyio
    async def test_employee_lookup_deterministic(self) -> None:
        r1 = await bench_get_employee_handler(employee_id="E001")
        r2 = await bench_get_employee_handler(employee_id="E001")
        assert r1 == r2
        assert r1["name"] == "Alice Johnson"

    @pytest.mark.anyio
    async def test_leave_balance_deterministic(self) -> None:
        r1 = await bench_get_leave_balance_handler(employee_id="E001", leave_type="pto")
        r2 = await bench_get_leave_balance_handler(employee_id="E001", leave_type="pto")
        assert r1 == r2
        assert r1["remaining_days"] == "15"

    @pytest.mark.anyio
    async def test_unknown_employee_returns_unknown(self) -> None:
        r = await bench_get_employee_handler(employee_id="E999")
        assert r["name"] == "Unknown"

    @pytest.mark.anyio
    async def test_unknown_leave_balance_returns_zero(self) -> None:
        r = await bench_get_leave_balance_handler(employee_id="E999", leave_type="pto")
        assert r["remaining_days"] == "0"

    def test_create_benchmark_tools_returns_three(self) -> None:
        tools = create_benchmark_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"get_weather", "get_employee", "get_leave_balance"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_matching_tool_calls_passes(self) -> None:
        scenario = BenchmarkScenario(
            scenario_id="test",
            name="Test",
            description="",
            input_text="test",
            expected_tool_calls=1,
        )
        result = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            actual_tool_calls=1,
            expected_tool_calls=1,
        )
        _validate(result, scenario)
        assert result.tool_call_match is True
        assert len(result.validation_errors) == 0

    def test_mismatched_tool_calls_fails(self) -> None:
        scenario = BenchmarkScenario(
            scenario_id="test",
            name="Test",
            description="",
            input_text="test",
            expected_tool_calls=1,
        )
        result = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            actual_tool_calls=0,
            expected_tool_calls=1,
        )
        _validate(result, scenario)
        assert result.tool_call_match is False
        assert len(result.validation_errors) == 1

    def test_zero_tool_calls_match(self) -> None:
        scenario = BenchmarkScenario(
            scenario_id="test",
            name="Test",
            description="",
            input_text="test",
            expected_tool_calls=0,
        )
        result = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            actual_tool_calls=0,
            expected_tool_calls=0,
        )
        _validate(result, scenario)
        assert result.tool_call_match is True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_empty_results(self) -> None:
        agg = aggregate_results([])
        assert agg.count == 0

    def test_single_result(self) -> None:
        r = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            llm_latency_ms=100.0,
            total_processing_ms=200.0,
        )
        agg = aggregate_results([r])
        assert agg.count == 1
        assert agg.success_count == 1
        assert agg.success_rate == 1.0

    def test_multiple_results_stats(self) -> None:
        results = [
            BenchmarkRunResult(
                run_id=f"r{i}",
                scenario_id="test",
                success=(i != 1),
                llm_latency_ms=100.0 + i * 50,
                total_processing_ms=200.0 + i * 50,
            )
            for i in range(3)
        ]
        agg = aggregate_results(results)
        assert agg.count == 3
        assert agg.success_count == 2
        assert agg.failure_count == 1
        assert agg.llm_latency_ms["min"] == 100.0
        assert agg.llm_latency_ms["max"] == 200.0
        assert agg.llm_latency_ms["avg"] == 150.0

    def test_aggregation_to_api_dict(self) -> None:
        r = BenchmarkRunResult(
            run_id="r1",
            scenario_id="test",
            success=True,
            llm_latency_ms=100.0,
            total_processing_ms=200.0,
        )
        agg = aggregate_results([r])
        d = agg.to_api_dict()
        assert d["count"] == 1
        assert d["scenario_id"] == "test"


# ---------------------------------------------------------------------------
# Benchmark registry isolation
# ---------------------------------------------------------------------------

class TestBenchmarkRegistry:
    def test_benchmark_registry_is_independent(self) -> None:
        """Benchmark tools must not affect the global registry."""
        from app.services.tool_service import get_tool_registry

        global_registry = get_tool_registry()
        global_tools_before = {t.name for t in global_registry.list_tools()}

        # Create benchmark registry
        bench_reg = ToolRegistry()
        for tool in create_benchmark_tools():
            bench_reg.register(tool)

        global_tools_after = {t.name for t in global_registry.list_tools()}
        assert global_tools_before == global_tools_after


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_run_result_to_api_dict(self) -> None:
        r = BenchmarkRunResult(
            run_id="abc-123",
            scenario_id="simple_conversation",
            success=True,
            llm_latency_ms=500.0,
            total_processing_ms=800.0,
        )
        d = r.to_api_dict()
        assert d["run_id"] == "abc-123"
        assert d["success"] is True
        assert d["benchmark_mode"] == "text_input"
        assert "validation_errors" in d


class TestToolExecutionTiming:
    def test_record_tool_call_captures_duration(self) -> None:
        """record_tool_call must accumulate duration_ms from ToolExecutor."""
        from app.services.metrics_service import VoiceTurnMetrics

        m = VoiceTurnMetrics()
        m.record_tool_call(success=True, duration_ms=42.5)
        m.record_tool_call(success=True, duration_ms=57.5)
        assert m.tool_count == 2
        assert m.tool_execution_ms == 100.0

    def test_record_tool_call_without_duration(self) -> None:
        """record_tool_call with no duration_ms must not break accumulation."""
        from app.services.metrics_service import VoiceTurnMetrics

        m = VoiceTurnMetrics()
        m.record_tool_call(success=True)
        assert m.tool_count == 1
        assert m.tool_execution_ms == 0.0
