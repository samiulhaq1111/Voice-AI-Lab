"""Tests for AgentRuntime — provider-independent agent loop."""

import json
from unittest.mock import AsyncMock

import pytest

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage, LLMResponse, ToolSchema
from app.tools.demo_tools import create_demo_tools
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# FakeLLM — proves AgentRuntime depends on the interface, not OpenRouter
# ---------------------------------------------------------------------------


class FakeLLM(LLMInterface):
    """A fake LLM that returns pre-programmed responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return self._call_count

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(content="No more responses programmed", finish_reason="stop")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def stream_chat(self, messages, *, model=None, tools=None, temperature=0.7, max_tokens=None):
        raise NotImplementedError("Streaming not needed for tests")

    async def close(self) -> None:
        pass


def _make_tool_call(tool_call_id: str, name: str, arguments: dict) -> dict:
    """Helper to build a tool call dict matching OpenAI format."""
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _build_runtime(
    llm: FakeLLM,
    max_rounds: int = 5,
) -> AgentRuntime:
    """Build an AgentRuntime with demo tools."""
    registry = ToolRegistry()
    for tool in create_demo_tools():
        registry.register(tool)
    executor = ToolExecutor(registry)
    config = AgentConfig(max_tool_rounds=max_rounds)
    return AgentRuntime(
        llm=llm,
        tool_registry=registry,
        tool_executor=executor,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleResponse:
    """Test: simple user → LLM → final response (no tool calls)."""

    @pytest.mark.asyncio
    async def test_simple_response(self) -> None:
        llm = FakeLLM([LLMResponse(content="Hello! How can I help?", finish_reason="stop")])
        runtime = _build_runtime(llm)

        result = await runtime.run("Hi there")

        assert isinstance(result, AgentResult)
        assert result.response == "Hello! How can I help?"
        assert result.tool_calls == []
        assert result.iterations == 1
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_process_user_input_returns_string(self) -> None:
        llm = FakeLLM([LLMResponse(content="OK", finish_reason="stop")])
        runtime = _build_runtime(llm)

        text = await runtime.process_user_input("test")
        assert text == "OK"


class TestSingleToolCall:
    """Test: LLM → one tool call → tool result → final response."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        tool_call = _make_tool_call("tc1", "get_weather", {"location": "Islamabad"})
        llm = FakeLLM([
            LLMResponse(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LLMResponse(content="The weather in Islamabad is sunny, 78°F.", finish_reason="stop"),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("What is the weather in Islamabad?")

        assert "Islamabad" in result.response
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.iterations == 2
        assert llm.call_count == 2


class TestMultipleToolCalls:
    """Test: LLM → multiple tool calls in one response → final response."""

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_round(self) -> None:
        tc1 = _make_tool_call("tc1", "get_weather", {"location": "Islamabad"})
        tc2 = _make_tool_call("tc2", "get_weather", {"location": "Lahore"})
        llm = FakeLLM([
            LLMResponse(
                content=None,
                tool_calls=[tc1, tc2],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Islamabad is sunny, Lahore is cloudy.", finish_reason="stop"),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("Compare weather in Islamabad and Lahore")

        assert len(result.tool_calls) == 2
        assert result.iterations == 2


class TestMultiStepToolCalls:
    """Test: LLM → tool1 → LLM → tool2 → LLM → final response."""

    @pytest.mark.asyncio
    async def test_multi_step(self) -> None:
        tc1 = _make_tool_call("tc1", "get_employee", {"employee_id": "E001"})
        tc2 = _make_tool_call("tc2", "get_leave_balance", {"employee_id": "E001", "leave_type": "pto"})
        llm = FakeLLM([
            LLMResponse(content=None, tool_calls=[tc1], finish_reason="tool_calls"),
            LLMResponse(content=None, tool_calls=[tc2], finish_reason="tool_calls"),
            LLMResponse(content="Alice Johnson has 12 PTO days remaining.", finish_reason="stop"),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("How many PTO days does E001 have?")

        assert result.iterations == 3
        assert len(result.tool_calls) == 2
        assert "Alice" in result.response or "PTO" in result.response


class TestUnknownTool:
    """Test: LLM calls a tool that doesn't exist."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        tc = _make_tool_call("tc1", "nonexistent_tool", {"arg": "value"})
        llm = FakeLLM([
            LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            LLMResponse(content="I couldn't find that tool.", finish_reason="stop"),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("Use nonexistent_tool")

        # Should still complete — tool error is fed back to LLM
        assert result.iterations == 2
        assert len(result.tool_calls) == 1


class TestInvalidToolArguments:
    """Test: LLM calls a tool with invalid JSON arguments."""

    @pytest.mark.asyncio
    async def test_invalid_json_args(self) -> None:
        tc = {
            "id": "tc1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "not valid json {{{",
            },
        }
        llm = FakeLLM([
            LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            LLMResponse(content="Sorry, the arguments were invalid.", finish_reason="stop"),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("Get weather with bad args")

        assert result.iterations == 2
        assert len(result.tool_calls) == 1


class TestToolExecutionFailure:
    """Test: tool handler raises an exception."""

    @pytest.mark.asyncio
    async def test_tool_failure(self) -> None:
        # Create a tool that always fails
        from app.tools.tool import Tool

        async def failing_handler(**kwargs) -> dict:
            raise RuntimeError("Tool exploded internally")

        registry = ToolRegistry()
        registry.register(Tool(
            name="failing_tool",
            description="A tool that always fails",
            parameters={"type": "object", "properties": {}},
            handler=failing_handler,
        ))
        executor = ToolExecutor(registry)

        tc = _make_tool_call("tc1", "failing_tool", {})
        llm = FakeLLM([
            LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            LLMResponse(content="The tool failed.", finish_reason="stop"),
        ])
        runtime = AgentRuntime(
            llm=llm,
            tool_registry=registry,
            tool_executor=executor,
            config=AgentConfig(max_tool_rounds=5),
        )

        result = await runtime.run("Use the failing tool")

        assert result.iterations == 2
        assert len(result.tool_calls) == 1


class TestMaxIterations:
    """Test: agent stops after max iteration limit."""

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self) -> None:
        # Every response requests a tool call — never gives a final answer
        tc = _make_tool_call("tc1", "get_weather", {"location": "test"})
        responses = [
            LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls")
            for _ in range(10)
        ]
        llm = FakeLLM(responses)
        runtime = _build_runtime(llm, max_rounds=3)

        result = await runtime.run("Keep calling tools forever")

        assert result.iterations == 3
        assert "unable to complete" in result.response.lower() or "apologize" in result.response.lower()


class TestLLMProviderFailure:
    """Test: LLM raises an exception."""

    @pytest.mark.asyncio
    async def test_llm_failure(self) -> None:
        class FailingLLM(LLMInterface):
            @property
            def provider_name(self) -> str:
                return "failing"

            async def chat(self, messages, **kwargs) -> LLMResponse:
                raise RuntimeError("LLM provider is down")

            async def stream_chat(self, messages, **kwargs):
                raise NotImplementedError

            async def close(self) -> None:
                pass

        runtime = AgentRuntime(llm=FailingLLM())

        with pytest.raises(RuntimeError, match="LLM provider is down"):
            await runtime.run("This will fail")


class TestUsageTracking:
    """Test: usage tokens are accumulated across iterations."""

    @pytest.mark.asyncio
    async def test_usage_accumulation(self) -> None:
        tc = _make_tool_call("tc1", "get_weather", {"location": "test"})
        llm = FakeLLM([
            LLMResponse(
                content=None,
                tool_calls=[tc],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
            LLMResponse(
                content="Done",
                finish_reason="stop",
                usage={"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180},
            ),
        ])
        runtime = _build_runtime(llm)

        result = await runtime.run("Get weather")

        assert result.usage["prompt_tokens"] == 250
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 300


class TestFakeLLMArchitecture:
    """Architectural test: AgentRuntime works with ANY LLMInterface implementation.

    This test proves AgentRuntime depends on the interface, NOT on OpenRouter.
    """

    @pytest.mark.asyncio
    async def test_fake_llm_full_loop(self) -> None:
        """Complete flow: user → tool call → tool result → final response.

        Uses only FakeLLM — no OpenRouter, no real API keys.
        """
        tc = _make_tool_call("call_1", "get_employee", {"employee_id": "E002"})

        fake = FakeLLM([
            LLMResponse(
                content=None,
                tool_calls=[tc],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            ),
            LLMResponse(
                content="Bob Smith works in Marketing.",
                finish_reason="stop",
                usage={"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
            ),
        ])

        registry = ToolRegistry()
        for tool in create_demo_tools():
            registry.register(tool)

        runtime = AgentRuntime(
            llm=fake,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            config=AgentConfig(max_tool_rounds=5),
        )

        result = await runtime.run("Who is employee E002?")

        assert result.response == "Bob Smith works in Marketing."
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.usage["total_tokens"] == 155
        assert fake.provider_name == "fake"
