"""End-to-end MCP tool call through the existing AgentRuntime loop.

Boots the real stdio MCP test server through the real MCP SDK (via
``startup_mcp``), lets the unmodified AgentRuntime tool-call loop execute an
allowlisted MCP tool through the native registry/executor, and verifies the
tool result is fed back into the next LLM iteration before the final
response is produced.

A scripted fake LLM drives the loop, so the test needs no OpenRouter
credentials, no network access, and no real LLM. The tool-call dicts and
tool schemas use the exact shapes the OpenRouter adapter sends/receives.
"""

import json
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.agents.runtime import AgentConfig, AgentRuntime
from app.core.config import settings
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage, LLMResponse, StreamChunk, ToolSchema
from app.services.mcp_service import shutdown_mcp, startup_mcp
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

_STDIO_SERVER = Path(__file__).parent / "_mcp_stdio_server.py"

# Registered name of the allowlisted tool: mcp_<label>_<tool>.
_TOOL_NAME = "mcp_test_lookup_employee"

_MCP_SETTINGS = (
    "mcp_enabled",
    "mcp_server_label",
    "mcp_server_command",
    "mcp_server_args",
    "mcp_allowed_tools",
    "mcp_tool_timeout",
    "mcp_connect_timeout",
)


@pytest.fixture(autouse=True)
async def _clean_mcp_state() -> AsyncGenerator[None, None]:
    """Reset MCP module state and restore MCP settings around every test."""
    await shutdown_mcp()
    saved = {name: getattr(settings, name) for name in _MCP_SETTINGS}
    yield
    for name, value in saved.items():
        setattr(settings, name, value)
    await shutdown_mcp()


def _enable_mcp(*, allowed_tools: str = "lookup_employee") -> None:
    """Point the MCP lifecycle at the local stdio test server."""
    settings.mcp_enabled = True
    settings.mcp_server_label = "test"
    settings.mcp_server_command = sys.executable
    settings.mcp_server_args = str(_STDIO_SERVER)
    settings.mcp_allowed_tools = allowed_tools
    settings.mcp_tool_timeout = 10.0
    settings.mcp_connect_timeout = 20.0


def _make_tool_call(tool_call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-format tool call dict (the shape OpenRouter normalizes)."""
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedLLM(LLMInterface):
    """Fake LLM that replays scripted responses and records every request."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.requests.append({"messages": list(messages), "tools": list(tools or [])})
        if not self._responses:
            raise AssertionError("ScriptedLLM received more chat() calls than scripted")
        return self._responses.pop(0)

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming is not exercised by the non-streaming agent loop."""
        raise NotImplementedError("Streaming not needed for this test")

    async def close(self) -> None:
        pass


class TestAgentCallsMCPTool:
    """The existing agent loop executes an allowlisted MCP tool end to end."""

    @pytest.mark.asyncio
    async def test_tool_result_reaches_next_llm_iteration(self) -> None:
        _enable_mcp()
        registry = ToolRegistry()
        assert await startup_mcp(registry) is True

        # The allowlisted tool is bridged; the server's other tool (`echo`)
        # is not allowlisted and must not exist as a native tool.
        assert registry.get(_TOOL_NAME) is not None
        assert registry.get("mcp_test_echo") is None

        tool_call = _make_tool_call("call_mcp_1", _TOOL_NAME, {"employee_id": "E100"})
        final_text = "Jane Example works in Engineering (employee E100)."
        llm = ScriptedLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                ),
                LLMResponse(
                    content=final_text,
                    finish_reason="stop",
                    usage={"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
                ),
            ]
        )
        runtime = AgentRuntime(
            llm=llm,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            config=AgentConfig(max_tool_rounds=5),
        )
        try:
            result = await runtime.run("Who is employee E100?")
        finally:
            await runtime.close()
            await shutdown_mcp()

        # Final response was generated after the MCP tool round.
        assert result.response == final_text
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == _TOOL_NAME
        assert len(llm.requests) == 2

        # Iteration 1: only the allowlisted MCP tool schema was offered.
        offered = {schema.name for schema in llm.requests[0]["tools"]}
        assert _TOOL_NAME in offered
        assert "mcp_test_echo" not in offered

        # Iteration 2: the real MCP tool result reached the LLM as a
        # tool-role message carrying the original tool_call_id.
        tool_messages = [m for m in llm.requests[1]["messages"] if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_mcp_1"
        assert tool_messages[0].name == _TOOL_NAME
        payload = json.loads(tool_messages[0].content or "")
        assert payload["name"] == "Jane Example"
        assert payload["department"] == "Engineering"

        # Usage from both LLM iterations is accumulated.
        assert result.usage["total_tokens"] == 155


class TestUnallowlistedToolRejected:
    """A server tool outside MCP_ALLOWED_TOOLS is unreachable via the registry."""

    @pytest.mark.asyncio
    async def test_unallowlisted_tool_is_not_registered_or_callable(self) -> None:
        _enable_mcp()  # allowlist: lookup_employee only; `echo` is also served
        registry = ToolRegistry()
        try:
            assert await startup_mcp(registry) is True
            assert registry.get("mcp_test_echo") is None

            executor = ToolExecutor(registry)
            rejected = await executor.execute("mcp_test_echo", {"text": "hi"})
            assert rejected.success is False
            assert rejected.error is not None
            assert "not found" in rejected.error

            # The allowlisted tool works through the same executor.
            approved = await executor.execute(_TOOL_NAME, {"employee_id": "E200"})
            assert approved.success is True
            payload = json.loads(approved.output)
            assert payload["name"] == "John Sample"
        finally:
            await shutdown_mcp()
