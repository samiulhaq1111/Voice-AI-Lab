"""Tests for the MCP service lifecycle (app/services/mcp_service.py).

Unit tests use fake sessions (no subprocess), per the project's fake-session
testing convention. One integration test at the end exercises the REAL SDK
(mcp 2.3.0) against a minimal local stdio server (tests/_mcp_stdio_server.py),
verifying the bridge against actual SDK result and call signatures.
"""

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.mcp_service as mcp_service
from app.core.config import Settings, settings
from app.services.mcp_service import shutdown_mcp, startup_mcp
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.tool import Tool

_STDIO_SERVER = Path(__file__).parent / "_mcp_stdio_server.py"
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


def _fake_session(tool_names: list[str]) -> SimpleNamespace:
    """Minimal fake satisfying MCPSessionLike (v2 snake_case field shapes)."""

    async def list_tools() -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name=name,
                    description=f"Test tool {name}",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                )
                for name in tool_names
            ]
        )

    async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"called {name}")],
            is_error=False,
        )

    return SimpleNamespace(list_tools=list_tools, call_tool=call_tool)


class _TrackingResource:
    """Async context manager recording closure (stands in for the SDK client)."""

    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_TrackingResource":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


def _install_fake_open(
    monkeypatch: pytest.MonkeyPatch,
    session: Any,
    resource: _TrackingResource,
) -> list[tuple[str, list[str]]]:
    """Replace _open_client with a fake recording spawn parameters.

    The fake enters the tracking resource into the provided stack so
    shutdown/cleanup observably closes it, standing in for the SDK client.
    """
    calls: list[tuple[str, list[str]]] = []

    async def fake_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
        calls.append((command, list(args)))
        await stack.enter_async_context(resource)
        return session

    monkeypatch.setattr(mcp_service, "_open_client", fake_open)
    return calls


def _install_failing_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _open_client with a fake that must never be called."""

    async def fail_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
        raise AssertionError("_open_client must not be called by this code path")

    monkeypatch.setattr(mcp_service, "_open_client", fail_open)


def _enable_mcp(**overrides: Any) -> None:
    """Point settings at a fake local stdio server with an explicit allowlist."""
    values: dict[str, Any] = {
        "mcp_enabled": True,
        "mcp_server_label": "local",
        "mcp_server_command": "fake-mcp-server",
        "mcp_server_args": "",
        "mcp_allowed_tools": "echo",
        "mcp_tool_timeout": 5.0,
        "mcp_connect_timeout": 5.0,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(settings, name, value)


def _native_tool(name: str) -> Tool:
    """Build a trivial native (non-MCP) tool."""

    async def handler(**kwargs: Any) -> str:
        return "native"

    return Tool(
        name=name,
        description="A native (non-MCP) tool",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPDisabled:
    """Requirement: MCP is disabled by default; no connection when disabled."""

    def test_disabled_by_default(self) -> None:
        # The shipped default, independent of any local .env overrides.
        assert Settings.model_fields["mcp_enabled"].default is False

    @pytest.mark.asyncio
    async def test_no_connection_attempt_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings.mcp_enabled = False
        _install_failing_open(monkeypatch)
        registry = ToolRegistry()

        assert await startup_mcp(registry) is False
        assert registry.count == 0


class TestSuccessfulStartup:
    """Requirement: successful startup registers only allowlisted tools."""

    @pytest.mark.asyncio
    async def test_registers_only_allowlisted_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The server also exposes a dangerous tool that must NOT be bridged.
        _enable_mcp(mcp_allowed_tools="echo")
        session = _fake_session(tool_names=["echo", "drop_table"])
        calls = _install_fake_open(monkeypatch, session, _TrackingResource())
        registry = ToolRegistry()

        assert await startup_mcp(registry) is True

        names = [tool.name for tool in registry.list_tools()]
        assert names == ["mcp_local_echo"]
        # The configured stdio command (no args) reached the spawn layer.
        assert calls == [("fake-mcp-server", [])]
        # The bridged tool executes through the existing native path.
        result = await ToolExecutor(registry).execute_from_json("mcp_local_echo", '{"text": "hi"}')
        assert result.success is True
        assert result.output == "called echo"

    @pytest.mark.asyncio
    async def test_allowlist_order_wins_over_listing_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_mcp(mcp_allowed_tools="beta,alpha")
        session = _fake_session(tool_names=["alpha", "beta"])
        _install_fake_open(monkeypatch, session, _TrackingResource())
        registry = ToolRegistry()

        assert await startup_mcp(registry) is True
        assert [tool.name for tool in registry.list_tools()] == [
            "mcp_local_beta",
            "mcp_local_alpha",
        ]


class TestStartupFailure:
    """Requirement: startup failures warn and leave native tools usable."""

    @pytest.mark.asyncio
    async def test_handshake_failure_is_fail_soft(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _enable_mcp()

        async def failing_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
            raise RuntimeError("spawn failed")

        monkeypatch.setattr(mcp_service, "_open_client", failing_open)
        registry = ToolRegistry()
        native = _native_tool("get_weather")
        registry.register(native)

        with caplog.at_level(logging.WARNING, logger="voice_ai_lab"):
            assert await startup_mcp(registry) is False

        assert any("startup failed" in r.message for r in caplog.records)
        # Native tools remain registered and executable.
        result = await ToolExecutor(registry).execute("get_weather", {})
        assert result.success is True
        assert registry.get("get_weather") is native

    @pytest.mark.asyncio
    async def test_missing_allowlisted_tool_is_fail_soft_and_cleans_up(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The allowlisted tool genuinely does not exist in the listing.
        _enable_mcp(mcp_allowed_tools="nonexistent_tool")
        session = _fake_session(tool_names=["echo"])
        resource = _TrackingResource()
        _install_fake_open(monkeypatch, session, resource)
        registry = ToolRegistry()

        with caplog.at_level(logging.WARNING, logger="voice_ai_lab"):
            assert await startup_mcp(registry) is False

        assert any("does not expose" in r.message for r in caplog.records)
        # Partially-opened resources were closed (no leaked session).
        assert resource.closed is True
        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_misconfigured_is_fail_soft(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Enabled but the command is missing (allowlist is set).
        _enable_mcp(mcp_server_command="", mcp_allowed_tools="echo")
        _install_failing_open(monkeypatch)
        registry = ToolRegistry()

        with caplog.at_level(logging.WARNING, logger="voice_ai_lab"):
            assert await startup_mcp(registry) is False
        assert any("not set" in r.message for r in caplog.records)

        # Enabled with a command but an empty allowlist.
        _enable_mcp(mcp_server_command="fake-mcp-server", mcp_allowed_tools="")
        with caplog.at_level(logging.WARNING, logger="voice_ai_lab"):
            assert await startup_mcp(registry) is False
        assert registry.count == 0


class TestShutdown:
    """Requirement: shutdown closes MCP resources; duplicate startup ignored."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_resources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_mcp()
        session = _fake_session(tool_names=["echo"])
        resource = _TrackingResource()
        _install_fake_open(monkeypatch, session, resource)
        registry = ToolRegistry()
        assert await startup_mcp(registry) is True

        await shutdown_mcp()
        assert resource.closed is True

        # Idempotent: a second shutdown (and one when never started) is quiet.
        await shutdown_mcp()

    @pytest.mark.asyncio
    async def test_duplicate_startup_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_mcp()
        session = _fake_session(tool_names=["echo"])
        calls = _install_fake_open(monkeypatch, session, _TrackingResource())
        registry = ToolRegistry()
        assert await startup_mcp(registry) is True
        count_before = registry.count

        assert await startup_mcp(registry) is True
        # No duplicate registration and no second connection attempt.
        assert registry.count == count_before
        assert len(calls) == 1


class TestNameCollision:
    """Requirement: MCP names never silently overwrite native tools."""

    @pytest.mark.asyncio
    async def test_full_collision_never_overwrites_native_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _enable_mcp(mcp_allowed_tools="echo")
        session = _fake_session(tool_names=["echo"])
        resource = _TrackingResource()
        _install_fake_open(monkeypatch, session, resource)
        registry = ToolRegistry()
        native = _native_tool("mcp_local_echo")
        registry.register(native)

        with caplog.at_level(logging.WARNING, logger="voice_ai_lab"):
            assert await startup_mcp(registry) is False

        # The native tool object is untouched — MCP never overwrote it.
        assert registry.get("mcp_local_echo") is native
        assert registry.count == 1
        # Nothing was bridged, so the session was torn down.
        assert resource.closed is True
        assert any("refusing to overwrite" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_partial_collision_registers_remaining_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_mcp(mcp_allowed_tools="alpha,beta")
        session = _fake_session(tool_names=["alpha", "beta"])
        resource = _TrackingResource()
        _install_fake_open(monkeypatch, session, resource)
        registry = ToolRegistry()
        native_beta = _native_tool("mcp_local_beta")
        registry.register(native_beta)

        assert await startup_mcp(registry) is True

        assert registry.get("mcp_local_beta") is native_beta
        assert registry.get("mcp_local_alpha") is not None
        assert registry.count == 2
        # Some tools were bridged, so the session stays open.
        assert resource.closed is False


class TestStdioIntegration:
    """Requirement: the bridge works against the real SDK over stdio.

    Spawns tests/_mcp_stdio_server.py through the SDK's stdio transport —
    the only subprocess-based test. Exercises spawn, initialize, list_tools
    (v2 snake_case fields), call_tool, the existing executor path, and
    shutdown end-to-end.
    """

    @pytest.mark.asyncio
    async def test_end_to_end_with_real_sdk(self) -> None:
        pytest.importorskip("mcp", reason="mcp SDK not installed")
        _enable_mcp(
            mcp_server_label="test",
            mcp_server_command=sys.executable,
            mcp_server_args=str(_STDIO_SERVER),
            mcp_allowed_tools="echo",
            mcp_tool_timeout=10.0,
            mcp_connect_timeout=20.0,
        )
        registry = ToolRegistry()

        assert await startup_mcp(registry) is True

        tool = registry.get("mcp_test_echo")
        assert tool is not None
        assert tool.description == "Echo the provided text back."
        assert tool.parameters["properties"]["text"]["type"] == "string"
        assert registry.count == 1  # only the allowlisted tool was bridged

        result = await ToolExecutor(registry).execute_from_json(
            "mcp_test_echo", '{"text": "hello"}'
        )
        assert result.success is True
        assert result.output == "echo: hello"

        # A bad-argument call surfaces the SDK's isError result as a normal
        # ToolResult failure (fail-soft for the agent loop).
        bad = await ToolExecutor(registry).execute_from_json("mcp_test_echo", '{"wrong": true}')
        assert bad.success is False
        assert "returned an error" in (bad.error or "")

        await shutdown_mcp()
