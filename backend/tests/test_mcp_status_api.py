"""Tests for the read-only MCP status endpoint (Step 4).

Covers the three required states — disabled (default), connected (fake
session, no subprocess), and failed startup — plus misconfiguration.
Assertions also verify that sensitive startup details (command lines,
arguments, raw exception text) never reach the response.
"""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.services.mcp_service as mcp_service
from app.core.config import settings
from app.main import app
from app.services.mcp_service import shutdown_mcp, startup_mcp
from app.services.tool_service import get_tool_registry
from app.tools.demo_tools import create_demo_tools
from app.tools.registry import ToolRegistry

_MCP_SETTINGS = (
    "mcp_enabled",
    "mcp_server_label",
    "mcp_server_command",
    "mcp_server_args",
    "mcp_allowed_tools",
    "mcp_tool_timeout",
    "mcp_connect_timeout",
)

_NATIVE_TOOLS = {"get_weather", "get_employee", "get_leave_balance"}

# Distinctive token planted in the fake command line and exception text;
# it must never appear in any response.
_SECRET = "supersecret-value"


@pytest.fixture(autouse=True)
async def _clean_mcp_state() -> AsyncGenerator[None, None]:
    """Reset MCP module state and restore MCP settings around every test."""
    await shutdown_mcp()
    saved = {name: getattr(settings, name) for name in _MCP_SETTINGS}
    yield
    for name, value in saved.items():
        setattr(settings, name, value)
    await shutdown_mcp()


def _enable_mcp(**overrides: Any) -> None:
    """Enable MCP against a fake server (never spawned — see fakes below)."""
    settings.mcp_enabled = True
    settings.mcp_server_label = "local"
    settings.mcp_server_command = "fake-mcp-server"
    settings.mcp_server_args = f"--api-key={_SECRET}"
    settings.mcp_allowed_tools = "echo"
    settings.mcp_tool_timeout = 5.0
    settings.mcp_connect_timeout = 5.0
    for name, value in overrides.items():
        setattr(settings, name, value)


class _FakeSession:
    """Minimal MCP session double exposing tools with SDK v2 field shapes."""

    def __init__(self, tool_names: list[str]) -> None:
        self._tools = [
            SimpleNamespace(
                name=name,
                description=f"Test tool {name}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in tool_names
        ]

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False)


class _TrackingResource:
    """Minimal async context manager standing in for the SDK client."""

    async def __aenter__(self) -> "_TrackingResource":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _install_fake_open(monkeypatch: pytest.MonkeyPatch, tool_names: list[str]) -> None:
    async def _fake_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> _FakeSession:
        await stack.enter_async_context(_TrackingResource())
        return _FakeSession(tool_names)

    monkeypatch.setattr(mcp_service, "_open_client", _fake_open)


def _install_failing_open(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _failing_open(
        stack: AsyncExitStack, *, command: str, args: list[str]
    ) -> _FakeSession:
        raise RuntimeError(f"spawn failed: token={_SECRET}")

    monkeypatch.setattr(mcp_service, "_open_client", _failing_open)


@pytest.fixture
async def client_and_registry() -> AsyncGenerator[tuple[AsyncClient, ToolRegistry], None]:
    """Async test client with the tool-registry dependency overridden."""
    registry = ToolRegistry()
    for tool in create_demo_tools():
        registry.register(tool)
    app.dependency_overrides[get_tool_registry] = lambda: registry
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, registry
    finally:
        app.dependency_overrides.pop(get_tool_registry, None)


async def _get_status(client: AsyncClient) -> tuple[dict[str, Any], str]:
    """GET the status endpoint; return (parsed JSON, raw response text)."""
    response = await client.get("/api/v1/mcp/status")
    assert response.status_code == 200
    return response.json(), response.text


class TestMCPStatusStates:
    """Disabled, connected, and failed states of the status endpoint."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(
        self, client_and_registry: tuple[AsyncClient, ToolRegistry]
    ) -> None:
        client, _ = client_and_registry
        # Pin the configured default: this test exercises the disabled
        # state, independent of any local .env overrides.
        settings.mcp_enabled = False
        data, _text = await _get_status(client)

        assert data["enabled"] is False
        assert data["connected"] is False
        assert data["state"] == "disabled"
        assert data["error"] is None
        assert data["mcp_tools"] == []
        assert set(data["native_tools"]) == _NATIVE_TOOLS

    @pytest.mark.asyncio
    async def test_connected_reports_session_and_mcp_tools(
        self,
        client_and_registry: tuple[AsyncClient, ToolRegistry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, registry = client_and_registry
        _enable_mcp()
        _install_fake_open(monkeypatch, ["echo"])
        assert await startup_mcp(registry) is True

        data, text = await _get_status(client)

        assert data["enabled"] is True
        assert data["connected"] is True
        assert data["state"] == "connected"
        assert data["error"] is None
        assert [t["name"] for t in data["mcp_tools"]] == ["mcp_local_echo"]
        assert data["mcp_tools"][0]["mcp_tool"] == "echo"
        assert "get_weather" in data["native_tools"]
        # Safe fields only: command line, arguments, and secrets never leak.
        assert "fake-mcp-server" not in text
        assert _SECRET not in text

        # Connection state tracks the live session, not the enable flag.
        await shutdown_mcp()
        data_after, _ = await _get_status(client)
        assert data_after["enabled"] is True
        assert data_after["connected"] is False
        assert data_after["state"] == "not_started"

    @pytest.mark.asyncio
    async def test_failed_startup_reports_sanitized_error(
        self,
        client_and_registry: tuple[AsyncClient, ToolRegistry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, registry = client_and_registry
        _enable_mcp()
        _install_failing_open(monkeypatch)
        assert await startup_mcp(registry) is False

        data, text = await _get_status(client)

        assert data["enabled"] is True
        assert data["connected"] is False
        assert data["state"] == "failed"
        assert data["error"] == "MCP startup failed (RuntimeError)"
        # Native tools remain fully available after the failure.
        assert set(data["native_tools"]) == _NATIVE_TOOLS
        # No command line, arguments, raw exception text, or secrets.
        assert "fake-mcp-server" not in text
        assert _SECRET not in text
        assert "spawn failed" not in text

    @pytest.mark.asyncio
    async def test_misconfigured_reports_not_configured(
        self,
        client_and_registry: tuple[AsyncClient, ToolRegistry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, registry = client_and_registry
        _enable_mcp(mcp_server_command="", mcp_allowed_tools="")
        _install_failing_open(monkeypatch)  # must never be reached
        assert await startup_mcp(registry) is False

        data, _ = await _get_status(client)

        assert data["enabled"] is True
        assert data["connected"] is False
        assert data["state"] == "failed"
        assert data["error"] is not None
        assert "not configured" in data["error"]
