"""Runtime MCP enable/disable control (service + endpoints).

Unit tests use fake sessions (no subprocess) per the project convention;
the last class exercises the REAL SDK against tests/_mcp_stdio_server.py.
Covers: allowlist-only registration, disable unregistering + closing,
native-tool preservation, repeated cycles, fail-soft startup, stale
schemas after disable, the in-flight drain policy, and the control
endpoint access rules (loopback by default, optional token header).
"""

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.services.mcp_service as mcp_service
from app.core.config import settings
from app.main import app
from app.services.mcp_service import (
    disable_mcp,
    enable_mcp,
    get_mcp_state,
    shutdown_mcp,
)
from app.services.tool_service import get_tool_registry
from app.tools.demo_tools import create_demo_tools
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
    "mcp_control_token",
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


def _configure_mcp(**overrides: Any) -> None:
    """Configure the trusted stdio server (runtime toggle must not need .env)."""
    values: dict[str, Any] = {
        "mcp_enabled": False,  # runtime control must work with the default off
        "mcp_server_label": "local",
        "mcp_server_command": "fake-mcp-server",
        "mcp_server_args": f"--api-key={_SECRET}",
        "mcp_allowed_tools": "echo",
        "mcp_tool_timeout": 5.0,
        "mcp_connect_timeout": 5.0,
        "mcp_control_token": "",
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(settings, name, value)


class _FakeSession:
    """Minimal MCP session double exposing SDK v2 field shapes."""

    def __init__(self, tool_names: list[str]) -> None:
        self._tools = [
            SimpleNamespace(
                name=name,
                description=f"Test tool {name}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in tool_names
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, dict(arguments or {})))
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], is_error=False)


class _SlowSession(_FakeSession):
    """Session whose call blocks until the test releases it."""

    def __init__(self, tool_names: list[str]) -> None:
        super().__init__(tool_names)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.started.set()
        await self.release.wait()
        self.calls.append((name, dict(arguments or {})))
        return SimpleNamespace(content=[SimpleNamespace(text="slow-ok")], is_error=False)


class _TrackingResource:
    """Minimal async context manager standing in for the SDK client."""

    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_TrackingResource":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False


def _install_fake_open(
    monkeypatch: pytest.MonkeyPatch,
    factory: Any,
) -> tuple[list[tuple[str, list[str]]], list[_TrackingResource]]:
    """Replace _open_client; record spawns and resources for leak checks."""
    spawns: list[tuple[str, list[str]]] = []
    resources: list[_TrackingResource] = []

    async def fake_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
        spawns.append((command, list(args)))
        resource = _TrackingResource()
        resources.append(resource)
        await stack.enter_async_context(resource)
        return factory()

    monkeypatch.setattr(mcp_service, "_open_client", fake_open)
    return spawns, resources


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


@pytest.fixture
async def client_and_registry() -> AsyncGenerator[tuple[AsyncClient, ToolRegistry], None]:
    """Loopback test client with the tool-registry dependency overridden."""
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


# ---------------------------------------------------------------------------
# Service: enable / disable / cycles
# ---------------------------------------------------------------------------


class TestRuntimeEnableDisable:
    """Requirements 1-4: enable registers allowlist, disable cleans up."""

    @pytest.mark.asyncio
    async def test_enable_connects_and_registers_only_allowlisted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()  # mcp_enabled stays False: runtime toggle needs no .env
        session = _FakeSession(["echo", "drop_table"])
        spawns, _resources = _install_fake_open(monkeypatch, lambda: session)
        registry = ToolRegistry()
        registry.register(_native_tool("get_weather"))

        assert settings.mcp_enabled is False
        assert await enable_mcp(registry) is True

        assert [tool.name for tool in registry.list_tools()] == [
            "get_weather",
            "mcp_local_echo",
        ]
        assert len(spawns) == 1 and spawns[0][0] == "fake-mcp-server"
        state = get_mcp_state()
        assert state["enabled"] is True
        assert state["connected"] is True
        assert state["state"] == "connected"
        # The existing executor path executes the bridged tool.
        result = await ToolExecutor(registry).execute_from_json("mcp_local_echo", '{"text": "hi"}')
        assert result.success is True
        assert result.output == "ok"

    @pytest.mark.asyncio
    async def test_enable_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure_mcp()
        spawns, resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        registry = ToolRegistry()

        assert await enable_mcp(registry) is True
        assert await enable_mcp(registry) is True

        assert len(spawns) == 1  # no duplicate session
        assert registry.count == 1  # no duplicate registration
        assert resources[0].closed is False

    @pytest.mark.asyncio
    async def test_disable_unregisters_and_closes_preserving_native(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp(mcp_enabled=True)  # simulates the configured-on .env
        session = _FakeSession(["echo"])
        _spawns, resources = _install_fake_open(monkeypatch, lambda: session)
        registry = ToolRegistry()
        registry.register(_native_tool("get_weather"))
        assert await enable_mcp(registry) is True

        await disable_mcp()

        assert resources[0].closed is True
        assert registry.count == 1 and registry.get("get_weather") is not None
        state = get_mcp_state()
        # Runtime disable overrides the configured default (MCP_ENABLED=true).
        assert state["enabled"] is False
        assert state["connected"] is False
        assert state["state"] == "disabled"
        # MCP tool is no longer callable; the native tool still is.
        executor = ToolExecutor(registry)
        mcp_result = await executor.execute("mcp_local_echo", {})
        assert mcp_result.success is False
        assert "not found" in (mcp_result.error or "")
        native_result = await executor.execute("get_weather", {})
        assert native_result.success is True

    @pytest.mark.asyncio
    async def test_disable_is_idempotent(self) -> None:
        await disable_mcp()
        await disable_mcp()
        assert get_mcp_state()["state"] == "disabled"

    @pytest.mark.asyncio
    async def test_repeated_cycles_do_not_duplicate_or_leak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        spawns, resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        registry = ToolRegistry()
        registry.register(_native_tool("get_weather"))

        for _cycle in range(3):
            assert await enable_mcp(registry) is True
            assert [tool.name for tool in registry.list_tools()] == [
                "get_weather",
                "mcp_local_echo",
            ]
            await disable_mcp()
            assert registry.count == 1

        assert len(spawns) == 3
        assert len(resources) == 3
        assert all(resource.closed for resource in resources)


# ---------------------------------------------------------------------------
# Service: failure handling / stale schemas / in-flight policy
# ---------------------------------------------------------------------------


class TestRuntimeFailure:
    """Requirement 5: a failed startup leaves native tools usable."""

    @pytest.mark.asyncio
    async def test_failed_enable_is_fail_soft_and_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        registry = ToolRegistry()
        registry.register(_native_tool("get_weather"))

        async def failing_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
            raise RuntimeError(f"spawn failed: token={_SECRET}")

        monkeypatch.setattr(mcp_service, "_open_client", failing_open)

        assert await enable_mcp(registry) is False
        state = get_mcp_state()
        assert state["enabled"] is True
        assert state["connected"] is False
        assert state["state"] == "failed"
        assert state["error"] == "MCP startup failed (RuntimeError)"
        assert _SECRET not in (state["error"] or "")
        assert (await ToolExecutor(registry).execute("get_weather", {})).success is True

        # A later attempt (server now reachable) recovers cleanly.
        _spawns, _resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        assert await enable_mcp(registry) is True
        assert registry.get("mcp_local_echo") is not None
        assert get_mcp_state()["state"] == "connected"

    @pytest.mark.asyncio
    async def test_full_collision_reports_failure_and_keeps_native(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        _spawns, resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        registry = ToolRegistry()
        native = _native_tool("mcp_local_echo")
        registry.register(native)

        assert await enable_mcp(registry) is False

        assert registry.get("mcp_local_echo") is native  # never overwritten
        assert registry.count == 1
        assert resources[0].closed is True  # nothing bridged -> session closed
        state = get_mcp_state()
        assert state["state"] == "failed"
        assert "no tools" in (state["error"] or "")


class TestStaleAgentSchemas:
    """Requirement 6: cached schemas cannot execute MCP tools after disable."""

    @pytest.mark.asyncio
    async def test_cached_schema_call_fails_after_disable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        session = _FakeSession(["echo"])
        _install_fake_open(monkeypatch, lambda: session)
        registry = ToolRegistry()
        assert await enable_mcp(registry) is True

        # An agent turn started before disable caches these schemas.
        cached_names = [schema.name for schema in registry.get_schemas()]
        assert cached_names == ["mcp_local_echo"]

        await disable_mcp()

        executor = ToolExecutor(registry)
        for name in cached_names:
            result = await executor.execute_from_json(name, '{"text": "stale"}')
            assert result.success is False
            assert "not found" in (result.error or "")
        assert session.calls == []  # the disabled session was never reached


class TestInflightDisablePolicy:
    """Requirement 7: disable drains active calls before closing resources."""

    @pytest.mark.asyncio
    async def test_disable_waits_for_inflight_call_and_rejects_new_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        session = _SlowSession(["echo"])
        _spawns, resources = _install_fake_open(monkeypatch, lambda: session)
        registry = ToolRegistry()
        assert await enable_mcp(registry) is True
        executor = ToolExecutor(registry)

        call = asyncio.create_task(executor.execute_from_json("mcp_local_echo", '{"text": "slow"}'))
        await asyncio.wait_for(session.started.wait(), timeout=2)

        disable_task = asyncio.create_task(disable_mcp())
        for _ in range(200):
            if get_mcp_state()["connected"] is False:
                break
            await asyncio.sleep(0.01)

        # Disable is waiting for the active call; the resource is untouched.
        assert not disable_task.done()
        assert resources[0].closed is False

        # NEW calls are rejected deterministically while draining.
        late = await executor.execute_from_json("mcp_local_echo", '{"text": "late"}')
        assert late.success is False
        assert "not found" in (late.error or "")

        session.release.set()
        await asyncio.wait_for(disable_task, timeout=5)

        result = await call
        assert result.success is True
        assert result.output == "slow-ok"  # the in-flight call completed
        assert resources[0].closed is True
        assert registry.count == 0
        assert get_mcp_state()["state"] == "disabled"


# ---------------------------------------------------------------------------
# Control endpoints + access control
# ---------------------------------------------------------------------------


class TestControlEndpoints:
    """Requirements 2, 5, 8: endpoint contracts and sanitized failures."""

    @pytest.mark.asyncio
    async def test_enable_disable_roundtrip(
        self,
        client_and_registry: tuple[AsyncClient, ToolRegistry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _registry = client_and_registry
        _configure_mcp(mcp_enabled=True)
        _install_fake_open(monkeypatch, lambda: _FakeSession(["echo", "drop_table"]))

        enabled = await client.post("/api/v1/mcp/enable")
        assert enabled.status_code == 200
        data = enabled.json()
        assert data["enabled"] is True
        assert data["connected"] is True
        assert data["state"] == "connected"
        assert [tool["name"] for tool in data["mcp_tools"]] == ["mcp_local_echo"]
        assert data["mcp_tools"][0]["mcp_tool"] == "echo"
        assert set(data["native_tools"]) == _NATIVE_TOOLS
        assert _SECRET not in enabled.text

        disabled = await client.post("/api/v1/mcp/disable")
        assert disabled.status_code == 200
        data_off = disabled.json()
        assert data_off["enabled"] is False
        assert data_off["connected"] is False
        assert data_off["state"] == "disabled"
        assert data_off["mcp_tools"] == []
        assert set(data_off["native_tools"]) == _NATIVE_TOOLS

    @pytest.mark.asyncio
    async def test_failed_enable_returns_safe_actual_state(
        self,
        client_and_registry: tuple[AsyncClient, ToolRegistry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _registry = client_and_registry
        _configure_mcp()

        async def failing_open(stack: AsyncExitStack, *, command: str, args: list[str]) -> Any:
            raise RuntimeError(f"spawn failed: token={_SECRET}")

        monkeypatch.setattr(mcp_service, "_open_client", failing_open)

        response = await client.post("/api/v1/mcp/enable")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["connected"] is False
        assert data["state"] == "failed"
        assert data["error"] == "MCP startup failed (RuntimeError)"
        assert set(data["native_tools"]) == _NATIVE_TOOLS
        # Command line, arguments, raw exception text, and secrets never leak.
        assert "fake-mcp-server" not in response.text
        assert "spawn failed" not in response.text
        assert _SECRET not in response.text


class TestControlAccess:
    """Requirement 8: loopback default, token mode when configured."""

    @pytest.mark.asyncio
    async def test_non_loopback_without_token_is_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        spawns, _resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        registry = ToolRegistry()
        for tool in create_demo_tools():
            registry.register(tool)
        app.dependency_overrides[get_tool_registry] = lambda: registry
        try:
            transport = ASGITransport(app=app, client=("203.0.113.9", 4567))
            async with AsyncClient(transport=transport, base_url="http://test") as remote:
                denied = await remote.post("/api/v1/mcp/enable")
                assert denied.status_code == 403
                # The read-only status stays available.
                status = await remote.get("/api/v1/mcp/status")
                assert status.status_code == 200
        finally:
            app.dependency_overrides.pop(get_tool_registry, None)

        assert spawns == []  # denied before any connection attempt

    @pytest.mark.asyncio
    async def test_token_mode_requires_matching_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_mcp()
        settings.mcp_control_token = _SECRET
        spawns, _resources = _install_fake_open(monkeypatch, lambda: _FakeSession(["echo"]))
        registry = ToolRegistry()
        app.dependency_overrides[get_tool_registry] = lambda: registry
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                missing = await client.post("/api/v1/mcp/enable")
                assert missing.status_code == 403
                wrong = await client.post(
                    "/api/v1/mcp/enable", headers={"X-MCP-Control-Token": "wrong"}
                )
                assert wrong.status_code == 403
                assert spawns == []

                ok = await client.post(
                    "/api/v1/mcp/enable", headers={"X-MCP-Control-Token": _SECRET}
                )
                assert ok.status_code == 200
                assert ok.json()["state"] == "connected"

            # A remote client with the token is allowed (token mode wins).
            remote = ASGITransport(app=app, client=("203.0.113.9", 4567))
            async with AsyncClient(transport=remote, base_url="http://test") as client:
                remote_ok = await client.post(
                    "/api/v1/mcp/disable", headers={"X-MCP-Control-Token": _SECRET}
                )
                assert remote_ok.status_code == 200
                assert remote_ok.json()["state"] == "disabled"
        finally:
            app.dependency_overrides.pop(get_tool_registry, None)


# ---------------------------------------------------------------------------
# Real SDK integration
# ---------------------------------------------------------------------------


class TestRuntimeControlStdioIntegration:
    """Runtime enable/disable against the real SDK and local stdio server."""

    @pytest.mark.asyncio
    async def test_runtime_cycle_with_real_server(self) -> None:
        pytest.importorskip("mcp", reason="mcp SDK not installed")
        _configure_mcp(
            mcp_server_label="test",
            mcp_server_command=sys.executable,
            mcp_server_args=str(_STDIO_SERVER),
            mcp_allowed_tools="lookup_employee",
            mcp_tool_timeout=10.0,
            mcp_connect_timeout=20.0,
        )
        registry = ToolRegistry()

        assert await enable_mcp(registry) is True
        tool = registry.get("mcp_test_lookup_employee")
        assert tool is not None
        assert registry.get("mcp_test_echo") is None  # allowlist only

        result = await ToolExecutor(registry).execute_from_json(
            "mcp_test_lookup_employee", '{"employee_id": "E200"}'
        )
        assert result.success is True
        assert '"name": "John Sample"' in (result.output or "")

        await disable_mcp()

        assert registry.count == 0
        state = get_mcp_state()
        assert state["enabled"] is False
        assert state["connected"] is False
        assert state["state"] == "disabled"
        stale = await ToolExecutor(registry).execute("mcp_test_lookup_employee", {})
        assert stale.success is False
