"""MCP service: optional stdio MCP client lifecycle and tool bridging.

Disabled by default (``MCP_ENABLED=false``). When enabled, startup
connects to ONE local stdio MCP server through the SDK's high-level
``mcp.Client``, then discovers and registers ONLY the explicitly
allowlisted tools into the tool registry via ``create_mcp_tools``. The
existing native tool path (registry -> executor -> agent loop) executes
them unchanged; there is no separate MCP execution system.

Lifecycle contract:
- ``startup_mcp`` is fail-soft: a spawn failure, handshake timeout, or
  missing allowlisted tool is logged as one concise warning and the app
  continues with native tools only. It never raises.
- ``shutdown_mcp`` closes the client session and its subprocess. It is
  idempotent and quiet when MCP was never started.
- Runtime control (``enable_mcp`` / ``disable_mcp``, surfaced by the
  ``/api/v1/mcp`` control endpoints) flips the session without editing
  ``.env``: enable spawns and registers the configured allowlist; disable
  stops accepting new MCP calls, unregisters the bridged tools from the
  registry, drains in-flight calls (bounded), then closes the session and
  subprocess. ``MCP_ENABLED`` still decides the startup default; a runtime
  toggle only overrides it until the process restarts.
- MCP tool names never overwrite an existing registry entry, and a
  second startup call while active is ignored (no duplicate
  registration).
- The SDK import is lazy (inside ``_open_client``) so the app runs
  unchanged without the ``mcp`` package whenever MCP is disabled.
- ``get_mcp_state`` reports configuration plus LIVE session state for
  the read-only status endpoint; connectivity is never inferred from
  configuration, and failure text never includes command lines,
  subprocess arguments, or raw exception messages.
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.services.tool_service import get_tool_registry
from app.tools.mcp_bridge import MCPSessionLike, MCPToolError, create_mcp_tools
from app.tools.registry import ToolRegistry
from app.tools.tool import Tool


class _SessionGate:
    """Counts in-flight MCP calls and rejects new ones once closing starts.

    Runtime-disable policy: ``stop_accepting`` rejects NEW calls
    deterministically (they surface through the executor as normal failure
    ToolResults), then ``wait_drained`` lets already-running calls finish
    before the session and subprocess are closed. The wait is bounded so a
    stuck call can never block teardown forever. The accepting check and
    the in-flight increment happen without an ``await`` between them, so
    they are atomic on the single asyncio event loop.
    """

    def __init__(self, session: MCPSessionLike) -> None:
        self._session = session
        self._accepting = True
        self._inflight = 0
        self._drained = asyncio.Event()

    async def list_tools(self) -> Any:
        if not self._accepting:
            raise MCPToolError("MCP is disabled; tool listing rejected")
        return await self._session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self._accepting:
            raise MCPToolError("MCP is disabled; tool call rejected")
        self._inflight += 1
        try:
            return await self._session.call_tool(name, arguments=arguments)
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._drained.set()

    def stop_accepting(self) -> None:
        """Reject all future calls; signal drain when nothing is running."""
        self._accepting = False
        if self._inflight == 0:
            self._drained.set()

    async def wait_drained(self, timeout: float) -> bool:
        """Wait (bounded) for in-flight calls; True when fully drained."""
        if self._inflight == 0:
            return True
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False


# Application-lifetime MCP resources, held at module scope like the global
# tool-registry singleton in tool_service. The AsyncExitStack owns the SDK
# client context (and with it the stdio subprocess).
_exit_stack: AsyncExitStack | None = None
_client: MCPSessionLike | None = None
# In-flight call gate of the CURRENT session (see ``_SessionGate``).
_gate: _SessionGate | None = None
# Registry and bridged tool objects of the CURRENT session. Tools are
# unregistered by identity, so a native tool can never be removed.
_registry: ToolRegistry | None = None
_registered_tools: list[Tool] = []
# Safe, concise description of the last startup failure. Exception TYPE
# names only — never command lines, argv, or raw exception text, which
# could carry secrets (see ``get_mcp_state``).
_last_error: str | None = None
# Runtime toggle override: None = follow ``settings.mcp_enabled`` (the
# configured startup default), True/False = explicit runtime enable/disable.
# Cleared on application shutdown; never written back to ``.env``.
_runtime_override: bool | None = None

# Lifecycle operations are serialized (concurrent enables cannot spawn
# duplicate sessions; disable cannot race enable). A module-level Lock
# binds to the first event loop that awaits it, so it is recreated when a
# different loop appears (each test runs on a fresh loop).
_lifecycle_lock: asyncio.Lock | None = None
_lifecycle_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_lifecycle_lock() -> asyncio.Lock:
    """Return a lifecycle lock bound to the CURRENT event loop."""
    global _lifecycle_lock, _lifecycle_lock_loop
    loop = asyncio.get_running_loop()
    if _lifecycle_lock is None or _lifecycle_lock_loop is not loop:
        _lifecycle_lock = asyncio.Lock()
        _lifecycle_lock_loop = loop
    return _lifecycle_lock


def _reset_state() -> None:
    """Clear module-level MCP state (shutdown path and test resets)."""
    global _exit_stack, _client, _gate, _registry, _registered_tools
    global _last_error, _runtime_override
    _exit_stack = None
    _client = None
    _gate = None
    _registry = None
    _registered_tools = []
    _last_error = None
    _runtime_override = None


async def startup_mcp(registry: ToolRegistry | None = None) -> bool:
    """Start MCP at app startup when ``MCP_ENABLED=true``.

    Fail-soft by contract: any failure is logged as a concise warning and
    ``False`` is returned; the caller continues with native tools only.
    This is the configured startup path (application lifespan); runtime
    toggling goes through ``enable_mcp``/``disable_mcp``.

    Args:
        registry: Registry to register into. Defaults to the global tool
            registry; tests pass a private registry to avoid pollution.

    Returns:
        True if the MCP session is live.
    """
    if not settings.mcp_enabled:
        logger.info("[MCP] startup skipped (disabled)")
        return False
    return await _start_session(registry if registry is not None else get_tool_registry())


async def enable_mcp(registry: ToolRegistry | None = None) -> bool:
    """Runtime enable: connect now and register the configured allowlist.

    Overrides the ``MCP_ENABLED`` startup default without editing ``.env``
    or restarting the backend. Idempotent: a live session is left untouched
    (no duplicate spawn or registration). Fail-soft: startup errors are
    recorded for ``get_mcp_state()`` and native tools remain usable.
    """
    return await _start_session(
        registry if registry is not None else get_tool_registry(), override=True
    )


async def _start_session(target: ToolRegistry, *, override: bool | None = None) -> bool:
    """Connect, discover, and register allowlisted tools (serialized).

    Shared by the configured lifespan startup and the runtime enable
    endpoint; the lifecycle lock guarantees a single live session. The
    runtime override (when given) is recorded inside the lock so it can
    never interleave with a concurrent disable and report a state that
    disagrees with the session actually held.
    """
    global _exit_stack, _client, _gate, _registry, _registered_tools, _last_error
    global _runtime_override

    async with _get_lifecycle_lock():
        if override is not None:
            _runtime_override = override
        if _exit_stack is not None:
            logger.info("[MCP] already connected — startup ignored (no duplicate session)")
            return True

        # A new attempt replaces any failure recorded by an earlier one.
        _last_error = None

        command = settings.mcp_server_command.strip()
        allowed_tools = settings.mcp_allowed_tool_list
        if not command or not allowed_tools:
            _last_error = (
                "MCP is enabled but not configured "
                "(MCP_SERVER_COMMAND and MCP_ALLOWED_TOOLS are required)"
            )
            logger.warning(
                "[MCP] startup skipped (enabled but MCP_SERVER_COMMAND and "
                "MCP_ALLOWED_TOOLS are not set) — continuing with native tools only"
            )
            return False

        stack = AsyncExitStack()
        try:
            # One bounded budget for spawn + initialize + tool discovery.
            async with asyncio.timeout(settings.mcp_connect_timeout):
                session = await _open_client(
                    stack, command=command, args=settings.mcp_server_arg_list
                )
                gate = _SessionGate(session)
                tools = await create_mcp_tools(
                    gate,
                    server_label=settings.mcp_server_label,
                    allowed_tools=allowed_tools,
                    timeout_seconds=settings.mcp_tool_timeout,
                )
        except Exception as e:
            await _close_stack_quietly(stack)
            _last_error = f"MCP startup failed ({type(e).__name__})"
            logger.warning(
                "[MCP] startup failed error_type=%s error=%s — continuing with native tools only",
                type(e).__name__,
                str(e)[:200],
            )
            return False

        registered_tools = _register_tools(target, tools)
        if not registered_tools:
            # Every allowlisted tool collided with an existing registry
            # name: nothing was bridged, so do not hold the session open.
            _last_error = (
                "MCP startup produced no tools (all allowlisted names are already registered)"
            )
            await _close_stack_quietly(stack)
            return False

        _exit_stack = stack
        _client = session
        _gate = gate
        _registry = target
        _registered_tools = registered_tools
        logger.info(
            "[MCP] connected server=%s registered_tools=%s",
            settings.mcp_server_label,
            ", ".join(tool.name for tool in registered_tools),
        )
        return True


async def shutdown_mcp() -> None:
    """Close the MCP session and its subprocess. Idempotent.

    Application-shutdown / test-reset path: stops accepting calls,
    unregisters the bridged tools by identity (no orphaned registry
    entries), restores the configured enable default, and closes the
    session without waiting for in-flight calls — the process is going
    away, and runtime disable provides the drain policy.
    """
    async with _get_lifecycle_lock():
        stack = _exit_stack
        gate = _gate
        registry = _registry
        tools = _registered_tools
        _reset_state()
        if gate is not None:
            gate.stop_accepting()
        _unregister_tools(registry, tools)
        if stack is None:
            return
        try:
            await stack.aclose()
            logger.info("[MCP] session closed")
        except Exception as e:
            logger.warning(
                "[MCP] shutdown incomplete error_type=%s error=%s",
                type(e).__name__,
                str(e)[:200],
            )


async def disable_mcp() -> None:
    """Runtime disable: unregister MCP tools and close the session.

    In-flight policy: NEW calls are rejected immediately (session gate),
    bridged tools are removed from the registry so cached agent schemas
    cannot execute them, then already-running calls are allowed to finish
    (bounded by the tool timeout + 1 s) before the session and subprocess
    are closed. Idempotent: disabling when nothing runs just records the
    disabled state.
    """
    global _runtime_override

    async with _get_lifecycle_lock():
        stack = _exit_stack
        gate = _gate
        registry = _registry
        tools = _registered_tools
        _reset_state()
        _runtime_override = False
        if gate is not None:
            gate.stop_accepting()
        _unregister_tools(registry, tools)
        if stack is None:
            logger.info("[MCP] disable requested (no live session)")
            return
        drained = True
        if gate is not None:
            drained = await gate.wait_drained(timeout=settings.mcp_tool_timeout + 1.0)
        if not drained:
            logger.warning("[MCP] disable proceeding with in-flight call(s) after drain timeout")
        await _close_stack_quietly(stack)
        logger.info("[MCP] disabled; MCP tools unregistered and session closed")


def _effective_enabled() -> bool:
    """Desired MCP state: runtime toggle when set, else configured default."""
    if _runtime_override is not None:
        return _runtime_override
    return settings.mcp_enabled


def get_mcp_state() -> dict[str, Any]:
    """Return safe MCP lifecycle state for status reporting.

    ``enabled`` is the desired state (configured default or runtime
    toggle); ``connected`` reflects live module state (a client and exit
    stack actually held), never a flag: enabled with a failed or absent
    session reports ``connected=False``. Error text is a sanitized
    category string (exception type only), never raw exception output.

    Returns:
        Dict with ``enabled``, ``connected``, ``state`` (one of
        ``disabled``/``connected``/``failed``/``not_started``),
        ``server_label``, and ``error`` (None when no failure recorded).
    """
    enabled = _effective_enabled()
    connected = _exit_stack is not None and _client is not None
    if not enabled:
        state = "disabled"
    elif connected:
        state = "connected"
    elif _last_error is not None:
        state = "failed"
    else:
        state = "not_started"
    return {
        "enabled": enabled,
        "connected": connected,
        "state": state,
        "server_label": settings.mcp_server_label,
        "error": _last_error,
    }


async def _open_client(stack: AsyncExitStack, *, command: str, args: list[str]) -> MCPSessionLike:
    """Open the stdio MCP client (subprocess + connect + initialize) on ``stack``.

    The ``mcp`` import is lazy so the app runs without the package when MCP
    is disabled. Entering the client's async context spawns the subprocess
    and negotiates the protocol; closing the stack later terminates the
    subprocess. The SDK-level read timeout backs up the executor's
    per-tool ``wait_for`` bound with a clean SDK-side timeout.
    """
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command=command, args=list(args))
    client = await stack.enter_async_context(
        Client(params, read_timeout_seconds=settings.mcp_tool_timeout)
    )
    logger.debug("[MCP] stdio client opened command=%s", command)
    return client


def _register_tools(registry: ToolRegistry, tools: list[Tool]) -> list[Tool]:
    """Register MCP tools without overwriting existing registry entries.

    Returns the tool objects actually registered. A name already present in
    the registry (a native tool, an earlier MCP registration, or a sibling
    MCP tool that sanitizes to the same name) is skipped with a warning —
    MCP tools must never silently replace native tools.
    """
    registered: list[Tool] = []
    for tool in tools:
        existing = registry.get(tool.name)
        if existing is not None:
            logger.warning(
                "[MCP] tool name '%s' is already registered (%s) — refusing "
                "to overwrite; skipping this MCP tool",
                tool.name,
                type(existing).__name__,
            )
            continue
        registry.register(tool)
        registered.append(tool)
    return registered


def _unregister_tools(registry: ToolRegistry | None, tools: list[Tool]) -> None:
    """Remove bridged tools by IDENTITY — never a native tool by name."""
    if registry is None or not tools:
        return
    removed = 0
    for tool in tools:
        if registry.get(tool.name) is tool:
            registry.unregister(tool.name)
            removed += 1
    logger.info("[MCP] unregistered %d MCP tool(s)", removed)


async def _close_stack_quietly(stack: AsyncExitStack) -> None:
    """Close a stack on fail-soft paths, logging instead of raising."""
    try:
        await stack.aclose()
    except Exception as e:
        logger.warning(
            "[MCP] cleanup incomplete error_type=%s error=%s",
            type(e).__name__,
            str(e)[:200],
        )
