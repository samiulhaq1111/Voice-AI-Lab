"""MCP integration endpoints: status + runtime enable/disable.

Status reports whether MCP is enabled (configured default or runtime
toggle), whether a session is LIVE (derived from the actual held
client/exit stack — never from a flag alone), the MCP tools currently
registered in the native tool registry (kept distinct from native
tools), and a sanitized failure message.

The control endpoints flip the session at runtime without editing ``.env``:
enable connects/initializes/discovers and registers only the configured
allowlist; disable unregisters the bridged tools and closes the session.
Clients never supply commands, arguments, paths, or tool names — the
trusted backend configuration and allowlist are always used — and the
response is the backend's actual resulting state (failures included), so
the UI cannot optimistically claim success.

Access control (the project ships no auth layer): when
``MCP_CONTROL_TOKEN`` is configured, control requests must present a
matching ``X-MCP-Control-Token`` header (constant-time comparison);
otherwise control is restricted to loopback clients — sufficient for the
local-first deployment and never open to the network by default. The
status endpoint stays read-only and open.
"""

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import settings
from app.schemas import MCPStatusResponse, MCPToolStatusResponse
from app.services.mcp_service import disable_mcp, enable_mcp, get_mcp_state
from app.services.tool_service import get_tool_registry
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/mcp", tags=["mcp"])

# Hosts considered local. ASGITransport/TestClient use "127.0.0.1" or
# "testclient"; IPv6 loopback arrives as "::1" (or the v4-mapped form).
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "testclient"}


def require_mcp_control_access(request: Request) -> None:
    """Authorize a runtime MCP control request.

    Token mode when ``MCP_CONTROL_TOKEN`` is set; otherwise loopback-only.
    Raises HTTPException(403) without revealing configuration values.
    """
    expected = settings.mcp_control_token.strip()
    if expected:
        supplied = request.headers.get("x-mcp-control-token", "")
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=403, detail="MCP control access denied")
        return
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="MCP control requires a loopback client or MCP_CONTROL_TOKEN",
        )


def _build_status(registry: ToolRegistry) -> MCPStatusResponse:
    """Assemble the status payload from live state and the registry."""
    state = get_mcp_state()

    mcp_tools: list[MCPToolStatusResponse] = []
    native_tools: list[str] = []
    for tool in registry.list_tools():
        if tool.metadata.get("mcp_server"):
            mcp_tools.append(
                MCPToolStatusResponse(
                    name=tool.name,
                    description=tool.description,
                    mcp_tool=str(tool.metadata.get("mcp_tool") or ""),
                )
            )
        else:
            native_tools.append(tool.name)

    return MCPStatusResponse(
        **state,
        mcp_tools=mcp_tools,
        native_tools=sorted(native_tools),
    )


@router.get("/status", response_model=MCPStatusResponse)
async def mcp_status(registry: ToolRegistry = Depends(get_tool_registry)) -> MCPStatusResponse:
    """Report MCP desired state, live session state, and registered tools."""
    return _build_status(registry)


@router.post("/enable", response_model=MCPStatusResponse)
async def mcp_enable(
    registry: ToolRegistry = Depends(get_tool_registry),
    _access: None = Depends(require_mcp_control_access),
) -> MCPStatusResponse:
    """Runtime-enable MCP and return the ACTUAL resulting state.

    Idempotent; fail-soft (a startup failure returns 200 with
    ``state='failed'`` and a sanitized ``error`` while native tools stay
    available).
    """
    await enable_mcp(registry)
    return _build_status(registry)


@router.post("/disable", response_model=MCPStatusResponse)
async def mcp_disable(
    registry: ToolRegistry = Depends(get_tool_registry),
    _access: None = Depends(require_mcp_control_access),
) -> MCPStatusResponse:
    """Runtime-disable MCP and return the ACTUAL resulting state.

    Idempotent; waits (bounded) for in-flight MCP calls before closing the
    session, and unregisters the bridged tools from the registry.
    """
    await disable_mcp()
    return _build_status(registry)
