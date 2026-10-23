"""MCP-to-native Tool bridge (Phase 1, Step 1).

Converts tools exposed by an already-connected MCP client session into
native ``Tool`` objects that register into the existing ``ToolRegistry``
and execute through the existing ``ToolExecutor`` — with zero changes to
the tool-calling pipeline (AgentRuntime, adapters, services, frontend).

Scope of this module (mapping only):
- Map explicitly allowlisted MCP tools (name / description / inputSchema)
  onto native Tool fields. Tools are NEVER selected implicitly from the
  server's listing order.
- Namespace each registered name as ``mcp_<label>_<tool>``; the handler
  closure retains the original MCP tool name for execution.
- Proxy each call to the session, converting text content blocks into a
  concise string. MCP ``isError`` results raise ``MCPToolError`` so the
  executor produces its normal failure ToolResult.

Out of scope: session lifecycle — the caller supplies an initialized
session (the SDK's high-level ``mcp.Client`` and the low-level
``ClientSession`` both satisfy the structural protocol) and owns
connect/close. This module spawns no subprocesses.

SDK interface, verified against mcp 2.3.0 (2026-07-28 protocol):
- ``list_tools()`` returns ``ListToolsResult`` with ``.tools``; each
  ``Tool`` carries ``.name``, ``.description`` and ``.input_schema``
  (snake_case Python attribute — the wire/v1 spelling ``inputSchema``
  is only a serialization alias in 2.x, never an attribute).
- ``call_tool(name, arguments)`` returns ``CallToolResult`` with
  ``.content`` blocks (text blocks carry ``.text``) and ``.is_error``.
  Fields are therefore read snake_case first with the v1 camelCase
  spelling as fallback, so both SDK majors and test fakes keep working.
- ``list_tools()`` returns a single page; pagination cursors are not
  followed (sufficient for one local server with a few tools).
- Concurrent ``call_tool()`` multiplexing and post-timeout session
  reusability are the session's responsibility; the bridge adds no
  locking, retries, or speculative concurrency.
"""

import re
from typing import Any, Protocol

from app.core.logging import logger
from app.tools.tool import Tool

# Registered (namespaced) tool names must satisfy the OpenAI function-name
# pattern [a-zA-Z0-9_-]. MCP tool names SHOULD already comply; sanitizing is
# defensive and affects registration only — execution always uses the
# original MCP name.
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Error detail is truncated before it reaches ToolResult (and with it the
# LLM context and logs), matching the providers' 200-char convention.
_MAX_DETAIL_CHARS = 200

# Fallback JSON Schema for MCP tools with a missing or empty inputSchema.
_EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


class MCPBridgeError(Exception):
    """Raised when the bridge cannot map a session's tools (invalid
    allowlist configuration or a failed tool listing)."""


class MCPToolError(Exception):
    """Raised when an MCP tool call returns an ``isError`` result."""


class MCPSessionLike(Protocol):
    """Structural subset of the MCP SDK client surface used by the bridge.

    Satisfied by the high-level ``mcp.Client`` and the low-level
    ``ClientSession`` alike (verified against mcp 2.3.0). Result types are
    intentionally ``Any``: protocol models are read via documented
    attribute names (see the module docstring) so SDK models and test
    fakes both work.
    """

    async def list_tools(self) -> Any:
        """Return the server's tool listing (result carries ``.tools``)."""

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke a tool; the result carries ``.content`` and ``.isError``."""


async def create_mcp_tools(
    session: MCPSessionLike,
    *,
    server_label: str,
    allowed_tools: list[str],
    timeout_seconds: float = 30.0,
) -> list[Tool]:
    """Create native ``Tool`` proxies for explicitly allowlisted MCP tools.

    Args:
        session: An already-initialized MCP client session. Lifecycle
            (connect/close) belongs to the caller.
        server_label: Short server label used to namespace registered
            names as ``mcp_<label>_<tool>``.
        allowed_tools: MCP tool names to expose. Tools are never selected
            implicitly from the server's listing order.
        timeout_seconds: Per-call timeout enforced by ``ToolExecutor``.

    Returns:
        One native Tool per allowlisted entry, in allowlist order.

    Raises:
        MCPBridgeError: If the configuration is invalid (blank label,
            empty or duplicated allowlist), the listing fails, or the
            server does not expose an allowlisted tool.
    """
    if not server_label.strip():
        raise MCPBridgeError("MCP bridge requires a non-empty server label")
    if not allowed_tools:
        raise MCPBridgeError(
            "MCP bridge requires an explicit, non-empty tool allowlist "
            "(tools are never selected implicitly)"
        )
    duplicates = sorted({name for name in allowed_tools if allowed_tools.count(name) > 1})
    if duplicates:
        raise MCPBridgeError(f"Duplicate MCP tool(s) in allowlist: {', '.join(duplicates)}")

    try:
        listing = await session.list_tools()
    except Exception as e:
        raise MCPBridgeError(f"MCP server '{server_label}' tool listing failed: {e}") from e

    available: dict[str, Any] = {
        str(getattr(tool, "name", "")): tool for tool in getattr(listing, "tools", None) or []
    }

    missing = [name for name in allowed_tools if name not in available]
    if missing:
        raise MCPBridgeError(
            f"MCP server '{server_label}' does not expose allowlisted tool(s): {', '.join(missing)}"
        )

    tools = [
        _build_proxy_tool(
            session=session,
            server_label=server_label,
            mcp_name=mcp_name,
            mcp_tool=available[mcp_name],
            timeout_seconds=timeout_seconds,
        )
        for mcp_name in allowed_tools
    ]
    logger.info(
        "[MCP] bridged %d tool(s) from server '%s': %s",
        len(tools),
        server_label,
        ", ".join(tool.name for tool in tools),
    )
    return tools


def _sanitize(name_part: str) -> str:
    """Replace characters that are invalid in OpenAI-style tool names."""
    return _INVALID_NAME_CHARS.sub("_", name_part)


def _field(obj: Any, name: str, *aliases: str) -> Any:
    """Read the first present attribute among ``name`` and ``aliases``.

    mcp 2.x exposes pydantic fields under snake_case Python names
    (``input_schema``, ``is_error``); the camelCase wire spellings
    (``inputSchema``, ``isError``) are only serialization aliases there
    but were the attribute names in 1.x. Both spellings are accepted,
    v2 first, so the bridge works with either SDK major and with fakes.
    """
    value = getattr(obj, name, None)
    if value is not None:
        return value
    for alias in aliases:
        value = getattr(obj, alias, None)
        if value is not None:
            return value
    return None


def _content_text(result: Any) -> str:
    """Join the text blocks of an MCP tool result into one string.

    Non-text content blocks (images, audio, embedded resources) carry no
    ``.text`` attribute and are skipped.
    """
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts)


def _build_proxy_tool(
    *,
    session: MCPSessionLike,
    server_label: str,
    mcp_name: str,
    mcp_tool: Any,
    timeout_seconds: float,
) -> Tool:
    """Build one native ``Tool`` that proxies a single MCP tool."""
    registered_name = f"mcp_{_sanitize(server_label)}_{_sanitize(mcp_name)}"

    description = str(getattr(mcp_tool, "description", None) or "").strip()
    if not description:
        description = f"MCP tool '{mcp_name}' (no description provided)"

    parameters = _field(mcp_tool, "input_schema", "inputSchema")
    if not isinstance(parameters, dict) or not parameters:
        parameters = dict(_EMPTY_PARAMETERS)

    async def handler(**arguments: Any) -> str:
        """Invoke the MCP tool and return its text content."""
        logger.debug("[MCP] invoking mcp_tool=%s via native=%s", mcp_name, registered_name)
        result = await session.call_tool(mcp_name, arguments=arguments)
        text = _content_text(result)
        if _field(result, "is_error", "isError"):
            detail = text.strip() or "no error detail provided"
            raise MCPToolError(
                f"MCP tool '{mcp_name}' returned an error: {detail[:_MAX_DETAIL_CHARS]}"
            )
        return text or "(no text content returned)"

    return Tool(
        name=registered_name,
        description=description,
        parameters=parameters,
        handler=handler,
        timeout_seconds=timeout_seconds,
        metadata={"mcp_server": server_label, "mcp_tool": mcp_name},
    )
