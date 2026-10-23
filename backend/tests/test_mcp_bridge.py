"""Tests for the MCP-to-native Tool bridge (app/tools/mcp_bridge.py).

Uses a fake session satisfying MCPSessionLike — no real MCP server or SDK
required. The fakes mirror the SDK wire-field shapes the bridge reads:
listing results carry ``.tools``; tools carry ``.name``, ``.description``
and ``.inputSchema``; call results carry ``.content`` blocks (text blocks
have ``.text``) and ``.isError``.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from app.tools.executor import ToolExecutor
from app.tools.mcp_bridge import MCPBridgeError, MCPToolError, create_mcp_tools
from app.tools.registry import ToolRegistry
from app.tools.tool import Tool


def _mcp_tool(
    name: str,
    description: str | None = "Search the documentation",
    input_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Build a fake MCP tool descriptor with SDK-style field names."""
    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    return SimpleNamespace(name=name, description=description, inputSchema=input_schema)


def _text_result(*texts: str, is_error: bool = False) -> SimpleNamespace:
    """Build a fake call result containing text content blocks."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=t) for t in texts],
        isError=is_error,
    )


class FakeMCPSession:
    """Minimal session double satisfying the MCPSessionLike protocol.

    ``results`` are returned in order; once exhausted the last result is
    repeated so multi-call tests stay simple.
    """

    def __init__(self, tools: list[Any], results: list[Any]) -> None:
        self._tools = tools
        self._results = list(results)
        self._call_index = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=list(self._tools))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, dict(arguments or {})))
        if not self._results:
            raise AssertionError("call_tool invoked but no results programmed")
        result = self._results[min(self._call_index, len(self._results) - 1)]
        self._call_index += 1
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaMapping:
    """Requirement: map MCP name/description/inputSchema onto Tool fields."""

    @pytest.mark.asyncio
    async def test_maps_schema_and_namespaced_name(self) -> None:
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Query text"}},
            "required": ["query"],
        }
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs", "Search the docs", schema)],
            results=[],
        )

        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, Tool)
        assert tool.name == "mcp_docs_search_docs"
        assert tool.description == "Search the docs"
        assert tool.parameters == schema
        assert tool.timeout_seconds == 30.0
        assert tool.metadata == {"mcp_server": "docs", "mcp_tool": "search_docs"}

        # Registry compatibility: the proxy registers like any native tool.
        registry = ToolRegistry()
        registry.register(tool)
        schemas = registry.get_schemas()
        assert schemas[0].name == "mcp_docs_search_docs"

    @pytest.mark.asyncio
    async def test_sanitizes_name_and_falls_back_description(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search.docs v2", description=None, input_schema={})],
            results=[],
        )

        tools = await create_mcp_tools(
            session, server_label="local dev", allowed_tools=["search.docs v2"]
        )

        # Invalid characters are sanitized for registration only; the
        # original MCP name is retained for execution (recorded in metadata).
        assert tools[0].name == "mcp_local_dev_search_docs_v2"
        assert tools[0].parameters == {"type": "object", "properties": {}}
        assert tools[0].description == ("MCP tool 'search.docs v2' (no description provided)")
        assert tools[0].metadata["mcp_tool"] == "search.docs v2"


class TestArgumentForwarding:
    """Requirement: the async handler invokes the original MCP tool."""

    @pytest.mark.asyncio
    async def test_forwards_arguments_through_tool_executor(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs")],
            results=[_text_result("42 pages matched")],
        )
        tools = await create_mcp_tools(
            session,
            server_label="docs",
            allowed_tools=["search_docs"],
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(tools[0])

        result = await ToolExecutor(registry).execute_from_json(
            "mcp_docs_search_docs", '{"query": "latency", "limit": 5}'
        )

        assert result.success is True
        assert result.output == "42 pages matched"
        # The namespaced tool forwarded args to the ORIGINAL MCP name.
        assert session.calls == [("search_docs", {"query": "latency", "limit": 5})]
        assert tools[0].timeout_seconds == 5.0


class TestResultConversion:
    """Requirement: successful text content becomes a concise string."""

    @pytest.mark.asyncio
    async def test_joins_text_blocks_and_skips_non_text(self) -> None:
        mixed = SimpleNamespace(
            content=[
                SimpleNamespace(text="first"),
                SimpleNamespace(data=b"fake-image"),  # no .text -> skipped
                SimpleNamespace(text="second"),
            ],
            isError=False,
        )
        session = FakeMCPSession(tools=[_mcp_tool("search_docs")], results=[mixed])
        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])

        output = await tools[0].handler(query="latency")

        assert output == "first\nsecond"

    @pytest.mark.asyncio
    async def test_empty_content_returns_placeholder(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs")],
            results=[SimpleNamespace(content=[], isError=False)],
        )
        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])

        output = await tools[0].handler(query="latency")

        assert output == "(no text content returned)"


class TestErrorResults:
    """Requirement: MCP isError results raise so ToolExecutor fails normally."""

    @pytest.mark.asyncio
    async def test_is_error_raises_concise_exception(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs")],
            results=[_text_result("x" * 1000, is_error=True)],
        )
        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])

        with pytest.raises(MCPToolError) as excinfo:
            await tools[0].handler(query="latency")

        message = str(excinfo.value)
        assert "returned an error" in message
        assert len(message) < 300  # concise: detail truncated to 200 chars

    @pytest.mark.asyncio
    async def test_executor_produces_failure_result_for_is_error(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs")],
            results=[_text_result("index unavailable", is_error=True)],
        )
        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])
        registry = ToolRegistry()
        registry.register(tools[0])

        result = await ToolExecutor(registry).execute("mcp_docs_search_docs", {"query": "latency"})

        assert result.success is False
        assert "index unavailable" in (result.error or "")


class TestAllowlisting:
    """Requirement: only explicitly allowlisted tools are bridged."""

    @pytest.mark.asyncio
    async def test_only_allowlisted_tools_bridged(self) -> None:
        session = FakeMCPSession(
            tools=[
                _mcp_tool("search_docs"),
                _mcp_tool("delete_docs", "Dangerous write tool"),
            ],
            results=[_text_result("ok")],
        )

        tools = await create_mcp_tools(session, server_label="docs", allowed_tools=["search_docs"])

        assert [t.name for t in tools] == ["mcp_docs_search_docs"]

        # The non-allowlisted tool was never bridged: it is not executable
        # through the registry, and the session was never called.
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        result = await ToolExecutor(registry).execute("mcp_docs_delete_docs", {})
        assert result.success is False
        assert "not found" in (result.error or "")
        assert session.calls == []

    @pytest.mark.asyncio
    async def test_allowlist_order_is_deterministic(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("alpha"), _mcp_tool("beta")],
            results=[],
        )

        tools = await create_mcp_tools(
            session, server_label="docs", allowed_tools=["beta", "alpha"]
        )

        # Allowlist order wins over the server's listing order.
        assert [t.name for t in tools] == ["mcp_docs_beta", "mcp_docs_alpha"]

    @pytest.mark.asyncio
    async def test_invalid_configuration_raises(self) -> None:
        session = FakeMCPSession(
            tools=[_mcp_tool("search_docs"), _mcp_tool("delete_docs")],
            results=[],
        )

        # Allowlisted tool missing from the server's listing.
        with pytest.raises(MCPBridgeError, match="nonexistent_tool"):
            await create_mcp_tools(
                session,
                server_label="docs",
                allowed_tools=["search_docs", "nonexistent_tool"],
            )

        # Empty allowlist: never fall back to implicit selection.
        with pytest.raises(MCPBridgeError, match="allowlist"):
            await create_mcp_tools(session, server_label="docs", allowed_tools=[])

        # Duplicate allowlist entries.
        with pytest.raises(MCPBridgeError, match="Duplicate"):
            await create_mcp_tools(
                session, server_label="docs", allowed_tools=["search_docs", "search_docs"]
            )

        # Blank server label.
        with pytest.raises(MCPBridgeError, match="server label"):
            await create_mcp_tools(session, server_label=" ", allowed_tools=["search_docs"])


class TestSDKv2FieldShapes:
    """Regression: mcp 2.x models expose snake_case attributes.

    In mcp 2.x, ``inputSchema``/``isError`` are only serialization aliases
    — attribute access is ``input_schema``/``is_error`` (verified against
    the installed mcp 2.3.0). The bridge must read the v2 names; the v1
    camelCase spellings are exercised by every other fake in this file.
    """

    @pytest.mark.asyncio
    async def test_maps_v2_input_schema_attribute(self) -> None:
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        tool = SimpleNamespace(name="echo", description="Echo back", input_schema=schema)
        session = FakeMCPSession(tools=[tool], results=[])

        tools = await create_mcp_tools(session, server_label="test", allowed_tools=["echo"])

        assert tools[0].parameters == schema
        assert tools[0].description == "Echo back"

    @pytest.mark.asyncio
    async def test_reads_v2_is_error_attribute(self) -> None:
        tool = SimpleNamespace(
            name="echo",
            description="Echo back",
            input_schema={"type": "object", "properties": {}},
        )
        result = SimpleNamespace(
            content=[SimpleNamespace(text="boom")],
            is_error=True,
        )
        session = FakeMCPSession(tools=[tool], results=[result])
        tools = await create_mcp_tools(session, server_label="test", allowed_tools=["echo"])

        with pytest.raises(MCPToolError, match="boom"):
            await tools[0].handler(text="hi")
