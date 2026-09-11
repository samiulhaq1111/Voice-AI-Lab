"""Tool registry for managing available tools."""

from app.core.logging import logger
from app.providers.types import ToolSchema
from app.tools.tool import Tool


class ToolRegistry:
    """Central registry for all available tools.

    The registry holds tool definitions and provides lookup by name.
    Tools are registered at startup and can be enabled/disabled dynamically.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' is already registered, overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def get_enabled(self) -> list[Tool]:
        """Return all enabled tools."""
        return [tool for tool in self._tools.values() if tool.enabled]

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        """Return tool schemas for all enabled tools."""
        return [tool.to_schema() for tool in self.get_enabled()]

    def enable(self, name: str) -> None:
        """Enable a tool by name."""
        tool = self._tools.get(name)
        if tool:
            tool.enabled = True

    def disable(self, name: str) -> None:
        """Disable a tool by name."""
        tool = self._tools.get(name)
        if tool:
            tool.enabled = False

    @property
    def count(self) -> int:
        """Return total number of registered tools."""
        return len(self._tools)
