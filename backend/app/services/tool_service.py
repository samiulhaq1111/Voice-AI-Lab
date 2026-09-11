"""Tool service: manages the global tool registry."""

from app.tools.demo_tools import create_demo_tools
from app.tools.registry import ToolRegistry

# Global singleton registry instance
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry, initializing it if needed."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        for tool in create_demo_tools():
            _registry.register(tool)
    return _registry
