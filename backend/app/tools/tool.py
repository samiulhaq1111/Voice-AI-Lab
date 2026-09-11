"""Tool definition and base interface."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.providers.types import ToolSchema


@dataclass
class Tool:
    """Represents a callable tool that can be exposed to the LLM.

    A tool has a name, description, input schema, and a handler function.
    The handler receives parsed input and returns a result.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, Any]]
    timeout_seconds: float = 30.0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> ToolSchema:
        """Convert this tool to a schema suitable for LLM function calling."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float | None = None
