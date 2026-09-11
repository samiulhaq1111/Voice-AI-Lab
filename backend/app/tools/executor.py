"""Tool executor for running tool calls with timeout and error handling."""

import asyncio
import json
import time
from typing import Any

from app.core.logging import logger
from app.tools.registry import ToolRegistry
from app.tools.tool import ToolResult


class ToolExecutor:
    """Executes tool calls from the LLM with timeout and error handling.

    The executor looks up tools in the registry, validates inputs,
    runs the handler with a timeout, and returns structured results.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Parsed arguments for the tool.

        Returns:
            ToolResult with success/failure and output/error.
        """
        tool = self._registry.get(tool_name)

        if tool is None:
            logger.error("Tool '%s' not found in registry", tool_name)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        if not tool.enabled:
            logger.warning("Tool '%s' is disabled", tool_name)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' is disabled",
            )

        start_time = time.monotonic()

        try:
            result = await asyncio.wait_for(
                tool.handler(**arguments),
                timeout=tool.timeout_seconds,
            )
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info("Tool '%s' executed successfully in %.1fms", tool_name, duration_ms)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=result,
                duration_ms=duration_ms,
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            error_msg = f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s"
            logger.error(error_msg)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            error_msg = f"Tool '{tool_name}' failed: {e!s}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )

    async def execute_from_json(self, tool_name: str, arguments_json: str) -> ToolResult:
        """Execute a tool with JSON string arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments_json: JSON string of arguments.

        Returns:
            ToolResult with success/failure and output/error.
        """
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Invalid JSON arguments: {e}",
            )

        return await self.execute(tool_name, arguments)
