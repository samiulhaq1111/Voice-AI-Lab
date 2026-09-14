"""Agent Runtime: the core orchestration loop.

The Agent Runtime manages conversation state, message history,
LLM invocation, tool-call detection, tool execution, and response generation.

It depends on interfaces (STTInterface, LLMInterface, TTSInterface)
and the ToolExecutor — NOT on concrete providers.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import logger
from app.providers.llm.interface import LLMInterface
from app.providers.stt.interface import STTInterface
from app.providers.tts.interface import TTSInterface
from app.providers.types import LLMMessage, LLMResponse
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an Agent Runtime instance."""

    system_prompt: str = "You are a helpful voice assistant."
    llm_model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    max_tool_rounds: int = 5  # Safety limit for tool-call loops


@dataclass
class AgentState:
    """Mutable conversation state for a session."""

    messages: list[LLMMessage] = field(default_factory=list)
    session_id: str | None = None
    is_active: bool = True


@dataclass
class AgentResult:
    """Structured result from an agent loop execution."""

    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    iterations: int = 0


# Type alias for tool-call event callbacks used by voice/external observers.
# Called with (tool_call_dict, tool_result_dict) after each tool execution.
ToolCallCallback = Callable[[dict[str, Any], dict[str, Any]], Any]


class AgentRuntime:
    """Core agent orchestration loop.

    Responsibilities:
    - Maintain conversation state and message history
    - Invoke the LLM with current history + tool schemas
    - Detect tool calls in LLM responses
    - Execute tool calls via ToolExecutor
    - Feed tool results back to the LLM
    - Return the final text response
    - Handle errors and retries
    """

    def __init__(
        self,
        *,
        stt: STTInterface | None = None,
        llm: LLMInterface,
        tts: TTSInterface | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._tool_registry = tool_registry or ToolRegistry()
        self._tool_executor = tool_executor or ToolExecutor(self._tool_registry)
        self._config = config or AgentConfig()
        self._state = AgentState()

    @property
    def state(self) -> AgentState:
        """Access the current agent conversation state."""
        return self._state

    def reset(self, system_prompt: str | None = None) -> None:
        """Reset the conversation state."""
        self._state = AgentState()
        if system_prompt:
            self._config.system_prompt = system_prompt

    def _build_messages(self) -> list[LLMMessage]:
        """Build the full message list including system prompt."""
        messages = [LLMMessage(role="system", content=self._config.system_prompt)]
        messages.extend(self._state.messages)
        return messages

    async def process_user_input(self, text: str) -> str:
        """Process user text input through the full agent loop.

        This is the main entry point for text-based interaction.
        It runs the LLM, handles any tool calls, and returns the final response.

        Args:
            text: The user's transcribed text input.

        Returns:
            The agent's text response.
        """
        result = await self.run(text)
        return result.response

    async def run(
        self,
        text: str,
        *,
        initial_messages: list[LLMMessage] | None = None,
        on_tool_call: ToolCallCallback | None = None,
    ) -> AgentResult:
        """Process user text and return a structured AgentResult.

        This is the preferred entry point — it returns the full result
        including tool calls, usage, and iteration count.

        Args:
            text: The user's text input.
            initial_messages: Optional prior conversation messages to seed
                the agent state with (loaded from history).
            on_tool_call: Optional callback invoked after each tool execution
                with (tool_call_dict, result_dict). Used by voice gateway
                to emit real-time tool events.

        Returns:
            AgentResult with response, tool_calls, usage, iterations.
        """
        logger.info("Agent start (input=%s...)", text[:80])

        # Seed state with prior conversation history if provided
        if initial_messages:
            self._state.messages.extend(initial_messages)

        # Add user message to history
        self._state.messages.append(LLMMessage(role="user", content=text))

        # Run the LLM with tool-call loop
        result = await self._run_agent_loop(on_tool_call=on_tool_call)

        # Add assistant response to history
        self._state.messages.append(LLMMessage(role="assistant", content=result.response))

        logger.info(
            "Agent completion (iterations=%d, tool_calls=%d)",
            result.iterations,
            len(result.tool_calls),
        )
        return result

    async def _run_agent_loop(self, *, on_tool_call: ToolCallCallback | None = None) -> AgentResult:
        """Run the LLM call loop, handling tool calls until a final text response.

        Returns:
            AgentResult with response, accumulated tool calls, usage, and iteration count.
        """
        tool_schemas = self._tool_registry.get_schemas()
        all_tool_calls: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {}
        iterations = 0

        for round_num in range(self._config.max_tool_rounds):
            iterations += 1
            messages = self._build_messages()

            logger.info(
                "[AGENT] Iteration started iteration=%d messages=%d tools=%d",
                iterations,
                len(messages),
                len(tool_schemas),
            )

            response: LLMResponse = await self._llm.chat(
                messages=messages,
                model=self._config.llm_model,
                tools=tool_schemas if tool_schemas else None,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            # Accumulate usage
            for key, val in response.usage.items():
                total_usage[key] = total_usage.get(key, 0) + val

            logger.info(
                "[AGENT] LLM response received has_content=%s tool_calls=%d finish=%s",
                bool(response.content),
                len(response.tool_calls or []),
                response.finish_reason,
            )

            # If no tool calls, return the text response
            if not response.tool_calls:
                return AgentResult(
                    response=response.content or "",
                    tool_calls=all_tool_calls,
                    usage=total_usage,
                    iterations=iterations,
                )

            # Handle tool calls
            logger.info(
                "[AGENT] Tool calls requested count=%d iteration=%d",
                len(response.tool_calls),
                iterations,
            )

            # Add assistant message with tool calls to history
            self._state.messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                all_tool_calls.append(tool_call)
                tool_name = tool_call.get("function", {}).get("name", "")
                logger.info("[AGENT] Tool call requested name=%s", tool_name)
                raw_result = await self._execute_tool_call(tool_call)
                logger.info("[AGENT] Tool call completed name=%s", tool_name)
                result_dict: dict[str, Any] = {
                    "name": tool_call.get("function", {}).get("name", ""),
                    "success": not (isinstance(raw_result, dict) and "error" in raw_result),
                    "output": raw_result,
                }
                if on_tool_call:
                    import inspect

                    maybe_coro = on_tool_call(tool_call, result_dict)
                    if inspect.iscoroutine(maybe_coro):
                        await maybe_coro
                self._state.messages.append(
                    LLMMessage(
                        role="tool",
                        content=(
                            json.dumps(raw_result)
                            if not isinstance(raw_result, str)
                            else raw_result
                        ),
                        tool_call_id=tool_call.get("id", ""),
                        name=tool_call.get("function", {}).get("name", ""),
                    )
                )

        # Safety: max rounds exceeded
        logger.warning(
            "[AGENT] Maximum iterations reached max_iterations=%d",
            self._config.max_tool_rounds,
        )
        msg = "I apologize, but I was unable to complete the request within the allowed steps."
        return AgentResult(
            response=msg,
            tool_calls=all_tool_calls,
            usage=total_usage,
            iterations=iterations,
        )

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> Any:
        """Execute a single tool call from the LLM response."""
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")

        logger.info("Executing tool: %s", tool_name)

        result = await self._tool_executor.execute_from_json(tool_name, arguments_str)

        if result.success:
            return result.output
        return {"error": result.error}

    async def close(self) -> None:
        """Release all provider resources."""
        if self._stt:
            await self._stt.close()
        await self._llm.close()
        if self._tts:
            await self._tts.close()
