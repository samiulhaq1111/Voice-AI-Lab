"""Telephony agent service (Phase 6C Milestone 6).

Processes finalized utterances from the Telnyx → Deepgram pipeline through
AgentRuntime. Manages session-level LLM provider lifecycle and conversation
history for multi-turn phone calls.

Architecture:
    bridge.utterance_queue
        ↓
    _agent_worker() — one utterance at a time
        ↓
    AgentRuntime.run(transcript, initial_messages=history)
        ↓
    Log text response (no TTS in this milestone)

The agent worker runs as a separate async task so that the Telnyx media
receive loop and Deepgram event pump are never blocked by LLM calls.

Session-level LLM provider lifecycle:
    - Created once when the telephony session starts
    - Reused across all utterances in the call
    - Closed exactly once during session cleanup
    - AgentRuntime must NOT close it
"""

import asyncio
import time
from typing import Any

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.core.config import settings
from app.core.logging import logger
from app.providers.factory import ProviderError, get_llm_provider
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor

# Telnyx telephony uses a fast, reliable model for low-latency phone calls.
# This is independent of the global default LLM model.
_TELNYX_LLM_MODEL = "openai/gpt-4o-mini"


class TelephonyAgentSession:
    """Manages agent processing for a single telephony call.

    Owns the session-level LLM provider and conversation history.
    Processes utterances one at a time via the agent worker.
    """

    def __init__(
        self,
        call_control_id: str,
        utterance_queue: asyncio.Queue[str | None],
    ) -> None:
        self._call_control_id = call_control_id
        self._utterance_queue = utterance_queue
        self._llm: LLMInterface | None = None
        self._history: list[LLMMessage] = []
        self._turn: int = 0
        self._worker_task: asyncio.Task | None = None
        self._running = False

    @property
    def call_control_id(self) -> str:
        return self._call_control_id

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def history(self) -> list[LLMMessage]:
        return list(self._history)

    async def start(self) -> None:
        """Initialize the session-level LLM provider and start the agent worker."""
        if self._running:
            logger.warning(
                "[VOICE:TELNYX:AGENT] Agent session already started "
                "call_control_id=%s",
                self._call_control_id,
            )
            return

        # Create session-level LLM provider
        # Telnyx telephony uses a dedicated model for low-latency phone calls
        resolved_provider = "openrouter"
        resolved_model = _TELNYX_LLM_MODEL
        try:
            self._llm = get_llm_provider(resolved_provider, model=resolved_model)
            logger.debug(
                "[VOICE:TELNYX:AGENT] LLM provider created "
                "provider=%s model=%s",
                resolved_provider,
                resolved_model,
            )
        except (ProviderError, ValueError) as e:
            logger.error(
                "[VOICE:TELNYX:AGENT] LLM provider creation failed "
                "call_control_id=%s error=%s",
                self._call_control_id,
                e,
            )
            raise

        self._running = True
        self._worker_task = asyncio.create_task(self._agent_worker())
        logger.debug(
            "[VOICE:TELNYX:AGENT] worker started call_control_id=%s",
            self._call_control_id,
        )

    async def stop(self) -> None:
        """Stop the agent worker and close the session-level LLM provider."""
        if not self._running:
            return

        self._running = False
        logger.info(
            "[VOICE:TELNYX:AGENT] stopping turns=%d",
            self._turn,
        )

        # Send shutdown sentinel to worker
        try:
            self._utterance_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Wait for worker to finish
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=10.0)
            except (TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
            self._worker_task = None

        # Close session-level LLM exactly once
        if self._llm is not None:
            try:
                await self._llm.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:TELNYX:AGENT] LLM close error=%s call_control_id=%s",
                    e,
                    self._call_control_id,
                )
            self._llm = None

        logger.debug(
            "[VOICE:TELNYX:AGENT] session stopped call_control_id=%s",
            self._call_control_id,
        )

    async def _agent_worker(self) -> None:
        """Serialized agent processing worker.

        Consumes utterances from the queue one at a time and processes
        them through AgentRuntime. Runs in a separate task so the Telnyx
        media receive loop is never blocked.
        """
        logger.debug(
            "[VOICE:TELNYX:AGENT] worker loop started "
            "call_control_id=%s",
            self._call_control_id,
        )
        while True:
            try:
                utterance = await self._utterance_queue.get()
            except asyncio.CancelledError:
                break

            # None sentinel — shutdown signal
            if utterance is None:
                logger.debug(
                    "[VOICE:TELNYX:AGENT] shutdown sentinel"
                )
                break

            self._turn += 1
            turn = self._turn
            turn_start = time.monotonic()
            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d start model=%s",
                turn,
                _TELNYX_LLM_MODEL,
            )
            logger.debug(
                "[VOICE:TELNYX:AGENT] turn=%d transcript='%s' history=%d",
                turn,
                utterance[:80],
                len(self._history),
            )

            try:
                result = await self._process_utterance(utterance)
                turn_ms = (time.monotonic() - turn_start) * 1000
                response_text = result.get("response", "")
                usage = result.get("usage", {})
                iterations = result.get("iterations", 0)
                tool_calls = result.get("tool_calls", [])

                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d completed "
                    "duration_ms=%.0f iterations=%d tools=%d",
                    turn,
                    turn_ms,
                    iterations,
                    len(tool_calls),
                )
                logger.debug(
                    "[VOICE:TELNYX:AGENT] turn=%d response='%s' "
                    "tokens_in=%s tokens_out=%s",
                    turn,
                    response_text[:200] if response_text else "",
                    usage.get("prompt_tokens", "n/a"),
                    usage.get("completion_tokens", "n/a"),
                )

            except Exception as e:
                turn_ms = (time.monotonic() - turn_start) * 1000
                logger.error(
                    "[VOICE:TELNYX:AGENT] turn=%d failed "
                    "duration_ms=%.0f error=%s",
                    turn,
                    turn_ms,
                    e,
                )
                # Do NOT crash — keep the media/Deepgram pipeline alive

        logger.debug(
            "[VOICE:TELNYX:AGENT] worker loop stopped "
            "total_turns=%d",
            self._turn,
        )

    async def _process_utterance(
        self, transcript: str
    ) -> dict[str, Any]:
        """Process one utterance through AgentRuntime.

        Uses the session-level LLM provider and accumulates conversation
        history. The current utterance is NOT duplicated in history.
        """
        if self._llm is None:
            raise RuntimeError("LLM provider not initialized")

        # Build agent with session-level LLM
        tool_registry = get_tool_registry()
        tool_executor = ToolExecutor(tool_registry)
        config = AgentConfig(
            llm_model=_TELNYX_LLM_MODEL,
            max_tool_rounds=settings.max_agent_iterations,
        )
        runtime = AgentRuntime(
            llm=self._llm,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            config=config,
        )

        # Run agent loop with conversation history
        # AgentRuntime.run() adds the user message to its internal state,
        # so we pass history as initial_messages (without the current utterance)
        result: AgentResult = await runtime.run(
            transcript,
            initial_messages=list(self._history),
        )

        # Update session conversation history with new messages
        # (user message + assistant response + any tool messages)
        new_start = len(self._history)
        new_msgs = runtime.state.messages[new_start:]
        for llm_msg in new_msgs:
            self._history.append(llm_msg)

        return {
            "response": result.response,
            "tool_calls": result.tool_calls,
            "usage": result.usage,
            "iterations": result.iterations,
        }
