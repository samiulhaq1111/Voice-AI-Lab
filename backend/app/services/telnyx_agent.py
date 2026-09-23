"""Telephony agent service (Phase 6C Milestone 6/7).

Processes finalized utterances from the Telnyx → Deepgram pipeline through
AgentRuntime, then synthesizes the text response via TTS and sends
telephone-compatible audio back through the Telnyx media WebSocket.

Architecture:
    bridge.utterance_queue
        ↓
    _agent_worker() — one utterance at a time
        ↓
    AgentRuntime.run(transcript, initial_messages=history)
        ↓
    TelnyxTTSService.synthesize_and_send(response_text)
        ↓
    Caller hears AI response

The agent worker runs as a separate async task so that the Telnyx media
receive loop and Deepgram event pump are never blocked by LLM calls.

Session-level lifecycle:
    - LLM + TTS providers created once when the telephony session starts
    - Reused across all utterances in the call
    - Closed exactly once during session cleanup
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
from app.services.sentence_buffer import SentenceBuffer
from app.services.telephony_events import broadcast_telephony_event
from app.services.telnyx_deepgram import Utterance
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor

# Telnyx telephony uses a fast, reliable model for low-latency phone calls.
# This is independent of the global default LLM model.
_TELNYX_LLM_MODEL = "openai/gpt-4o-mini"

# Default system prompt for telephone voice assistant
_TELNYX_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and conversational."
)


class TelephonyAgentSession:
    """Manages agent processing for a single telephony call.

    Owns the session-level LLM provider, TTS service, and conversation
    history. Processes utterances one at a time via the agent worker.
    """

    def __init__(
        self,
        call_control_id: str,
        utterance_queue: asyncio.Queue[Utterance | None],
        *,
        tts_service: Any | None = None,
        websocket: Any | None = None,
    ) -> None:
        self._call_control_id = call_control_id
        self._utterance_queue = utterance_queue
        self._llm: LLMInterface | None = None
        self._history: list[LLMMessage] = []
        self._turn: int = 0
        self._worker_task: asyncio.Task | None = None
        self._running = False
        self._tts_service = tts_service
        self._websocket = websocket

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
            # Compute end-of-speech latency deltas from the bridge's
            # monotonic timestamps. These are the same clock used by
            # the bridge, so the deltas are exact.
            speech_final_to_utterance_end_ms: float | None = None
            utterance_end_to_agent_ms: float | None = None
            speech_final_to_agent_ms: float | None = None
            if (
                utterance.speech_final_at is not None
                and utterance.utterance_end_at is not None
            ):
                speech_final_to_utterance_end_ms = (
                    utterance.utterance_end_at - utterance.speech_final_at
                ) * 1000
                utterance_end_to_agent_ms = (
                    turn_start - utterance.utterance_end_at
                ) * 1000
                speech_final_to_agent_ms = (
                    turn_start - utterance.speech_final_at
                ) * 1000
            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d start model=%s "
                "speech_final_to_utterance_end_ms=%.0f "
                "utterance_end_to_agent_ms=%.0f",
                turn,
                _TELNYX_LLM_MODEL,
                speech_final_to_utterance_end_ms or 0.0,
                utterance_end_to_agent_ms or 0.0,
            )
            await broadcast_telephony_event(
                "agent_processing",
                self._call_control_id,
                "Processing with GPT-4o-mini",
                turn=turn,
                metadata={
                    "provider": "openrouter",
                    "model": _TELNYX_LLM_MODEL,
                    "release_reason": utterance.release_reason,
                    "settle_ms": utterance.settle_ms,
                    "speech_final_to_utterance_end_ms": (
                        round(speech_final_to_utterance_end_ms)
                        if speech_final_to_utterance_end_ms is not None
                        else None
                    ),
                    "utterance_end_to_agent_ms": (
                        round(utterance_end_to_agent_ms)
                        if utterance_end_to_agent_ms is not None
                        else None
                    ),
                    "speech_final_to_agent_ms": (
                        round(speech_final_to_agent_ms)
                        if speech_final_to_agent_ms is not None
                        else None
                    ),
                },
            )
            logger.debug(
                "[VOICE:TELNYX:AGENT] turn=%d transcript='%s' history=%d",
                turn,
                utterance.text[:80],
                len(self._history),
            )

            try:
                # Use streaming path for reduced latency
                result = await self._process_utterance_streaming(
                    utterance.text, turn
                )
                turn_ms = (time.monotonic() - turn_start) * 1000
                response_text = result.get("response", "")
                usage = result.get("usage", {})
                iterations = result.get("iterations", 0)
                tool_calls = result.get("tool_calls", [])
                streamed = result.get("streamed", False)

                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d completed "
                    "duration_ms=%.0f iterations=%d tools=%d streamed=%s",
                    turn,
                    turn_ms,
                    iterations,
                    len(tool_calls),
                    streamed,
                )
                await broadcast_telephony_event(
                    "agent_response",
                    self._call_control_id,
                    "Response generated",
                    turn=turn,
                    metadata={
                        "duration_ms": round(turn_ms),
                        "iterations": iterations,
                        "tool_calls": len(tool_calls),
                        "streamed": streamed,
                        "first_token_ms": (
                            round(result["first_token_ms"])
                            if result.get("first_token_ms") is not None
                            else None
                        ),
                        "first_sentence_ms": (
                            round(result["first_sentence_ms"])
                            if result.get("first_sentence_ms") is not None
                            else None
                        ),
                    },
                )
                logger.debug(
                    "[VOICE:TELNYX:AGENT] turn=%d response='%s' "
                    "tokens_in=%s tokens_out=%s",
                    turn,
                    response_text[:200] if response_text else "",
                    usage.get("prompt_tokens", "n/a"),
                    usage.get("completion_tokens", "n/a"),
                )

                # For non-streamed responses (tool calls), handle TTS here
                if not streamed and response_text and self._tts_service and self._websocket:
                    try:
                        await self._tts_service.synthesize_and_send(
                            text=response_text,
                            turn=turn,
                            websocket=self._websocket,
                            call_id=self._call_control_id,
                        )
                    except Exception as e:
                        logger.error(
                            "[VOICE:TELNYX:AGENT] turn=%d tts_send_error=%s",
                            turn,
                            e,
                        )
                elif response_text and not self._tts_service:
                    logger.debug(
                        "[VOICE:TELNYX:AGENT] turn=%d TTS skipped — no service",
                        turn,
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
                await broadcast_telephony_event(
                    "error",
                    self._call_control_id,
                    "Agent processing failed",
                    turn=turn,
                    metadata={"stage": "agent"},
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

    async def _process_utterance_streaming(
        self, transcript: str, turn: int
    ) -> dict[str, Any]:
        """Process utterance with LLM streaming for reduced latency.

        Uses a two-phase approach:
        1. First, use non-streaming chat() to check for tool calls
        2. If no tools, use stream_chat() to stream the response
        3. Sentence buffer feeds TTS incrementally

        Args:
            transcript: The user's transcribed text.
            turn: Current turn number for observability.

        Returns:
            Dict with response, timing metrics, and streaming info.
        """
        if self._llm is None:
            raise RuntimeError("LLM provider not initialized")

        stream_start = time.monotonic()
        first_token_ms: float | None = None
        first_sentence_ms: float | None = None

        # Build messages for LLM
        messages = [LLMMessage(role="system", content=_TELNYX_SYSTEM_PROMPT)]
        messages.extend(self._history)
        messages.append(LLMMessage(role="user", content=transcript))

        # Phase 1: Check for tool calls using non-streaming chat()
        tool_schemas = get_tool_registry().get_schemas()
        logger.debug(
            "[VOICE:TELNYX:AGENT] turn=%d streaming phase1: tool check",
            turn,
        )

        response = await self._llm.chat(
            messages=messages,
            model=_TELNYX_LLM_MODEL,
            tools=tool_schemas if tool_schemas else None,
        )

        # If tool calls detected, fall back to non-streaming path
        if response.tool_calls:
            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d tool_calls=%d — "
                "falling back to non-streaming",
                turn,
                len(response.tool_calls),
            )
            # Use existing AgentRuntime for tool execution
            result = await self._process_utterance(transcript)
            result["streamed"] = False
            result["fallback_reason"] = "tool_calls"
            return result

        # Phase 2: No tools — stream the response
        logger.info(
            "[VOICE:TELNYX:AGENT] turn=%d streaming phase2: no tools, "
            "using stream_chat",
            turn,
        )

        await broadcast_telephony_event(
            "llm_stream_start",
            self._call_control_id,
            "LLM streaming response",
            turn=turn,
            metadata={"provider": "openrouter"},
        )

        # Stream the response
        sentence_buffer = SentenceBuffer()
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        accumulated_text = ""

        async def stream_llm() -> None:
            """Stream LLM tokens and feed sentence buffer."""
            nonlocal first_token_ms, first_sentence_ms, accumulated_text

            try:
                async for chunk in self._llm.stream_chat(
                    messages=messages,
                    model=_TELNYX_LLM_MODEL,
                ):
                    if first_token_ms is None:
                        first_token_ms = (
                            time.monotonic() - stream_start
                        ) * 1000
                        await broadcast_telephony_event(
                            "llm_first_token",
                            self._call_control_id,
                            f"First LLM token after {first_token_ms:.0f}ms",
                            turn=turn,
                            metadata={"ttft_ms": round(first_token_ms)},
                        )

                    accumulated_text += chunk
                    sentences = sentence_buffer.add(chunk)

                    for sentence in sentences:
                        if first_sentence_ms is None:
                            first_sentence_ms = (
                                time.monotonic() - stream_start
                            ) * 1000
                            await broadcast_telephony_event(
                                "llm_first_sentence",
                                self._call_control_id,
                                f"First sentence after {first_sentence_ms:.0f}ms",
                                turn=turn,
                                metadata={"ttfs_ms": round(first_sentence_ms)},
                            )
                        await sentence_queue.put(sentence)

                # Flush remaining text
                remaining = sentence_buffer.flush()
                if remaining:
                    await sentence_queue.put(remaining)

            except Exception as e:
                logger.error(
                    "[VOICE:TELNYX:AGENT] turn=%d LLM stream error=%s",
                    turn,
                    e,
                )
            finally:
                await sentence_queue.put(None)  # Sentinel

        async def sentence_iterator():
            """Yield sentences from the queue."""
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                yield sentence

        # Start LLM streaming in background
        stream_task = asyncio.create_task(stream_llm())

        # Stream sentences to TTS
        tts_result = {"success": False}
        if self._tts_service and self._websocket:
            tts_result = await self._tts_service.stream_sentences(
                sentences=sentence_iterator(),
                turn=turn,
                websocket=self._websocket,
                call_id=self._call_control_id,
            )

        # Wait for stream to complete
        await stream_task

        # Update history
        self._history.append(LLMMessage(role="user", content=transcript))
        self._history.append(
            LLMMessage(role="assistant", content=accumulated_text)
        )

        total_ms = (time.monotonic() - stream_start) * 1000
        logger.info(
            "[VOICE:TELNYX:AGENT] turn=%d streaming completed "
            "ttft_ms=%s ttfs_ms=%s total_ms=%.0f text_length=%d",
            turn,
            f"{first_token_ms:.0f}" if first_token_ms is not None else "n/a",
            f"{first_sentence_ms:.0f}" if first_sentence_ms is not None else "n/a",
            total_ms,
            len(accumulated_text),
        )

        return {
            "response": accumulated_text,
            "tool_calls": [],
            "usage": response.usage,
            "iterations": 1,
            "streamed": True,
            "first_token_ms": first_token_ms,
            "first_sentence_ms": first_sentence_ms,
            "tts_first_audio_ms": tts_result.get("first_audio_ms"),
            "total_sentences": tts_result.get("total_sentences", 0),
        }
