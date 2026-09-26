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
import json
import time
from typing import Any

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
                speech_end_at = utterance.utterance_end_at
                result = await self._process_utterance_streaming(
                    utterance.text, turn, speech_end_at=speech_end_at
                )
                response_text = result.get("response", "")
                usage = result.get("usage", {})
                iterations = result.get("iterations", 0)
                tool_calls = result.get("tool_calls", [])
                streamed = result.get("streamed", False)
                timing = result.get("timing", {})

                # Extract tool-turn timing from AgentRuntime
                tool_execution_ms = timing.get("tool_execution_total_ms")
                final_llm_complete_ms = timing.get("final_llm_complete_ms")
                tool_call_detected_at = result.get("tool_call_detected_at")
                initial_llm_request_start = result.get("initial_llm_request_start")

                # Compute end-to-end metrics from speech_end_at
                first_audio_sent_at = result.get("first_audio_sent_at")
                tts_complete_at = result.get("tts_complete_at")
                speech_to_first_audio_ms: float | None = None
                if first_audio_sent_at is not None:
                    speech_to_first_audio_ms = (
                        first_audio_sent_at - speech_end_at
                    ) * 1000
                tts_ttfa_ms: float | None = None
                tts_total_ms: float | None = None
                tts_start_abs = result.get("tts_start")
                if tts_start_abs is not None:
                    if first_audio_sent_at is not None:
                        tts_ttfa_ms = (first_audio_sent_at - tts_start_abs) * 1000
                    if tts_complete_at is not None:
                        tts_total_ms = (tts_complete_at - tts_start_abs) * 1000

                # Build agent_response metadata with accurate LLM metrics
                llm_total_ms = result.get("llm_complete_ms")
                # Compute tool detection latency
                tool_detection_ms: float | None = None
                if (
                    tool_call_detected_at is not None
                    and initial_llm_request_start is not None
                ):
                    tool_detection_ms = (
                        tool_call_detected_at - initial_llm_request_start
                    ) * 1000
                # initial_llm_total_ms: initial streaming LLM duration
                # (request_start → stream complete / tool detected)
                initial_llm_total_ms = result.get("initial_llm_complete_ms")
                agent_meta: dict[str, Any] = {
                    "iterations": iterations,
                    "tool_calls": len(tool_calls),
                    "streamed": streamed,
                    "llm_ttft_ms": (
                        round(result["llm_ttft_ms"])
                        if result.get("llm_ttft_ms") is not None
                        else None
                    ),
                    "llm_first_sentence_ms": (
                        round(result["llm_first_sentence_ms"])
                        if result.get("llm_first_sentence_ms") is not None
                        else None
                    ),
                    "llm_total_ms": (
                        round(llm_total_ms) if llm_total_ms is not None else None
                    ),
                    "initial_llm_total_ms": (
                        round(initial_llm_total_ms)
                        if initial_llm_total_ms is not None
                        else None
                    ),
                    "tool_execution_ms": (
                        round(tool_execution_ms)
                        if tool_execution_ms is not None
                        else None
                    ),
                    "tool_detection_ms": (
                        round(tool_detection_ms)
                        if tool_detection_ms is not None
                        else None
                    ),
                    "final_llm_total_ms": (
                        round(final_llm_complete_ms)
                        if final_llm_complete_ms is not None
                        else None
                    ),
                }
                await broadcast_telephony_event(
                    "agent_response",
                    self._call_control_id,
                    "Response generated",
                    turn=turn,
                    metadata=agent_meta,
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
                        tts_result = await self._tts_service.synthesize_and_send(
                            text=response_text,
                            turn=turn,
                            websocket=self._websocket,
                            call_id=self._call_control_id,
                        )
                        # Capture TTS timing for tool turns
                        tts_start_abs = tts_result.get("tts_start")
                        tts_complete_at = tts_result.get("tts_complete_at")
                        tool_first_audio_sent_at = tts_result.get("first_audio_sent_at")
                        if tts_start_abs is not None and tts_complete_at is not None:
                            tts_total_ms = (tts_complete_at - tts_start_abs) * 1000
                        # For tool turns, speech_to_first_audio_ms uses the
                        # TTS first audio timestamp (streaming path had no audio)
                        if tool_first_audio_sent_at is not None:
                            first_audio_sent_at = tool_first_audio_sent_at
                            speech_to_first_audio_ms = (
                                first_audio_sent_at - speech_end_at
                            ) * 1000
                            if tts_start_abs is not None:
                                tts_ttfa_ms = (
                                    first_audio_sent_at - tts_start_abs
                                ) * 1000
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

                # turn_complete_at: AFTER all TTS audio has been sent
                turn_complete_at = time.monotonic()
                turn_ms = (turn_complete_at - turn_start) * 1000
                turn_total_ms = None
                if speech_end_at is not None:
                    turn_total_ms = (turn_complete_at - speech_end_at) * 1000

                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d completed "
                    "duration_ms=%.0f iterations=%d tools=%d streamed=%s "
                    "speech_to_first_audio_ms=%s turn_total_ms=%s",
                    turn,
                    turn_ms,
                    iterations,
                    len(tool_calls),
                    streamed,
                    f"{speech_to_first_audio_ms:.0f}"
                    if speech_to_first_audio_ms is not None
                    else "n/a",
                    f"{turn_total_ms:.0f}" if turn_total_ms is not None else "n/a",
                )

                # Diagnostic: log full tool-turn timestamp trace
                if tool_calls:
                    logger.info(
                        "[VOICE:TELNYX:AGENT] turn=%d TOOL_TURN_TRACE "
                        "initial_llm_total_ms=%s "
                        "tool_detection_ms=%s "
                        "tool_execution_ms=%s "
                        "final_llm_total_ms=%s "
                        "tts_ttfa_ms=%s "
                        "tts_total_ms=%s "
                        "speech_to_first_audio_ms=%s "
                        "turn_total_ms=%s",
                        turn,
                        f"{initial_llm_total_ms:.0f}"
                        if initial_llm_total_ms is not None
                        else "n/a",
                        f"{tool_detection_ms:.0f}"
                        if tool_detection_ms is not None
                        else "n/a",
                        f"{tool_execution_ms:.0f}"
                        if tool_execution_ms is not None
                        else "n/a",
                        f"{final_llm_complete_ms:.0f}"
                        if final_llm_complete_ms is not None
                        else "n/a",
                        f"{tts_ttfa_ms:.0f}"
                        if tts_ttfa_ms is not None
                        else "n/a",
                        f"{tts_total_ms:.0f}"
                        if tts_total_ms is not None
                        else "n/a",
                        f"{speech_to_first_audio_ms:.0f}"
                        if speech_to_first_audio_ms is not None
                        else "n/a",
                        f"{turn_total_ms:.0f}"
                        if turn_total_ms is not None
                        else "n/a",
                    )

                # Emit turn_completed with all end-to-end metrics
                await broadcast_telephony_event(
                    "turn_completed",
                    self._call_control_id,
                    f"Turn {turn} completed",
                    turn=turn,
                    metadata={
                        "speech_to_first_audio_ms": (
                            round(speech_to_first_audio_ms)
                            if speech_to_first_audio_ms is not None
                            else None
                        ),
                        "llm_ttft_ms": agent_meta.get("llm_ttft_ms"),
                        "llm_first_sentence_ms": agent_meta.get(
                            "llm_first_sentence_ms"
                        ),
                        "llm_total_ms": agent_meta.get("llm_total_ms"),
                        "initial_llm_total_ms": agent_meta.get(
                            "initial_llm_total_ms"
                        ),
                        "tool_execution_ms": agent_meta.get("tool_execution_ms"),
                        "tool_detection_ms": agent_meta.get("tool_detection_ms"),
                        "final_llm_total_ms": agent_meta.get("final_llm_total_ms"),
                        "tts_ttfa_ms": (
                            round(tts_ttfa_ms)
                            if tts_ttfa_ms is not None
                            else None
                        ),
                        "tts_total_ms": (
                            round(tts_total_ms)
                            if tts_total_ms is not None
                            else None
                        ),
                        "turn_total_ms": (
                            round(turn_total_ms)
                            if turn_total_ms is not None
                            else None
                        ),
                    },
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

    async def _process_utterance_streaming(
        self, transcript: str, turn: int, *, speech_end_at: float | None = None
    ) -> dict[str, Any]:
        """Process utterance with single stream_chat() call.

        Streams the LLM response directly through the sentence buffer
        into TTS. Tool calls are detected from stream chunks; if found,
        falls back to the non-streaming AgentRuntime path.

        Args:
            transcript: The user's transcribed text.
            turn: Current turn number for observability.
            speech_end_at: Monotonic timestamp when speech ended (for
                end-to-end latency calculation).

        Returns:
            Dict with response, timing metrics, and streaming info.
        """
        if self._llm is None:
            raise RuntimeError("LLM provider not initialized")

        llm_request_start = time.monotonic()
        llm_ttft_ms: float | None = None
        llm_first_sentence_ms: float | None = None
        llm_complete_ms: float | None = None

        # Build messages for LLM
        messages = [LLMMessage(role="system", content=_TELNYX_SYSTEM_PROMPT)]
        messages.extend(self._history)
        messages.append(LLMMessage(role="user", content=transcript))

        # Single stream_chat() call WITH tool schemas
        tool_schemas = get_tool_registry().get_schemas()
        logger.info(
            "[VOICE:TELNYX:AGENT] turn=%d stream_chat start tools=%d",
            turn,
            len(tool_schemas),
        )

        await broadcast_telephony_event(
            "llm_stream_start",
            self._call_control_id,
            f"Streaming from GPT-4o-mini ({len(tool_schemas)} tools)",
            turn=turn,
            metadata={"provider": "openrouter", "tools": len(tool_schemas)},
        )

        # Stream and detect tool calls
        sentence_buffer = SentenceBuffer()
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        accumulated_text = ""
        tool_detected = False
        tool_call_detected_at: float | None = None
        usage_data: dict[str, int] = {}
        # Tool call delta accumulation (OpenAI streaming protocol)
        tool_call_deltas: dict[int, dict[str, Any]] = {}

        async def stream_llm() -> None:
            """Stream LLM tokens, accumulate tool calls, feed sentence buffer."""
            nonlocal llm_ttft_ms, llm_first_sentence_ms
            nonlocal llm_complete_ms, accumulated_text, tool_detected
            nonlocal usage_data, tool_call_detected_at

            try:
                async for chunk in self._llm.stream_chat(
                    messages=messages,
                    model=_TELNYX_LLM_MODEL,
                    tools=tool_schemas if tool_schemas else None,
                ):
                    # Accumulate tool call deltas from stream
                    if chunk.tool_calls:
                        if not tool_detected:
                            tool_detected = True
                            tool_call_detected_at = time.monotonic()
                            logger.info(
                                "[VOICE:TELNYX:AGENT] turn=%d tool_calls "
                                "detected in stream — accumulating",
                                turn,
                            )
                        # Accumulate deltas by index
                        for tc_delta in chunk.tool_calls:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_deltas:
                                tool_call_deltas[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            assembled = tool_call_deltas[idx]
                            if tc_delta.get("id"):
                                assembled["id"] = tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                assembled["function"]["name"] = func["name"]
                            if func.get("arguments"):
                                assembled["function"]["arguments"] += func["arguments"]
                        continue

                    # Track first token timing
                    if chunk.content and llm_ttft_ms is None:
                        llm_ttft_ms = (
                            time.monotonic() - llm_request_start
                        ) * 1000
                        await broadcast_telephony_event(
                            "llm_first_token",
                            self._call_control_id,
                            f"First LLM token after {llm_ttft_ms:.0f}ms",
                            turn=turn,
                            metadata={"ttft_ms": round(llm_ttft_ms)},
                        )

                    if chunk.content:
                        accumulated_text += chunk.content
                        sentences = sentence_buffer.add(chunk.content)

                        for sentence in sentences:
                            if llm_first_sentence_ms is None:
                                llm_first_sentence_ms = (
                                    time.monotonic() - llm_request_start
                                ) * 1000
                                await broadcast_telephony_event(
                                    "llm_first_sentence",
                                    self._call_control_id,
                                    f"First sentence after "
                                    f"{llm_first_sentence_ms:.0f}ms",
                                    turn=turn,
                                    metadata={
                                        "ttfs_ms": round(
                                            llm_first_sentence_ms
                                        ),
                                    },
                                )
                            await sentence_queue.put(sentence)

                    if chunk.finish_reason:
                        usage_data = {"finish_reason": chunk.finish_reason}

                llm_complete_ms = (
                    time.monotonic() - llm_request_start
                ) * 1000

                # Flush remaining text (only if no tool calls)
                if not tool_detected:
                    remaining = sentence_buffer.flush()
                    if remaining:
                        if llm_first_sentence_ms is None:
                            llm_first_sentence_ms = (
                                time.monotonic() - llm_request_start
                            ) * 1000
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
        tts_result: dict[str, Any] = {"success": False}
        if self._tts_service and self._websocket:
            tts_result = await self._tts_service.stream_sentences(
                sentences=sentence_iterator(),
                turn=turn,
                websocket=self._websocket,
                call_id=self._call_control_id,
            )

        # Wait for stream task to complete
        await stream_task

        # If tool calls detected, execute directly — no redundant LLM call
        if tool_detected:
            # Assemble complete tool calls from accumulated deltas
            detected_tool_calls = [
                {
                    "id": tool_call_deltas[idx]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call_deltas[idx]["function"]["name"],
                        "arguments": tool_call_deltas[idx]["function"]["arguments"],
                    },
                }
                for idx in sorted(tool_call_deltas.keys())
            ]
            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d assembled %d tool call(s) "
                "from stream deltas",
                turn,
                len(detected_tool_calls),
            )
            for tc in detected_tool_calls:
                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d tool_call name=%s "
                    "args=%s",
                    turn,
                    tc["function"]["name"],
                    tc["function"]["arguments"][:200],
                )

            # Add user message to conversation history
            self._history.append(LLMMessage(role="user", content=transcript))

            # Add assistant message with tool calls to history
            self._history.append(
                LLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=detected_tool_calls,
                )
            )

            # Execute each tool call
            tool_registry = get_tool_registry()
            tool_executor = ToolExecutor(tool_registry)
            tool_execution_total_ms: float = 0
            tool_details: list[dict[str, Any]] = []

            for tool_call in detected_tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                tool_start = time.monotonic()
                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d executing tool=%s",
                    turn,
                    tool_name,
                )
                tool_result = await tool_executor.execute_from_json(
                    tool_name, arguments_str
                )
                tool_ms = (time.monotonic() - tool_start) * 1000
                tool_execution_total_ms += tool_ms
                if tool_result.success:
                    raw_result = tool_result.output
                else:
                    raw_result = {"error": tool_result.error}
                tool_success = tool_result.success
                tool_details.append({
                    "name": tool_name,
                    "execution_ms": round(tool_ms),
                    "success": tool_success,
                })
                logger.info(
                    "[VOICE:TELNYX:AGENT] turn=%d tool=%s "
                    "duration_ms=%.0f success=%s",
                    turn,
                    tool_name,
                    tool_ms,
                    tool_success,
                )
                # Add tool result to conversation history
                self._history.append(
                    LLMMessage(
                        role="tool",
                        content=(
                            json.dumps(raw_result)
                            if not isinstance(raw_result, str)
                            else raw_result
                        ),
                        tool_call_id=tool_call.get("id", ""),
                        name=tool_name,
                    )
                )

            # Final LLM call with tool results in history
            final_llm_start = time.monotonic()
            final_messages = [
                LLMMessage(role="system", content=_TELNYX_SYSTEM_PROMPT)
            ]
            final_messages.extend(self._history)

            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d final LLM call "
                "messages=%d",
                turn,
                len(final_messages),
            )
            final_response = await self._llm.chat(
                messages=final_messages,
                model=_TELNYX_LLM_MODEL,
            )
            final_llm_complete_ms = (
                time.monotonic() - final_llm_start
            ) * 1000
            logger.info(
                "[VOICE:TELNYX:AGENT] turn=%d final LLM completed "
                "duration_ms=%.0f response_length=%d",
                turn,
                final_llm_complete_ms,
                len(final_response.content or ""),
            )

            # Add final assistant response to history
            self._history.append(
                LLMMessage(
                    role="assistant",
                    content=final_response.content or "",
                )
            )

            # Accumulate usage from final LLM
            total_usage = dict(usage_data)
            for key, val in (final_response.usage or {}).items():
                total_usage[key] = total_usage.get(key, 0) + val

            return {
                "response": final_response.content or "",
                "tool_calls": detected_tool_calls,
                "usage": total_usage,
                "iterations": 2,  # initial stream + final chat
                "streamed": False,
                "llm_ttft_ms": llm_ttft_ms,
                "llm_first_sentence_ms": None,
                "tool_call_detected_at": tool_call_detected_at,
                "initial_llm_request_start": llm_request_start,
                "initial_llm_complete_ms": llm_complete_ms,
                "timing": {
                    "tool_execution_total_ms": tool_execution_total_ms,
                    "tool_details": tool_details,
                    "final_llm_start_at": final_llm_start,
                    "final_llm_complete_ms": final_llm_complete_ms,
                },
            }

        # Update conversation history
        self._history.append(LLMMessage(role="user", content=transcript))
        self._history.append(
            LLMMessage(role="assistant", content=accumulated_text)
        )

        # Capture absolute timestamps for end-to-end metric computation
        # in _agent_worker(). The actual speech_to_first_audio_ms is
        # computed there using speech_end_at as the anchor.
        tts_first_audio = tts_result.get("first_audio_ms")
        first_audio_sent_at = tts_result.get("first_audio_sent_at")
        tts_start_abs = tts_result.get("tts_start")
        tts_complete_at = time.monotonic()  # TTS stream_sentences just finished

        logger.info(
            "[VOICE:TELNYX:AGENT] turn=%d streaming completed "
            "ttft_ms=%s ttfs_ms=%s llm_total_ms=%.0f "
            "text_length=%d tts_ttfa_ms=%s",
            turn,
            f"{llm_ttft_ms:.0f}" if llm_ttft_ms is not None else "n/a",
            f"{llm_first_sentence_ms:.0f}"
            if llm_first_sentence_ms is not None
            else "n/a",
            llm_complete_ms or 0,
            len(accumulated_text),
            f"{tts_first_audio:.0f}" if tts_first_audio is not None else "n/a",
        )

        return {
            "response": accumulated_text,
            "tool_calls": [],
            "usage": usage_data,
            "iterations": 1,
            "streamed": True,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_first_sentence_ms": llm_first_sentence_ms,
            "llm_complete_ms": llm_complete_ms,
            "tts_first_audio_ms": tts_first_audio,
            "first_audio_sent_at": first_audio_sent_at,
            "tts_start": tts_start_abs,
            "tts_complete_at": tts_complete_at,
            "total_sentences": tts_result.get("total_sentences", 0),
        }
