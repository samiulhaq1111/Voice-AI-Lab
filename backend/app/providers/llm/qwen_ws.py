"""Qwen WebSocket LLM adapter implementing LLMInterface.

Talks to a persistent, stateless Qwen3.5-4B inference server over a single
WebSocket connection (e.g. a Kaggle tunnel). One adapter instance owns one
connection for the whole Voice AI realtime session; the server keeps no
conversation state, so the complete message history is sent with every
request.

Client -> server frames:
    {"type": "chat", "request_id": ..., "messages": [...],
     "temperature": ..., "max_tokens": ...}
    {"type": "cancel", "request_id": ...}

Server -> client frames:
    {"type": "start", "request_id": ...}
    {"type": "timing", "event": "first_token", "request_id": ..., "ttft": ...}
    {"type": "token", "request_id": ..., "text": ...}
    {"type": "timing", "event": "complete", "request_id": ..., "ttft": ...,
     "generation_time": ..., "total_time": ..., "chunks": ...}
    {"type": "done", "request_id": ...}
    {"type": "cancel_ack", "request_id": ...}
    {"type": "cancelled", "request_id": ..., "tokens": ..., "total_time": ...}
    {"type": "error", ...}
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import websockets

from app.core.config import settings
from app.core.logging import logger
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage, LLMResponse, StreamChunk, ToolSchema

# Bound the WebSocket handshake so a dead Kaggle tunnel fails fast.
_OPEN_TIMEOUT_CAP_S = 10.0
_PING_INTERVAL_S = 20.0
_PING_TIMEOUT_S = 20.0

# Bounded deadline for the best-effort remote cancellation handshake.
_CANCEL_DRAIN_TIMEOUT_S = 2.0

# Safety valve against a server flooding stale frames for old requests.
_MAX_STALE_FRAMES = 1000


class QwenWebSocketAdapter(LLMInterface):
    """Streaming adapter for a stateless Qwen WebSocket inference server.

    - ONE persistent WebSocket per adapter instance (= per Voice AI realtime
      session), established lazily on the first request.
    - The server holds no conversation state: the full message history is
      sent with every request.
    - Requests are serialized through an asyncio.Lock; the realtime gateway
      already runs one agent worker per session, the lock is defense in
      depth.
    - Tool calls are not part of the Kaggle protocol; the ``tools`` argument
      is accepted for interface compatibility and ignored.
    - No automatic retry/replay: if the connection fails mid-request the
      request fails; the next request may reconnect lazily.
    """

    def __init__(
        self,
        *,
        ws_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._ws_url = ws_url or settings.qwen_ws_url
        self._api_key = api_key or settings.qwen_ws_api_key
        self._default_model = default_model
        self._timeout = timeout or settings.llm_timeout
        if not self._ws_url:
            raise ValueError("Qwen WebSocket URL not configured. Set QWEN_WS_URL.")

        self._ws: Any = None
        self._lock = asyncio.Lock()
        self._closed = False
        logger.debug("Qwen WebSocket adapter initialized url=%s", self._ws_url)

    @property
    def provider_name(self) -> str:
        return "qwen_ws"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Non-streaming completion: consume stream_chat() and join content."""
        parts: list[str] = []
        async for chunk in self.stream_chat(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk.content:
                parts.append(chunk.content)
        return LLMResponse(
            content="".join(parts),
            tool_calls=None,
            finish_reason="stop",
            usage={},
            metadata={},
        )

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion tokens from the Qwen WebSocket server.

        Yields one StreamChunk(content=...) per ``token`` frame and exactly
        one terminal StreamChunk(finish_reason="stop") on ``done``. Frames
        carrying an unknown type are logged and skipped.
        """
        if tools:
            logger.debug(
                "[LLM] provider=qwen_ws tools_ignored count=%d (protocol has no tool calls)",
                len(tools),
            )
        resolved_model = model or self._default_model
        request_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "type": "chat",
            "request_id": request_id,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request_start = time.monotonic()

        # The realtime gateway serializes turns through a single agent
        # worker; the lock is defense in depth so concurrent stream_chat
        # calls can never interleave frames on the shared connection.
        async with self._lock:
            ws = await self._ensure_connected()
            try:
                await ws.send(json.dumps(payload))
            except websockets.ConnectionClosed as e:
                # Lazy reconnect policy: drop the dead socket so the NEXT
                # request can reconnect. This request is never replayed.
                self._ws = None
                logger.error(
                    "[LLM] provider=qwen_ws send_failed request_id=%s error=%s",
                    request_id,
                    e,
                )
                raise RuntimeError(
                    f"Qwen WebSocket connection lost while sending request: {e}"
                ) from e

            logger.info(
                "[LLM] provider=qwen_ws request_sent request_id=%s messages=%d model=%s",
                request_id,
                len(payload["messages"]),
                resolved_model or "server-default",
            )

            request_finished = False
            try:
                async for chunk in self._receive_stream(ws, request_id):
                    if chunk.finish_reason:
                        request_finished = True
                    yield chunk
            except GeneratorExit:
                # The consumer stopped early (session STOP/disconnect, or the
                # usual break right after the finish_reason chunk). Generator
                # close runs in the event loop's finalizer task, so the caller
                # is never delayed. If the request did not reach ``done``,
                # tell the server to stop generating (bounded, best effort).
                # Cancellation is never suppressed.
                if not request_finished:
                    await self._cancel(request_id)
                raise
            except websockets.ConnectionClosed as e:
                self._ws = None
                logger.error(
                    "[LLM] provider=qwen_ws stream_closed request_id=%s error=%s",
                    request_id,
                    e,
                )
                raise RuntimeError(
                    f"Qwen WebSocket closed during streaming request: {e}"
                ) from e

            logger.info(
                "[LLM] provider=qwen_ws request_complete request_id=%s duration_ms=%.0f",
                request_id,
                (time.monotonic() - request_start) * 1000,
            )

    async def close(self) -> None:
        """Close the persistent WebSocket connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            await ws.close()
            logger.info("[LLM] provider=qwen_ws connection_closed")
        except Exception as e:
            logger.warning(
                "[LLM] provider=qwen_ws close_failed error_type=%s", type(e).__name__
            )

    async def _ensure_connected(self) -> Any:
        """Lazily establish (or reuse) the persistent WebSocket connection."""
        if self._closed:
            raise RuntimeError("Qwen WebSocket adapter is closed")
        if self._ws is not None:
            return self._ws
        kwargs: dict[str, Any] = {
            "open_timeout": min(self._timeout, _OPEN_TIMEOUT_CAP_S),
            "ping_interval": _PING_INTERVAL_S,
            "ping_timeout": _PING_TIMEOUT_S,
        }
        if self._api_key:
            # Optional: the current Kaggle server needs no auth; send a
            # standard bearer header only when a key is configured.
            kwargs["additional_headers"] = {"Authorization": f"Bearer {self._api_key}"}
        try:
            self._ws = await websockets.connect(self._ws_url, **kwargs)
        except Exception as e:
            self._ws = None
            logger.error(
                "[LLM] provider=qwen_ws connect_failed url=%s error_type=%s error=%s",
                self._ws_url,
                type(e).__name__,
                str(e)[:200],
            )
            raise RuntimeError(f"Failed to connect to Qwen WebSocket server: {e}") from e
        logger.info("[LLM] provider=qwen_ws connected url=%s", self._ws_url)
        return self._ws

    async def _receive_stream(
        self, ws: Any, request_id: str
    ) -> AsyncIterator[StreamChunk]:
        """Receive server frames for one request and translate them.

        Frames whose request_id does not match the active request are stale
        leftovers from an earlier (cancelled/completed) request and are
        discarded so they can never contaminate the current stream.
        """
        stale_discarded = 0
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._timeout)
            except TimeoutError as e:
                raise RuntimeError(
                    f"Qwen server sent no frame for {self._timeout:g}s "
                    f"(request_id={request_id})"
                ) from e
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "[LLM] provider=qwen_ws non_json_frame request_id=%s raw_type=%s",
                    request_id,
                    type(raw).__name__,
                )
                continue
            if not isinstance(message, dict):
                continue

            msg_type = message.get("type")
            frame_id = message.get("request_id")
            if frame_id is not None and frame_id != request_id:
                stale_discarded += 1
                logger.debug(
                    "[LLM] provider=qwen_ws stale_frame_discarded type=%s request_id=%s",
                    msg_type,
                    frame_id,
                )
                if stale_discarded > _MAX_STALE_FRAMES:
                    raise RuntimeError(
                        f"Qwen server flooded stale frames (> {_MAX_STALE_FRAMES})"
                    )
                continue

            if msg_type == "token":
                text = message.get("text") or ""
                if text:
                    yield StreamChunk(content=text)
            elif msg_type == "done":
                logger.info("[LLM] provider=qwen_ws done request_id=%s", request_id)
                yield StreamChunk(finish_reason="stop")
                return
            elif msg_type == "cancelled":
                # Server confirms remote cancellation for THIS request: the
                # stream ends without a finish_reason chunk.
                logger.info(
                    "[LLM] provider=qwen_ws server_cancelled request_id=%s tokens=%s",
                    request_id,
                    message.get("tokens"),
                )
                return
            elif msg_type == "error":
                detail = (
                    message.get("message") or message.get("error") or json.dumps(message)
                )
                raise RuntimeError(f"Qwen server error: {detail}")
            elif msg_type == "start":
                logger.debug("[LLM] provider=qwen_ws start request_id=%s", request_id)
            elif msg_type == "timing":
                self._log_timing(message, request_id)
            else:
                logger.debug(
                    "[LLM] provider=qwen_ws unhandled_frame type=%s request_id=%s",
                    msg_type,
                    request_id,
                )

    @staticmethod
    def _log_timing(message: dict[str, Any], request_id: str) -> None:
        """Log server-side timing events (no StreamChunk is produced)."""
        if message.get("event") == "first_token":
            logger.info(
                "[LLM] provider=qwen_ws server_ttft request_id=%s ttft_ms=%.0f",
                request_id,
                QwenWebSocketAdapter._to_ms(message.get("ttft")),
            )
        elif message.get("event") == "complete":
            logger.info(
                "[LLM] provider=qwen_ws server_complete request_id=%s "
                "ttft_ms=%.0f generation_ms=%.0f total_ms=%.0f chunks=%s",
                request_id,
                QwenWebSocketAdapter._to_ms(message.get("ttft")),
                QwenWebSocketAdapter._to_ms(message.get("generation_time")),
                QwenWebSocketAdapter._to_ms(message.get("total_time")),
                message.get("chunks"),
            )

    @staticmethod
    def _to_ms(raw: Any) -> float:
        """Best-effort conversion of a server-side seconds value to ms."""
        try:
            return float(raw) * 1000.0
        except (TypeError, ValueError):
            return 0.0

    async def _cancel(self, request_id: str) -> None:
        """Best-effort remote cancellation; bounded and never raises.

        Sends {"type": "cancel"} and waits (bounded) for the matching
        cancel_ack -> cancelled handshake, discarding stale frames. Tolerates
        an already-closed socket. asyncio.CancelledError is never swallowed.
        """
        ws = self._ws
        if ws is None:
            return
        try:
            async with asyncio.timeout(_CANCEL_DRAIN_TIMEOUT_S):
                await ws.send(json.dumps({"type": "cancel", "request_id": request_id}))
                logger.info(
                    "[LLM] provider=qwen_ws cancel_sent request_id=%s", request_id
                )
                while True:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=_CANCEL_DRAIN_TIMEOUT_S
                    )
                    try:
                        message = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(message, dict):
                        continue
                    frame_id = message.get("request_id")
                    if frame_id is not None and frame_id != request_id:
                        continue
                    msg_type = message.get("type")
                    if msg_type == "cancel_ack":
                        logger.debug(
                            "[LLM] provider=qwen_ws cancel_ack request_id=%s",
                            request_id,
                        )
                    elif msg_type == "cancelled":
                        logger.info(
                            "[LLM] provider=qwen_ws cancelled request_id=%s",
                            request_id,
                        )
                        return
                    elif msg_type == "done":
                        # Generation finished before the cancel landed.
                        return
        except TimeoutError:
            logger.warning(
                "[LLM] provider=qwen_ws cancel_drain_timeout request_id=%s",
                request_id,
            )
        except Exception as e:
            logger.warning(
                "[LLM] provider=qwen_ws cancel_failed request_id=%s error_type=%s",
                request_id,
                type(e).__name__,
            )

    @staticmethod
    def _serialize_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
        """Convert LLMMessage objects to the Kaggle JSON format.

        The server is stateless and expects plain role/content pairs;
        ordering is preserved. Tool-call metadata has no protocol mapping
        and is omitted (the Qwen server does not support tool calls).
        """
        return [
            {"role": message.role, "content": message.content or ""}
            for message in messages
        ]
