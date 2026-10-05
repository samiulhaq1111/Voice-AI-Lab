"""Protocol tests for the Qwen WebSocket adapter (qwen_ws).

Verifies QwenWebSocketAdapter against an in-process mock WebSocket server
(real loopback socket on 127.0.0.1, ephemeral port) speaking the Kaggle
Qwen wire format. No external network calls, no Kaggle URL.

Covers: construction (no I/O), outgoing chat payload, token streaming
order, finish_reason, server error mapping, stale request_id filtering,
cancellation (cancel/cancel_ack/cancelled), persistent connection reuse,
idempotent close(), and chat() concatenation.
"""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import serve

from app.providers.llm.interface import LLMInterface
from app.providers.llm.qwen_ws import QwenWebSocketAdapter
from app.providers.types import LLMMessage, LLMResponse

MESSAGES = [
    LLMMessage(role="system", content="You are helpful."),
    LLMMessage(role="user", content="Hello"),
]

EXPECTED_SERIALIZED_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

# ChatScript: async callable (ws connection, chat request) -> None
ChatScript = Callable[[Any, dict], Awaitable[None]]


async def _send(ws: Any, payload: dict) -> None:
    await ws.send(json.dumps(payload))


def standard_chat_script(
    tokens: tuple[str, ...] = ("Hello", " ", "there"),
) -> ChatScript:
    """Full happy-path responder: start, ttft, tokens, complete, done."""

    async def script(ws: Any, request: dict) -> None:
        rid = request["request_id"]
        await _send(ws, {"type": "start", "request_id": rid})
        await _send(
            ws,
            {"type": "timing", "event": "first_token", "request_id": rid, "ttft": 0.57},
        )
        for token in tokens:
            await _send(ws, {"type": "token", "request_id": rid, "text": token})
        await _send(
            ws,
            {
                "type": "timing",
                "event": "complete",
                "request_id": rid,
                "ttft": 0.57,
                "generation_time": 0.08,
                "total_time": 0.65,
                "chunks": len(tokens),
            },
        )
        await _send(ws, {"type": "done", "request_id": rid})

    return script


class MockQwenServer:
    """In-process Kaggle-protocol server driven by a per-test chat script."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.connections = 0
        self.chat_script: ChatScript | None = None
        self.url = ""

    async def handler(self, ws: Any) -> None:
        self.connections += 1
        async for raw in ws:
            message = json.loads(raw)
            self.requests.append(message)
            if message.get("type") == "chat" and self.chat_script is not None:
                await self.chat_script(ws, message)
            elif message.get("type") == "cancel":
                rid = message.get("request_id")
                await _send(ws, {"type": "cancel_ack", "request_id": rid})
                await _send(
                    ws,
                    {
                        "type": "cancelled",
                        "request_id": rid,
                        "tokens": 2,
                        "total_time": 0.05,
                    },
                )

    def chat_requests(self) -> list[dict]:
        return [r for r in self.requests if r.get("type") == "chat"]

    def cancel_requests(self) -> list[dict]:
        return [r for r in self.requests if r.get("type") == "cancel"]


@pytest.fixture
async def mock_server() -> AsyncIterator[MockQwenServer]:
    server_obj = MockQwenServer()
    server = await serve(server_obj.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server_obj.url = f"ws://127.0.0.1:{port}/ws"
    server_obj.chat_script = standard_chat_script()
    yield server_obj
    server.close()
    await server.wait_closed()


def _adapter(url: str) -> QwenWebSocketAdapter:
    # Short timeout so a stalled protocol surfaces as a fast failure.
    return QwenWebSocketAdapter(ws_url=url, timeout=5.0)


async def _collect(adapter: QwenWebSocketAdapter) -> list[Any]:
    chunks = []
    async for chunk in adapter.stream_chat(MESSAGES, max_tokens=100):
        chunks.append(chunk)
    return chunks


class TestConstruction:
    """Requirement 1: instantiation with explicit local URL, no I/O."""

    @pytest.mark.asyncio
    async def test_construction_performs_no_io(self, mock_server: MockQwenServer) -> None:
        adapter = _adapter(mock_server.url)

        assert isinstance(adapter, LLMInterface)
        assert adapter.provider_name == "qwen_ws"
        # No socket opened and nothing sent before the first stream_chat call.
        assert adapter._ws is None
        assert mock_server.connections == 0
        assert mock_server.requests == []

        await adapter.close()


class TestChatProtocol:
    """Requirements 2-6: request payload, streaming, error, stale ids."""

    @pytest.mark.asyncio
    async def test_outgoing_chat_request_payload(
        self, mock_server: MockQwenServer
    ) -> None:
        adapter = _adapter(mock_server.url)
        await _collect(adapter)
        await adapter.close()

        assert len(mock_server.chat_requests()) == 1
        request = mock_server.chat_requests()[0]
        assert request["type"] == "chat"
        request_id = request["request_id"]
        assert isinstance(request_id, str) and len(request_id) == 32
        int(request_id, 16)  # uuid4().hex: valid lowercase hex
        assert request["messages"] == EXPECTED_SERIALIZED_MESSAGES
        assert request["temperature"] == 0.7
        assert request["max_tokens"] == 100
        assert set(request.keys()) == {
            "type",
            "request_id",
            "messages",
            "temperature",
            "max_tokens",
        }

    @pytest.mark.asyncio
    async def test_token_streaming_order_and_completion(
        self, mock_server: MockQwenServer
    ) -> None:
        adapter = _adapter(mock_server.url)
        chunks = await _collect(adapter)
        await adapter.close()

        # start/timing frames produce no chunks; exactly one chunk per token
        # plus one terminal finish_reason chunk (which carries content=None).
        assert [c.content for c in chunks[:-1]] == ["Hello", " ", "there"]
        assert all(c.finish_reason is None for c in chunks[:-1])
        assert chunks[-1].content is None
        assert chunks[-1].finish_reason == "stop"
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_server_error_raises_runtime_error(
        self, mock_server: MockQwenServer
    ) -> None:
        async def error_script(ws: Any, request: dict) -> None:
            rid = request["request_id"]
            await _send(ws, {"type": "start", "request_id": rid})
            await _send(
                ws, {"type": "error", "request_id": rid, "message": "kaboom"}
            )

        mock_server.chat_script = error_script
        adapter = _adapter(mock_server.url)

        with pytest.raises(RuntimeError, match="kaboom"):
            await _collect(adapter)
        await adapter.close()

    @pytest.mark.asyncio
    async def test_stale_request_ids_ignored(
        self, mock_server: MockQwenServer
    ) -> None:
        async def stale_frame_script(ws: Any, request: dict) -> None:
            rid = request["request_id"]
            # Leftover frame from an earlier request arrives first.
            await _send(
                ws,
                {
                    "type": "token",
                    "request_id": "stale-old-request",
                    "text": "IGNORED",
                },
            )
            await _send(ws, {"type": "start", "request_id": rid})
            await _send(ws, {"type": "token", "request_id": rid, "text": "Hi"})
            await _send(ws, {"type": "done", "request_id": rid})

        mock_server.chat_script = stale_frame_script
        adapter = _adapter(mock_server.url)
        chunks = await _collect(adapter)
        await adapter.close()

        contents = [c.content for c in chunks if c.content]
        assert contents == ["Hi"]
        assert "IGNORED" not in contents
        assert chunks[-1].finish_reason == "stop"


class TestCancellationAndLifecycle:
    """Requirements 7-10: cancel handshake, reuse, close, chat()."""

    @pytest.mark.asyncio
    async def test_cancellation_sends_cancel_and_handles_ack(
        self, mock_server: MockQwenServer
    ) -> None:
        async def one_token_script(ws: Any, request: dict) -> None:
            rid = request["request_id"]
            await _send(ws, {"type": "start", "request_id": rid})
            await _send(ws, {"type": "token", "request_id": rid, "text": "partial"})

        mock_server.chat_script = one_token_script
        adapter = _adapter(mock_server.url)

        agen = adapter.stream_chat(MESSAGES, max_tokens=100)
        collected = []
        async for chunk in agen:
            collected.append(chunk)
            break  # consumer stops mid-generation
        await agen.aclose()

        assert collected[0].content == "partial"
        # The adapter asked the server to stop the matching generation and
        # consumed the cancel_ack -> cancelled handshake (aclose returned).
        cancels = mock_server.cancel_requests()
        assert len(cancels) == 1
        assert cancels[0]["request_id"] == mock_server.chat_requests()[0]["request_id"]
        await adapter.close()

    @pytest.mark.asyncio
    async def test_persistent_connection_reused(
        self, mock_server: MockQwenServer
    ) -> None:
        adapter = _adapter(mock_server.url)

        await _collect(adapter)
        ws_after_first = adapter._ws
        await _collect(adapter)

        assert mock_server.connections == 1  # ONE socket for both turns
        assert adapter._ws is ws_after_first
        chat_requests = mock_server.chat_requests()
        assert len(chat_requests) == 2
        assert chat_requests[0]["request_id"] != chat_requests[1]["request_id"]
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, mock_server: MockQwenServer) -> None:
        adapter = _adapter(mock_server.url)
        await _collect(adapter)

        await adapter.close()
        assert adapter._ws is None
        await adapter.close()  # second close is a no-op, must not raise
        assert adapter._ws is None

    @pytest.mark.asyncio
    async def test_close_without_connection_is_safe(
        self, mock_server: MockQwenServer
    ) -> None:
        adapter = _adapter(mock_server.url)
        await adapter.close()
        await adapter.close()
        assert mock_server.connections == 0

    @pytest.mark.asyncio
    async def test_chat_concatenates_content(
        self, mock_server: MockQwenServer
    ) -> None:
        adapter = _adapter(mock_server.url)
        result = await adapter.chat(MESSAGES, max_tokens=100)
        await adapter.close()

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there"
        assert result.finish_reason == "stop"
        assert result.tool_calls is None
        assert result.usage == {}
        assert result.metadata == {}
