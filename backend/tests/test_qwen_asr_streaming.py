"""Focused tests for the Qwen ASR streaming STT provider (STT_PROVIDER=qwen).

Exactly the four mandatory checks:
    1. Provider lifecycle: connect → start → audio → partial → final → close
    2. Realtime integration: a Qwen final starts the existing LLM turn; a
       partial transcript alone does NOT
    3. Cleanup: cancellation/disconnect closes the provider, no leaked task
    4. Default guard: with no override, the existing Deepgram session is
       still selected

All Qwen interaction goes through an in-process loopback WebSocket server
speaking the Kaggle ASR wire format — no external network calls, no Kaggle URL.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from app.api.voice_realtime import _handle_realtime_session
from app.core.config import settings
from app.providers.stt.deepgram_streaming import DeepgramStreamingSession
from app.providers.stt.qwen_streaming import QwenKaggleStreamingSession
from app.providers.stt.streaming import StreamConfig, open_streaming_session
from app.providers.types import TTSResult

# Kaggle ASR server wire shapes (per the working server contract).
PARTIAL_RESULTS = {
    "type": "Results",
    "is_final": False,
    "speech_final": False,
    "text": "hello there",
    "confidence": 0.9,
    "language": "English",
}

FINAL_RESULTS = {
    "type": "Results",
    "is_final": True,
    "speech_final": True,
    "text": "hello there",
    "confidence": 0.9,
    "language": "English",
}

UTTERANCE_END = {"type": "UtteranceEnd", "last_word_end": 1.234}

_START_WITH_LLM: dict[str, Any] = {
    "type": "start",
    "sample_rate": 16000,
    "channels": 1,
    "encoding": "linear16",
    "language": "en",
    "llm_provider": "openrouter",
    "llm_model": "test-model",
}


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01
) -> None:
    """Poll ``predicate`` until it is true — observes async progress."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(interval)


class MockKaggleASRServer:
    """In-process Kaggle-protocol ASR server recording what the provider sends."""

    def __init__(self, *, auto_final: bool = True, silent: bool = False) -> None:
        self.requests: list[dict] = []
        self.audio_chunks: list[bytes] = []
        self.close_streams = 0
        self.connections = 0
        self.connection_closed = asyncio.Event()
        # Test-controlled gate: with auto_final=False the final/UtteranceEnd
        # are only sent after the test sets this event.
        self.final_trigger = asyncio.Event()
        self._auto_final = auto_final
        self._silent = silent
        self._partial_sent = False

    async def handler(self, ws: Any) -> None:
        self.connections += 1
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    self.audio_chunks.append(raw)
                    await self._maybe_respond(ws)
                    continue
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                self.requests.append(message)
                if message.get("type") == "CloseStream":
                    self.close_streams += 1
        except ConnectionClosed:
            pass
        finally:
            self.connection_closed.set()

    async def _maybe_respond(self, ws: Any) -> None:
        if self._silent or self._partial_sent:
            return
        self._partial_sent = True
        try:
            await ws.send(json.dumps(PARTIAL_RESULTS))
            if not self._auto_final:
                await self.final_trigger.wait()
            await ws.send(json.dumps(FINAL_RESULTS))
            await ws.send(json.dumps(UTTERANCE_END))
        except Exception:
            # Client closed mid-script (failure/test-teardown paths).
            return

    def start_messages(self) -> list[dict]:
        return [r for r in self.requests if r.get("type") == "start"]


@asynccontextmanager
async def _loopback_server(
    *, auto_final: bool = True, silent: bool = False
) -> AsyncIterator[MockKaggleASRServer]:
    server_obj = MockKaggleASRServer(auto_final=auto_final, silent=silent)
    server = await serve(server_obj.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server_obj.url = f"ws://127.0.0.1:{port}/asr"
    try:
        yield server_obj
    finally:
        # Unblock any handler waiting on the test gate before tearing down.
        server_obj.final_trigger.set()
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Mandatory 1 — provider lifecycle
# ---------------------------------------------------------------------------


async def test_qwen_provider_lifecycle(monkeypatch) -> None:
    """connect → send start → send audio → partial → final → close."""
    async with _loopback_server() as server:
        monkeypatch.setattr(settings, "qwen_asr_ws_url", server.url)
        session = QwenKaggleStreamingSession(StreamConfig())
        try:
            await session.start()
            await _wait_until(lambda: len(server.start_messages()) == 1)

            start_msg = server.start_messages()[0]
            assert start_msg == {
                "type": "start",
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
                "endpointing_ms": 300,
                "utterance_end_ms": 1000,
                "vad_threshold": 0.012,
                "inference_step_ms": 500,
                "language": "en",
            }

            chunk = b"\x10\x00" * 320  # 20 ms PCM16 mono @ 16 kHz
            await session.send_audio(chunk)
            await _wait_until(lambda: len(server.audio_chunks) == 1)
            assert server.audio_chunks[0] == chunk

            partial = await asyncio.wait_for(session.receive(), timeout=5)
            assert partial is not None
            assert partial.type == "partial"
            assert partial.text == "hello there"
            assert abs(partial.confidence - 0.9) < 1e-6
            assert session._first_partial_at is not None  # qwen_first_partial_ms

            final = await asyncio.wait_for(session.receive(), timeout=5)
            assert final is not None
            assert final.type == "final"
            assert final.text == "hello there"
            assert final.metadata["speech_final"] is True
            assert session._first_final_at is not None  # qwen_final_ms

            utterance_end = await asyncio.wait_for(session.receive(), timeout=5)
            assert utterance_end is not None
            assert utterance_end.type == "utterance_end"
            assert utterance_end.metadata["last_word_end"] == 1.234

            await session.finish()
            await _wait_until(lambda: server.close_streams == 1)

            await session.close()
            await session.close()  # idempotent
            assert session._ws is None
            await _wait_until(server.connection_closed.is_set)
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Mandatory 2 — realtime integration (real provider + real gateway pump)
# ---------------------------------------------------------------------------


class _BrowserDouble:
    """Minimal ASGI socket double driving ``_handle_realtime_session`` directly."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []

    def queue_text(self, payload: dict[str, Any]) -> None:
        self.inbound.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(payload)}
        )

    def queue_bytes(self, data: bytes) -> None:
        self.inbound.put_nowait({"type": "websocket.receive", "bytes": data})

    async def receive(self) -> dict[str, Any]:
        return await self.inbound.get()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def event_types(self) -> list[str]:
        return [e.get("type", "?") for e in self.sent]


def _patch_gateway_deps() -> tuple[dict[str, Any], AsyncMock]:
    """Patch every dependency the realtime agent worker needs.

    Mirrors tests/test_voice_realtime.py ``_patch_realtime_deps``; only the
    LLM stream mock is observed by the caller.
    """
    mock_db_session = MagicMock()
    mock_db_session.id = "test-qwen-voice-session"
    mock_db_session.llm_provider = "openrouter"
    mock_db_session.llm_model = "test-model"
    mock_db_session.message_count = 0
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_db_session
    )

    mock_llm = MagicMock()
    mock_llm.close = AsyncMock()

    mock_tts = MagicMock()
    mock_tts.provider_name = "elevenlabs"
    mock_tts.synthesize = AsyncMock(
        return_value=TTSResult(audio_data=b"fake-mp3", content_type="audio/mpeg")
    )
    mock_tts.close = AsyncMock()

    mock_stream = AsyncMock(
        return_value={
            "response": "Hello! How can I help?",
            "tool_calls": [],
            "usage": {},
            "iterations": 1,
            "streamed": True,
            "first_token_ms": 150.0,
            "streamed_tokens": 5,
            "sentences": ["Hello! How can I help?"],
        }
    )

    patchers = {
        "session_local": patch(
            "app.api.voice_realtime.SessionLocal",
            MagicMock(return_value=mock_db),
        ),
        "create_session": patch(
            "app.api.voice_realtime.create_realtime_session",
            return_value=mock_db_session,
        ),
        "get_llm": patch(
            "app.api.voice_realtime.get_llm_provider", return_value=mock_llm
        ),
        "get_tts": patch(
            "app.api.voice_realtime.get_tts_provider", return_value=mock_tts
        ),
        "stream": patch(
            "app.api.voice_realtime._stream_llm_response", mock_stream
        ),
    }
    for p in patchers.values():
        p.start()
    return patchers, mock_stream


def _stop_patchers(patchers: dict[str, Any]) -> None:
    for p in patchers.values():
        p.stop()


async def test_qwen_final_starts_llm_turn_partial_does_not(monkeypatch) -> None:
    """Qwen final → existing realtime STT handling → LLM turn starts.

    The Qwen session is created through the REAL factory (STT_PROVIDER=qwen)
    against the loopback Kaggle server; only LLM/TTS/DB are mocked.
    """
    async with _loopback_server(auto_final=False) as server:
        monkeypatch.setattr(settings, "qwen_asr_ws_url", server.url)
        monkeypatch.setattr(settings, "stt_provider", "qwen")
        patchers, mock_stream = _patch_gateway_deps()
        ws = _BrowserDouble()
        ws.queue_text(_START_WITH_LLM)
        ws.queue_bytes(b"\x10\x00" * 320)
        handler = asyncio.create_task(_handle_realtime_session(ws))
        try:
            # The Qwen partial reaches the browser through the existing pump…
            await _wait_until(lambda: "transcript_partial" in ws.event_types())
            # …and must NOT start the LLM: only the final/speech_final (or
            # UtteranceEnd fallback) contract releases a turn. The final is
            # gated server-side, so no release can even be scheduled yet.
            await asyncio.sleep(0.15)
            assert mock_stream.await_count == 0
            assert "agent_processing" not in ws.event_types()

            # Release the Qwen final (is_final + speech_final) → existing
            # speech_final early release → agent worker → LLM turn.
            server.final_trigger.set()
            await _wait_until(
                lambda: "agent_response" in ws.event_types(), timeout=10.0
            )
            assert mock_stream.await_count == 1
            assert mock_stream.call_args.kwargs["transcript"] == "hello there"

            types = ws.event_types()
            assert types.index("transcript_partial") < types.index("agent_processing")
            assert types.index("transcript_final") < types.index("agent_processing")

            ws.queue_text({"type": "stop"})
            await asyncio.wait_for(handler, timeout=10.0)
        finally:
            if not handler.done():
                handler.cancel()
            _stop_patchers(patchers)


# ---------------------------------------------------------------------------
# Mandatory 3 — cleanup (cancellation / disconnect, no leaked task)
# ---------------------------------------------------------------------------


async def test_qwen_cancellation_and_disconnect_close_provider(monkeypatch) -> None:
    """Cancellation/disconnect → provider closes → no leaked task."""
    async with _loopback_server(silent=True) as server:
        monkeypatch.setattr(settings, "qwen_asr_ws_url", server.url)
        session = QwenKaggleStreamingSession(StreamConfig())
        await session.start()
        await _wait_until(lambda: server.connections == 1)

        # Task inventory with the connection live: nothing the provider does
        # from here on may leave a NEW pending task behind.
        before = set(asyncio.all_tasks())

        # (a) Cancellation while a receive() is in flight — exactly how the
        # gateway pump is cancelled on session shutdown.
        consumer = asyncio.create_task(session.receive())
        await asyncio.sleep(0.05)  # let it block in recv()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        assert consumer.cancelled()

        # (b) Disconnect while a receive() is in flight: close() tears the
        # connection down and the pending receive ENDS (None) — never hangs.
        pending = asyncio.create_task(session.receive())
        await asyncio.sleep(0.05)
        await session.close()
        assert session._ws is None
        result = await asyncio.wait_for(pending, timeout=5.0)
        assert result is None

        await session.close()  # idempotent
        await _wait_until(server.connection_closed.is_set)

        await asyncio.sleep(0.1)  # settle any finishing internal task
        current = asyncio.current_task()
        leaked = [
            t
            for t in asyncio.all_tasks()
            if t not in before and t is not current and not t.done()
        ]
        assert leaked == [], f"leaked tasks: {leaked}"


# ---------------------------------------------------------------------------
# Mandatory 4 — existing Deepgram default guard
# ---------------------------------------------------------------------------


def test_default_selection_remains_deepgram(monkeypatch) -> None:
    """Default config (no STT_PROVIDER override) still selects Deepgram."""
    monkeypatch.setattr(settings, "stt_provider", "")
    monkeypatch.setattr(settings, "default_stt_provider", "deepgram")
    monkeypatch.setattr(settings, "deepgram_api_key", "test-deepgram-key")
    session = open_streaming_session(StreamConfig())  # no I/O at construction
    assert isinstance(session, DeepgramStreamingSession)
    assert session.provider_name == "deepgram"
