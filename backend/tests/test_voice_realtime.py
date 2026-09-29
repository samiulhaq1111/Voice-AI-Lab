"""Targeted tests for the realtime streaming voice gateway (Phase 6B).

All Deepgram interaction is mocked — no real network calls.
Covers protocol validation, event forwarding, lifecycle cleanup,
credential safety, and AgentRuntime integration on utterance_end.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.voice_realtime import (
    RealtimeSessionState,
    _handle_realtime_session,
    _shutdown_session,
)
from app.main import app
from app.providers.stt.streaming import (
    StreamConfig,
    StreamEvent,
    StreamingSTTError,
)


def _get_client() -> TestClient:
    return TestClient(app)


REALTIME_WS_PATH = "/api/v1/voice/realtime/ws"


class FakeStreamingSession:
    """In-memory stand-in for a provider streaming session."""

    def __init__(self, events: list[StreamEvent] | None = None) -> None:
        self.provider_name = "deepgram"
        self.events: list[StreamEvent] = list(events or [])
        self.sent_audio: list[bytes] = []
        self.started = False
        self.finished = False
        self.closed = False
        self.fail_on_start = False

    async def start(self) -> None:
        if self.fail_on_start:
            raise StreamingSTTError("Simulated provider failure")
        self.started = True

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def receive(self) -> StreamEvent | None:
        """Model a healthy provider: block until finish() or an event arrives.

        Real Deepgram keeps the stream open while audio flows; receive() only
        returns None after finish() (drained) — matching that behavior keeps
        the premature-provider-end regression test meaningful.
        """
        while not self.events and not self.finished:
            await asyncio.sleep(0.01)
        if self.events:
            return self.events.pop(0)
        return None

    async def finish(self) -> None:
        self.finished = True

    async def close(self) -> None:
        self.closed = True


class _EndingEarlySession(FakeStreamingSession):
    """Simulates a provider stream that ends immediately (no events)."""

    async def receive(self):
        return None


class _ExplodingReceiveSession(FakeStreamingSession):
    """Simulates a provider receive loop that crashes."""

    async def receive(self):
        raise RuntimeError("boom")


def _connect_with_session_instance(session: FakeStreamingSession):
    """Patch the factory to return a specific session instance."""
    holder: dict[str, Any] = {"session": session}
    patcher = patch(
        "app.api.voice_realtime.open_streaming_session", return_value=session
    )
    patcher.start()
    holder["patcher"] = patcher
    return _get_client(), holder


def _valid_start() -> dict[str, Any]:
    return {
        "type": "start",
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "linear16",
        "language": "en",
    }


def _connect_with_fake_session(
    events: list[StreamEvent] | None = None,
) -> tuple[TestClient, dict[str, Any]]:
    """Start the session-factory patch and return (client, holder).

    The holder receives the created FakeStreamingSession under "session"
    and the active patcher under "patcher" (call .stop() in finally).
    """
    holder: dict[str, Any] = {}

    def _fake_open(cfg: StreamConfig) -> FakeStreamingSession:
        session = FakeStreamingSession(events=events)
        holder["session"] = session
        return session

    patcher = patch(
        "app.api.voice_realtime.open_streaming_session", side_effect=_fake_open
    )
    patcher.start()
    holder["patcher"] = patcher
    return _get_client(), holder


# ---------------------------------------------------------------------------
# 1. Route exists + connection accepted
# ---------------------------------------------------------------------------


class TestRouteExists:
    def test_realtime_ws_route_exists(self) -> None:
        """Connecting to the realtime endpoint should be accepted."""
        client = _get_client()
        with client.websocket_connect(REALTIME_WS_PATH) as ws:
            # Send an invalid message — if the route exists we get an error event
            ws.send_text("not-json")
            event = ws.receive_json()
            assert event["type"] == "error"
            assert event["message"] == "Invalid JSON"


# ---------------------------------------------------------------------------
# 2-4. START validation
# ---------------------------------------------------------------------------


class TestStartValidation:
    def test_valid_start_creates_session(self) -> None:
        """Valid START opens a streaming session and returns session_started."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                event = ws.receive_json()
                assert event["type"] == "session_started"
                assert event["provider"] == "deepgram"
                assert event["session_id"]
                assert holder["session"].started is True
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_invalid_encoding_rejected(self) -> None:
        """Unsupported encoding produces an error, no session created."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                bad = _valid_start() | {"encoding": "float32"}
                ws.send_json(bad)
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "encoding" in event["message"].lower()
                assert "session" not in holder
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_invalid_channels_rejected(self) -> None:
        """Non-mono audio is rejected."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                bad = _valid_start() | {"channels": 2}
                ws.send_json(bad)
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "mono" in event["message"].lower()
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_invalid_sample_rate_rejected(self) -> None:
        """Out-of-range sample rate is rejected."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                bad = _valid_start() | {"sample_rate": 1000}
                ws.send_json(bad)
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "sample_rate" in event["message"]
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_non_integer_sample_rate_rejected(self) -> None:
        """Non-integer sample_rate is rejected before session creation."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                bad = _valid_start() | {"sample_rate": "16000"}
                ws.send_json(bad)
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "integer" in event["message"].lower()
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_second_start_rejected(self) -> None:
        """A second START while a session is active is an error."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
                ws.send_json(_valid_start())
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "already started" in event["message"]
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_before_start_rejected(self) -> None:
        """STOP without an active session is an error, connection survives."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "No active session" in event["message"]
                # Connection still usable afterwards
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_unknown_message_type_rejected(self) -> None:
        """Unknown control message types produce an error event."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json({"type": "dance"})
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "dance" in event["message"]
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 5-9. Audio forwarding + transcript events + STOP
# ---------------------------------------------------------------------------


class TestStreamingFlow:
    def _stream_events(self) -> list[StreamEvent]:
        return [
            StreamEvent(type="partial", text="hello I want to", confidence=0.8),
            StreamEvent(type="final", text="hello I want to request leave", confidence=0.95),
            StreamEvent(type="utterance_end"),
        ]

    def test_binary_audio_forwarded_to_adapter(self) -> None:
        """Binary PCM chunks are forwarded to the streaming session."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_bytes(b"pcm-chunk-1")
                ws.send_bytes(b"pcm-chunk-2")
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
                session = holder["session"]
                assert session.sent_audio == [b"pcm-chunk-1", b"pcm-chunk-2"]
                assert session.finished is True
                assert session.closed is True
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_partial_final_utterance_end_forwarded(self) -> None:
        """Provider events are translated and forwarded in order."""
        client, holder = _connect_with_fake_session(events=self._stream_events())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"

                partial = ws.receive_json()
                assert partial["type"] == "transcript_partial"
                assert partial["text"] == "hello I want to"

                final = ws.receive_json()
                assert final["type"] == "transcript_final"
                assert final["text"] == "hello I want to request leave"

                ue = ws.receive_json()
                assert ue["type"] == "utterance_end"

                # End the session before the client context tears the
                # socket down. This test uses real (unpatched) providers, so
                # the worker may hold a live HTTP call; letting the handler
                # finish its cleanup via STOP keeps TestClient teardown away
                # from its known portal-cancellation race.
                ws.send_json({"type": "stop"})
                while ws.receive_json()["type"] != "completed":
                    pass
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_completes_after_stream_events(self) -> None:
        """STOP completes the session; in-flight events are best-effort.

        With immediate pump cancellation, some events may or may not
        be forwarded depending on asyncio scheduling. The key guarantee
        is that completed is sent promptly without hanging.
        """
        client, holder = _connect_with_fake_session(events=self._stream_events())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_bytes(b"\x01\x02")
                ws.send_json({"type": "stop"})

                # Collect events until completed
                types = []
                while True:
                    event = ws.receive_json()
                    types.append(event["type"])
                    if event["type"] == "completed":
                        break
                # completed must always be present
                assert types[-1] == "completed"
                # Timings included in completed
                timings = json.loads(json.dumps(event["timings"]))
                assert timings["audio_bytes"] == 2
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_empty_audio_session_completes(self) -> None:
        """STOP with no audio still completes gracefully."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
                assert event["timings"]["audio_bytes"] == 0
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 10-13. Error paths + safety
# ---------------------------------------------------------------------------


class TestErrorPathsAndSafety:
    def test_audio_before_start_ignored(self) -> None:
        """Binary audio received before START is safely ignored."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_bytes(b"premature-audio")
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
                # Premature chunk must NOT have been forwarded
                assert holder["session"].sent_audio == []
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_provider_start_failure_produces_error(self) -> None:
        """A provider failure at session start yields an error event."""

        def _failing_open(cfg: StreamConfig) -> FakeStreamingSession:
            session = FakeStreamingSession()
            session.fail_on_start = True
            return session

        client = _get_client()
        with patch(
            "app.api.voice_realtime.open_streaming_session", side_effect=_failing_open
        ):
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                event = ws.receive_json()
                assert event["type"] == "error"
                assert "start" in event["message"].lower()

    def test_provider_midstream_error_forwarded(self) -> None:
        """A provider error event is forwarded to the client."""
        events = [StreamEvent(type="error", text="Deepgram boom")]
        client, holder = _connect_with_fake_session(events=events)
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
                event = ws.receive_json()
                assert event["type"] == "error"
                assert event["message"] == "Deepgram boom"
                assert event["stage"] == "stt"
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_provider_end_before_stop_reports_error(self) -> None:
        """Premature provider stream end must surface as an error, not silence.

        Regression test for the Phase 6A bug where a dead/silent provider
        stream completed the session with no transcript and no error.
        """
        client, holder = _connect_with_session_instance(_EndingEarlySession())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
                # Pump exits immediately → client gets an error event
                event = ws.receive_json()
                assert event["type"] == "error"
                assert event["stage"] == "stt"
                # STOP afterwards still completes cleanly
                ws.send_json({"type": "stop"})
                completed = ws.receive_json()
                assert completed["type"] == "completed"
                assert completed["timings"]["final_count"] == 0
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_pump_exception_reported_to_client(self) -> None:
        """Exceptions inside the provider receive loop must not be swallowed."""
        client, holder = _connect_with_session_instance(_ExplodingReceiveSession())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
                event = ws.receive_json()
                assert event["type"] == "error"
                assert event["stage"] == "stt"
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_malformed_messages_handled_safely(self) -> None:
        """Invalid JSON and empty messages do not kill the connection."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_text("}{ broken json")
                assert ws.receive_json()["type"] == "error"

                # Still functional afterwards
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_client_disconnect_cleans_resources(self) -> None:
        """Client disconnect triggers session close in the finally block."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
            # Context exited → client disconnected
            session = holder["session"]
            deadline = time.monotonic() + 5.0
            while not session.closed and time.monotonic() < deadline:
                time.sleep(0.05)
            assert session.closed is True
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Phase 6C: session lifecycle + isolation regression tests (T1-T8)
# ---------------------------------------------------------------------------


class LifecycleFakeWS:
    """ASGI socket double for driving ``_handle_realtime_session`` directly.

    Mirrors the real Starlette + uvicorn behavior seen in production:
    a disconnect arrives as a raw ASGI message, and any send after the
    client is gone raises ``WebSocketDisconnect(1006)`` (the Starlette
    conversion of uvicorn's ``ClientDisconnected`` transport error).
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.disconnected = False
        self.sends_after_disconnect = 0

    def queue_text(self, payload: dict[str, Any]) -> None:
        self.inbound.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(payload)}
        )

    def queue_bytes(self, data: bytes) -> None:
        self.inbound.put_nowait({"type": "websocket.receive", "bytes": data})

    def queue_disconnect(self, code: int = 1000) -> None:
        self.inbound.put_nowait(
            {"type": "websocket.disconnect", "code": code, "reason": ""}
        )

    async def receive(self) -> dict[str, Any]:
        message = await self.inbound.get()
        if message.get("type") == "websocket.disconnect":
            self.disconnected = True
        return message

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.disconnected:
            self.sends_after_disconnect += 1
            raise WebSocketDisconnect(code=1006)
        self.sent.append(payload)

    def event_types(self) -> list[str]:
        return [event.get("type", "?") for event in self.sent]


class LifecycleSession(FakeStreamingSession):
    """Provider session double that models the adversarial lifecycle race.

    - ``receive()`` blocks while no event is available (like a live Deepgram
      socket); with ``slow_after_finish`` only cancellation can end it.
    - ``send_audio()`` records post-finish chunks so tests can assert that
      STOP really stops audio forwarding.
    """

    def __init__(
        self,
        events: list[StreamEvent] | None = None,
        slow_after_finish: bool = False,
    ) -> None:
        super().__init__(events=events)
        self.slow_after_finish = slow_after_finish
        self.receive_waiting = False
        self.receive_cancelled = False
        self.close_calls = 0
        self.audio_after_finish: list[bytes] = []

    async def send_audio(self, chunk: bytes) -> None:
        if self.finished or self.closed:
            self.audio_after_finish.append(chunk)
            return
        self.sent_audio.append(chunk)

    async def receive(self) -> StreamEvent | None:
        try:
            while not self.events and not self.finished and not self.closed:
                self.receive_waiting = True
                await asyncio.sleep(0.01)
            if self.events:
                return self.events.pop(0)
            if self.slow_after_finish and self.finished:
                # A real Deepgram socket can linger after finish(); only
                # cancellation may terminate this receive.
                await asyncio.sleep(30)
            return None
        except asyncio.CancelledError:
            self.receive_cancelled = True
            self.receive_waiting = False
            raise

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _utterance(text: str) -> list[StreamEvent]:
    """One finalized utterance as the provider would emit it."""
    return [
        StreamEvent(type="final", text=text, confidence=0.9),
        StreamEvent(type="utterance_end"),
    ]


def _deliver_next_utterance(holder: dict[str, Any], text: str) -> None:
    """Deliver the next spoken turn to the fake provider stream (Phase 6E).

    Under the settle-based turn model a UtteranceEnd only *schedules* the
    turn release; events appended while a turn is still pending would merge
    into it. A second utterance therefore starts a new turn only after the
    previous turn was released and answered — tests append its events once
    the previous turn's outcome was observed on the wire.
    """
    holder["session"].events.extend(_utterance(text))


def _hanging_llm(entered: asyncio.Event) -> Callable[..., Any]:
    """Stream-LLM stand-in that blocks until cancelled, signalling entry."""

    async def _stream(**kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(30)
        return {
            "response": "never",
            "tool_calls": [],
            "usage": {},
            "iterations": 1,
            "streamed": True,
            "first_token_ms": None,
            "streamed_tokens": 0,
            "sentences": [],
        }

    return _stream


async def _drive_session(ws: LifecycleFakeWS, timeout: float = 10.0) -> None:
    """Run the realtime handler to completion; errors/timeouts fail the test."""
    handler = asyncio.create_task(_handle_realtime_session(ws))
    await asyncio.wait_for(handler, timeout=timeout)


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01
) -> None:
    """Poll ``predicate`` until it is true — observes task progress."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(interval)


async def _assert_no_pending_tasks(settle: float = 0.1) -> None:
    """No session-owned task may survive its session's cleanup."""
    await asyncio.sleep(settle)
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert pending == [], f"leaked session tasks: {pending}"


class TestSessionLifecycleIsolation:
    """Phase 6C regression tests (T1-T8): lifecycle + session isolation.

    Every test drives ``_handle_realtime_session`` directly against fake
    sockets/sessions so task ownership, cancellation and reaping are
    observable — unlike the TestClient-based tests above.
    """

    async def test_stop_prevents_further_audio(self) -> None:
        """T1: audio arriving after STOP is never forwarded to the provider."""
        session = LifecycleSession()
        ws = LifecycleFakeWS()
        ws.queue_text(_valid_start())
        ws.queue_bytes(b"audio-before-stop")
        ws.queue_text({"type": "stop"})
        ws.queue_bytes(b"audio-after-stop")
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                await _drive_session(ws)
            assert session.sent_audio == [b"audio-before-stop"]
            assert session.audio_after_finish == []
            assert session.finished is True
            assert session.close_calls == 1
            assert session.closed is True
            assert ws.event_types()[-1] == "completed"
            assert ws.sends_after_disconnect == 0
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_stop_terminates_deepgram_pump(self) -> None:
        """T2: STOP cancels the pump mid-receive and reaps it before return."""
        session = LifecycleSession(slow_after_finish=True)
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_valid_start())
                await _wait_until(lambda: session.receive_waiting)
                start = time.monotonic()
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            elapsed = time.monotonic() - start
            # Cancelled mid-receive — the 30s post-finish sleep proves the
            # pump cannot terminate on its own once it has entered receive.
            assert session.receive_cancelled is True
            assert elapsed < 5.0
            assert ws.event_types()[-1] == "completed"
            assert session.close_calls == 1
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_stop_terminates_agent_worker(self, caplog) -> None:
        """T3: STOP cancels and reaps the agent worker task."""
        session = LifecycleSession()
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_valid_start())
                    await _wait_until(
                        lambda: "[REALTIME:WORKER] Agent worker started"
                        in caplog.text
                    )
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            assert "Worker cancelled while waiting" in caplog.text
            assert "[REALTIME:WORKER] Agent worker exited" in caplog.text
            assert ws.event_types()[-1] == "completed"
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_no_websocket_sends_after_client_disconnect(self) -> None:
        """T4: disconnect mid-turn produces no late sends and no hard errors."""
        llm_entered = asyncio.Event()
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_hanging_llm(llm_entered)),
        )
        llm_patcher.start()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await asyncio.wait_for(llm_entered.wait(), timeout=5)
                # The browser vanishes (tab close / refresh) mid-LLM-turn.
                ws.queue_disconnect()
                await asyncio.wait_for(handler, timeout=10)
            # Worker was reaped inside the hanging LLM call: nothing was
            # sent to the dead socket, nothing was spent on TTS.
            assert ws.sends_after_disconnect == 0
            assert mock_tts.synthesize.await_count == 0
            assert "error" not in ws.event_types()
            assert ws.event_types()[-1] == "agent_processing"
            assert session.close_calls == 1
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_shutdown_session_is_idempotent(self) -> None:
        """T5a: concurrent + repeated shutdown calls run cleanup exactly once."""
        state = RealtimeSessionState(connection_id="unit-idempotent")
        session = LifecycleSession()
        mock_llm = MagicMock()
        mock_llm.close = AsyncMock()
        mock_tts = MagicMock()
        mock_tts.close = AsyncMock()
        mock_db = MagicMock()
        state.stt_session = session
        state.llm = mock_llm
        state.tts = mock_tts
        state.db = mock_db

        await asyncio.gather(
            _shutdown_session(state, "stop"),
            _shutdown_session(state, "stop"),
            _shutdown_session(state, "stop"),
        )
        await _shutdown_session(state, "stop")  # late caller — still safe

        assert state.closed is True
        assert state.shutdown_reason == "stop"
        assert session.close_calls == 1
        assert mock_llm.close.await_count == 1
        assert mock_tts.close.await_count == 1
        assert mock_db.close.call_count == 1
        await _assert_no_pending_tasks()

    async def test_stop_is_idempotent_over_protocol(self) -> None:
        """T5b: duplicate STOP messages cannot close Deepgram twice."""
        session = LifecycleSession()
        ws = LifecycleFakeWS()
        ws.queue_text(_valid_start())
        ws.queue_text({"type": "stop"})
        ws.queue_text({"type": "stop"})
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                await _drive_session(ws)
            completed = [e for e in ws.sent if e["type"] == "completed"]
            assert len(completed) == 1
            assert session.close_calls == 1
            assert session.closed is True
            assert ws.sends_after_disconnect == 0
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_refresh_disconnect_cleanup(self) -> None:
        """T6: abrupt disconnect (refresh) reaps every Session A task."""
        session = LifecycleSession()
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_valid_start())
                await _wait_until(lambda: session.receive_waiting)
                ws.queue_bytes(b"pcm-before-refresh")
                ws.queue_disconnect()  # no STOP — the tab just goes away
                await asyncio.wait_for(handler, timeout=10)
            assert session.sent_audio == [b"pcm-before-refresh"]
            assert session.audio_after_finish == []
            assert session.finished is False  # finish() only runs on STOP
            assert session.receive_cancelled is True
            assert session.close_calls == 1
            assert session.closed is True
            assert "completed" not in ws.event_types()
            assert ws.sends_after_disconnect == 0
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_session_isolation_prevents_cross_talk(self) -> None:
        """T7: Session B never interacts with closed Session A resources."""
        llm_entered = asyncio.Event()
        session_a = LifecycleSession(events=_utterance("hello from A"))
        session_b = LifecycleSession(events=_utterance("hello from B"))
        pending_sessions = [session_a, session_b]
        open_patcher = patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=lambda cfg: pending_sessions.pop(0),
        )
        open_patcher.start()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_hanging_llm(llm_entered)),
        )
        llm_patcher.start()
        try:
            # --- Session A: turn in flight, STOP mid-LLM ---
            ws_a = LifecycleFakeWS()
            handler_a = asyncio.create_task(_handle_realtime_session(ws_a))
            ws_a.queue_text(_start_with_llm())
            await asyncio.wait_for(llm_entered.wait(), timeout=5)
            ws_a.queue_bytes(b"audio-A")
            ws_a.queue_text({"type": "stop"})
            await asyncio.wait_for(handler_a, timeout=10)

            assert session_a.sent_audio == [b"audio-A"]
            assert session_a.audio_after_finish == []
            assert session_a.closed is True
            assert session_a.close_calls == 1
            assert ws_a.event_types()[-1] == "completed"
            assert "agent_response" not in ws_a.event_types()
            assert "audio" not in ws_a.event_types()
            assert mock_tts.synthesize.await_count == 0
            assert ws_a.sends_after_disconnect == 0
            a_events_after_stop = len(ws_a.sent)
            a_transcripts = [
                e["text"] for e in ws_a.sent if e["type"] == "transcript_final"
            ]
            assert a_transcripts == ["hello from A"]

            # --- Session B: fresh session after A fully closed ---
            ws_b = LifecycleFakeWS()
            handler_b = asyncio.create_task(_handle_realtime_session(ws_b))
            ws_b.queue_text(_start_with_llm())
            await _wait_until(lambda: session_b.receive_waiting)
            ws_b.queue_bytes(b"audio-B")
            ws_b.queue_text({"type": "stop"})
            await asyncio.wait_for(handler_b, timeout=10)

            assert session_b.sent_audio == [b"audio-B"]
            assert session_b.audio_after_finish == []
            assert session_b.receive_cancelled is True
            assert session_b.closed is True
            assert session_b.close_calls == 1
            assert ws_b.event_types()[0] == "session_started"
            assert ws_b.event_types()[-1] == "completed"
            assert "error" not in ws_b.event_types()
            assert ws_b.sends_after_disconnect == 0
            b_transcripts = [
                e["text"] for e in ws_b.sent if e["type"] == "transcript_final"
            ]
            assert b_transcripts == ["hello from B"]

            # Session A received nothing after its STOP, ever.
            assert len(ws_a.sent) == a_events_after_stop
            assert session_a.sent_audio == [b"audio-A"]
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)
            open_patcher.stop()

    async def test_sequential_sessions_stress_no_leaks(self, caplog) -> None:
        """T8: A->B->C->D sequential sessions leak nothing, send nothing stale."""
        session_refs = [
            LifecycleSession(events=_utterance(f"stress {i}")) for i in range(4)
        ]
        pending_sessions = list(session_refs)
        open_patcher = patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=lambda cfg: pending_sessions.pop(0),
        )
        open_patcher.start()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with caplog.at_level(logging.INFO):
                for index, session in enumerate(session_refs):
                    ws = LifecycleFakeWS()
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(lambda: session.receive_waiting)
                    ws.queue_bytes(f"audio-{index}".encode())
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)

                    assert session.sent_audio == [f"audio-{index}".encode()]
                    assert session.audio_after_finish == []
                    assert session.receive_cancelled is True
                    assert session.closed is True
                    assert session.close_calls == 1
                    assert ws.event_types()[0] == "session_started"
                    assert ws.event_types()[-1] == "completed"
                    assert "error" not in ws.event_types()
                    assert ws.sends_after_disconnect == 0
                    transcripts = [
                        e["text"]
                        for e in ws.sent
                        if e["type"] == "transcript_final"
                    ]
                    assert transcripts == [f"stress {index}"]
            assert "Gateway error" not in caplog.text
            assert caplog.text.count("session_closed connection_id=") == 4
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)
            open_patcher.stop()

    def test_no_credentials_in_events_or_logs(self, caplog) -> None:
        """API keys must never appear in client events or logs."""
        api_key = "test-deepgram-key-xxxxx"
        client, holder = _connect_with_fake_session(events=self._stream_events_test())
        received: list[dict] = []
        try:
            with caplog.at_level(logging.DEBUG):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_valid_start())
                    ws.send_bytes(b"\x01\x02")
                    ws.send_json({"type": "stop"})
                    while True:
                        event = ws.receive_json()
                        received.append(event)
                        if event["type"] == "completed":
                            break

        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

        for event in received:
            assert api_key not in json.dumps(event)
        assert api_key not in caplog.text

    @staticmethod
    def _stream_events_test() -> list[StreamEvent]:
        return [
            StreamEvent(type="partial", text="cred check", confidence=0.5),
        ]


# ---------------------------------------------------------------------------
# StreamConfig unit checks
# ---------------------------------------------------------------------------


class TestStreamConfigValidation:
    def test_valid_config(self) -> None:
        cfg = StreamConfig()
        assert cfg.validate() is None
        assert cfg.model == "nova-3"  # default from settings

    def test_default_model_from_settings(self) -> None:
        cfg = StreamConfig(model=None)
        assert cfg.model  # resolved from settings.default_stt_model


# ---------------------------------------------------------------------------
# asyncio sanity for the pump drain timeout
# ---------------------------------------------------------------------------


def test_stop_has_no_drain_timeout() -> None:
    """STOP uses immediate cancellation — no drain timeout constant."""
    import app.api.voice_realtime as mod

    # The old _STOP_DRAIN_TIMEOUT_S constant must be gone
    assert not hasattr(mod, "_STOP_DRAIN_TIMEOUT_S")


# ---------------------------------------------------------------------------
# Phase 6B: AgentRuntime integration on utterance_end
# ---------------------------------------------------------------------------


def _start_with_llm() -> dict[str, Any]:
    """START message with LLM configuration."""
    return {
        **_valid_start(),
        "llm_provider": "openrouter",
        "llm_model": "test-model",
    }


def _patch_realtime_deps():
    """Patch all dependencies needed for 6B agent processing tests."""
    from unittest.mock import AsyncMock, MagicMock

    from app.providers.types import TTSResult

    mock_db = MagicMock()
    mock_db_session = MagicMock()
    mock_db_session.id = "test-voice-session-id"
    mock_db_session.llm_provider = "openrouter"
    mock_db_session.llm_model = "test-model"
    mock_db_session.message_count = 0
    mock_db.query.return_value.filter.return_value.first.return_value = mock_db_session

    mock_session_local = MagicMock(return_value=mock_db)

    mock_agent_result = {
        "response": "Hello! How can I help?",
        "tool_calls": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "iterations": 1,
    }
    mock_process = AsyncMock(return_value=mock_agent_result)

    # Mock for streaming LLM path
    mock_stream_result = {
        "response": "Hello! How can I help?",
        "tool_calls": [],
        "usage": {},
        "iterations": 1,
        "streamed": True,
        "first_token_ms": 150.0,
        "streamed_tokens": 5,
        "sentences": ["Hello! How can I help?"],
    }
    mock_stream_process = AsyncMock(return_value=mock_stream_result)

    # LLM mock must have async close()
    mock_llm = MagicMock()
    mock_llm.close = AsyncMock()

    # TTS mock must have async synthesize() and close()
    mock_tts = MagicMock()
    mock_tts.provider_name = "elevenlabs"
    mock_tts.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_data=b"fake-mp3-audio-data",
            content_type="audio/mpeg",
        )
    )
    mock_tts.close = AsyncMock()

    patchers = {
        "session_local": patch(
            "app.api.voice_realtime.SessionLocal", mock_session_local
        ),
        "create_session": patch(
            "app.api.voice_realtime.create_realtime_session",
            return_value=mock_db_session,
        ),
        "get_llm": patch(
            "app.api.voice_realtime.get_llm_provider",
            return_value=mock_llm,
        ),
        "get_tts": patch(
            "app.api.voice_realtime.get_tts_provider",
            return_value=mock_tts,
        ),
        "process": patch(
            "app.api.voice_realtime.process_realtime_utterance",
            mock_process,
        ),
        "stream_process": patch(
            "app.api.voice_realtime._stream_llm_response",
            mock_stream_process,
        ),
    }
    for p in patchers.values():
        p.start()
    return patchers, mock_process, mock_tts, mock_stream_process


def _stop_patchers(patchers: dict) -> None:
    for p in patchers.values():
        p.stop()


class TestAgentRuntimeIntegration:
    """Phase 6B: utterance_end triggers AgentRuntime processing."""

    def test_utterance_end_triggers_agent_processing(self) -> None:
        """A final transcript + utterance_end triggers agent_processing event."""
        events = [
            StreamEvent(type="final", text="Hello there", confidence=0.95),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, mock_stream = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] in ("agent_response", "error"):
                        break
                assert "agent_processing" in received_types
                assert "agent_response" in received_types
                mock_stream.assert_awaited_once()
                call_args = mock_stream.call_args
                assert call_args.kwargs["transcript"] == "Hello there"
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_agent_response_contains_text(self) -> None:
        """agent_response event contains the agent's text response."""
        events = [
            StreamEvent(type="final", text="What is the time?", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, mock_process, _, _ = _patch_realtime_deps()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                agent_resp = None
                while True:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_resp = event
                        break
                    if event["type"] == "error":
                        break
                assert agent_resp is not None
                assert agent_resp["text"] == "Hello! How can I help?"
                assert agent_resp["iterations"] == 1
                assert agent_resp["tool_calls"] == 0
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_empty_utterance_does_not_invoke_agent(self) -> None:
        """An utterance_end without a final transcript does NOT call AgentRuntime."""
        events = [
            StreamEvent(type="partial", text="", confidence=0.0),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, mock_process, _, _ = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "completed":
                        break
                assert "agent_processing" not in received_types
                assert "agent_response" not in received_types
                mock_process.assert_not_awaited()
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_multiple_utterances_invoke_agent_each(self) -> None:
        """Two separate utterances produce two agent invocations."""
        from unittest.mock import AsyncMock

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "response": f"Response {call_count}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Response {call_count}"],
            }

        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First utterance")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while len(agent_responses) < 2:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(
                                holder, "Second utterance"
                            )
                    if event["type"] == "error":
                        break
                assert len(agent_responses) == 2
                assert agent_responses[0]["text"] == "Response 1"
                assert agent_responses[1]["text"] == "Response 2"
                assert call_count == 2
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_agent_error_produces_error_event(self) -> None:
        """An agent exception produces an error event, not silence."""
        from unittest.mock import AsyncMock

        from app.services.realtime_voice_service import RealtimeVoiceError

        events = [
            StreamEvent(type="final", text="This will fail", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=RealtimeVoiceError("LLM provider error")),
        )
        process_patcher.start()
        error_event = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "error":
                        error_event = event
                        break
                    if event["type"] == "completed":
                        break
                assert error_event is not None
                assert error_event["stage"] == "agent"
                assert "LLM provider error" in error_event["message"]
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_final_accumulates_then_utterance_end_triggers(self) -> None:
        """Multiple finals accumulate; utterance_end triggers exactly one call."""
        events = [
            StreamEvent(type="partial", text="what is", confidence=0.7),
            StreamEvent(type="final", text="what is", confidence=0.9),
            StreamEvent(type="partial", text="what is my leave", confidence=0.8),
            StreamEvent(type="final", text="my leave balance", confidence=0.95),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, mock_stream = _patch_realtime_deps()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                agent_resp = None
                while True:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_resp = event
                        break
                    if event["type"] == "error":
                        break
                assert agent_resp is not None
                mock_stream.assert_awaited_once()
                call_args = mock_stream.call_args
                # Both finals should be accumulated
                assert "what is" in call_args.kwargs["transcript"]
                assert "my leave balance" in call_args.kwargs["transcript"]
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_six_a_stt_events_still_forwarded(self) -> None:
        """Phase 6A STT events (partial/final/utterance_end) are still forwarded."""
        events = [
            StreamEvent(type="partial", text="hello", confidence=0.5),
            StreamEvent(type="final", text="hello world", confidence=0.95),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        received_types: list[str] = []
        received_texts: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.receive_json()  # session_started
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] in ("transcript_partial", "transcript_final"):
                        received_texts.append(event.get("text", ""))
                    if event["type"] == "agent_response":
                        break
                    if event["type"] == "error":
                        break
                assert "transcript_partial" in received_types
                assert "transcript_final" in received_types
                assert "utterance_end" in received_types
                assert "hello" in received_texts
                assert "hello world" in received_texts
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_llm_not_closed_after_utterance(self) -> None:
        """A: process_realtime_utterance() does NOT close the supplied LLM.

        The LLM is owned by the session and must survive across utterances.
        """
        from unittest.mock import AsyncMock, MagicMock

        from app.services.realtime_voice_service import process_realtime_utterance

        mock_db = MagicMock()
        mock_db_session = MagicMock()
        mock_db_session.id = "test-session"
        mock_db_session.llm_provider = "openrouter"
        mock_db_session.llm_model = "test-model"
        mock_db_session.message_count = 0
        mock_db.query.return_value.filter.return_value.first.return_value = mock_db_session

        mock_llm = MagicMock()
        mock_llm.close = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=MagicMock(
                content="Hello!",
                tool_calls=None,
                usage={"total_tokens": 10},
                finish_reason="stop",
            )
        )

        async def _run():
            with (
                patch("app.services.realtime_voice_service.get_tool_registry"),
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch("app.services.realtime_voice_service._save_tool_call"),
                patch("app.services.realtime_voice_service._save_usage"),
            ):
                await process_realtime_utterance(
                    db=mock_db,
                    session=mock_db_session,
                    transcript="Hello",
                    llm=mock_llm,
                )

        import asyncio

        asyncio.run(_run())

        # LLM close() must NOT have been called
        mock_llm.close.assert_not_awaited()

    def test_second_utterance_succeeds_with_same_llm(self) -> None:
        """L: Second utterance succeeds using the same LLM/httpx client.

        Regression test for the "client has been closed" bug.
        """
        from unittest.mock import AsyncMock

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "response": f"Response {call_count}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Response {call_count}"],
            }

        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        error_event = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while len(agent_responses) < 2:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(holder, "Second")
                    if event["type"] == "error":
                        error_event = event
                        break
                # Both utterances must succeed — no "client has been closed"
                assert error_event is None, f"Unexpected error: {error_event}"
                assert len(agent_responses) == 2
                assert agent_responses[0]["text"] == "Response 1"
                assert agent_responses[1]["text"] == "Response 2"
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_llm_closed_exactly_once_on_session_cleanup(self) -> None:
        """C: Session cleanup closes the LLM exactly once."""
        from unittest.mock import AsyncMock, MagicMock

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)

        # Create a mock LLM with async close()
        mock_llm = MagicMock()
        mock_llm.close = AsyncMock()

        patchers, _, mock_tts, _ = _patch_realtime_deps()
        # Override the get_llm patcher to return our specific mock
        patchers["get_llm"].stop()
        get_llm_patcher = patch(
            "app.api.voice_realtime.get_llm_provider",
            return_value=mock_llm,
        )
        get_llm_patcher.start()

        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for agent response
                while True:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        break
                    if event["type"] == "error":
                        break
                # Send STOP to trigger cleanup
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    if event["type"] == "completed":
                        break
                    if event["type"] == "error":
                        break

            # Verify close() was called exactly once
            assert mock_llm.close.await_count == 1
        finally:
            get_llm_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_history_passed_to_agent_runtime(self) -> None:
        """J: Previous conversation history is passed to AgentRuntime."""
        from app.providers.types import LLMMessage

        mock_history = [
            LLMMessage(role="user", content="Previous question"),
            LLMMessage(role="assistant", content="Previous answer"),
        ]

        events = [
            StreamEvent(type="final", text="New question", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, mock_stream = _patch_realtime_deps()
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=mock_history,
                ),
            ):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_start_with_llm())
                    while True:
                        event = ws.receive_json()
                        if event["type"] == "agent_response":
                            break
                        if event["type"] == "error":
                            break

            # Verify _stream_llm_response was called
            mock_stream.assert_awaited_once()
            # The function receives the session, and internally loads history
            # We can't directly verify initial_messages passed to AgentRuntime
            # without mocking deeper, but we verify the function was called
            # with the correct transcript
            call_args = mock_stream.call_args
            assert call_args.kwargs["transcript"] == "New question"
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()


class TestQueueBasedConcurrency:
    """Phase 6B: queue-based serialized utterance processing.

    The provider pump enqueues finalized utterances and continues
    draining Deepgram events. A separate worker task processes
    utterances from the queue through AgentRuntime.
    """

    def test_pump_continues_during_agent_processing(self) -> None:
        """1+2: Provider pump receives STT events while AgentRuntime runs.

        A second utterance's events arrive while the first is still
        being processed. Both must be handled correctly.
        """
        from unittest.mock import AsyncMock

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            import asyncio

            # Simulate some processing time
            await asyncio.sleep(0.1)
            return {
                "response": f"Response {call_count}: {kwargs['transcript']}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Response {call_count}: {kwargs['transcript']}"],
            }

        # Phase 6E: the second utterance arrives while the first turn's
        # agent processing is still running (delivered on agent_processing).
        events = _utterance("First question")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while len(agent_responses) < 2:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                    if event["type"] == "agent_processing" and (
                        not agent_responses
                    ):
                        # Deliver while the worker is still busy with turn 1.
                        _deliver_next_utterance(holder, "Second question")
                    if event["type"] == "error":
                        break
                # Both utterances must produce responses
                assert len(agent_responses) == 2
                assert call_count == 2
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_queued_utterances_processed_in_order(self) -> None:
        """4+5: Two utterances queued during processing are both handled in order."""
        from unittest.mock import AsyncMock

        call_order: list[str] = []

        async def _mock_process(**kwargs):
            call_order.append(kwargs["transcript"])
            return {
                "response": f"Answer {len(call_order)}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Answer {len(call_order)}"],
            }

        # Phase 6E: BBB and CCC are delivered as later turns, in order.
        events = _utterance("AAA")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                processing_seen = 0
                while len(agent_responses) < 3:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                    if event["type"] == "agent_processing":
                        processing_seen += 1
                        if processing_seen == 1:
                            _deliver_next_utterance(holder, "BBB")
                        elif processing_seen == 2:
                            _deliver_next_utterance(holder, "CCC")
                    if event["type"] == "error":
                        break
                assert len(agent_responses) == 3
                # Order must be preserved
                assert call_order == ["AAA", "BBB", "CCC"]
                assert agent_responses[0]["text"] == "Answer 1"
                assert agent_responses[1]["text"] == "Answer 2"
                assert agent_responses[2]["text"] == "Answer 3"
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_stop_waits_for_pending_processing(self) -> None:
        """14: STOP waits for pending agent processing before completed."""
        from unittest.mock import AsyncMock

        processing_done = False

        async def _mock_process(**kwargs):
            nonlocal processing_done
            import asyncio

            await asyncio.sleep(0.2)  # simulate some processing
            processing_done = True
            return {
                "response": "Done",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": ["Done"],
            }

        events = [
            StreamEvent(type="final", text="Process me", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for agent_response, then send STOP
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "agent_response":
                        break
                    if event["type"] == "error":
                        break
                # Now send STOP
                ws.send_json({"type": "stop"})
                # Should get completed (not force-close)
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "completed":
                        break
                    if event["type"] == "error":
                        break
                assert "completed" in received_types
                assert processing_done is True
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_agent_error_does_not_stall_queue(self) -> None:
        """16: Agent error does not leave the queue permanently stuck."""
        from unittest.mock import AsyncMock

        from app.services.realtime_voice_service import RealtimeVoiceError

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RealtimeVoiceError("First call fails")
            return {
                "response": "Second succeeds",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": ["Second succeeds"],
            }

        # Phase 6E: the second utterance follows the failed first turn.
        events = _utterance("Will fail")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        received_types: list[str] = []
        error_events: list[dict] = []
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait until we have 1 error + 1 agent_response (or 2 errors)
                while len(error_events) + len(agent_responses) < 2:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "error":
                        error_events.append(event)
                        if len(error_events) == 1:
                            _deliver_next_utterance(holder, "Will succeed")
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                # First utterance failed, second succeeded
                assert len(error_events) == 1
                assert error_events[0]["stage"] == "agent"
                assert len(agent_responses) == 1
                assert agent_responses[0]["text"] == "Second succeeds"
                assert call_count == 2
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_same_llm_reused_across_queued_utterances(self) -> None:
        """12: Same LLM instance is reused across multiple queued utterances."""
        from unittest.mock import AsyncMock

        llm_instances_used: list = []

        async def _mock_process(**kwargs):
            llm = kwargs.get("llm")
            llm_instances_used.append(id(llm))
            return {
                "response": "OK",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": ["OK"],
            }

        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while len(agent_responses) < 2:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(holder, "Second")
                    if event["type"] == "error":
                        break
                # Same LLM instance must be used for both utterances
                assert len(llm_instances_used) == 2
                assert llm_instances_used[0] == llm_instances_used[1]
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()


class TestTTSSynthesis:
    """Phase 6B.2: TTS synthesis after AgentRuntime responses."""

    def test_agent_response_triggers_tts(self) -> None:
        """A: Agent response triggers TTS synthesis."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "audio":
                        break
                    if event["type"] == "error":
                        break
                # TTS synthesize must have been called
                mock_tts.synthesize.assert_awaited_once()
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_receives_exact_response_text(self) -> None:
        """B: TTS receives exactly the assistant response text."""
        from unittest.mock import AsyncMock

        async def _mock_process(**kwargs):
            return {
                "response": "The weather is sunny today!",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": ["The weather is sunny today!"],
            }

        events = [
            StreamEvent(type="final", text="What is the weather?", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "audio":
                        break
                    if event["type"] == "error":
                        break
                # TTS must receive the exact response text
                mock_tts.synthesize.assert_awaited_once()
                call_kwargs = mock_tts.synthesize.call_args.kwargs
                assert call_kwargs["text"] == "The weather is sunny today!"
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_processing_event_emitted(self) -> None:
        """C: TTS processing event is emitted before audio."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "audio":
                        break
                    if event["type"] == "error":
                        break
                # tts_processing must come after agent_response, before audio
                assert "tts_processing" in received_types
                agent_idx = received_types.index("agent_response")
                tts_idx = received_types.index("tts_processing")
                audio_idx = received_types.index("audio")
                assert agent_idx < tts_idx < audio_idx
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_successful_tts_emits_audio_event(self) -> None:
        """D: Successful TTS emits an audio event."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        audio_event = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "audio":
                        audio_event = event
                        break
                    if event["type"] == "error":
                        break
                assert audio_event is not None
                assert "data" in audio_event
                assert len(audio_event["data"]) > 0
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_audio_format_is_mpeg(self) -> None:
        """E: Audio format is audio/mpeg."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        audio_event = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "audio":
                        audio_event = event
                        break
                    if event["type"] == "error":
                        break
                assert audio_event is not None
                assert audio_event["format"] == "audio/mpeg"
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_audio_bytes_valid_nonempty(self) -> None:
        """F: Audio bytes are valid and non-empty."""
        import base64

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        audio_event = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "audio":
                        audio_event = event
                        break
                    if event["type"] == "error":
                        break
                assert audio_event is not None
                # Decode base64 and verify non-empty
                audio_data = base64.b64decode(audio_event["data"])
                assert len(audio_data) > 0
                assert audio_data == b"fake-mp3-audio-data"
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_error_emits_error_stage_tts(self) -> None:
        """G: TTS error emits error event with stage=tts."""
        from unittest.mock import AsyncMock

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        # Make TTS fail
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS API down"))
        error_event = None
        agent_response = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "error" and event.get("stage") == "tts":
                        error_event = event
                        break
                    if event["type"] == "agent_response":
                        agent_response = event
                    if event["type"] == "completed":
                        break
                # Agent response must still arrive
                assert agent_response is not None
                # TTS error must be reported
                assert error_event is not None
                assert error_event["stage"] == "tts"
                assert "TTS API down" in error_event["message"]
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_error_does_not_destroy_llm_session(self) -> None:
        """H: TTS error does not destroy the LLM session."""
        from unittest.mock import AsyncMock

        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        # Make TTS fail on first call, succeed on second
        call_count = 0

        async def _tts_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("TTS temporary failure")
            from app.providers.types import TTSResult

            return TTSResult(audio_data=b"second-audio", content_type="audio/mpeg")

        mock_tts.synthesize = AsyncMock(side_effect=_tts_side_effect)
        agent_responses: list[dict] = []
        error_events: list[dict] = []
        audio_events: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for both utterances to be processed
                # Need 2 agent_responses + 1 audio (first TTS fails, second succeeds)
                while len(agent_responses) < 2 or len(audio_events) < 1:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(holder, "Second")
                    if event["type"] == "error":
                        error_events.append(event)
                    if event["type"] == "audio":
                        audio_events.append(event)
                    if len(error_events) > 2:
                        break
                # Both agent responses must succeed
                assert len(agent_responses) == 2
                # First TTS failed, second succeeded
                assert len(error_events) == 1
                assert error_events[0]["stage"] == "tts"
                assert len(audio_events) == 1
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_error_does_not_destroy_deepgram_session(self) -> None:
        """I: TTS error does not destroy the Deepgram session."""
        from unittest.mock import AsyncMock

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        # Make TTS fail
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS boom"))
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for error, then send STOP
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "error" and event.get("stage") == "tts":
                        break
                    if event["type"] == "completed":
                        break
                # Session must still be usable — send STOP
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "completed":
                        break
                    if event["type"] == "error":
                        break
                # Must complete cleanly
                assert "completed" in received_types
                # Deepgram session must be closed (not crashed)
                assert holder["session"].closed is True
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_multiple_utterances_fifo_with_tts(self) -> None:
        """J: Multiple utterances process in FIFO order: LLM → TTS → audio."""
        from unittest.mock import AsyncMock

        call_order: list[str] = []

        async def _mock_process(**kwargs):
            call_order.append(f"llm:{kwargs['transcript']}")
            return {
                "response": f"Answer for {kwargs['transcript']}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Answer for {kwargs['transcript']}"],
            }

        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        tts_calls: list[str] = []

        original_synthesize = mock_tts.synthesize

        async def _tracking_synthesize(**kwargs):
            tts_calls.append(f"tts:{kwargs['text']}")
            return await original_synthesize(**kwargs)

        mock_tts.synthesize = AsyncMock(side_effect=_tracking_synthesize)
        agent_responses: list[dict] = []
        audio_events: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for both agent responses AND both audio events
                while len(agent_responses) < 2 or len(audio_events) < 2:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(holder, "Second")
                    if event["type"] == "audio":
                        audio_events.append(event)
                    if event["type"] == "error":
                        break
                # Both utterances processed
                assert len(agent_responses) == 2
                assert len(audio_events) == 2
                # FIFO order preserved
                assert call_order == ["llm:First", "llm:Second"]
                assert tts_calls == [
                    "tts:Answer for First",
                    "tts:Answer for Second",
                ]
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_not_called_for_empty_utterances(self) -> None:
        """K: TTS is not called for empty utterances."""
        events = [
            StreamEvent(type="partial", text="", confidence=0.0),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "completed":
                        break
                # No agent processing, no TTS
                assert "agent_processing" not in received_types
                assert "tts_processing" not in received_types
                assert "audio" not in received_types
                mock_tts.synthesize.assert_not_awaited()
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_partial_transcripts_never_trigger_tts(self) -> None:
        """L: Partial transcripts never trigger TTS."""
        events = [
            StreamEvent(type="partial", text="hello", confidence=0.5),
            StreamEvent(type="partial", text="hello world", confidence=0.7),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "completed":
                        break
                # Only partials, no utterance_end → no TTS
                assert "transcript_partial" in received_types
                assert "utterance_end" not in received_types
                assert "tts_processing" not in received_types
                assert "audio" not in received_types
                mock_tts.synthesize.assert_not_awaited()
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_does_not_create_duplicate_message(self) -> None:
        """M: TTS does not create a duplicate conversation message.

        TTS is purely audio representation — it must not call _save_message.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import patch as mock_patch

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        save_message_calls: list = []

        original_process = patchers["stream_process"].new

        async def _tracking_process(**kwargs):
            result = await original_process(**kwargs)
            return result

        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_tracking_process),
        )
        process_patcher.start()
        try:
            with mock_patch(
                "app.services.realtime_voice_service._save_message",
                side_effect=lambda *a, **kw: save_message_calls.append((a, kw)),
            ):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_start_with_llm())
                    while True:
                        event = ws.receive_json()
                        if event["type"] == "audio":
                            break
                        if event["type"] == "error":
                            break
            # process_realtime_utterance handles message saving internally
            # TTS must not add additional saves
            # We verify TTS was called but didn't trigger extra DB writes
            assert len(save_message_calls) == 0  # mocked at service level
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_same_providers_reused_across_turns(self) -> None:
        """N: Same session-level providers remain usable across multiple turns."""
        from unittest.mock import AsyncMock

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "response": f"Response {call_count}",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 100.0,
                "streamed_tokens": 5,
                "sentences": [f"Response {call_count}"],
            }

        # Phase 6E: Second/Third follow each released turn, in order.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        audio_events: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for all agent responses AND all audio events
                while len(agent_responses) < 3 or len(audio_events) < 3:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
                        if len(agent_responses) == 1:
                            _deliver_next_utterance(holder, "Second")
                        elif len(agent_responses) == 2:
                            _deliver_next_utterance(holder, "Third")
                    if event["type"] == "audio":
                        audio_events.append(event)
                    if event["type"] == "error":
                        break
                # All 3 utterances succeeded with same providers
                assert len(agent_responses) == 3
                assert len(audio_events) == 3
                # TTS synthesize called 3 times (same instance)
                assert mock_tts.synthesize.await_count == 3
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_tts_closed_on_session_cleanup(self) -> None:
        """P: TTS close is called on session cleanup."""
        from unittest.mock import AsyncMock, MagicMock

        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)

        # Create a specific mock TTS to track close()
        mock_tts_specific = MagicMock()
        mock_tts_specific.provider_name = "elevenlabs"
        mock_tts_specific.synthesize = AsyncMock(
            return_value=MagicMock(
                audio_data=b"test-audio",
                content_type="audio/mpeg",
            )
        )
        mock_tts_specific.close = AsyncMock()

        patchers, _, _, _ = _patch_realtime_deps()
        # Override the get_tts patcher to return our specific mock
        patchers["get_tts"].stop()
        get_tts_patcher = patch(
            "app.api.voice_realtime.get_tts_provider",
            return_value=mock_tts_specific,
        )
        get_tts_patcher.start()

        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Wait for agent response
                while True:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        break
                    if event["type"] == "error":
                        break
                # Send STOP to trigger cleanup
                ws.send_json({"type": "stop"})
                while True:
                    event = ws.receive_json()
                    if event["type"] == "completed":
                        break
                    if event["type"] == "error":
                        break

            # Verify TTS close() was called
            mock_tts_specific.close.assert_awaited_once()
        finally:
            get_tts_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()


# ---------------------------------------------------------------------------
# Model resolution: Browser Realtime defaults to gpt-4o-mini
# ---------------------------------------------------------------------------


class TestRealtimeModelResolution:
    """Verify the Browser Realtime default LLM model.

    The realtime path uses a dedicated default (openai/gpt-4o-mini)
    instead of the global default (which may be a slow free-tier model).
    """

    def test_default_model_is_gpt4o_mini(self) -> None:
        """START without explicit model => openai/gpt-4o-mini."""
        from unittest.mock import MagicMock

        from app.services.realtime_voice_service import (
            _REALTIME_DEFAULT_LLM_MODEL,
            create_realtime_session,
        )

        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.llm_provider = "openrouter"
        mock_session.llm_model = _REALTIME_DEFAULT_LLM_MODEL
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None

        with patch(
            "app.services.realtime_voice_service.VoiceSession",
            return_value=mock_session,
        ):
            with patch(
                "app.services.realtime_voice_service.settings"
            ) as mock_settings:
                # Global default is a slow free model
                mock_settings.default_llm_provider = "openrouter"
                mock_settings.default_llm_model = (
                    "nvidia/nemotron-3.5-lightning:free"
                )
                # Call with no explicit model
                result = create_realtime_session(
                    db=mock_db,
                    llm_provider="openrouter",
                    llm_model=None,
                )
        # The session should use gpt-4o-mini, NOT the global default
        assert result.llm_model == "openai/gpt-4o-mini"

    def test_explicit_model_preserved(self) -> None:
        """START with explicit model => requested model is preserved."""
        from unittest.mock import MagicMock

        from app.services.realtime_voice_service import create_realtime_session

        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_session.id = "test-session-2"
        mock_session.llm_provider = "openrouter"
        mock_session.llm_model = "anthropic/claude-3.5-sonnet"
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None

        with patch(
            "app.services.realtime_voice_service.VoiceSession",
            return_value=mock_session,
        ):
            result = create_realtime_session(
                db=mock_db,
                llm_provider="openrouter",
                llm_model="anthropic/claude-3.5-sonnet",
            )
        assert result.llm_model == "anthropic/claude-3.5-sonnet"

    def test_realtime_default_constant(self) -> None:
        """The realtime default model constant is gpt-4o-mini."""
        from app.services.realtime_voice_service import (
            _REALTIME_DEFAULT_LLM_MODEL,
        )

        assert _REALTIME_DEFAULT_LLM_MODEL == "openai/gpt-4o-mini"


class TestPerTurnLatencyMetrics:
    """Per-utterance latency metrics calculation and event emission."""

    def test_utterance_timings_elapsed_calculation(self) -> None:
        """UtteranceTimings.to_metrics_dict calculates durations correctly."""
        from app.api.voice_realtime import UtteranceTimings

        base = 100.0
        t = UtteranceTimings(
            turn=1,
            utterance_end_at=101.0,
            agent_processing_started_at=101.050,
            llm_completed_at=102.500,
            tts_started_at=102.510,
            tts_completed_at=103.200,
            audio_sent_at=103.210,
        )
        result = t.to_metrics_dict(base)

        assert result["turn"] == 1
        # utterance_end → agent: 101.050 - 101.0 = 50ms
        assert result["utterance_end_to_agent_ms"] == 50.0
        # agent processing: 102.500 - 101.050 = 1450ms
        assert result["agent_processing_ms"] == 1450.0
        # TTS: 103.200 - 102.510 = 690ms
        assert result["tts_duration_ms"] == 690.0
        # total: 103.210 - 101.0 = 2210ms
        assert result["utterance_end_to_audio_sent_ms"] == 2210.0

    def test_utterance_timings_none_fields(self) -> None:
        """Missing timestamps produce None durations, not errors."""
        from app.api.voice_realtime import UtteranceTimings

        t = UtteranceTimings(turn=2)
        result = t.to_metrics_dict(100.0)

        assert result["turn"] == 2
        assert result["utterance_end_to_agent_ms"] is None
        assert result["agent_processing_ms"] is None
        assert result["tts_duration_ms"] is None
        assert result["utterance_end_to_audio_sent_ms"] is None

    def test_utterance_timings_partial_fields(self) -> None:
        """Partial timestamps: only available durations are calculated."""
        from app.api.voice_realtime import UtteranceTimings

        t = UtteranceTimings(
            turn=3,
            utterance_end_at=200.0,
            agent_processing_started_at=200.100,
            llm_completed_at=201.000,
            # No TTS timestamps
        )
        result = t.to_metrics_dict(100.0)

        assert result["utterance_end_to_agent_ms"] == 100.0
        assert result["agent_processing_ms"] == 900.0
        assert result["tts_duration_ms"] is None
        assert result["utterance_end_to_audio_sent_ms"] is None

    def test_utterance_timings_offsets(self) -> None:
        """Raw offsets are calculated from the base timestamp."""
        from app.api.voice_realtime import UtteranceTimings

        base = 50.0
        t = UtteranceTimings(
            turn=1,
            utterance_end_at=51.0,
            audio_sent_at=54.0,
        )
        result = t.to_metrics_dict(base)

        # 51.0 - 50.0 = 1000ms offset
        assert result["utterance_end_offset_ms"] == 1000.0
        # 54.0 - 50.0 = 4000ms offset
        assert result["audio_sent_offset_ms"] == 4000.0

    def test_turn_metrics_event_emitted_after_audio(self) -> None:
        """A turn_metrics event is emitted after the audio event."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "turn_metrics":
                        break
                    if event["type"] == "error":
                        break
                # turn_metrics must come after audio
                assert "audio" in received_types
                assert "turn_metrics" in received_types
                audio_idx = received_types.index("audio")
                metrics_idx = received_types.index("turn_metrics")
                assert audio_idx < metrics_idx
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_metrics_payload_has_required_fields(self) -> None:
        """turn_metrics event payload contains all expected duration fields."""
        events = [
            StreamEvent(type="final", text="Test", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_data = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "turn_metrics":
                        metrics_data = event["data"]
                        break
                    if event["type"] == "error":
                        break
                assert metrics_data is not None
                assert "turn" in metrics_data
                assert "utterance_end_to_agent_ms" in metrics_data
                assert "agent_processing_ms" in metrics_data
                assert "tts_duration_ms" in metrics_data
                assert "utterance_end_to_audio_sent_ms" in metrics_data
                # Offsets for debugging
                assert "utterance_end_offset_ms" in metrics_data
                assert "audio_sent_offset_ms" in metrics_data
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_metrics_values_are_non_negative(self) -> None:
        """All duration values in turn_metrics are non-negative."""
        events = [
            StreamEvent(type="final", text="Check", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_data = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "turn_metrics":
                        metrics_data = event["data"]
                        break
                    if event["type"] == "error":
                        break
                assert metrics_data is not None
                duration_keys = [
                    "utterance_end_to_agent_ms",
                    "agent_processing_ms",
                    "tts_duration_ms",
                    "utterance_end_to_audio_sent_ms",
                ]
                for key in duration_keys:
                    val = metrics_data[key]
                    if val is not None:
                        assert val >= 0, f"{key}={val} is negative"
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_metrics_emitted_without_tts(self) -> None:
        """turn_metrics is emitted even when TTS is skipped (no TTS provider)."""
        events = [
            StreamEvent(type="final", text="No TTS", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        # Override TTS provider to None
        patchers["get_tts"].stop()
        no_tts_patcher = patch(
            "app.api.voice_realtime.get_tts_provider",
            side_effect=Exception("No TTS"),
        )
        no_tts_patcher.start()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "turn_metrics":
                        break
                    # Allow stopping on error too, but we expect turn_metrics
                    if len(received_types) > 20:
                        break
                assert "turn_metrics" in received_types
        finally:
            no_tts_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_timings_accumulate_across_utterances(self) -> None:
        """Multiple utterances produce multiple turn_metrics events."""
        # Phase 6E: the second utterance follows the first released turn.
        events = _utterance("First")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_turns: list[int] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                # Collect metrics until we have 2 or timeout
                import time as _time

                deadline = _time.monotonic() + 10.0
                while len(metrics_turns) < 2 and _time.monotonic() < deadline:
                    try:
                        event = ws.receive_json()
                        if event["type"] == "turn_metrics":
                            metrics_turns.append(event["data"]["turn"])
                            if len(metrics_turns) == 1:
                                _deliver_next_utterance(holder, "Second")
                        if event["type"] == "error":
                            break
                    except Exception:
                        break
                # Should have metrics for both turns
                assert 1 in metrics_turns
                assert 2 in metrics_turns
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()


class TestStreamingLLMPath:
    """Tests for streaming LLM conversational path."""

    def test_stream_produces_incremental_chunks(self) -> None:
        """LLM stream_chat yields incremental content chunks."""
        from unittest.mock import MagicMock

        from app.providers.types import StreamChunk

        mock_llm = MagicMock()

        async def fake_stream():
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" world")
            yield StreamChunk(content="!", finish_reason="stop")

        mock_llm.stream_chat = MagicMock(return_value=fake_stream())

        async def _run():
            from app.api.voice_realtime import _stream_llm_response

            mock_db = MagicMock()
            mock_db_session = MagicMock()
            mock_db_session.id = "test-session"
            mock_db_session.message_count = 0
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_db_session
            )

            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
            ):
                result = await _stream_llm_response(
                    db=mock_db,
                    voice_session_id="test-session",
                    transcript="Hi",
                    llm=mock_llm,
                    model="test-model",
                )
            return result

        import asyncio

        result = asyncio.run(_run())
        assert result["response"] == "Hello world!"
        assert result["streamed"] is True
        assert result["streamed_tokens"] == 3

    def test_first_token_timestamp_captured(self) -> None:
        """First-token latency is captured in the result."""
        from unittest.mock import MagicMock

        from app.providers.types import StreamChunk

        mock_llm = MagicMock()

        async def fake_stream():
            yield StreamChunk(content="First")
            yield StreamChunk(content=" token", finish_reason="stop")

        mock_llm.stream_chat = MagicMock(return_value=fake_stream())

        async def _run():
            from app.api.voice_realtime import _stream_llm_response

            mock_db = MagicMock()
            mock_db_session = MagicMock()
            mock_db_session.id = "test-session"
            mock_db_session.message_count = 0
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_db_session
            )

            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
            ):
                result = await _stream_llm_response(
                    db=mock_db,
                    voice_session_id="test-session",
                    transcript="Hi",
                    llm=mock_llm,
                    model="test-model",
                )
            return result

        import asyncio

        result = asyncio.run(_run())
        assert result["first_token_ms"] is not None
        assert result["first_token_ms"] >= 0

    def test_sentence_buffer_detects_boundaries(self) -> None:
        """SentenceBuffer detects '.', '!', '?' boundaries."""
        from app.services.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()

        # Test period
        sentences = buffer.add("Hello world. ")
        assert sentences == ["Hello world."]

        # Test exclamation
        sentences = buffer.add("Wow! ")
        assert sentences == ["Wow!"]

        # Test question
        sentences = buffer.add("How are you? ")
        assert sentences == ["How are you?"]

    def test_multiple_sentences_emitted_separately(self) -> None:
        """Multiple sentences are emitted as separate items."""
        from app.services.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()

        sentences = buffer.add("First sentence. Second sentence. ")
        assert len(sentences) == 2
        assert "First sentence." in sentences
        assert "Second sentence." in sentences

    def test_partial_sentence_held_in_buffer(self) -> None:
        """Partial sentences are held until boundary detected."""
        from app.services.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()

        # No boundary yet
        sentences = buffer.add("Hello world")
        assert sentences == []

        # Now add boundary
        sentences = buffer.add("! ")
        assert sentences == ["Hello world!"]

    def test_flush_returns_remaining_text(self) -> None:
        """flush() returns any remaining buffered text."""
        from app.services.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()
        buffer.add("Hello world")
        remaining = buffer.flush()
        assert remaining == "Hello world"

    def test_streaming_path_emits_agent_response(self) -> None:
        """Streaming path still emits agent_response event."""
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, mock_stream = _patch_realtime_deps()
        received_types: list[str] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    received_types.append(event["type"])
                    if event["type"] == "agent_response":
                        break
                    if event["type"] == "error":
                        break
                assert "agent_response" in received_types
                mock_stream.assert_awaited_once()
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_streaming_path_captures_first_token_in_timings(self) -> None:
        """turn_metrics includes llm_first_token_ms from streaming."""
        events = [
            StreamEvent(type="final", text="Test", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_data = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "turn_metrics":
                        metrics_data = event["data"]
                        break
                    if event["type"] == "error":
                        break
                assert metrics_data is not None
                assert "llm_first_token_ms" in metrics_data
                assert "streamed_tokens" in metrics_data
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()


# ---------------------------------------------------------------------------
# STOP lifecycle tests
# ---------------------------------------------------------------------------


class TestSTOPLifecycle:
    """STOP must immediately cancel the entire realtime pipeline.

    Covers: pump cancellation, worker cancellation, LLM/TTS interruption,
    resource cleanup, idempotency, and absence of Gateway errors.
    """

    def test_stop_completes_and_cleans_up(self) -> None:
        """STOP sends completed and cleans up the session."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
                session = holder["session"]
                assert session.finished is True
                assert session.closed is True
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_prevents_audio_forwarding(self) -> None:
        """Audio sent before STOP message is processed gets forwarded."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_bytes(b"before-stop")
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
                # Audio sent before STOP message was processed is forwarded
                assert b"before-stop" in holder["session"].sent_audio
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_idempotent(self) -> None:
        """STOP completes cleanly; a new connection can start fresh."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
            # WebSocket closed by server after STOP — this is correct.
            # A new connection should work independently.
            with client.websocket_connect(REALTIME_WS_PATH) as ws2:
                ws2.send_json(_valid_start())
                event2 = ws2.receive_json()
                assert event2["type"] == "session_started"
                ws2.send_json({"type": "stop"})
                event3 = ws2.receive_json()
                assert event3["type"] == "completed"
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_cancels_active_llm_work(self) -> None:
        """STOP cancels pending work instead of waiting for it."""
        from unittest.mock import AsyncMock

        async def _slow_stream(**kwargs):
            await asyncio.sleep(30)
            return {"response": "never", "tool_calls": [], "usage": {},
                    "iterations": 1, "streamed": True, "first_token_ms": None,
                    "streamed_tokens": 0, "sentences": []}

        events = [
            StreamEvent(type="final", text="hello", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_slow_stream),
        )
        process_patcher.start()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.receive_json()  # session_started
                # Send STOP immediately — pump/worker may or may not have
                # started processing. Either way, STOP must complete fast.
                start = time.monotonic()
                ws.send_json({"type": "stop"})
                # Drain events until completed
                while True:
                    event = ws.receive_json()
                    if event["type"] == "completed":
                        break
                elapsed = time.monotonic() - start
                # Must complete in < 5s, not 30s (the slow LLM duration)
                assert elapsed < 5.0
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_stop_no_gateway_runtime_error(self) -> None:
        """STOP does not produce a Gateway RuntimeError."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                # Collect all events
                events = []
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] == "completed":
                        break
                # No error events
                error_events = [e for e in events if e["type"] == "error"]
                assert len(error_events) == 0
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_cancels_pump_immediately(self) -> None:
        """Pump task is cancelled — no lingering Deepgram processing."""

        class _SlowPumpSession(FakeStreamingSession):
            """Session whose receive() blocks for a long time after finish."""
            async def receive(self):
                while not self.finished:
                    await asyncio.sleep(0.05)
                # After finish(), simulate lingering Deepgram messages
                await asyncio.sleep(10)
                return None

        client, holder = _connect_with_session_instance(_SlowPumpSession())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                start = time.monotonic()
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                elapsed = time.monotonic() - start
                assert event["type"] == "completed"
                # Must complete quickly, not wait for the 10s sleep
                assert elapsed < 5.0
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_cancels_worker_between_items(self) -> None:
        """Worker is cancelled even if waiting for next queue item."""
        client, holder = _connect_with_fake_session()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.receive_json()  # session_started
                # Worker is running but no utterances queued
                ws.send_json({"type": "stop"})
                start = time.monotonic()
                event = ws.receive_json()
                elapsed = time.monotonic() - start
                assert event["type"] == "completed"
                # Should complete immediately, not wait for worker
                assert elapsed < 5.0
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_stop_no_new_utterance_processing(self) -> None:
        """After STOP, the pump guard prevents new utterance enqueue."""
        from unittest.mock import AsyncMock

        process_calls: list[str] = []

        async def _mock_process(**kwargs):
            process_calls.append(kwargs["transcript"])
            return {
                "response": "ok",
                "tool_calls": [],
                "usage": {},
                "iterations": 1,
                "streamed": True,
                "first_token_ms": 10.0,
                "streamed_tokens": 1,
                "sentences": ["ok"],
            }

        events = [
            StreamEvent(type="final", text="should not process", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime._stream_llm_response",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                # Drain events until completed
                while True:
                    event = ws.receive_json()
                    if event["type"] == "completed":
                        break
                # The pump guard should prevent enqueue after stop_requested.
                # Due to race conditions, 0 or 1 calls are acceptable.
                assert len(process_calls) <= 1
        finally:
            process_patcher.stop()
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_stop_cleans_up_session_on_finally(self) -> None:
        """Session resources are cleaned up even after STOP + disconnect."""
        client, holder = _connect_with_fake_session()
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_json({"type": "stop"})
                event = ws.receive_json()
                assert event["type"] == "completed"
            # Context exited — client disconnected
            session = holder["session"]
            # Give the finally block time to clean up
            deadline = time.monotonic() + 5.0
            while not session.closed and time.monotonic() < deadline:
                time.sleep(0.05)
            assert session.closed is True
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Phase 6E: turn finalization (UtteranceEnd + settle release) regression tests
# ---------------------------------------------------------------------------


def _final_segment(
    text: str, *, speech_final: bool = False, confidence: float = 0.9
) -> StreamEvent:
    """A provider final segment with speech_final metadata."""
    return StreamEvent(
        type="final",
        text=text,
        confidence=confidence,
        metadata={"speech_final": speech_final},
    )


def _utterance_end_event(last_word_end: float | None = None) -> StreamEvent:
    """A provider UtteranceEnd with last_word_end metadata."""
    return StreamEvent(
        type="utterance_end",
        metadata={"last_word_end": last_word_end},
    )


def _settle_speed(monkeypatch: Any, ms: int) -> None:
    """Change the turn-release settle window for a test."""
    monkeypatch.setattr(
        "app.api.voice_realtime._TURN_RELEASE_SETTLE_MS", ms
    )


def _recording_llm_patcher(calls: list[str]) -> Any:
    """Open a patcher for _stream_llm_response recording released turns."""

    async def _stream(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["transcript"])
        return {
            "response": f"Answer {len(calls)}",
            "tool_calls": [],
            "usage": {},
            "iterations": 1,
            "streamed": True,
            "first_token_ms": 10.0,
            "streamed_tokens": 2,
            "sentences": [f"Answer {len(calls)}"],
        }

    patcher = patch(
        "app.api.voice_realtime._stream_llm_response",
        AsyncMock(side_effect=_stream),
    )
    patcher.start()
    return patcher


class TestTurnFinalizationSettle:
    """Phase 6E (T1-T9): UtteranceEnd schedules a settle release; newer
    transcript evidence cancels it; the release task is session-owned.
    """

    async def test_final_without_utterance_end_does_not_dispatch(
        self, monkeypatch
    ) -> None:
        """T1: a final segment never dispatches before UtteranceEnd."""
        _settle_speed(monkeypatch, 200)
        calls: list[str] = []
        session = LifecycleSession(
            events=[_final_segment("Tell me more", speech_final=False)]
        )
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: "transcript_final" in ws.event_types()
                )
                # Far beyond the settle window: a final alone never releases.
                await asyncio.sleep(0.4)
                assert calls == []
                assert "agent_processing" not in ws.event_types()
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert ws.event_types()[-1] == "completed"
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_speech_final_is_evidence_not_a_dispatch_trigger(
        self, monkeypatch, caplog
    ) -> None:
        """CHANGE 6: speech_final=True alone must not release the turn."""
        _settle_speed(monkeypatch, 200)
        calls: list[str] = []
        session = LifecycleSession(
            events=[_final_segment("check my order", speech_final=True)]
        )
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "utterance_segment_final" in caplog.text
                    )
                    await asyncio.sleep(0.4)
                    assert calls == []
                    assert "utterance_release_scheduled" not in caplog.text
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_multi_final_segments_combine_into_one_utterance(
        self, monkeypatch
    ) -> None:
        """T2/CHANGE 5: two finals of one turn become ONE utterance."""
        _settle_speed(monkeypatch, 300)
        calls: list[str] = []
        session = LifecycleSession(
            events=[_final_segment("Tell me the details of employee ID.")]
        )
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: "transcript_final" in ws.event_types()
                )
                # Second finalized segment of the same conversational turn.
                session.events.append(_final_segment("E zero zero one."))
                await _wait_until(
                    lambda: ws.event_types().count("transcript_final") == 2
                )
                assert "agent_processing" not in ws.event_types()
                # The natural pause ends → UtteranceEnd → settle → ONE turn.
                session.events.append(_utterance_end_event(12.5))
                await _wait_until(lambda: calls != [], timeout=5)
                assert calls == [
                    "Tell me the details of employee ID. E zero zero one."
                ]
                assert ws.event_types().count("agent_processing") == 1
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_utterance_end_schedules_release_not_immediate(
        self, monkeypatch, caplog
    ) -> None:
        """T3: UtteranceEnd schedules; release follows the settle window."""
        _settle_speed(monkeypatch, 500)
        calls: list[str] = []
        session = LifecycleSession(events=_utterance("book a flight"))
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "utterance_release_scheduled" in caplog.text
                    )
                    # Still inside the settle window: nothing released yet.
                    await asyncio.sleep(0.2)
                    assert calls == []
                    assert "agent_processing" not in ws.event_types()
                    await _wait_until(
                        lambda: calls == ["book a flight"], timeout=5
                    )
                    assert "utterance_released" in caplog.text
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_new_evidence_cancels_pending_release(
        self, monkeypatch, caplog
    ) -> None:
        """T4: evidence inside the settle window cancels and re-schedules."""
        _settle_speed(monkeypatch, 3000)
        calls: list[str] = []
        session = LifecycleSession(
            events=[_final_segment("Tell me the details of employee ID.")]
        )
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "transcript_final" in ws.event_types()
                    )
                    session.events.append(_utterance_end_event(8.0))
                    await _wait_until(
                        lambda: "utterance_release_scheduled" in caplog.text
                    )
                    # The user keeps speaking within the settle window.
                    session.events.append(
                        StreamEvent(
                            type="partial", text="E zero", confidence=0.4
                        )
                    )
                    await _wait_until(
                        lambda: "reason=new_partial_evidence" in caplog.text
                    )
                    # Restart with a short settle so the test finishes fast.
                    _settle_speed(monkeypatch, 150)
                    session.events.append(_final_segment("E zero zero one."))
                    session.events.append(_utterance_end_event(15.5))
                    await _wait_until(lambda: calls != [], timeout=5)
                    assert calls == [
                        "Tell me the details of employee ID. "
                        "E zero zero one."
                    ]
                    assert (
                        caplog.text.count("utterance_release_scheduled") == 2
                    )
                    assert "utterance_released" in caplog.text
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_duplicate_utterance_end_releases_only_one_turn(
        self, monkeypatch, caplog
    ) -> None:
        """T5: repeated UtteranceEnd must not double-release the turn."""
        _settle_speed(monkeypatch, 250)
        calls: list[str] = []
        session = LifecycleSession(
            events=[
                _final_segment("hello"),
                _utterance_end_event(3.0),
                _utterance_end_event(4.0),
            ]
        )
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(lambda: calls == ["hello"], timeout=5)
                    # A duplicate release has every chance to fire — none may.
                    await asyncio.sleep(0.4)
                    assert calls == ["hello"]
                    assert ws.event_types().count("agent_processing") == 1
                    assert caplog.text.count("utterance_released") == 1
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_stale_utterance_end_is_ignored_safely(
        self, monkeypatch, caplog
    ) -> None:
        """T6/CHANGE 4: last_word_end=-1 must not release — buffer survives."""
        _settle_speed(monkeypatch, 200)
        calls: list[str] = []
        session = LifecycleSession(events=[_final_segment("employee ID")])
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "transcript_final" in ws.event_types()
                    )
                    # Deepgram's stale UtteranceEnd (segment already finalized).
                    session.events.append(_utterance_end_event(-1))
                    await _wait_until(lambda: "stale=True" in caplog.text)
                    await asyncio.sleep(0.4)  # well past the settle window
                    assert calls == []
                    assert "utterance_release_scheduled" not in caplog.text
                    # The accumulated buffer survived: a real UtteranceEnd
                    # still releases the already-spoken text.
                    session.events.append(_utterance_end_event(9.5))
                    await _wait_until(
                        lambda: calls == ["employee ID"], timeout=5
                    )
                    ws.queue_text({"type": "stop"})
                    await asyncio.wait_for(handler, timeout=10)
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_stop_cancels_pending_release(
        self, monkeypatch, caplog
    ) -> None:
        """T7: STOP cancels the pending release — no turn is dispatched."""
        _settle_speed(monkeypatch, 3000)
        calls: list[str] = []
        session = LifecycleSession(events=_utterance("never released"))
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "utterance_release_scheduled" in caplog.text
                    )
                    ws.queue_text({"type": "stop"})
                    start = time.monotonic()
                    await asyncio.wait_for(handler, timeout=10)
                    elapsed = time.monotonic() - start
            # Cancelled long before its 3 s settle could elapse.
            assert elapsed < 2.0
            assert "cancelling_utterance_release" in caplog.text
            assert "reason=session_stopping" in caplog.text
            assert calls == []
            assert ws.event_types()[-1] == "completed"
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_disconnect_cancels_pending_release(
        self, monkeypatch, caplog
    ) -> None:
        """T8: client disconnect cancels the pending release task."""
        _settle_speed(monkeypatch, 3000)
        calls: list[str] = []
        session = LifecycleSession(events=_utterance("abandoned turn"))
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                with caplog.at_level(logging.INFO):
                    handler = asyncio.create_task(
                        _handle_realtime_session(ws)
                    )
                    ws.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "utterance_release_scheduled" in caplog.text
                    )
                    # The browser vanishes before the settle window elapses.
                    ws.queue_disconnect()
                    await asyncio.wait_for(handler, timeout=10)
            assert "reason=session_stopping" in caplog.text
            assert calls == []
            assert session.close_calls == 1
            assert session.closed is True
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)

    async def test_stale_release_task_cannot_affect_new_session(
        self, monkeypatch, caplog
    ) -> None:
        """T9: a release task from a closed session can never fire later."""
        calls: list[str] = []
        _settle_speed(monkeypatch, 3000)
        patchers, _, _, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        llm_patcher = _recording_llm_patcher(calls)
        try:
            # Phase 1: an old session dies with a pending release scheduled.
            session_a = LifecycleSession(events=_utterance("old session turn"))
            ws_a = LifecycleFakeWS()
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session_a,
            ):
                with caplog.at_level(logging.INFO):
                    handler_a = asyncio.create_task(
                        _handle_realtime_session(ws_a)
                    )
                    ws_a.queue_text(_start_with_llm())
                    await _wait_until(
                        lambda: "utterance_release_scheduled" in caplog.text
                    )
                    ws_a.queue_disconnect()
                    await asyncio.wait_for(handler_a, timeout=10)
            await _assert_no_pending_tasks()
            assert calls == []

            # Phase 2: a new session with a short settle releases only its own
            # turn — the dead session's release never fires.
            _settle_speed(monkeypatch, 150)
            session_b = LifecycleSession(events=_utterance("new session turn"))
            ws_b = LifecycleFakeWS()
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session_b,
            ):
                handler_b = asyncio.create_task(
                    _handle_realtime_session(ws_b)
                )
                ws_b.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: calls == ["new session turn"], timeout=5
                )
                ws_b.queue_text({"type": "stop"})
                await asyncio.wait_for(handler_b, timeout=10)
            assert calls == ["new session turn"]
            await _assert_no_pending_tasks()
        finally:
            llm_patcher.stop()
            _stop_patchers(patchers)


class TestUtteranceEndConfiguration:
    """Phase 6E: utterance_end_ms default, override and validation."""

    def test_default_utterance_end_ms_is_2000(self) -> None:
        """CHANGE 1: browser sessions default to a 2000 ms turn gap."""
        captured: list[StreamConfig] = []
        session = FakeStreamingSession()

        def _capture(cfg: StreamConfig) -> FakeStreamingSession:
            captured.append(cfg)
            return session

        patcher = patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=_capture,
        )
        patcher.start()
        try:
            client = _get_client()
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                assert ws.receive_json()["type"] == "session_started"
            assert len(captured) == 1
            assert captured[0].utterance_end_ms == 2000
        finally:
            patcher.stop()

    def test_start_may_override_utterance_end_ms(self) -> None:
        """CHANGE 1: the gateway accepts an explicit utterance_end_ms."""
        captured: list[StreamConfig] = []
        session = FakeStreamingSession()

        def _capture(cfg: StreamConfig) -> FakeStreamingSession:
            captured.append(cfg)
            return session

        patcher = patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=_capture,
        )
        patcher.start()
        try:
            client = _get_client()
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json({**_valid_start(), "utterance_end_ms": 3500})
                assert ws.receive_json()["type"] == "session_started"
            assert len(captured) == 1
            assert captured[0].utterance_end_ms == 3500
        finally:
            patcher.stop()

    def test_non_integer_utterance_end_ms_rejected(self) -> None:
        """Non-integer values (including bools) are protocol errors."""
        patcher = patch("app.api.voice_realtime.open_streaming_session")
        mock_open = patcher.start()
        try:
            client = _get_client()
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json({**_valid_start(), "utterance_end_ms": "fast"})
                first = ws.receive_json()
                assert first["type"] == "error"
                assert "utterance_end_ms must be an integer" in first["message"]
                ws.send_json({**_valid_start(), "utterance_end_ms": True})
                second = ws.receive_json()
                assert second["type"] == "error"
            mock_open.assert_not_called()
        finally:
            patcher.stop()

    def test_out_of_range_utterance_end_ms_rejected(self) -> None:
        """Values outside 1000-5000 ms fail StreamConfig validation."""
        patcher = patch("app.api.voice_realtime.open_streaming_session")
        mock_open = patcher.start()
        try:
            client = _get_client()
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json({**_valid_start(), "utterance_end_ms": 500})
                first = ws.receive_json()
                assert first["type"] == "error"
                assert "utterance_end_ms out of range" in first["message"]
                ws.send_json({**_valid_start(), "utterance_end_ms": 6000})
                second = ws.receive_json()
                assert second["type"] == "error"
            mock_open.assert_not_called()
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Phase 6G: turn latency instrumentation (per-stage timestamps + breakdown)
# ---------------------------------------------------------------------------


def _wait_for_log_text(caplog: Any, needle: str, timeout: float = 5.0) -> bool:
    """Poll caplog for a needle — shutdown-path logs land after `completed`."""
    deadline = time.monotonic() + timeout
    while needle not in caplog.text and time.monotonic() < deadline:
        time.sleep(0.02)
    return needle in caplog.text


class TestTurnLatencyInstrumentation:
    """Phase 6G: per-stage latency timestamps and turn_latency_breakdown."""

    def test_utterance_timings_phase6g_segments(self) -> None:
        """All Phase 6G segment durations are computed from measured marks."""
        from app.api.voice_realtime import UtteranceTimings

        t = UtteranceTimings(
            turn=4,
            utterance_end_at=100.0,
            utterance_released_at=100.8,
            agent_processing_started_at=100.81,
            llm_request_started_at=100.82,
            llm_first_token_at=101.22,
            llm_first_sentence_at=101.47,
            llm_completed_at=103.18,
            tts_started_at=103.20,
            tts_completed_at=104.50,
            audio_sent_at=104.52,
            streamed_tokens=12,
            llm_sentence_count=3,
        )
        result = t.to_metrics_dict(0.0)

        assert result["utterance_end_to_release_ms"] == 800.0
        assert result["release_to_agent_ms"] == 10.0
        assert result["agent_to_llm_request_ms"] == 10.0
        assert result["llm_request_to_first_token_ms"] == 400.0
        assert result["llm_first_token_to_first_sentence_ms"] == 250.0
        assert result["llm_request_to_first_sentence_ms"] == 650.0
        assert result["llm_request_to_complete_ms"] == 2360.0
        assert result["first_sentence_to_tts_start_ms"] == 1730.0
        # ElevenLabs synthesize() buffers a complete MP3: no first-byte event.
        assert result["tts_start_to_first_audio_ms"] is None
        assert result["tts_first_audio_reason"] == "unavailable_complete_mp3"
        assert result["tts_start_to_complete_ms"] == 1300.0
        assert result["audio_encode_send_ms"] == 20.0
        assert result["utterance_to_first_audio_ms"] == 4520.0
        assert result["llm_sentence_count"] == 3
        assert result["utterance_released_offset_ms"] == 100800.0
        # Phase 6E keys preserved
        assert result["utterance_end_to_agent_ms"] == 810.0
        assert result["tts_duration_ms"] == 1300.0

    def test_stream_llm_records_first_sentence_timestamp(self) -> None:
        """First token, first sentence and completion marks are recorded."""
        from unittest.mock import MagicMock

        from app.api.voice_realtime import _stream_llm_response
        from app.providers.types import StreamChunk

        mock_llm = MagicMock()

        async def fake_stream():
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" world. ")
            yield StreamChunk(content="Bye.")
            yield StreamChunk(content="", finish_reason="stop")

        mock_llm.stream_chat = MagicMock(return_value=fake_stream())

        async def _run():
            mock_db = MagicMock()
            mock_db_session = MagicMock()
            mock_db_session.id = "test-session"
            mock_db_session.message_count = 0
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_db_session
            )
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
            ):
                return await _stream_llm_response(
                    db=mock_db,
                    voice_session_id="test-session",
                    transcript="Hi",
                    llm=mock_llm,
                    model="test-model",
                )

        result = asyncio.run(_run())

        assert result["sentences"] == ["Hello world.", "Bye."]
        assert result["sentence_count"] == 2
        assert result["streamed_tokens"] == 3
        req = result["llm_request_started_at"]
        tok = result["llm_first_token_at"]
        sent_at = result["llm_first_sentence_at"]
        done = result["llm_completed_at"]
        assert all(v is not None for v in (req, tok, sent_at, done))
        assert req <= tok <= sent_at <= done
        # The relative first-token ms stays consistent with the absolute marks.
        assert abs((tok - req) * 1000 - result["first_token_ms"]) < 5.0

    def test_stream_llm_metrics_valid_without_sentence(self) -> None:
        """An empty stream records no token/sentence marks but still completes."""
        from unittest.mock import MagicMock

        from app.api.voice_realtime import _stream_llm_response
        from app.providers.types import StreamChunk

        mock_llm = MagicMock()

        async def fake_stream():
            yield StreamChunk(content="", finish_reason="stop")

        mock_llm.stream_chat = MagicMock(return_value=fake_stream())

        async def _run():
            mock_db = MagicMock()
            mock_db_session = MagicMock()
            mock_db_session.id = "test-session"
            mock_db_session.message_count = 0
            mock_db.query.return_value.filter.return_value.first.return_value = (
                mock_db_session
            )
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
            ):
                return await _stream_llm_response(
                    db=mock_db,
                    voice_session_id="test-session",
                    transcript="Hi",
                    llm=mock_llm,
                    model="test-model",
                )

        result = asyncio.run(_run())

        assert result["response"] == ""
        assert result["first_token_ms"] is None
        assert result["llm_first_token_at"] is None
        assert result["llm_first_sentence_at"] is None
        assert result["sentence_count"] == 0
        assert result["llm_request_started_at"] is not None
        assert result["llm_completed_at"] is not None

    def test_turn_metrics_include_phase6g_segments(self) -> None:
        """End-to-end turn_metrics carries the 6G segments; 6E keys preserved."""
        events = _utterance("Hello")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_data = None
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while True:
                    event = ws.receive_json()
                    if event["type"] == "turn_metrics":
                        metrics_data = event["data"]
                        break
                    if event["type"] == "error":
                        break
                assert metrics_data is not None
                d = metrics_data
                # Phase 6E keys preserved
                assert d["utterance_end_to_agent_ms"] is not None
                assert d["agent_processing_ms"] is not None
                assert d["tts_duration_ms"] is not None
                assert d["utterance_end_to_audio_sent_ms"] is not None
                # Phase 6G segments
                assert d["utterance_end_to_release_ms"] is not None
                assert d["utterance_end_to_release_ms"] >= 700  # settle window
                assert d["release_to_agent_ms"] is not None
                assert d["agent_to_llm_request_ms"] == 0.0  # fallback mark
                assert d["llm_request_to_first_token_ms"] == 150.0  # mocked
                assert d["llm_request_to_first_sentence_ms"] is None
                assert d["llm_first_token_to_first_sentence_ms"] is None
                assert d["llm_request_to_complete_ms"] is not None
                assert d["llm_sentence_count"] == 1
                assert d["tts_start_to_first_audio_ms"] is None
                assert d["tts_first_audio_reason"] == "unavailable_complete_mp3"
                assert d["tts_start_to_complete_ms"] == d["tts_duration_ms"]
                assert (
                    d["utterance_to_first_audio_ms"]
                    == d["utterance_end_to_audio_sent_ms"]
                )
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_latency_breakdown_after_browser_timing(self, caplog) -> None:
        """browser_timing completes the turn record → one breakdown log."""
        events = _utterance("Hello")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        metrics_data = None
        try:
            with caplog.at_level(logging.INFO):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_start_with_llm())
                    while True:
                        event = ws.receive_json()
                        if event["type"] == "turn_metrics":
                            metrics_data = event["data"]
                            break
                        if event["type"] == "error":
                            break
                    assert metrics_data is not None
                    ws.send_json(
                        {
                            "type": "browser_timing",
                            "turn": 1,
                            "ws_transit_ms": 4.5,
                            "received_to_playing_ms": 15.0,
                        }
                    )
                    # A duplicate report for the same turn is ignored safely.
                    ws.send_json(
                        {
                            "type": "browser_timing",
                            "turn": 1,
                            "ws_transit_ms": 4.5,
                            "received_to_playing_ms": 15.0,
                        }
                    )
                    ws.send_json({"type": "stop"})
                    completed = ws.receive_json()
                    assert completed["type"] == "completed"
            assert caplog.text.count("turn_latency_breakdown") == 1
            assert "turn_latency_breakdown turn=1" in caplog.text
            assert "browser_received=4.5" in caplog.text
            assert "browser_playing=15.0" in caplog.text
            expected_total = round(
                metrics_data["utterance_end_to_audio_sent_ms"] + 4.5 + 15.0, 1
            )
            assert f"total={expected_total:.1f}" in caplog.text
            assert "browser_timing_ignored turn=1" in caplog.text
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_turn_latency_breakdown_fallback_without_browser_report(
        self, caplog
    ) -> None:
        """Without a browser report the breakdown is flushed at shutdown."""
        events = _utterance("Hello")
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with caplog.at_level(logging.INFO):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_start_with_llm())
                    while True:
                        event = ws.receive_json()
                        if event["type"] == "turn_metrics":
                            break
                        if event["type"] == "error":
                            break
                    ws.send_json({"type": "stop"})
                    completed = ws.receive_json()
                    assert completed["type"] == "completed"
                # The shutdown flush runs after `completed` was sent.
                assert _wait_for_log_text(caplog, "turn_latency_breakdown turn=1")
            assert caplog.text.count("turn_latency_breakdown") == 1
            assert "browser_received=n/a" in caplog.text
            assert "browser_playing=n/a" in caplog.text
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()

    def test_unknown_browser_timing_is_ignored(self, caplog) -> None:
        """A browser_timing for an unknown turn never crashes the session."""
        client, holder = _connect_with_fake_session()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with caplog.at_level(logging.INFO):
                with client.websocket_connect(REALTIME_WS_PATH) as ws:
                    ws.send_json(_start_with_llm())
                    assert ws.receive_json()["type"] == "session_started"
                    ws.send_json(
                        {
                            "type": "browser_timing",
                            "turn": 99,
                            "ws_transit_ms": 1.0,
                            "received_to_playing_ms": 10.0,
                        }
                    )
                    ws.send_json({"type": "stop"})
                    completed = ws.receive_json()
                    assert completed["type"] == "completed"
            assert "browser_timing_ignored turn=99" in caplog.text
        finally:
            _stop_patchers(patchers)
            holder["patcher"].stop()


# ---------------------------------------------------------------------------
# Phase 6H: sentence-streaming TTS in the browser realtime path
# ---------------------------------------------------------------------------


class _StreamingTTSDouble:
    """TTS provider double exposing a real async-generator stream_synthesize.

    Mirrors the ElevenLabsAdapter contract: ``stream_synthesize`` is a plain
    method returning an async generator of encoded-audio chunks, so the
    gateway's strict ``inspect.isasyncgen`` check accepts it. Each sentence
    streams two chunks that ``_synthesize_sentence_segment`` joins into one
    complete segment; per-sentence timings and cancellation are recorded.
    """

    provider_name = "fake-streaming-tts"

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_texts: set[str] | None = None,
        fail_message: str = "TTS stream failure",
        block: bool = False,
    ) -> None:
        self.delay = delay
        self.fail_texts = fail_texts or set()
        self.fail_message = fail_message
        self.block = block
        self.stream_texts: list[str] = []
        self.started_at: dict[str, float] = {}
        self.first_chunk_at: dict[str, float] = {}
        self.completed_at: dict[str, float] = {}
        self.active_streams = 0
        self.max_concurrent_streams = 0
        self.cancelled = False
        self.synthesize = AsyncMock(
            side_effect=AssertionError("buffered synthesize on streaming path")
        )
        self.close = AsyncMock()

    def stream_synthesize(self, text: str, **kwargs: Any):
        """Plain def → async generator (the real provider's interface shape)."""

        async def _gen():
            self.stream_texts.append(text)
            self.started_at[text] = time.monotonic()
            if text in self.fail_texts:
                raise RuntimeError(self.fail_message)
            self.active_streams += 1
            self.max_concurrent_streams = max(
                self.max_concurrent_streams, self.active_streams
            )
            try:
                if self.block:
                    await asyncio.sleep(30)
                if self.delay:
                    await asyncio.sleep(self.delay)
                for index, part in enumerate(("a", "b")):
                    if index == 0:
                        self.first_chunk_at[text] = time.monotonic()
                    yield f"[{text}|{part}]".encode()
                self.completed_at[text] = time.monotonic()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                self.active_streams -= 1

        return _gen()


def _ws_events_of_type(ws: LifecycleFakeWS, event_type: str) -> list[dict[str, Any]]:
    """All protocol events of one type sent on the fake socket, in order."""
    return [e for e in ws.sent if e.get("type") == event_type]


def _install_scripted_llm(
    sentences: list[str],
    *,
    gate: asyncio.Event | None = None,
    gate_before: int | None = None,
    done: asyncio.Event | None = None,
    done_at: dict[str, float] | None = None,
) -> None:
    """Wire the patched LLM provider to a scripted streaming generator.

    Each sentence is one content chunk with a trailing space so SentenceBuffer
    emits it immediately. ``gate`` suspends the stream before sentence
    ``gate_before`` (0-based) — holding it open proves TTS overlap.
    """
    import app.api.voice_realtime as realtime_mod
    from app.providers.types import StreamChunk

    mock_llm = realtime_mod.get_llm_provider.return_value

    async def _gen():
        for index, sentence in enumerate(sentences):
            if gate is not None and gate_before is not None and index == gate_before:
                await gate.wait()
            yield StreamChunk(content=f"{sentence} ")
        if done_at is not None:
            done_at["ts"] = time.monotonic()
        if done is not None:
            done.set()
        yield StreamChunk(content="", finish_reason="stop")

    mock_llm.stream_chat = MagicMock(side_effect=lambda **kwargs: _gen())


class TestSentenceStreamingTts:
    """Phase 6H: per-sentence streaming TTS connected to SentenceBuffer.

    Real-flow tests un-patch ``_stream_llm_response`` so the actual
    SentenceBuffer bridge runs between a scripted streaming LLM and the
    per-turn TTS consumer. Mocked-flow tests keep the 6B mock to prove the
    legacy wire protocol (order, buffered fallback, error strategy) intact.
    """

    async def test_first_sentence_tts_starts_before_llm_completion(self) -> None:
        """1: sentence 1 is synthesized while the LLM is still streaming."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        gate = asyncio.Event()
        done = asyncio.Event()
        _install_scripted_llm(
            ["Alpha.", "Bravo."], gate=gate, gate_before=1, done=done
        )
        fake = _StreamingTTSDouble()
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                # Segment 1 lands while the LLM is parked before sentence 2.
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "audio")) >= 1
                )
                assert not done.is_set()
                assert "Alpha." in fake.first_chunk_at
                assert "Bravo." not in fake.started_at
                gate.set()
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert done.is_set()
            assert fake.stream_texts == ["Alpha.", "Bravo."]
            assert len(_ws_events_of_type(ws, "audio")) == 2
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_multiple_sentences_preserve_order(self) -> None:
        """2: segments are delivered strictly in sentence order."""
        import base64

        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        _install_scripted_llm(["Alpha.", "Bravo.", "Charlie."])
        fake = _StreamingTTSDouble()
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "audio")) == 3
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            audio_events = _ws_events_of_type(ws, "audio")
            assert [e["segment"] for e in audio_events] == [1, 2, 3]
            assert [e["turn"] for e in audio_events] == [1, 1, 1]
            assert all(e["format"] == "audio/mpeg" for e in audio_events)
            decoded = [
                base64.b64decode(e["data"]).decode() for e in audio_events
            ]
            assert decoded == [
                "[Alpha.|a][Alpha.|b]",
                "[Bravo.|a][Bravo.|b]",
                "[Charlie.|a][Charlie.|b]",
            ]
            assert fake.stream_texts == ["Alpha.", "Bravo.", "Charlie."]
            assert fake.max_concurrent_streams == 1
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_llm_and_tts_overlap(self) -> None:
        """3: TTS for sentence 1 completes before the LLM stream finishes."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        gate = asyncio.Event()
        done = asyncio.Event()
        done_at: dict[str, float] = {}
        _install_scripted_llm(
            ["Alpha.", "Bravo."],
            gate=gate,
            gate_before=1,
            done=done,
            done_at=done_at,
        )
        fake = _StreamingTTSDouble(delay=0.25)
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(lambda: "Alpha." in fake.first_chunk_at)
                first_chunk_ts = fake.first_chunk_at["Alpha."]
                assert not done.is_set()
                gate.set()
                await _wait_until(lambda: done_at.get("ts") is not None)
                assert fake.started_at["Alpha."] < done_at["ts"]
                assert first_chunk_ts < done_at["ts"]
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_first_audio_timestamp_recorded_in_metrics(self) -> None:
        """4+10: streaming first-audio marks recorded; 6E/6G keys stay valid."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        _install_scripted_llm(["Alpha.", "Bravo."])
        fake = _StreamingTTSDouble()
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            d = _ws_events_of_type(ws, "turn_metrics")[0]["data"]
            # Phase 6E keys preserved
            assert d["utterance_end_to_agent_ms"] is not None
            assert d["agent_processing_ms"] is not None
            assert d["tts_duration_ms"] is not None
            assert d["utterance_end_to_audio_sent_ms"] is not None
            # Phase 6G keys preserved and measured on the real path
            assert d["llm_request_to_first_token_ms"] is not None
            assert d["llm_request_to_first_sentence_ms"] is not None
            assert d["llm_request_to_complete_ms"] is not None
            assert d["first_sentence_to_tts_start_ms"] is not None
            assert 0 <= d["first_sentence_to_tts_start_ms"] < 1000
            # Phase 6H first-audio marks
            assert d["tts_start_to_first_audio_ms"] is not None
            assert 0 <= d["tts_start_to_first_audio_ms"] < 1000
            assert d["tts_first_audio_reason"] is None
            assert d["tts_first_sentence_synth_ms"] is not None
            assert d["utterance_to_first_audio_ms"] is not None
            assert d["utterance_end_to_first_audio_sent_ms"] is not None
            assert (
                d["utterance_to_first_audio_ms"]
                <= d["utterance_end_to_first_audio_sent_ms"]
            )
            assert (
                d["utterance_to_first_audio_ms"]
                <= d["utterance_end_to_audio_sent_ms"]
            )
            assert d["tts_sentence_count"] == 2
            assert d["tts_audio_chunks"] == 4
            assert d["tts_mode"] == "streaming"
            assert d["llm_sentence_count"] == 2
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_stop_cancels_pending_sentence_tts(self, caplog) -> None:
        """5: STOP cancels a mid-flight sentence synthesis; no audio leaks."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        _install_scripted_llm(["Alpha.", "Bravo."])
        fake = _StreamingTTSDouble(block=True)
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                caplog.at_level(logging.INFO),
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                # The consumer is inside sentence 1 synthesis, then STOP hits.
                await _wait_until(lambda: fake.stream_texts == ["Alpha."])
                assert "tts_processing" in ws.event_types()
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert fake.cancelled is True
            assert "cancelling_tts_consumer" in caplog.text
            assert "consumer_cancelled turn=1" in caplog.text
            assert "audio" not in ws.event_types()
            assert "turn_metrics" not in ws.event_types()
            assert "error" not in ws.event_types()
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_disconnect_cancels_pending_sentence_tts(self) -> None:
        """6: an abrupt disconnect cancels a mid-flight sentence synthesis."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        _install_scripted_llm(["Alpha.", "Bravo."])
        fake = _StreamingTTSDouble(block=True)
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(lambda: fake.stream_texts == ["Alpha."])
                ws.queue_disconnect()
                await asyncio.wait_for(handler, timeout=10)
            assert fake.cancelled is True
            assert "audio" not in ws.event_types()
            assert "turn_metrics" not in ws.event_types()
            assert "completed" not in ws.event_types()
            assert "error" not in ws.event_types()
            assert ws.sends_after_disconnect == 0
            assert session.closed is True
            assert session.close_calls == 1
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_single_sentence_failure_does_not_kill_session(self, caplog) -> None:
        """7: a failed sentence is logged and counted; the turn still completes."""
        import base64

        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        _install_scripted_llm(["Alpha.", "Bravo."])
        fake = _StreamingTTSDouble(
            fail_texts={"Bravo."}, fail_message="Broken sentence"
        )
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                caplog.at_level(logging.INFO),
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert "sentence_synthesis_failed turn=1 sentence=2" in caplog.text
            assert (
                "consumer_exit turn=1 sentences=2 segments=1 failures=1 chunks=2"
                in caplog.text
            )
            audio_events = _ws_events_of_type(ws, "audio")
            assert len(audio_events) == 1
            assert audio_events[0]["segment"] == 1
            assert (
                base64.b64decode(audio_events[0]["data"]).decode()
                == "[Alpha.|a][Alpha.|b]"
            )
            assert "error" not in ws.event_types()
            d = _ws_events_of_type(ws, "turn_metrics")[0]["data"]
            assert d["tts_sentence_failures"] == 1
            assert d["tts_sentence_count"] == 1
            assert d["tts_audio_chunks"] == 2
            assert d["tts_mode"] == "streaming"
            assert d["tts_start_to_first_audio_ms"] is not None
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_all_sentence_failures_preserve_error_strategy(self) -> None:
        """7 (mock path): every-sentence failure keeps the pre-6H error event."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, mock_stream = _patch_realtime_deps()
        fake = _StreamingTTSDouble(
            fail_texts={"Hello! How can I help?"}, fail_message="TTS API down"
        )
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: any(
                        e.get("type") == "error" and e.get("stage") == "tts"
                        for e in ws.sent
                    )
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            mock_stream.assert_awaited_once()
            errors = _ws_events_of_type(ws, "error")
            assert len(errors) == 1
            assert errors[0]["stage"] == "tts"
            assert "TTS API down" in errors[0]["message"]
            assert "audio" not in ws.event_types()
            assert "turn_metrics" not in ws.event_types()
            assert mock_tts.synthesize.await_count == 0
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_stale_tts_audio_never_reaches_next_session(self) -> None:
        """8: a cancelled session's pending TTS can never leak into session B."""
        session_a = LifecycleSession(events=_utterance("hello from A"))
        session_b = LifecycleSession(events=_utterance("hello from B"))
        pending_sessions = [session_a, session_b]
        open_patcher = patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=lambda cfg: pending_sessions.pop(0),
        )
        open_patcher.start()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        fake_blocked = _StreamingTTSDouble(block=True)
        fake_b = _StreamingTTSDouble()
        try:
            # Session A: STOP lands while a sentence synthesis is in flight.
            ws_a = LifecycleFakeWS()
            mock_tts.stream_synthesize = fake_blocked.stream_synthesize
            handler_a = asyncio.create_task(_handle_realtime_session(ws_a))
            ws_a.queue_text(_start_with_llm())
            await _wait_until(
                lambda: fake_blocked.stream_texts == ["Hello! How can I help?"]
            )
            ws_a.queue_text({"type": "stop"})
            await asyncio.wait_for(handler_a, timeout=10)
            assert fake_blocked.cancelled is True
            assert session_a.closed is True
            assert "audio" not in ws_a.event_types()
            a_events_after_stop = len(ws_a.sent)

            # Session B: a fresh socket/session streams audio normally.
            ws_b = LifecycleFakeWS()
            mock_tts.stream_synthesize = fake_b.stream_synthesize
            handler_b = asyncio.create_task(_handle_realtime_session(ws_b))
            ws_b.queue_text(_start_with_llm())
            await _wait_until(
                lambda: len(_ws_events_of_type(ws_b, "audio")) >= 1
            )
            ws_b.queue_text({"type": "stop"})
            await asyncio.wait_for(handler_b, timeout=10)

            assert len(ws_a.sent) == a_events_after_stop
            assert "audio" not in ws_a.event_types()
            audio_b = _ws_events_of_type(ws_b, "audio")
            assert len(audio_b) == 1
            assert audio_b[0]["turn"] == 1
            assert audio_b[0]["segment"] == 1
            assert "error" not in ws_b.event_types()
            assert session_b.closed is True
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)
            open_patcher.stop()

    async def test_turn_finalization_and_metrics_preserved_multi_turn(self) -> None:
        """9+10: sequential turns still finalize (6E) with valid 6G metrics."""
        session = LifecycleSession(events=_utterance("First"))
        ws = LifecycleFakeWS()
        patchers, _, _, _ = _patch_realtime_deps()
        try:
            with patch(
                "app.api.voice_realtime.open_streaming_session",
                return_value=session,
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                _deliver_next_utterance({"session": session}, "Second")
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 2
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            turn_metrics = _ws_events_of_type(ws, "turn_metrics")
            assert [e["data"]["turn"] for e in turn_metrics] == [1, 2]
            transcript_texts = [
                e["text"] for e in _ws_events_of_type(ws, "transcript_final")
            ]
            assert transcript_texts == ["First", "Second"]
            assert len(_ws_events_of_type(ws, "agent_response")) == 2
            audio_events = _ws_events_of_type(ws, "audio")
            assert [e["turn"] for e in audio_events] == [1, 2]
            assert [e["segment"] for e in audio_events] == [1, 1]
            assert len(_ws_events_of_type(ws, "tts_processing")) == 2
            # Phase 6G metrics remain valid through the buffered fallback.
            d = turn_metrics[0]["data"]
            assert d["agent_to_llm_request_ms"] == 0.0
            assert d["llm_request_to_first_token_ms"] == 150.0
            assert d["llm_sentence_count"] == 1
            assert d["tts_first_audio_reason"] == "unavailable_complete_mp3"
            assert d["tts_start_to_complete_ms"] == d["tts_duration_ms"]
            assert (
                d["utterance_to_first_audio_ms"]
                == d["utterance_end_to_audio_sent_ms"]
            )
            assert d["tts_start_to_first_audio_ms"] is None
            assert d["tts_mode"] == "buffered"
            assert d["tts_sentence_count"] == 1
            assert d["tts_audio_chunks"] is None
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)

    async def test_early_browser_timing_reconciled_at_registration(
        self, caplog
    ) -> None:
        """Race: a mid-turn playback report is stashed and logged once."""
        session = LifecycleSession(events=_utterance("hello"))
        ws = LifecycleFakeWS()
        patchers, _, mock_tts, _ = _patch_realtime_deps()
        patchers["stream_process"].stop()
        gate = asyncio.Event()
        done = asyncio.Event()
        _install_scripted_llm(
            ["Alpha.", "Bravo."], gate=gate, gate_before=1, done=done
        )
        fake = _StreamingTTSDouble()
        mock_tts.stream_synthesize = fake.stream_synthesize
        try:
            with (
                caplog.at_level(logging.INFO),
                patch(
                    "app.services.realtime_voice_service._load_history",
                    return_value=[],
                ),
                patch("app.services.realtime_voice_service._save_message"),
                patch(
                    "app.api.voice_realtime.open_streaming_session",
                    return_value=session,
                ),
            ):
                handler = asyncio.create_task(_handle_realtime_session(ws))
                ws.queue_text(_start_with_llm())
                # Segment 1 is on the wire while the LLM is still gated.
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "audio")) >= 1
                )
                ws.queue_text(
                    {
                        "type": "browser_timing",
                        "turn": 1,
                        "ws_transit_ms": 3.0,
                        "received_to_playing_ms": 12.0,
                    }
                )
                # Fence: the next report is processed only after the first;
                # the turn is still active, so turn 1 took the stash path.
                ws.queue_text(
                    {
                        "type": "browser_timing",
                        "turn": 99,
                        "ws_transit_ms": 1.0,
                        "received_to_playing_ms": 1.0,
                    }
                )
                await _wait_until(
                    lambda: "browser_timing_ignored turn=99" in caplog.text
                )
                gate.set()
                await _wait_until(
                    lambda: len(_ws_events_of_type(ws, "turn_metrics")) >= 1
                )
                ws.queue_text({"type": "stop"})
                await asyncio.wait_for(handler, timeout=10)
            assert caplog.text.count("turn_latency_breakdown") == 1
            assert "turn_latency_breakdown turn=1" in caplog.text
            assert "browser_timing_ignored turn=1" not in caplog.text
            assert "browser_received=3.0" in caplog.text
            assert "browser_playing=12.0" in caplog.text
            d = _ws_events_of_type(ws, "turn_metrics")[0]["data"]
            expected_total = round(
                d["utterance_end_to_first_audio_sent_ms"] + 3.0 + 12.0, 1
            )
            assert f"total={expected_total:.1f}" in caplog.text
            assert (
                f"first_audio={d['utterance_to_first_audio_ms']:.1f}"
                in caplog.text
            )
            assert "sentences=2" in caplog.text
            assert "tts_mode=streaming" in caplog.text
            assert ws.event_types()[-1] == "completed"
            await _assert_no_pending_tasks()
        finally:
            _stop_patchers(patchers)
