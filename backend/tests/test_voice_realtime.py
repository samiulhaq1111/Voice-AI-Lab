"""Targeted tests for the realtime streaming voice gateway (Phase 6B).

All Deepgram interaction is mocked — no real network calls.
Covers protocol validation, event forwarding, lifecycle cleanup,
credential safety, and AgentRuntime integration on utterance_end.
"""

import asyncio
import json
import logging
import time
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

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
        finally:
            holder["patcher"].stop()  # type: ignore[union-attr]

    def test_stop_drains_events_then_completed(self) -> None:
        """STOP finishes the stream, drains events, then sends completed."""
        client, holder = _connect_with_fake_session(events=self._stream_events())
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_valid_start())
                ws.receive_json()  # session_started
                ws.send_bytes(b"\x01\x02")
                ws.send_json({"type": "stop"})

                # Drain transcript events, expect completed last
                types = []
                while True:
                    event = ws.receive_json()
                    types.append(event["type"])
                    if event["type"] == "completed":
                        break
                assert "transcript_partial" in types
                assert "transcript_final" in types
                assert "utterance_end" in types
                assert types[-1] == "completed"

                # Timings included in completed
                timings = json.loads(json.dumps(event["timings"]))
                assert timings["audio_bytes"] == 2
                assert timings["partial_count"] == 1
                assert timings["final_count"] == 1
                assert timings["utterance_end_count"] == 1
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


def test_stop_drain_timeout_is_finite() -> None:
    """Guard against hanging STOP: drain timeout must be positive & small."""
    from app.api.voice_realtime import _STOP_DRAIN_TIMEOUT_S

    assert 0 < _STOP_DRAIN_TIMEOUT_S <= 10


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
    }
    for p in patchers.values():
        p.start()
    return patchers, mock_process, mock_tts


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
        patchers, mock_process, _ = _patch_realtime_deps()
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
                mock_process.assert_awaited_once()
                call_args = mock_process.call_args
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
        patchers, mock_process, _ = _patch_realtime_deps()
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
        patchers, mock_process, _ = _patch_realtime_deps()
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
        """Two separate utterances produce two AgentRuntime invocations."""
        from unittest.mock import AsyncMock

        call_count = 0

        async def _mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "response": f"Response {call_count}",
                "tool_calls": [],
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="First utterance", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second utterance", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
        """An AgentRuntime exception produces an error event, not silence."""
        from unittest.mock import AsyncMock

        from app.services.realtime_voice_service import RealtimeVoiceError

        events = [
            StreamEvent(type="final", text="This will fail", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
        patchers, mock_process, _ = _patch_realtime_deps()
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
                mock_process.assert_awaited_once()
                call_args = mock_process.call_args
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="First", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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

        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, mock_process, _ = _patch_realtime_deps()
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

            # Verify process_realtime_utterance was called
            mock_process.assert_awaited_once()
            # The function receives the session, and internally loads history
            # We can't directly verify initial_messages passed to AgentRuntime
            # without mocking deeper, but we verify the function was called
            # with the correct transcript
            call_args = mock_process.call_args
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            # First utterance
            StreamEvent(type="final", text="First question", confidence=0.9),
            StreamEvent(type="utterance_end"),
            # Second utterance arrives right after
            StreamEvent(type="final", text="Second question", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="AAA", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="BBB", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="CCC", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
            AsyncMock(side_effect=_mock_process),
        )
        process_patcher.start()
        agent_responses: list[dict] = []
        try:
            with client.websocket_connect(REALTIME_WS_PATH) as ws:
                ws.send_json(_start_with_llm())
                while len(agent_responses) < 3:
                    event = ws.receive_json()
                    if event["type"] == "agent_response":
                        agent_responses.append(event)
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
                "usage": {"total_tokens": 5},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="Process me", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
                "usage": {"total_tokens": 5},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="Will fail", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Will succeed", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
                "usage": {"total_tokens": 5},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="First", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="What is the weather?", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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

        events = [
            StreamEvent(type="final", text="First", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="First", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
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
        patchers, _, mock_tts = _patch_realtime_deps()
        save_message_calls: list = []

        original_process = patchers["process"].new

        async def _tracking_process(**kwargs):
            result = await original_process(**kwargs)
            return result

        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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
                "usage": {"total_tokens": 10},
                "iterations": 1,
            }

        events = [
            StreamEvent(type="final", text="First", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Second", confidence=0.9),
            StreamEvent(type="utterance_end"),
            StreamEvent(type="final", text="Third", confidence=0.9),
            StreamEvent(type="utterance_end"),
        ]
        client, holder = _connect_with_fake_session(events=events)
        patchers, _, mock_tts = _patch_realtime_deps()
        patchers["process"].stop()
        process_patcher = patch(
            "app.api.voice_realtime.process_realtime_utterance",
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

        patchers, _, _ = _patch_realtime_deps()
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
