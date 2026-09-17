"""Unit tests for DeepgramStreamingSession message translation (Phase 6A fix).

Uses REAL Deepgram streaming message shapes captured from live sessions
(verified via the provider-path diagnostic). All websocket interaction is
mocked — no real API calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from websockets.exceptions import ConnectionClosed

from app.providers.stt.deepgram_streaming import DeepgramStreamingSession
from app.providers.stt.streaming import StreamConfig

# Real Deepgram message shapes (captured live)
PARTIAL_RESULTS = {
    "type": "Results",
    "channel_index": [0, 1],
    "duration": 1.04,
    "start": 0.0,
    "is_final": False,
    "speech_final": False,
    "channel": {
        "alternatives": [
            {"transcript": "This is a test of", "confidence": 0.87}
        ],
        "language": "en",
    },
}

FINAL_RESULTS = {
    "type": "Results",
    "channel_index": [0, 1],
    "duration": 4.22,
    "start": 0.0,
    "is_final": True,
    "speech_final": True,
    "channel": {
        "alternatives": [
            {
                "transcript": "This is a test of the real time streaming pipeline.",
                "confidence": 0.99,
            }
        ],
        "language": "en",
    },
}

SILENCE_RESULTS = {
    "type": "Results",
    "is_final": False,
    "speech_final": False,
    "channel": {"alternatives": [{"transcript": "", "confidence": 0.0}]},
}

NO_ALTERNATIVES_RESULTS = {
    "type": "Results",
    "is_final": False,
    "channel": {"alternatives": []},
}

UTTERANCE_END = {"type": "UtteranceEnd", "channel": [0, 1], "last_word": ""}

ERROR_MESSAGE = {
    "type": "Error",
    "code": "BAD_REQUEST",
    "description": "Invalid language code",
}

METADATA_MESSAGE = {
    "type": "Metadata",
    "transaction_key": "deprecated",
    "request_id": "abc-123",
    "sha256": "deadbeef",
}

CLOSE_STREAM = {"type": "CloseStream"}


def _session() -> DeepgramStreamingSession:
    return DeepgramStreamingSession(StreamConfig())


def _translate(payload: dict):
    return _session()._translate_message(json.dumps(payload))


# ---------------------------------------------------------------------------
# Results translation (the Phase 6A bug path)
# ---------------------------------------------------------------------------


class TestResultsTranslation:
    def test_partial_transcript(self) -> None:
        event = _translate(PARTIAL_RESULTS)
        assert event is not None
        assert event.type == "partial"
        assert event.text == "This is a test of"
        assert abs(event.confidence - 0.87) < 1e-6

    def test_final_transcript(self) -> None:
        event = _translate(FINAL_RESULTS)
        assert event is not None
        assert event.type == "final"
        assert event.text == "This is a test of the real time streaming pipeline."
        assert event.metadata["speech_final"] is True

    def test_silence_results_return_none(self) -> None:
        """Empty-transcript Results (silence ticks) must not produce events."""
        assert _translate(SILENCE_RESULTS) is None

    def test_no_alternatives_returns_none(self) -> None:
        assert _translate(NO_ALTERNATIVES_RESULTS) is None

    def test_utterance_end(self) -> None:
        event = _translate(UTTERANCE_END)
        assert event is not None
        assert event.type == "utterance_end"

    def test_error_message(self) -> None:
        event = _translate(ERROR_MESSAGE)
        assert event is not None
        assert event.type == "error"
        assert "Invalid language code" in event.text

    def test_metadata_ignored(self) -> None:
        assert _translate(METADATA_MESSAGE) is None

    def test_close_stream_ignored(self) -> None:
        assert _translate(CLOSE_STREAM) is None

    def test_non_json_ignored(self) -> None:
        assert _session()._translate_message("not json at all") is None

    def test_unknown_type_ignored(self) -> None:
        assert _translate({"type": "SomethingNew"}) is None


# ---------------------------------------------------------------------------
# receive() loop with mocked websocket
# ---------------------------------------------------------------------------


class TestReceiveLoop:
    async def test_receive_translates_text_frame(self) -> None:
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(return_value=json.dumps(PARTIAL_RESULTS))
        event = await session.receive()
        assert event is not None
        assert event.type == "partial"

    async def test_receive_ignores_binary_frame(self) -> None:
        """A stray binary frame alone produces no event and no stream end."""
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=[
                b"\x00\x01\x02",
                ConnectionClosed(rcvd=None, sent=None),
            ]
        )
        assert await session.receive() is None

    async def test_receive_skips_metadata_then_returns_event(self) -> None:
        """Regression: a Metadata message must NOT end the stream.

        Root cause of the Phase 6A 'no transcripts' bug — the pump treated
        any None (ignorable message) as stream end.
        """
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=[
                json.dumps(METADATA_MESSAGE),
                json.dumps(PARTIAL_RESULTS),
            ]
        )
        event = await session.receive()
        assert event is not None
        assert event.type == "partial"

    async def test_receive_skips_silence_tick_then_returns_event(self) -> None:
        """Regression: empty-transcript Results (ambient noise) must NOT end
        the stream — this fired 'Provider stream ended unexpectedly' in the
        reported manual test about 2s after session start.
        """
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=[
                json.dumps(SILENCE_RESULTS),
                json.dumps(FINAL_RESULTS),
            ]
        )
        event = await session.receive()
        assert event is not None
        assert event.type == "final"

    async def test_receive_skips_binary_then_returns_event(self) -> None:
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=[b"\x00\x01", json.dumps(PARTIAL_RESULTS)]
        )
        event = await session.receive()
        assert event is not None
        assert event.type == "partial"

    async def test_receive_skips_mixed_ignorable_sequence(self) -> None:
        """A realistic burst: silence, metadata, binary, then a real result."""
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=[
                json.dumps(SILENCE_RESULTS),
                json.dumps(METADATA_MESSAGE),
                b"\x00",
                json.dumps(UTTERANCE_END),
            ]
        )
        event = await session.receive()
        assert event is not None
        assert event.type == "utterance_end"

    async def test_receive_returns_none_on_connection_closed(self) -> None:
        session = _session()
        session._ws = MagicMock()
        session._ws.recv = AsyncMock(
            side_effect=ConnectionClosed(rcvd=None, sent=None)
        )
        assert await session.receive() is None

    async def test_receive_returns_none_when_not_started(self) -> None:
        session = _session()  # _ws is None
        assert await session.receive() is None

    async def test_receive_returns_none_after_close(self) -> None:
        session = _session()
        session._closed = True
        assert await session.receive() is None


# ---------------------------------------------------------------------------
# send_audio counters (periodic deepgram_audio_sent log source)
# ---------------------------------------------------------------------------


class TestSendAudioCounters:
    async def test_send_audio_counts_chunks_and_bytes(self) -> None:
        session = _session()
        session._ws = MagicMock()
        session._ws.send = AsyncMock()
        await session.send_audio(b"\x01" * 10)
        await session.send_audio(b"\x02" * 20)
        assert session._chunks_sent == 2
        assert session._bytes_sent == 30

    async def test_send_audio_noop_after_finish(self) -> None:
        session = _session()
        session._ws = MagicMock()
        session._ws.send = AsyncMock()
        session._finished = True
        await session.send_audio(b"\x01")
        assert session._chunks_sent == 0
        session._ws.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# URL construction (safe — no credentials in URL)
# ---------------------------------------------------------------------------


class TestStreamUrl:
    def test_url_contains_streaming_params(self) -> None:
        session = _session()
        url = session._build_url()
        assert url.startswith("wss://api.deepgram.com/v1/listen?")
        for param in (
            "model=nova-3",
            "encoding=linear16",
            "sample_rate=16000",
            "channels=1",
            "interim_results=true",
            "endpointing=300",
            "utterance_end_ms=1000",
            "smart_format=true",
        ):
            assert param in url

    def test_url_contains_no_api_key(self) -> None:
        session = _session()
        assert "api_key" not in session._build_url()
