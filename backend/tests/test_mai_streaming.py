"""Focused tests for the MAI streaming adapter's commit gating.

Covers the minimum above-threshold duration gate: a brief audio excursion
must not trigger a commit (MAI hallucinates on committed mostly-silent
audio), while a sustained utterance followed by the endpointing silence
window still commits exactly once and resets the tracking state.
"""

import asyncio
import json
import struct

import pytest

from app.providers.stt.mai_streaming import MaiTranscribeStreamingSession
from app.providers.stt.streaming import StreamConfig

_SAMPLE_RATE = 16000  # adapter input rate — no resampler in these tests
_FRAME_MS = 50
_LOUD_AMPLITUDE = 1000  # constant level → RMS 1000 ≥ threshold (250)


def _frame(duration_ms: int, amplitude: int) -> bytes:
    """One PCM16 mono frame at 16 kHz with a constant sample level."""
    count = int(_SAMPLE_RATE * duration_ms / 1000)
    return struct.pack(f"<{count}h", *([amplitude] * count))


class _RecordingWebSocket:
    """Minimal stand-in that records outbound JSON frames (no network)."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _make_session(
    monkeypatch: pytest.MonkeyPatch,
) -> MaiTranscribeStreamingSession:
    from app.core.config import settings

    monkeypatch.setattr(settings, "mai_stt_api_key", "test-mai-key", raising=False)
    monkeypatch.setattr(
        settings, "mai_stt_resource_url", "https://test.example.com", raising=False
    )
    return MaiTranscribeStreamingSession(StreamConfig(sample_rate=_SAMPLE_RATE))


def test_brief_excursion_does_not_trigger_commit(monkeypatch) -> None:
    """A single 50 ms above-threshold frame must not arm a commit."""
    session = _make_session(monkeypatch)
    session._ws = _RecordingWebSocket()

    assert session._track_endpointing(_frame(_FRAME_MS, _LOUD_AMPLITUDE)) == ""
    # One second of trailing silence — far past the 300 ms endpoint window.
    for _ in range(20):
        assert session._track_endpointing(_frame(_FRAME_MS, 0)) == ""
    # The excursion armed speech detection but stays below the minimum.
    assert session._has_speech is True
    assert session._speech_ms == pytest.approx(float(_FRAME_MS))


def test_sustained_utterance_commits_after_endpoint_silence(monkeypatch) -> None:
    """200 ms above threshold + the silence window → exactly one commit."""
    session = _make_session(monkeypatch)
    ws = _RecordingWebSocket()
    session._ws = ws

    async def run() -> None:
        for _ in range(4):  # 4 × 50 ms = 200 ms above threshold
            await session.send_audio(_frame(_FRAME_MS, _LOUD_AMPLITUDE))
        for _ in range(6):  # silence reaches the 300 ms endpoint window
            await session.send_audio(_frame(_FRAME_MS, 0))

    asyncio.run(run())

    commits = [m for m in ws.sent if m["type"] == "input_audio_buffer.commit"]
    assert len(commits) == 1
    # The commit resets the endpointing state (existing convention).
    assert session._speech_ms == 0.0
    assert session._has_speech is False
    assert session._silence_ms == 0.0
