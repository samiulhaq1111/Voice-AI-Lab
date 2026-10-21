"""Targeted tests for per-session realtime TTS provider selection.

Covers the Voice Chat TTS dropdown contract:
- ``tts_provider`` is optional and defaults to the backend default.
- ``elevenlabs`` and ``azure`` are accepted; anything else is rejected.
- The selected provider is passed to ``get_tts_provider`` at session start.
- ``tts_mode`` still controls the server/browser transport.

No real Azure or ElevenLabs calls are made — the factory is mocked.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.voice_realtime import _parse_start_message
from app.main import app

REALTIME_WS_PATH = "/api/v1/voice/realtime/ws"


def _start(**overrides: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "start",
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "linear16",
        "language": "en",
    }
    msg.update(overrides)
    return msg


class _IdleStreamingSession:
    """In-memory STT session that stays open until finish()/close()."""

    provider_name = "deepgram"

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    async def start(self) -> None:
        pass

    async def send_audio(self, chunk: bytes) -> None:
        pass

    async def receive(self):
        await self._closed.wait()
        return None

    async def finish(self) -> None:
        self._closed.set()

    async def close(self) -> None:
        self._closed.set()


@pytest.mark.parametrize("provider", ["elevenlabs", "azure"])
def test_tts_provider_accepted(provider: str) -> None:
    """Both server providers parse into the per-session field."""
    cfg, llm, model, mode, tts_provider, stt_provider = _parse_start_message(
        _start(tts_provider=provider)
    )
    assert cfg.sample_rate == 16000
    assert llm == ""
    assert model is None
    assert mode == "elevenlabs"
    assert tts_provider == provider
    assert stt_provider == ""


def test_tts_provider_defaults_to_backend_default() -> None:
    """Omitting tts_provider preserves the existing default behavior."""
    *_, mode, tts_provider, stt_provider = _parse_start_message(_start())
    assert mode == "elevenlabs"
    assert tts_provider == ""
    assert stt_provider == ""


def test_browser_mode_omits_provider() -> None:
    """Browser TTS parses without a server provider (client-side speech)."""
    *_, mode, tts_provider, _ = _parse_start_message(
        _start(tts_mode="browser")
    )
    assert mode == "browser"
    assert tts_provider == ""


def test_unknown_tts_provider_rejected() -> None:
    """A non-server value (e.g. browser) is rejected as a provider."""
    with pytest.raises(ValueError, match="tts_provider"):
        _parse_start_message(_start(tts_provider="browser"))


def test_non_string_tts_provider_rejected() -> None:
    """Malformed (unhashable) provider values fail validation, not the handler."""
    with pytest.raises(ValueError, match="tts_provider"):
        _parse_start_message(_start(tts_provider=["azure"]))


@pytest.mark.parametrize(
    ("selected", "expected"),
    [(None, None), ("elevenlabs", "elevenlabs"), ("azure", "azure")],
)
def test_selected_tts_provider_reaches_factory(
    selected: str | None, expected: str | None
) -> None:
    """The START selection is forwarded to get_tts_provider for this session."""
    session = _IdleStreamingSession()
    fake_tts = MagicMock()
    fake_tts.provider_name = expected or "elevenlabs"

    with (
        patch(
            "app.api.voice_realtime.open_streaming_session",
            side_effect=lambda cfg, provider=None: session,
        ),
        patch(
            "app.api.voice_realtime.get_tts_provider",
            return_value=fake_tts,
        ) as get_tts,
    ):
        client = TestClient(app)
        extra = {"tts_provider": selected} if selected else {}
        with client.websocket_connect(REALTIME_WS_PATH) as ws:
            ws.send_json(_start(**extra))
            event = ws.receive_json()
            assert event["type"] == "session_started"
            get_tts.assert_called_once_with(expected)
