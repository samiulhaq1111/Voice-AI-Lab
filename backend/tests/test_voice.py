"""Tests for the voice service and WebSocket endpoint."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.types import LLMMessage, LLMResponse, STTResult, ToolSchema, TTSResult
from app.services.voice_service import VoiceConfig, VoiceError, _validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_stt_result(text: str = "Hello world") -> STTResult:
    return STTResult(text=text, confidence=0.95, is_final=True)


def _fake_tts_result() -> TTSResult:
    return TTSResult(
        audio_data=b"fake-mp3-audio-bytes",
        content_type="audio/mpeg",
    )


def _mock_llm_response(
    content: str | None = "Hello",
    tool_calls: list | None = None,
    usage: dict | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


# ---------------------------------------------------------------------------
# Tests: VoiceConfig
# ---------------------------------------------------------------------------


class TestVoiceConfig:
    """Test VoiceConfig parsing and defaults."""

    def test_from_dict_empty(self) -> None:
        cfg = VoiceConfig.from_dict({})
        assert cfg.stt_provider == "deepgram"
        assert cfg.llm_provider == "openrouter"
        assert cfg.tts_provider == "elevenlabs"

    def test_from_dict_none(self) -> None:
        cfg = VoiceConfig.from_dict(None)
        assert cfg.stt_provider == "deepgram"

    def test_from_dict_overrides(self) -> None:
        cfg = VoiceConfig.from_dict({
            "stt_provider": "deepgram",
            "stt_model": "nova-2",
            "llm_provider": "openrouter",
            "llm_model": "gpt-4o-mini",
        })
        assert cfg.stt_model == "nova-2"
        assert cfg.llm_model == "gpt-4o-mini"
        # TTS should still use defaults
        assert cfg.tts_provider == "elevenlabs"


# ---------------------------------------------------------------------------
# Tests: Config Validation
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Test provider configuration validation."""

    def test_valid_config(self) -> None:
        cfg = VoiceConfig.from_dict({})
        _validate_config(cfg)  # Should not raise

    def test_unsupported_stt(self) -> None:
        cfg = VoiceConfig(stt_provider="whisper", llm_provider="openrouter", tts_provider="elevenlabs")
        with pytest.raises(VoiceError, match="Unsupported STT"):
            _validate_config(cfg)

    def test_unsupported_llm(self) -> None:
        cfg = VoiceConfig(stt_provider="deepgram", llm_provider="anthropic", tts_provider="elevenlabs")
        with pytest.raises(VoiceError, match="Unsupported LLM"):
            _validate_config(cfg)

    def test_unsupported_tts(self) -> None:
        cfg = VoiceConfig(stt_provider="deepgram", llm_provider="openrouter", tts_provider="google")
        with pytest.raises(VoiceError, match="Unsupported TTS"):
            _validate_config(cfg)


# ---------------------------------------------------------------------------
# Tests: WebSocket Endpoint
# ---------------------------------------------------------------------------


class TestVoiceWebSocket:
    """Test the /api/v1/voice/ws WebSocket endpoint."""

    def _get_client(self) -> TestClient:
        return TestClient(app)

    def test_ws_start_sends_session_started(self) -> None:
        """START event should return session_started."""
        client = self._get_client()
        with client.websocket_connect("/api/v1/voice/ws") as ws:
            ws.send_json({
                "type": "start",
                "session_id": None,
                "configuration": {},
            })
            data = json.loads(ws.receive_text())
            assert data["type"] == "session_started"
            assert "session_id" in data

    def test_ws_stop_without_start_returns_error(self) -> None:
        """STOP without START should return an error."""
        client = self._get_client()
        with client.websocket_connect("/api/v1/voice/ws") as ws:
            ws.send_json({"type": "stop"})
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"

    def test_ws_unknown_type_returns_error(self) -> None:
        """Unknown message type should return an error."""
        client = self._get_client()
        with client.websocket_connect("/api/v1/voice/ws") as ws:
            ws.send_json({"type": "foobar"})
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "foobar" in data["message"]

    def test_ws_stop_without_audio_returns_error(self) -> None:
        """STOP without audio should return an error."""
        client = self._get_client()
        with client.websocket_connect("/api/v1/voice/ws") as ws:
            ws.send_json({"type": "start", "session_id": None, "configuration": {}})
            json.loads(ws.receive_text())  # session_started
            ws.send_json({"type": "stop"})
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "No audio" in data["message"]

    def test_ws_invalid_session_returns_error(self) -> None:
        """START with non-existent session_id should return an error."""
        client = self._get_client()
        with client.websocket_connect("/api/v1/voice/ws") as ws:
            ws.send_json({
                "type": "start",
                "session_id": "nonexistent-id",
                "configuration": {},
            })
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"

    def test_ws_full_turn_with_mocked_providers(self) -> None:
        """Complete voice turn: START → audio → STOP → events."""
        client = self._get_client()

        with (
            patch("app.services.voice_service.get_stt_provider") as mock_stt_factory,
            patch("app.services.voice_service.get_llm_provider") as mock_llm_factory,
            patch("app.services.voice_service.get_tts_provider") as mock_tts_factory,
        ):
            # Mock STT
            mock_stt = AsyncMock()
            mock_stt.transcribe = AsyncMock(return_value=_fake_stt_result("My ID is E00"))
            mock_stt.close = AsyncMock()
            mock_stt_factory.return_value = mock_stt

            # Mock LLM
            mock_llm = AsyncMock()
            mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Got it, E00."))
            mock_llm.close = AsyncMock()
            mock_llm_factory.return_value = mock_llm

            # Mock TTS
            mock_tts = AsyncMock()
            mock_tts.synthesize = AsyncMock(return_value=_fake_tts_result())
            mock_tts.close = AsyncMock()
            mock_tts_factory.return_value = mock_tts

            with client.websocket_connect("/api/v1/voice/ws") as ws:
                # START
                ws.send_json({
                    "type": "start",
                    "session_id": None,
                    "configuration": {},
                })
                start_event = json.loads(ws.receive_text())
                assert start_event["type"] == "session_started"
                session_id = start_event["session_id"]

                # Send fake audio (binary)
                ws.send_bytes(b"fake-webm-audio-data")

                # STOP
                ws.send_json({"type": "stop"})

                # Collect all events
                events = []
                for _ in range(15):
                    try:
                        raw = ws.receive_text()
                        data = json.loads(raw)
                        events.append(data)
                        if data["type"] in ("completed", "error"):
                            break
                    except Exception:
                        break

        event_types = [e["type"] for e in events]
        assert "processing" in event_types, f"Missing processing event. Got: {event_types}"
        assert "transcript" in event_types
        assert "agent_response" in event_types
        assert "audio" in event_types
        assert "completed" in event_types

        # Verify transcript
        transcript_events = [e for e in events if e["type"] == "transcript"]
        assert len(transcript_events) >= 1
        assert transcript_events[0]["text"] == "My ID is E00"
        assert transcript_events[0]["final"] is True

        # Verify agent response
        response_events = [e for e in events if e["type"] == "agent_response"]
        assert len(response_events) >= 1
        assert "E00" in response_events[0]["text"]

        # Verify audio is base64
        audio_events = [e for e in events if e["type"] == "audio"]
        assert len(audio_events) >= 1
        assert audio_events[0]["format"] == "audio/mpeg"
        decoded = base64.b64decode(audio_events[0]["data"])
        assert decoded == b"fake-mp3-audio-bytes"

    def test_ws_tts_failure_sends_error_not_completed(self) -> None:
        """TTS failure should emit error and NOT completed."""
        client = self._get_client()

        with (
            patch("app.services.voice_service.get_stt_provider") as mock_stt_factory,
            patch("app.services.voice_service.get_llm_provider") as mock_llm_factory,
            patch("app.services.voice_service.get_tts_provider") as mock_tts_factory,
        ):
            # Mock STT
            mock_stt = AsyncMock()
            mock_stt.transcribe = AsyncMock(return_value=_fake_stt_result("Hello"))
            mock_stt.close = AsyncMock()
            mock_stt_factory.return_value = mock_stt

            # Mock LLM
            mock_llm = AsyncMock()
            mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Hi there"))
            mock_llm.close = AsyncMock()
            mock_llm_factory.return_value = mock_llm

            # Mock TTS — raises error
            mock_tts = AsyncMock()
            mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("ElevenLabs API error: HTTP 400"))
            mock_tts.close = AsyncMock()
            mock_tts_factory.return_value = mock_tts

            with client.websocket_connect("/api/v1/voice/ws") as ws:
                # START
                ws.send_json({
                    "type": "start",
                    "session_id": None,
                    "configuration": {},
                })
                start_event = json.loads(ws.receive_text())
                assert start_event["type"] == "session_started"

                # Send fake audio
                ws.send_bytes(b"fake-webm-audio-data")

                # STOP
                ws.send_json({"type": "stop"})

                # Collect all events
                events = []
                for _ in range(15):
                    try:
                        raw = ws.receive_text()
                        data = json.loads(raw)
                        events.append(data)
                        if data["type"] in ("completed", "error"):
                            break
                    except Exception:
                        break

        event_types = [e["type"] for e in events]
        # Should have processing, transcript, agent_response, then error
        assert "transcript" in event_types
        assert "agent_response" in event_types
        assert "error" in event_types
        # Must NOT have completed after error
        assert "completed" not in event_types
        # Must NOT have audio
        assert "audio" not in event_types

        # Verify error message mentions TTS
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1
        assert "TTS" in error_events[0]["message"]

    def test_ws_session_reuse(self) -> None:
        """Second START with session_id should reuse the session."""
        client = self._get_client()

        with (
            patch("app.services.voice_service.get_stt_provider") as mock_stt_factory,
            patch("app.services.voice_service.get_llm_provider") as mock_llm_factory,
            patch("app.services.voice_service.get_tts_provider") as mock_tts_factory,
        ):
            mock_stt = AsyncMock()
            mock_stt.transcribe = AsyncMock(return_value=_fake_stt_result("Hello"))
            mock_stt.close = AsyncMock()
            mock_stt_factory.return_value = mock_stt

            mock_llm = AsyncMock()
            mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Hi"))
            mock_llm.close = AsyncMock()
            mock_llm_factory.return_value = mock_llm

            mock_tts = AsyncMock()
            mock_tts.synthesize = AsyncMock(return_value=_fake_tts_result())
            mock_tts.close = AsyncMock()
            mock_tts_factory.return_value = mock_tts

            with client.websocket_connect("/api/v1/voice/ws") as ws:
                # First turn
                ws.send_json({"type": "start", "session_id": None, "configuration": {}})
                r1 = json.loads(ws.receive_text())
                assert r1["type"] == "session_started"
                sid = r1["session_id"]

                ws.send_bytes(b"audio1")
                ws.send_json({"type": "stop"})

                # Drain events
                for _ in range(15):
                    try:
                        data = json.loads(ws.receive_text())
                        if data["type"] in ("completed", "error"):
                            break
                    except Exception:
                        break

                # Second turn with same session
                ws.send_json({"type": "start", "session_id": sid, "configuration": {}})
                r2 = json.loads(ws.receive_text())
                assert r2["type"] == "session_started"
                assert r2["session_id"] == sid

                ws.send_bytes(b"audio2")
                ws.send_json({"type": "stop"})

                for _ in range(15):
                    try:
                        data = json.loads(ws.receive_text())
                        if data["type"] in ("completed", "error"):
                            break
                    except Exception:
                        break

    def test_ws_stt_failure_sends_error(self) -> None:
        """STT failure should send an error event."""
        client = self._get_client()

        with patch("app.services.voice_service.get_stt_provider") as mock_stt_factory:
            mock_stt = AsyncMock()
            mock_stt.transcribe = AsyncMock(side_effect=RuntimeError("STT down"))
            mock_stt.close = AsyncMock()
            mock_stt_factory.return_value = mock_stt

            with client.websocket_connect("/api/v1/voice/ws") as ws:
                ws.send_json({"type": "start", "session_id": None, "configuration": {}})
                json.loads(ws.receive_text())  # session_started

                ws.send_bytes(b"audio")
                ws.send_json({"type": "stop"})

                events = []
                for _ in range(10):
                    try:
                        data = json.loads(ws.receive_text())
                        events.append(data)
                        if data["type"] == "error":
                            break
                    except Exception:
                        break

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1, f"Expected error event, got: {[e['type'] for e in events]}"
        assert "STT" in error_events[0]["message"]
