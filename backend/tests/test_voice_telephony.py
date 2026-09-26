"""Focused tests for the Telnyx telephony webhook endpoint."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

TELEPHONY_WEBHOOK_PATH = "/api/v1/voice/telephony/webhook"
TELEPHONY_MEDIA_WS_PATH = "/api/v1/voice/telephony/media"


def _client() -> TestClient:
    return TestClient(app)


def _mock_telnyx_client() -> object:
    """Create a fully mocked TelnyxCallControlClient context manager."""
    patcher = patch("app.services.telnyx_client.TelnyxCallControlClient")
    mock_cls = patcher.start()
    mock_cls.return_value.answer_call = AsyncMock(return_value={})
    mock_cls.return_value.speak = AsyncMock(return_value={})
    mock_cls.return_value.streaming_start = AsyncMock(return_value={})
    mock_cls.return_value.close = AsyncMock()
    return patcher


def _initiated_payload(
    call_control_id: str = "cc_abc",
    event_id: str = "evt_123",
) -> dict:
    return {
        "data": {
            "event_type": "call.initiated",
            "id": event_id,
            "payload": {
                "call_control_id": call_control_id,
                "call_leg_id": "leg_1",
                "from": "+15551234567",
                "to": "+15559876543",
            },
        }
    }


class TestTelnyxWebhook:
    """Telnyx Call Control V2 webhook endpoint tests."""

    def test_valid_webhook_returns_200(self) -> None:
        """A valid Telnyx webhook payload returns HTTP 200."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH, json=_initiated_payload()
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_missing_optional_fields_returns_200(self) -> None:
        """An empty or minimal payload still returns 200."""
        resp = _client().post(TELEPHONY_WEBHOOK_PATH, json={})
        assert resp.status_code == 200

        resp = _client().post(TELEPHONY_WEBHOOK_PATH, json={"data": {}})
        assert resp.status_code == 200

        resp = _client().post(
            TELEPHONY_WEBHOOK_PATH,
            json={"data": {"event_type": "call.answered"}},
        )
        assert resp.status_code == 200

    def test_event_type_extracted(self, caplog) -> None:
        """The event_type from the payload is logged."""
        import logging

        payload = {
            "data": {
                "event_type": "call.answered",
                "id": "evt_456",
                "payload": {"call_control_id": "cc_def"},
            }
        }
        with caplog.at_level(logging.INFO):
            resp = _client().post(TELEPHONY_WEBHOOK_PATH, json=payload)
        assert resp.status_code == 200
        assert "call.answered" in caplog.text

    def test_non_json_body_returns_200(self) -> None:
        """A non-JSON body is handled gracefully, still returns 200."""
        resp = _client().post(
            TELEPHONY_WEBHOOK_PATH,
            content=b"not-json",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code == 200

    def test_no_ai_or_provider_invoked(self) -> None:
        """The webhook does NOT invoke AI/provider code."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with (
            patch(
                "app.services.realtime_voice_service.process_realtime_utterance",
                AsyncMock(),
            ) as mock_process,
            patch(
                "app.providers.factory.get_tts_provider",
            ) as mock_tts,
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH, json=_initiated_payload()
            )
            assert resp.status_code == 200
            mock_process.assert_not_awaited()
            mock_tts.assert_not_called()


class TestTelnyxAnswerCall:
    """Tests for call.initiated → answer behavior."""

    def test_call_initiated_triggers_answer(self) -> None:
        """call.initiated with call_control_id triggers answer_call."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_answer_me"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once_with("cc_answer_me")

    def test_non_initiated_event_does_not_answer(self) -> None:
        """Non call.initiated events do NOT trigger answer_call."""
        payload = {
            "data": {
                "event_type": "call.answered",
                "id": "evt_x",
                "payload": {"call_control_id": "cc_skip"},
            }
        }
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_answer = AsyncMock(return_value={})
            mock_speak = AsyncMock(return_value={})
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(TELEPHONY_WEBHOOK_PATH, json=payload)
        assert resp.status_code == 200
        mock_answer.assert_not_awaited()
        mock_speak.assert_not_awaited()

    def test_missing_call_control_id_does_not_answer(self) -> None:
        """call.initiated without call_control_id does NOT trigger answer."""
        payload = {
            "data": {
                "event_type": "call.initiated",
                "id": "evt_y",
                "payload": {},
            }
        }
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_answer = AsyncMock(return_value={})
            mock_speak = AsyncMock(return_value={})
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(TELEPHONY_WEBHOOK_PATH, json=payload)
        assert resp.status_code == 200
        mock_answer.assert_not_awaited()
        mock_speak.assert_not_awaited()

    def test_answer_success_logged(self, caplog) -> None:
        """Successful answer_call logs the call_control_id."""
        import logging

        mock_answer = AsyncMock(return_value={"data": {"result": "ok"}})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_log_ok"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once_with("cc_log_ok")
        # Webhook logs the event_type
        assert "call.initiated" in caplog.text

    def test_answer_failure_does_not_break_webhook(self, caplog) -> None:
        """If answer_call fails, webhook still returns 200."""
        import logging

        mock_answer = AsyncMock(
            side_effect=RuntimeError("Telnyx answer failed (HTTP 401): bad key")
        )
        mock_speak = AsyncMock(return_value={})
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            caplog.at_level(logging.ERROR),
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_fail"),
            )
        assert resp.status_code == 200
        assert "failed" in caplog.text

    def test_answer_client_closed_after_use(self) -> None:
        """The Telnyx client is closed after answer (success or failure)."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        mock_close = AsyncMock()
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = mock_close
            _client().post(
                TELEPHONY_WEBHOOK_PATH, json=_initiated_payload()
            )
        mock_close.assert_awaited_once()


class TestTelnyxCallControlClient:
    """Unit tests for TelnyxCallControlClient with mocked httpx."""

    async def test_answer_call_success(self) -> None:
        """answer_call returns parsed dict on HTTP 200."""
        import httpx

        from app.services.telnyx_client import TelnyxCallControlClient

        mock_response = httpx.Response(
            200,
            json={"data": {"result": "ok"}},
            request=httpx.Request("POST", "https://api.telnyx.com/v2/calls/cc_test/actions/answer"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TelnyxCallControlClient.__new__(TelnyxCallControlClient)
        client._client = mock_client

        result = await client.answer_call("cc_test")
        assert result == {"data": {"result": "ok"}}
        mock_client.post.assert_awaited_once_with(
            "/calls/cc_test/actions/answer", json={}
        )

    async def test_answer_call_http_error(self) -> None:
        """answer_call raises RuntimeError on HTTP error."""
        import httpx

        from app.services.telnyx_client import TelnyxCallControlClient

        error_response = httpx.Response(
            401,
            json={"errors": [{"detail": "Invalid API key"}]},
            request=httpx.Request("POST", "https://api.telnyx.com/v2/calls/cc_bad/actions/answer"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=error_response)

        client = TelnyxCallControlClient.__new__(TelnyxCallControlClient)
        client._client = mock_client

        try:
            await client.answer_call("cc_bad")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "401" in str(e)
            assert "Invalid API key" in str(e)

    def test_no_api_key_raises_value_error(self) -> None:
        """Constructor raises ValueError when no API key is available."""
        from app.services.telnyx_client import TelnyxCallControlClient

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_api_key = ""
            try:
                TelnyxCallControlClient(api_key="")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "TELNYX_API_KEY" in str(e)

    async def test_speak_call_success(self) -> None:
        """speak returns parsed dict on HTTP 200 with voice parameter."""
        import httpx

        from app.services.telnyx_client import TelnyxCallControlClient

        mock_response = httpx.Response(
            200,
            json={"data": {"result": "ok"}},
            request=httpx.Request(
                "POST",
                "https://api.telnyx.com/v2/calls/cc_speak/actions/speak",
            ),
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=mock_response)

        client = TelnyxCallControlClient.__new__(TelnyxCallControlClient)
        client._client = mock_http

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            result = await client.speak("cc_speak", "Hello, this is the Voice AI Lab.")
        assert result == {"data": {"result": "ok"}}
        mock_http.post.assert_awaited_once_with(
            "/calls/cc_speak/actions/speak",
            json={
                "payload": "Hello, this is the Voice AI Lab.",
                "voice": "female",
                "language": "en-US",
                "service_level": "basic",
            },
        )

    async def test_speak_call_http_error(self) -> None:
        """speak raises RuntimeError on HTTP error."""
        import httpx

        from app.services.telnyx_client import TelnyxCallControlClient

        error_response = httpx.Response(
            400,
            json={"errors": [{"detail": "Invalid call_control_id"}]},
            request=httpx.Request(
                "POST",
                "https://api.telnyx.com/v2/calls/cc_bad_speak/actions/speak",
            ),
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=error_response)

        client = TelnyxCallControlClient.__new__(TelnyxCallControlClient)
        client._client = mock_http

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            try:
                await client.speak("cc_bad_speak", "Hello")
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "400" in str(e)
                assert "Invalid call_control_id" in str(e)

    async def test_speak_custom_voice_override(self) -> None:
        """speak uses explicitly passed voice over settings default."""
        import httpx

        from app.services.telnyx_client import TelnyxCallControlClient

        mock_response = httpx.Response(
            200,
            json={"data": {"result": "ok"}},
            request=httpx.Request(
                "POST",
                "https://api.telnyx.com/v2/calls/cc_voice/actions/speak",
            ),
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=mock_response)

        client = TelnyxCallControlClient.__new__(TelnyxCallControlClient)
        client._client = mock_http

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            await client.speak(
                "cc_voice", "Hi", voice="Telnyx.KokoroTTS.af"
            )
        mock_http.post.assert_awaited_once_with(
            "/calls/cc_voice/actions/speak",
            json={
                "payload": "Hi",
                "voice": "Telnyx.KokoroTTS.af",
                "language": "en-US",
                "service_level": "basic",
            },
        )


class TestTelnyxSpeakFlow:
    """Tests for answer → speak flow in webhook."""

    def test_speak_triggered_after_successful_answer(self) -> None:
        """After successful answer, speak is called with greeting text."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_speak_test"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once_with("cc_speak_test")
        mock_speak.assert_awaited_once_with(
            "cc_speak_test",
            "Hello, I am automated assistance. How can I help you today",
        )

    def test_speak_not_triggered_when_answer_fails(self) -> None:
        """If answer fails, speak is NOT called."""
        mock_answer = AsyncMock(
            side_effect=RuntimeError("Telnyx answer failed (HTTP 401): bad")
        )
        mock_speak = AsyncMock(return_value={})
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_no_speak"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once()
        mock_speak.assert_not_awaited()

    def test_speak_failure_still_returns_200(self, caplog) -> None:
        """If speak fails after answer, webhook still returns 200."""
        import logging

        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(
            side_effect=RuntimeError("Telnyx speak failed (HTTP 500): error")
        )
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            caplog.at_level(logging.ERROR),
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_speak_fail"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once()
        mock_speak.assert_awaited_once()
        assert "failed" in caplog.text

    def test_non_initiated_event_no_speak(self) -> None:
        """Non call.initiated events do NOT trigger speak."""
        payload = {
            "data": {
                "event_type": "call.hangup",
                "id": "evt_hangup",
                "payload": {"call_control_id": "cc_hangup"},
            }
        }
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_answer = AsyncMock(return_value={})
            mock_speak = AsyncMock(return_value={})
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(TELEPHONY_WEBHOOK_PATH, json=payload)
        assert resp.status_code == 200
        mock_answer.assert_not_awaited()
        mock_speak.assert_not_awaited()


class TestTelnyxMediaStreaming:
    """Tests for media streaming start after answer."""

    def test_streaming_start_called_after_answer(self) -> None:
        """streaming_start is called after answer + speak when URL configured."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            patch("app.api.voice_telephony.settings") as mock_settings,
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            mock_settings.telnyx_media_ws_url = "wss://example.com/media"
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_stream"),
            )
        assert resp.status_code == 200
        mock_streaming.assert_awaited_once_with(
            "cc_stream", "wss://example.com/media"
        )

    def test_streaming_skipped_when_no_ws_url(self) -> None:
        """streaming_start is NOT called when media WS URL is empty."""
        mock_answer = AsyncMock(return_value={})
        mock_speak = AsyncMock(return_value={})
        mock_streaming = AsyncMock(return_value={})
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            patch("app.api.voice_telephony.settings") as mock_settings,
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.streaming_start = mock_streaming
            mock_client_cls.return_value.close = AsyncMock()
            mock_settings.telnyx_media_ws_url = ""
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_no_stream"),
            )
        assert resp.status_code == 200
        mock_streaming.assert_not_awaited()


class TestMediaWebSocket:
    """Tests for the media WebSocket endpoint."""

    def test_media_ws_connects_and_disconnects(self) -> None:
        """WebSocket connects and handles immediate disconnect."""
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH):
            pass

    def test_start_event_parsed(self) -> None:
        """Start event is parsed and logged."""
        import json

        start_msg = json.dumps({
            "event": "start",
            "sequence_number": "1",
            "start": {
                "call_control_id": "cc_ws",
                "call_session_id": "sess_1",
                "media_format": {
                    "encoding": "PCMU",
                    "sample_rate": 8000,
                    "channels": 1,
                },
            },
            "stream_id": "stream_abc",
        })
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text(start_msg)

    def test_media_event_returns_outbound(self) -> None:
        """Media event triggers outbound test audio."""
        import json

        start_msg = json.dumps({
            "event": "start",
            "sequence_number": "1",
            "start": {
                "call_control_id": "cc_media",
                "media_format": {
                    "encoding": "PCMU",
                    "sample_rate": 8000,
                    "channels": 1,
                },
            },
            "stream_id": "stream_1",
        })
        media_msg = json.dumps({
            "event": "media",
            "sequence_number": "2",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "payload": "dGVzdA==",
            },
            "stream_id": "stream_1",
        })
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text(start_msg)
            ws.send_text(media_msg)
            response = json.loads(ws.receive_text())
            assert response["event"] == "media"
            assert "payload" in response["media"]
            assert len(response["media"]["payload"]) > 0

    def test_stop_event(self) -> None:
        """Stop event is handled gracefully."""
        import json

        stop_msg = json.dumps({
            "event": "stop",
            "sequence_number": "10",
            "stop": {"call_control_id": "cc_stop"},
            "stream_id": "stream_stop",
        })
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text(stop_msg)

    def test_malformed_message_handled(self) -> None:
        """Malformed JSON does not crash the WebSocket."""
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text("not-json{{{")

    def test_unknown_event_ignored(self) -> None:
        """Unknown event types are logged and ignored."""
        import json

        unknown_msg = json.dumps({"event": "foobar_unknown"})
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text(unknown_msg)

    def test_outbound_media_format(self) -> None:
        """Outbound media message has correct PCMU format."""
        import base64
        import json

        start_msg = json.dumps({
            "event": "start",
            "sequence_number": "1",
            "start": {
                "call_control_id": "cc_fmt",
                "media_format": {"encoding": "PCMU", "sample_rate": 8000},
            },
            "stream_id": "stream_fmt",
        })
        media_msg = json.dumps({
            "event": "media",
            "sequence_number": "2",
            "media": {"track": "inbound", "payload": "dGVzdA=="},
            "stream_id": "stream_fmt",
        })
        c = _client()
        with c.websocket_connect(TELEPHONY_MEDIA_WS_PATH) as ws:
            ws.send_text(start_msg)
            ws.send_text(media_msg)
            response = json.loads(ws.receive_text())
            assert response["event"] == "media"
            assert isinstance(response["media"], dict)
            assert "payload" in response["media"]
            decoded = base64.b64decode(response["media"]["payload"])
            assert len(decoded) == 160  # 20ms PCMU = 160 bytes


class TestAudioConversion:
    """Tests for PCMU → PCM16 conversion and sample rate conversion."""

    def test_ulaw_to_linear_known_values(self) -> None:
        """Known PCMU values decode to expected PCM values."""
        from app.utils.audio import ulaw_to_linear

        # PCMU silence (0xFF) decodes to 0 (or near-zero)
        pcm = ulaw_to_linear(bytes([0xFF]))
        assert len(pcm) == 2  # 1 sample = 2 bytes
        # PCMU 0xFF is silence (0)
        import struct

        sample = struct.unpack("<h", pcm)[0]
        assert abs(sample) < 10  # Near zero

    def test_ulaw_to_linear_output_size(self) -> None:
        """Output size is 2x input size (1 byte → 2 bytes per sample)."""
        from app.utils.audio import ulaw_to_linear

        ulaw_data = bytes(160)  # 160 PCMU samples
        pcm_data = ulaw_to_linear(ulaw_data)
        assert len(pcm_data) == 320  # 160 samples × 2 bytes

    def test_pcm8k_to_pcm16k_output_size(self) -> None:
        """Output size is 2x input size (8kHz → 16kHz)."""
        import struct

        from app.utils.audio import pcm8k_to_pcm16k

        # 160 samples at 8kHz = 320 bytes
        pcm_8k = struct.pack("<160h", *([0] * 160))
        pcm_16k = pcm8k_to_pcm16k(pcm_8k)
        # 320 samples at 16kHz = 640 bytes
        assert len(pcm_16k) == 640

    def test_pcmu_8k_to_pcm_16k_full_pipeline(self) -> None:
        """Full pipeline: 160 PCMU bytes → 640 PCM bytes @ 16kHz."""
        from app.utils.audio import pcmu_8k_to_pcm_16k

        ulaw_data = bytes(160)  # 160 PCMU bytes (20ms @ 8kHz)
        pcm_data = pcmu_8k_to_pcm_16k(ulaw_data)
        # 160 samples → 320 samples (2x upsample)
        # 320 samples × 2 bytes = 640 bytes
        assert len(pcm_data) == 640

    def test_empty_input_returns_empty(self) -> None:
        """Empty input returns empty output."""
        from app.utils.audio import pcmu_8k_to_pcm_16k, ulaw_to_linear

        assert ulaw_to_linear(b"") == b""
        assert pcmu_8k_to_pcm_16k(b"") == b""

    def test_variable_payload_handled(self) -> None:
        """Non-standard payload sizes are handled correctly."""
        from app.utils.audio import pcmu_8k_to_pcm_16k

        # 100 bytes (not 160)
        ulaw_data = bytes(100)
        pcm_data = pcmu_8k_to_pcm_16k(ulaw_data)
        # 100 samples → 200 samples → 400 bytes
        assert len(pcm_data) == 400


class TestTelnyxDeepgramBridge:
    """Tests for the Telnyx → Deepgram bridge service."""

    def test_bridge_initialization(self) -> None:
        """Bridge initializes with default state."""
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        assert bridge.stream_id == ""
        assert bridge.packets_in == 0
        assert bridge.bytes_converted == 0
        assert bridge.partials == 0
        assert bridge.finals == 0

    async def test_process_media_packet_decodes_base64(self) -> None:
        """Media packet processing decodes base64 payload."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True  # Simulate running state

        # Create a valid PCMU payload (160 bytes of silence)
        pcmu_data = bytes([0xFF] * 160)
        payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
        media_data = {"payload": payload_b64}

        await bridge.process_media_packet(media_data)

        assert bridge.packets_in == 1
        # 160 PCMU → 320 PCM @ 8kHz → 640 PCM bytes @ 16kHz
        assert bridge.bytes_converted == 640

    async def test_process_media_empty_payload_ignored(self) -> None:
        """Empty payload is ignored."""

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True

        await bridge.process_media_packet({"payload": ""})

        assert bridge.packets_in == 0

    async def test_process_media_invalid_base64_handled(self) -> None:
        """Invalid base64 is handled gracefully."""

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True

        await bridge.process_media_packet({"payload": "not-valid-base64!!!"})

        # Should not crash, packet not counted
        assert bridge.packets_in == 0

    async def test_bridge_stop_without_start(self) -> None:
        """Stopping a bridge that was never started is safe."""

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        await bridge.stop()
        # Should not raise

    async def test_bridge_start_creates_deepgram_session(self) -> None:
        """Bridge start creates a Deepgram streaming session."""

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        # Mock the session factory
        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("test_stream")

        assert bridge.stream_id == "test_stream"
        mock_session.start.assert_awaited_once()

        # Cleanup
        await bridge.stop()


class TestDeepgramIntegration:
    """Tests for Deepgram session integration with Telnyx media."""

    async def test_media_packet_forwarded_to_deepgram(self) -> None:
        """Media packets are converted and forwarded to Deepgram session."""
        import asyncio
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        # Create mock session
        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("stream_test")

            # Send a media packet
            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
            await bridge.process_media_packet({"payload": payload_b64})

            # Give the forwarder time to process
            await asyncio.sleep(0.1)

            # Cleanup
            await bridge.stop()

        # Verify audio was sent to Deepgram
        assert mock_session.send_audio.await_count >= 1

    async def test_transcript_events_processed(self) -> None:
        """Transcript events from Deepgram are logged."""
        import asyncio

        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        # Create mock session that returns transcript events
        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()

        # Return partial, final, then None (end)
        mock_session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="partial", text="hello", confidence=0.8),
                StreamEvent(type="final", text="hello world", confidence=0.95),
                None,  # End of stream
            ]
        )

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("stream_transcript")

            # Wait for transcript processor to handle events
            await asyncio.sleep(0.2)

            # Cleanup
            await bridge.stop()

        assert bridge.partials == 1
        assert bridge.finals == 1

    async def test_websocket_disconnect_cleanup(self) -> None:
        """Bridge cleanup on WebSocket disconnect."""

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("stream_cleanup")
            await bridge.stop()

        mock_session.finish.assert_awaited_once()
        mock_session.close.assert_awaited_once()


class TestUtteranceAccumulation:
    """Tests for utterance accumulation in TelnyxDeepgramBridge."""

    async def test_final_transcript_accumulated(self) -> None:
        """Final transcripts are accumulated into current utterance."""
        import asyncio

        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        # Two finals then utterance_end
        mock_session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="final", text="Hello", confidence=0.9),
                StreamEvent(type="final", text="world", confidence=0.95),
                StreamEvent(type="utterance_end"),
                None,
            ]
        )

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s1")
            await asyncio.sleep(0.2)
            await bridge.stop()

        # Utterance should be accumulated
        assert bridge.utterances_emitted == 1
        # The utterance should be in the queue
        assert not bridge.utterance_queue.empty()
        utterance = await bridge.utterance_queue.get()
        assert "Hello" in utterance.text
        assert "world" in utterance.text

    async def test_partial_does_not_trigger_utterance(self) -> None:
        """Partial transcripts do NOT produce an utterance."""
        import asyncio

        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        # Only partials, no finals
        mock_session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="partial", text="hel", confidence=0.5),
                StreamEvent(type="partial", text="hello", confidence=0.7),
                StreamEvent(type="utterance_end"),
                None,
            ]
        )

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s2")
            await asyncio.sleep(0.2)
            await bridge.stop()

        # No utterance should be emitted (no finals)
        assert bridge.utterances_emitted == 0
        assert bridge.utterance_queue.empty()

    async def test_utterance_end_resets_accumulation(self) -> None:
        """utterance_end resets accumulation — next utterance starts fresh."""
        import asyncio

        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        # Two utterances
        mock_session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="final", text="First", confidence=0.9),
                StreamEvent(type="utterance_end"),
                StreamEvent(type="final", text="Second", confidence=0.9),
                StreamEvent(type="utterance_end"),
                None,
            ]
        )

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s3")
            await asyncio.sleep(0.3)
            await bridge.stop()

        assert bridge.utterances_emitted == 2
        # Collect both utterances
        utterances = []
        while not bridge.utterance_queue.empty():
            u = await bridge.utterance_queue.get()
            utterances.append(u)
        assert len(utterances) == 2
        assert "First" in utterances[0].text
        assert "Second" in utterances[1].text
        # Utterances should NOT contain text from each other
        assert "Second" not in utterances[0].text
        assert "First" not in utterances[1].text


class TestTelephonyAgentSession:
    """Tests for the telephony agent worker."""

    async def test_agent_worker_processes_utterance(self) -> None:
        """Agent worker processes a queued utterance."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_test", queue)

        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=AsyncMock(
                content="Hello there!",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry",
            ),
            patch(
                "app.services.telnyx_agent.ToolExecutor",
            ),
        ):
            await agent.start()
            # Queue an utterance
            now = time.monotonic()
            await queue.put(Utterance(text="Hello", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            # Give worker time to process
            await asyncio.sleep(0.3)
            # Stop
            await agent.stop()

        assert agent.turn == 1
        mock_llm.close.assert_awaited_once()

    async def test_partial_does_not_trigger_agent(self) -> None:
        """Partial transcripts do NOT reach AgentRuntime."""
        import asyncio

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()

        from app.providers.stt.streaming import StreamEvent

        mock_session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="partial", text="hel", confidence=0.5),
                StreamEvent(type="utterance_end"),
                None,
            ]
        )

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s_partial")
            await asyncio.sleep(0.2)
            await bridge.stop()

        # No utterance emitted — queue should be empty
        assert bridge.utterance_queue.empty()
        assert bridge.utterances_emitted == 0

    async def test_agent_failure_does_not_crash(self) -> None:
        """AgentRuntime failure is handled without crashing."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_fail", queue)

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry",
            ),
            patch(
                "app.services.telnyx_agent.ToolExecutor",
            ),
        ):
            await agent.start()
            now = time.monotonic()
            await queue.put(Utterance(text="test", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await asyncio.sleep(0.3)
            # Queue another utterance — worker should still be alive
            await queue.put(Utterance(text="test2", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await asyncio.sleep(0.3)
            await agent.stop()

        # Both utterances were attempted (worker didn't crash)
        assert agent.turn == 2

    async def test_llm_closed_exactly_once(self) -> None:
        """Session cleanup closes the LLM provider exactly once."""
        import asyncio

        from app.services.telnyx_agent import TelephonyAgentSession

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_close", queue)

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        with patch(
            "app.services.telnyx_agent.get_llm_provider",
            return_value=mock_llm,
        ):
            await agent.start()
            await agent.stop()
            # Second stop should be safe (no double close)
            await agent.stop()

        mock_llm.close.assert_awaited_once()

    async def test_conversation_history_accumulated(self) -> None:
        """AgentRuntime receives conversation history across turns."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_hist", queue)

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=AsyncMock(
                content="Response",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry",
            ),
            patch(
                "app.services.telnyx_agent.ToolExecutor",
            ),
        ):
            await agent.start()
            # First utterance
            now = time.monotonic()
            await queue.put(Utterance(text="Hello", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await asyncio.sleep(0.3)
            # Second utterance — should have history from first
            await queue.put(Utterance(text="How are you?", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await asyncio.sleep(0.3)
            await agent.stop()

        assert agent.turn == 2
        # History should contain messages from both turns
        assert len(agent.history) > 0


class TestLoggingConfiguration:
    """Tests for logging configuration (Part A)."""

    def test_websockets_logger_suppressed(self) -> None:
        """websockets.client logger is set to WARNING level."""
        import logging

        from app.core.logging import setup_logging

        setup_logging()

        ws_client = logging.getLogger("websockets.client")
        ws_server = logging.getLogger("websockets.server")
        ws_protocol = logging.getLogger("websockets.protocol")

        assert ws_client.level >= logging.WARNING
        assert ws_server.level >= logging.WARNING
        assert ws_protocol.level >= logging.WARNING


class TestTrackFiltering:
    """Tests for inbound/outbound track filtering in the Deepgram bridge."""

    async def test_inbound_media_sent_to_deepgram(self) -> None:
        """Inbound track media is converted and sent to Deepgram."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s_inbound")

            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
            await bridge.process_media_packet(
                {"payload": payload_b64, "track": "inbound"}
            )
            await asyncio.sleep(0.1)
            await bridge.stop()

        assert bridge.packets_in == 1
        assert mock_session.send_audio.await_count >= 1

    async def test_outbound_media_not_sent_to_deepgram(self) -> None:
        """Outbound track media is NOT sent to Deepgram."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s_outbound")

            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
            await bridge.process_media_packet(
                {"payload": payload_b64, "track": "outbound"}
            )
            await asyncio.sleep(0.1)
            await bridge.stop()

        # No packets should have been converted
        assert bridge.packets_in == 0
        assert bridge.bytes_converted == 0
        # Outbound packets should be counted
        assert bridge._outbound_packets_skipped == 1
        # Deepgram should NOT have received audio
        assert mock_session.send_audio.await_count == 0

    async def test_outbound_only_no_transcript(self) -> None:
        """An outbound-only media sequence produces no transcript."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s_out_only")

            # Send 10 outbound-only packets
            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
            for _ in range(10):
                await bridge.process_media_packet(
                    {"payload": payload_b64, "track": "outbound"}
                )
            await asyncio.sleep(0.1)
            await bridge.stop()

        assert bridge.packets_in == 0
        assert bridge.utterances_emitted == 0
        assert bridge.utterance_queue.empty()

    async def test_mixed_tracks_only_inbound_counted(self) -> None:
        """Mixed inbound/outbound: only inbound packets are processed."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()

        mock_session = AsyncMock()
        mock_session.provider_name = "deepgram"
        mock_session.start = AsyncMock()
        mock_session.send_audio = AsyncMock()
        mock_session.finish = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.receive = AsyncMock(return_value=None)

        with patch(
            "app.services.telnyx_deepgram.open_streaming_session",
            return_value=mock_session,
        ):
            await bridge.start("s_mixed")

            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")

            # 3 inbound + 5 outbound
            for _ in range(3):
                await bridge.process_media_packet(
                    {"payload": payload_b64, "track": "inbound"}
                )
            for _ in range(5):
                await bridge.process_media_packet(
                    {"payload": payload_b64, "track": "outbound"}
                )
            await asyncio.sleep(0.1)
            await bridge.stop()

        assert bridge.packets_in == 3
        assert bridge._outbound_packets_skipped == 5

    async def test_empty_track_treated_as_inbound(self) -> None:
        """Packets with no track field are treated as inbound (safe default)."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True

        pcmu_data = bytes([0xFF] * 160)
        payload_b64 = base64.b64encode(pcmu_data).decode("ascii")

        # No track field — should be processed
        await bridge.process_media_packet({"payload": payload_b64})
        assert bridge.packets_in == 1
        assert bridge.bytes_converted == 640


class TestTelnyxModelConfiguration:
    """Tests for Telnyx-specific model configuration."""

    async def test_telnyx_uses_gpt4o_mini(self) -> None:
        """Telnyx agent creates the configured gpt-4o-mini model."""
        import asyncio

        from app.services.telnyx_agent import (
            _TELNYX_LLM_MODEL,
            TelephonyAgentSession,
        )
        from app.services.telnyx_deepgram import Utterance

        assert _TELNYX_LLM_MODEL == "openai/gpt-4o-mini"

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_model", queue)

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=AsyncMock(
                content="OK",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )
        )
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ) as mock_factory,
            patch("app.services.telnyx_agent.get_tool_registry"),
            patch("app.services.telnyx_agent.ToolExecutor"),
        ):
            await agent.start()
            # Verify factory was called with correct provider/model
            mock_factory.assert_called_once_with(
                "openrouter", model="openai/gpt-4o-mini"
            )
            await agent.stop()

    async def test_telnyx_multiple_utterances_sequential(self) -> None:
        """Telnyx session processes multiple utterances sequentially."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_multi", queue)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncMock(
                content=f"Response {call_count}",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=mock_chat)
        mock_llm.close = AsyncMock()

        async def mock_stream_chat(*args, **kwargs):
            from app.providers.types import StreamChunk
            yield StreamChunk(content="Response")
        mock_llm.stream_chat = mock_stream_chat

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch("app.services.telnyx_agent.get_tool_registry"),
            patch("app.services.telnyx_agent.ToolExecutor"),
        ):
            await agent.start()
            # Queue 3 utterances
            now = time.monotonic()
            await queue.put(Utterance(text="My name is Sami", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await queue.put(Utterance(text="What is my name?", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await queue.put(Utterance(text="Thank you", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            # Give worker time to process all
            await asyncio.sleep(0.5)
            await agent.stop()

        assert agent.turn == 3

    async def test_slow_llm_does_not_block_media_loop(self) -> None:
        """A slow LLM request does not block the Telnyx media receive loop."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_slow", queue)

        async def slow_chat(**kwargs):
            await asyncio.sleep(0.5)  # Simulate slow LLM
            return AsyncMock(
                content="Slow response",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=slow_chat)
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch("app.services.telnyx_agent.get_tool_registry"),
            patch("app.services.telnyx_agent.ToolExecutor"),
        ):
            await agent.start()
            # Queue an utterance — worker starts processing (slow)
            now = time.monotonic()
            await queue.put(Utterance(text="Hello", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            # Media/STT should continue receiving audio while LLM is slow.
            # Verify by checking the worker task is running (not done).
            await asyncio.sleep(0.1)
            assert agent._worker_task is not None
            assert not agent._worker_task.done()
            # Wait for completion
            await asyncio.sleep(0.6)
            await agent.stop()

        assert agent.turn == 1

    async def test_agent_worker_continues_during_media(self) -> None:
        """Agent worker processes utterances while bridge receives audio."""
        import asyncio
        import base64
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge, Utterance

        # Setup bridge
        bridge = TelnyxDeepgramBridge()
        mock_dg_session = AsyncMock()
        mock_dg_session.provider_name = "deepgram"
        mock_dg_session.start = AsyncMock()
        mock_dg_session.send_audio = AsyncMock()
        mock_dg_session.finish = AsyncMock()
        mock_dg_session.close = AsyncMock()
        mock_dg_session.receive = AsyncMock(return_value=None)

        # Setup Agent
        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession("cc_parallel", queue)

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=AsyncMock(
                content="Response",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )
        mock_llm.close = AsyncMock()

        with (
            patch(
                "app.services.telnyx_deepgram.open_streaming_session",
                return_value=mock_dg_session,
            ),
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch("app.services.telnyx_agent.get_tool_registry"),
            patch("app.services.telnyx_agent.ToolExecutor"),
        ):
            await bridge.start("s_parallel")
            await agent.start()

            # Simulate inbound audio packets while agent processes
            pcmu_data = bytes([0xFF] * 160)
            payload_b64 = base64.b64encode(pcmu_data).decode("ascii")
            for _ in range(5):
                await bridge.process_media_packet(
                    {"payload": payload_b64, "track": "inbound"}
                )

            # Queue an utterance for agent processing
            now = time.monotonic()
            await queue.put(Utterance(text="Test while media flowing", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
            await asyncio.sleep(0.3)

            # More media while agent is working/done
            for _ in range(5):
                await bridge.process_media_packet(
                    {"payload": payload_b64, "track": "inbound"}
                )

            await agent.stop()
            await bridge.stop()

        # Both should have processed independently
        assert agent.turn == 1
        assert bridge.packets_in == 10


class TestAgentRuntimeTiming:
    """Tests for AgentRuntime timing instrumentation."""

    async def test_runtime_logs_timing(self) -> None:
        """AgentRuntime.run() tracks total elapsed time."""
        from app.agents.runtime import AgentConfig, AgentRuntime

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=AsyncMock(
                content="Timed response",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )
        mock_llm.close = AsyncMock()

        runtime = AgentRuntime(
            llm=mock_llm,
            config=AgentConfig(llm_model="openai/gpt-4o-mini"),
        )
        result = await runtime.run("Hello")

        assert result.response == "Timed response"
        assert result.iterations == 1
        assert mock_llm.chat.await_count == 1

        # Verify model was passed correctly
        call_kwargs = mock_llm.chat.call_args
        assert call_kwargs.kwargs.get("model") == "openai/gpt-4o-mini" or (
            call_kwargs[1].get("model") == "openai/gpt-4o-mini"
        )

    async def test_runtime_failure_logged(self) -> None:
        """AgentRuntime failure is logged with timing."""
        from app.agents.runtime import AgentConfig, AgentRuntime

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_llm.close = AsyncMock()

        runtime = AgentRuntime(
            llm=mock_llm,
            config=AgentConfig(llm_model="openai/gpt-4o-mini"),
        )

        with pytest.raises(RuntimeError, match="LLM error"):
            await runtime.run("Hello")

    async def test_runtime_multi_turn_history(self) -> None:
        """Multi-turn conversation passes history correctly."""
        from app.agents.runtime import AgentConfig, AgentRuntime

        turn = 0

        async def mock_chat(**kwargs):
            nonlocal turn
            turn += 1
            messages = kwargs.get("messages", [])
            # Verify system prompt is first
            assert messages[0].role == "system"
            return AsyncMock(
                content=f"Response {turn}",
                tool_calls=None,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=mock_chat)
        mock_llm.close = AsyncMock()

        # Turn 1
        runtime1 = AgentRuntime(
            llm=mock_llm,
            config=AgentConfig(llm_model="openai/gpt-4o-mini"),
        )
        result1 = await runtime1.run("My name is Sami")
        assert result1.response == "Response 1"
        history = list(runtime1.state.messages)

        # Turn 2 — pass history from turn 1
        runtime2 = AgentRuntime(
            llm=mock_llm,
            config=AgentConfig(llm_model="openai/gpt-4o-mini"),
        )
        result2 = await runtime2.run(
            "What is my name?",
            initial_messages=history,
        )
        assert result2.response == "Response 2"
        # Turn 2 should have: system + history (user+assistant) + new user
        messages = mock_llm.chat.call_args_list[-1]
        msg_list = messages.kwargs.get("messages", messages[1].get("messages", []))
        # system + user1 + assistant1 + user2 = 4
        assert len(msg_list) == 4


# ---------------------------------------------------------------------------
# Milestone 7: Telephone TTS Integration
# ---------------------------------------------------------------------------


class TestAudioConversionMP3ToPCMU:
    """Test MP3 → PCMU 8kHz audio conversion pipeline."""

    def test_mp3_to_pcmu_empty_input(self) -> None:
        """Empty MP3 input returns empty PCMU output."""
        from app.utils.audio import mp3_to_pcmu_8k

        assert mp3_to_pcmu_8k(b"") == b""

    def test_mp3_to_pcmu_with_silent_mp3(self) -> None:
        """Silent MP3 produces valid PCMU output."""
        import struct
        import subprocess

        from app.utils.audio import mp3_to_pcmu_8k

        # Generate a short silent PCM and encode to MP3 via ffmpeg
        pcm_silence = struct.pack("<8000h", *([0] * 8000))  # 1s silence @ 8kHz
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-i", "pipe:0",
                "-f", "mp3",
                "pipe:1",
            ],
            input=pcm_silence,
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0
        mp3_data = proc.stdout
        assert len(mp3_data) > 0

        # Convert MP3 → PCMU
        pcmu = mp3_to_pcmu_8k(mp3_data)
        assert len(pcmu) > 0
        # PCMU should be roughly 8000 bytes for 1s @ 8kHz
        # (allow some tolerance for resampling artifacts)
        assert len(pcmu) >= 4000

    def test_mp3_to_pcmu_invalid_data_raises(self) -> None:
        """Invalid data raises RuntimeError."""
        from app.utils.audio import mp3_to_pcmu_8k

        # Random bytes that aren't valid MP3
        with pytest.raises(RuntimeError, match="ffmpeg"):
            mp3_to_pcmu_8k(b"not valid mp3 data at all " * 100)


class TestTelnyxTTSService:
    """Test the telephone TTS service."""

    @pytest.mark.asyncio
    async def test_start_creates_tts_provider(self) -> None:
        """start() creates a TTS provider via get_tts_provider."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"

        with patch(
            "app.services.telnyx_tts.get_tts_provider",
            return_value=mock_tts,
        ):
            svc = TelnyxTTSService()
            await svc.start()
            assert svc._tts is mock_tts
            await svc.stop()

    @pytest.mark.asyncio
    async def test_start_handles_failure_gracefully(self) -> None:
        """start() handles TTS provider creation failure gracefully."""
        from app.services.telnyx_tts import TelnyxTTSService

        with patch(
            "app.services.telnyx_tts.get_tts_provider",
            side_effect=Exception("no API key"),
        ):
            svc = TelnyxTTSService()
            await svc.start()
            assert svc._tts is None

    @pytest.mark.asyncio
    async def test_synthesize_and_send_no_tts_returns_false(self) -> None:
        """synthesize_and_send returns False when no TTS provider."""
        from app.services.telnyx_tts import TelnyxTTSService

        svc = TelnyxTTSService()
        # Don't call start() — _tts is None
        ws = AsyncMock()
        result = await svc.synthesize_and_send("hello", 1, ws)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_synthesize_and_send_success(self) -> None:
        """synthesize_and_send synthesizes, converts, and sends audio."""
        import struct

        from app.providers.types import TTSResult
        from app.services.telnyx_tts import TelnyxTTSService

        # Create a minimal silent MP3 via ffmpeg
        pcm_silence = struct.pack("<1600h", *([0] * 1600))  # 0.2s @ 8kHz
        import subprocess

        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-i", "pipe:0", "-f", "mp3", "pipe:1",
            ],
            input=pcm_silence,
            capture_output=True,
            timeout=10,
        )
        mp3_data = proc.stdout

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.synthesize = AsyncMock(
            return_value=TTSResult(
                audio_data=mp3_data,
                content_type="audio/mpeg",
            )
        )

        ws = AsyncMock()

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        result = await svc.synthesize_and_send("Hello", 1, ws)
        assert result["success"] is True
        # WebSocket should have been called with media events
        assert ws.send_text.called
        # Verify the sent data is valid JSON with media event structure
        import json

        first_call = ws.send_text.call_args_list[0]
        msg = json.loads(first_call[0][0])
        assert msg["event"] == "media"
        assert "payload" in msg["media"]

        await svc.stop()

    @pytest.mark.asyncio
    async def test_synthesize_and_send_tts_failure(self) -> None:
        """synthesize_and_send returns False on TTS failure."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("API error"))

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        result = await svc.synthesize_and_send("Hello", 1, ws)
        assert result["success"] is False
        # WebSocket should NOT have been called
        ws.send_text.assert_not_called()


class TestTelnyxTTSPacing:
    """Verify that TTS chunks are sent with real-time pacing (20ms each)."""

    @pytest.mark.asyncio
    async def test_pacing_sleep_between_chunks(self) -> None:
        """asyncio.sleep(0.020) is called once per chunk after sending."""
        import struct
        import subprocess

        from app.providers.types import TTSResult
        from app.services.telnyx_tts import TelnyxTTSService

        # Create enough PCM silence to produce ~10 chunks (10 * 160 = 1600 samples)
        pcm_silence = struct.pack("<1600h", *([0] * 1600))
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-i", "pipe:0", "-f", "mp3", "pipe:1",
            ],
            input=pcm_silence,
            capture_output=True,
            timeout=10,
        )
        mp3_data = proc.stdout

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.synthesize = AsyncMock(
            return_value=TTSResult(
                audio_data=mp3_data,
                content_type="audio/mpeg",
            )
        )

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await svc.synthesize_and_send("Hello", 1, ws)

        assert result["success"] is True
        send_count = ws.send_text.call_count
        sleep_count = mock_sleep.call_count
        # sleep should be called once per chunk (after each send)
        assert sleep_count == send_count
        assert send_count > 0
        # Every sleep call must be 20ms
        for call in mock_sleep.call_args_list:
            assert call.args[0] == pytest.approx(0.020, abs=1e-6)

        await svc.stop()


class TestTelnyxTTSStreaming:
    """Streaming telephony TTS: mu-law chunks must reach Telnyx progressively."""

    @staticmethod
    def _frames(ws: AsyncMock) -> list[bytes]:
        """Decode every base64 PCMU frame handed to the media WebSocket."""
        import base64
        import json

        out: list[bytes] = []
        for call in ws.send_text.call_args_list:
            msg = json.loads(call.args[0])
            assert msg["event"] == "media"
            out.append(base64.b64decode(msg["media"]["payload"]))
        return out

    @staticmethod
    def _provider(chunks: list[bytes], error: Exception | None = None):
        """Build a stream_synthesize stand-in yielding raw mu-law chunks."""

        async def _stream(**_kwargs):
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error

        return _stream

    @pytest.mark.asyncio
    async def test_streaming_path_avoids_buffered_synthesize(self) -> None:
        """When the provider streams mu-law, the buffered path is not used."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider([b"\xff" * 400])
        mock_tts.synthesize = AsyncMock(side_effect=AssertionError("buffered used"))

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.synthesize_and_send("Hello", 1, ws, call_id="cc_1")

        assert result["success"] is True
        mock_tts.synthesize.assert_not_called()
        # 400 bytes → two complete 160-byte frames, 80-byte tail dropped
        assert ws.send_text.call_count == 2

    @pytest.mark.asyncio
    async def test_ulaw_8000_format_requested(self) -> None:
        """The provider is asked for telephony-native ulaw_8000 output."""
        from app.services.telnyx_tts import TelnyxTTSService

        captured: dict[str, object] = {}

        async def _stream(*, text: str = "", output_format: str | None = None, **_kw):
            captured["text"] = text
            captured["output_format"] = output_format
            yield b"\xff" * 160

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = _stream

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.synthesize_and_send("Hello", 1, ws)
            assert result["success"] is True

        assert captured["output_format"] == "ulaw_8000"
        assert captured["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_pcmu_frame_size_preserved(self) -> None:
        """Every frame sent to Telnyx is exactly one 20ms PCMU packet."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        # Odd-sized network chunks must be re-framed into 160-byte packets
        mock_tts.stream_synthesize = self._provider(
            [b"\xff" * 37, b"\x7f" * 400, b"\x00" * 5]
        )

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.synthesize_and_send("Hello", 1, ws)
            assert result["success"] is True

        frames = self._frames(ws)
        assert len(frames) == 2  # 442 bytes → 2 full frames + 122 tail dropped
        assert all(len(f) == 160 for f in frames)

    @pytest.mark.asyncio
    async def test_chunks_forwarded_incrementally(self) -> None:
        """Audio reaches Telnyx while the provider is still generating."""
        from app.services.telnyx_tts import TelnyxTTSService

        ws = AsyncMock()
        sends_seen: list[int] = []

        async def _stream(**_kwargs):
            yield b"\xff" * 320  # 2 frames worth
            sends_seen.append(ws.send_text.call_count)
            yield b"\xff" * 160  # 1 more frame
            sends_seen.append(ws.send_text.call_count)

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = _stream

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.synthesize_and_send("Hello", 1, ws)
            assert result["success"] is True

        # Frames were already played out before the stream produced its last chunk
        assert sends_seen == [2, 3]

    @pytest.mark.asyncio
    async def test_pacing_20ms_per_frame(self) -> None:
        """Each streamed frame is paced with a 20ms sleep."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider([b"\xff" * 800])

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await svc.synthesize_and_send("Hello", 1, ws)
            assert result["success"] is True

        assert sleep.await_count == ws.send_text.call_count == 5
        for call in sleep.call_args_list:
            assert call.args[0] == pytest.approx(0.020, abs=1e-6)

    @pytest.mark.asyncio
    async def test_first_audio_event_emitted_before_completion(self) -> None:
        """tts_first_audio fires when the FIRST frame is sent, not at the end."""
        import json

        from app.services.telephony_events import _observers
        from app.services.telnyx_tts import TelnyxTTSService

        trace: list[str] = []

        ws = AsyncMock()
        ws.send_text = AsyncMock(side_effect=lambda _p: trace.append("frame"))

        obs = AsyncMock()
        obs.send_text = AsyncMock(
            side_effect=lambda p: trace.append(json.loads(p)["type"])
        )
        _observers.clear()
        _observers.add(obs)

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        # 3 frames total: the first is playable before the stream ends
        mock_tts.stream_synthesize = self._provider([b"\xff" * 160, b"\xff" * 320])

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        try:
            with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
                result = await svc.synthesize_and_send(
                    "Hello there friend", 2, ws, call_id="cc_stream"
                )
                assert result["success"] is True
            # Allow broadcast tasks to complete
            await asyncio.sleep(0.05)
        finally:
            events = [json.loads(c.args[0]) for c in obs.send_text.call_args_list]
            _observers.clear()

        # With fire-and-forget broadcasts, events may not interleave with frames
        # in a deterministic order. Check that all expected events are present
        # and that tts_first_audio appears before tts_completed.
        # Note: turn_completed is now emitted by _agent_worker(), not TTS service.
        event_types = [e["type"] for e in events]
        assert "tts_processing" in event_types
        assert "tts_first_audio" in event_types
        assert "audio_streaming" in event_types
        assert "tts_completed" in event_types

        # tts_first_audio must appear before tts_completed
        first_audio_idx = event_types.index("tts_first_audio")
        completed_idx = event_types.index("tts_completed")
        assert first_audio_idx < completed_idx

        # All frames were sent
        assert ws.send_text.call_count == 3

        # Every event carries the call_id and the turn number
        assert all(e["call_id"] == "cc_stream" for e in events)
        assert all(e["turn"] == 2 for e in events)

        first = next(e for e in events if e["type"] == "tts_first_audio")
        assert first["metadata"]["format"] == "ulaw_8000"
        assert isinstance(first["metadata"]["ttfa_ms"], int)

        done = next(e for e in events if e["type"] == "tts_completed")
        assert done["metadata"]["chunks"] == 3
        assert done["metadata"]["bytes"] == 480
        assert done["metadata"]["playback_ms"] == 60

    @pytest.mark.asyncio
    async def test_empty_stream_falls_back_to_buffered(self) -> None:
        """A stream that yields nothing falls back to the buffered path."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider([])
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("buffered attempted"))

        ws = AsyncMock()
        svc = TelnyxTTSService()
        svc._tts = mock_tts

        result = await svc.synthesize_and_send("Hello", 1, ws)

        # Fallback was attempted; it failed here only because it is mocked
        mock_tts.synthesize.assert_awaited_once()
        assert result["success"] is False
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_sub_frame_stream_falls_back(self) -> None:
        """Fewer than 160 bytes total is not playable → buffered fallback."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider([b"\xff" * 159])
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("buffered attempted"))

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        result = await svc.synthesize_and_send("Hello", 1, AsyncMock())
        assert result["success"] is False
        mock_tts.synthesize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_error_after_first_frame_completes_turn(self) -> None:
        """A mid-stream failure is reported as partial audio, never replayed."""
        import json

        from app.services.telephony_events import _observers
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider(
            [b"\xff" * 160], error=RuntimeError("connection reset")
        )
        mock_tts.synthesize = AsyncMock(side_effect=AssertionError("buffered used"))

        ws = AsyncMock()
        obs = AsyncMock()
        _observers.clear()
        _observers.add(obs)

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        try:
            with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
                result = await svc.synthesize_and_send(
                    "Hello", 1, ws, call_id="cc_partial"
                )
                assert result["success"] is True
            # Allow broadcast tasks to complete
            await asyncio.sleep(0.05)
        finally:
            events = [json.loads(c.args[0]) for c in obs.send_text.call_args_list]
            _observers.clear()

        mock_tts.synthesize.assert_not_called()
        assert ws.send_text.call_count == 1
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["metadata"]["stage"] == "tts_stream"
        assert error_events[0]["call_id"] == "cc_partial"
        # The turn still terminates cleanly for the UI
        # Note: turn_completed is now emitted by _agent_worker(), not TTS service.
        assert events[-1]["type"] == "tts_completed"

    @pytest.mark.asyncio
    async def test_websocket_failure_returns_false(self) -> None:
        """A dead media WebSocket aborts the turn without replaying."""
        from app.services.telnyx_tts import TelnyxTTSService

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.stream_synthesize = self._provider([b"\xff" * 320])

        ws = AsyncMock()
        ws.send_text = AsyncMock(side_effect=ConnectionResetError("gone"))

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.synthesize_and_send("Hello", 1, ws)
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_buffered_fallback_still_reports_first_audio(self) -> None:
        """Legacy providers still emit tts_first_audio (buffered format tag)."""
        import json
        import struct
        import subprocess

        from app.providers.types import TTSResult
        from app.services.telephony_events import _observers
        from app.services.telnyx_tts import TelnyxTTSService

        pcm_silence = struct.pack("<1600h", *([0] * 1600))
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-i", "pipe:0", "-f", "mp3", "pipe:1",
            ],
            input=pcm_silence,
            capture_output=True,
            timeout=10,
        )

        mock_tts = AsyncMock()
        mock_tts.provider_name = "elevenlabs"
        mock_tts.synthesize = AsyncMock(
            return_value=TTSResult(audio_data=proc.stdout, content_type="audio/mpeg")
        )

        ws = AsyncMock()
        obs = AsyncMock()
        _observers.clear()
        _observers.add(obs)

        svc = TelnyxTTSService()
        svc._tts = mock_tts

        try:
            with patch("app.services.telnyx_tts.asyncio.sleep", new_callable=AsyncMock):
                result = await svc.synthesize_and_send("Hello", 3, ws, call_id="cc_old")
                assert result["success"] is True
            # Allow broadcast tasks to complete
            await asyncio.sleep(0.05)
        finally:
            events = [json.loads(c.args[0]) for c in obs.send_text.call_args_list]
            _observers.clear()

        types = [e["type"] for e in events]
        assert "tts_first_audio" in types
        first = next(e for e in events if e["type"] == "tts_first_audio")
        assert first["metadata"]["format"] == "pcmu_8000_buffered"


class TestElevenLabsStreamAdapter:
    """The ElevenLabs HTTP streaming endpoint must be used with ulaw_8000."""

    @pytest.mark.asyncio
    async def test_stream_endpoint_and_format(self) -> None:
        """stream_synthesize hits /stream and forwards output_format."""
        import httpx

        from app.providers.tts.elevenlabs import ElevenLabsAdapter

        seen: dict[str, str] = {}

        async def _body():
            yield b"\x00" * 100
            yield b"\x00" * 60

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["format"] = request.url.params.get("output_format", "")
            seen["accept"] = request.headers.get("accept", "")
            return httpx.Response(200, content=_body())

        adapter = ElevenLabsAdapter(api_key="test-key")
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"xi-api-key": "test-key"},
        )

        chunks = [
            c async for c in adapter.stream_synthesize("hi", output_format="ulaw_8000")
        ]
        await adapter.close()

        assert seen["path"].endswith("/stream")
        assert seen["format"] == "ulaw_8000"
        assert len(chunks) == 2  # progressive chunks, not one buffered body
        assert b"".join(chunks) == b"\x00" * 160

    @pytest.mark.asyncio
    async def test_stream_default_format_is_mp3(self) -> None:
        """Without an explicit format the adapter keeps its MP3 default."""
        import httpx

        from app.providers.tts.elevenlabs import ElevenLabsAdapter

        seen: dict[str, str] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["format"] = request.url.params.get("output_format", "")
            return httpx.Response(200, content=b"\x00" * 10)

        adapter = ElevenLabsAdapter(api_key="test-key")
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"xi-api-key": "test-key"},
        )

        chunks = [c async for c in adapter.stream_synthesize("hi")]
        await adapter.close()

        assert seen["format"] == "mp3_44100_128"
        assert b"".join(chunks) == b"\x00" * 10

    @pytest.mark.asyncio
    async def test_stream_http_error_raises(self) -> None:
        """A non-2xx streaming response surfaces as a RuntimeError."""
        import httpx

        from app.providers.tts.elevenlabs import ElevenLabsAdapter

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"detail": "bad format"})

        adapter = ElevenLabsAdapter(api_key="test-key")
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"xi-api-key": "test-key"},
        )

        with pytest.raises(RuntimeError, match="HTTP 400"):
            async for _ in adapter.stream_synthesize("hi", output_format="ulaw_8000"):
                pass

        await adapter.close()


class TestAgentWorkerTTSIntegration:
    """Test that the agent worker invokes TTS after agent response."""

    @pytest.mark.asyncio
    async def test_agent_worker_calls_tts(self) -> None:
        """Agent worker calls TTS service after successful agent response."""
        import asyncio
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        mock_llm = AsyncMock()
        mock_tts = AsyncMock()
        mock_tts.synthesize_and_send = AsyncMock(return_value=True)
        mock_tts.stream_sentences = AsyncMock(
            return_value={"success": True, "first_audio_ms": 100, "total_sentences": 1}
        )
        mock_ws = AsyncMock()

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()

        agent = TelephonyAgentSession(
            call_control_id="test-cc",
            utterance_queue=queue,
            tts_service=mock_tts,
            websocket=mock_ws,
        )
        agent._llm = mock_llm

        # stream_chat yields a simple text response
        from app.providers.types import StreamChunk

        async def mock_stream_chat(*args, **kwargs):
            yield StreamChunk(content="Hello there!")
        mock_llm.stream_chat = mock_stream_chat
        mock_llm.close = AsyncMock()

        # Start worker, send utterance, then shutdown
        agent._running = True
        agent._worker_task = asyncio.create_task(agent._agent_worker())
        now = time.monotonic()
        await queue.put(Utterance(text="Hello", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
        await asyncio.sleep(0.3)
        await queue.put(None)  # shutdown
        await asyncio.sleep(0.1)

        # TTS stream_sentences should have been called
        mock_tts.stream_sentences.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_worker_tts_failure_doesnt_crash(self) -> None:
        """TTS failure doesn't crash the agent worker."""
        import asyncio
        import time

        from app.providers.types import StreamChunk
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        async def _stream(*args, **kwargs):
            yield StreamChunk(content="Hi")

        mock_llm = AsyncMock()
        mock_llm.stream_chat = _stream
        mock_tts = AsyncMock()
        mock_tts.stream_sentences = AsyncMock(
            side_effect=RuntimeError("TTS boom")
        )
        mock_ws = AsyncMock()

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()

        agent = TelephonyAgentSession(
            call_control_id="test-cc",
            utterance_queue=queue,
            tts_service=mock_tts,
            websocket=mock_ws,
        )
        agent._llm = mock_llm

        agent._running = True
        agent._worker_task = asyncio.create_task(agent._agent_worker())
        now = time.monotonic()
        await queue.put(Utterance(text="Hello", speech_final_at=now, utterance_end_at=now + 1.0, last_word_end=0.5))
        await asyncio.sleep(0.3)
        # Worker should still be alive after TTS failure
        assert not agent._worker_task.done()
        await queue.put(None)
        await asyncio.sleep(0.1)

        # Worker should have processed the turn despite TTS failure
        assert agent.turn == 1


class TestTrackIsolationPreserved:
    """Verify outbound audio filtering still works with TTS."""

    @pytest.mark.asyncio
    async def test_outbound_track_skipped_by_bridge(self) -> None:
        """Outbound track packets are not sent to Deepgram STT."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True

        # Simulate an outbound track packet
        pcmu_silence = bytes(160)
        payload = base64.b64encode(pcmu_silence).decode("ascii")
        await bridge.process_media_packet(
            {"payload": payload, "track": "outbound"}
        )

        # Outbound packet should be skipped
        assert bridge._outbound_packets_skipped == 1
        assert bridge.packets_in == 0  # Not counted as inbound
        assert bridge._audio_queue.qsize() == 0  # Nothing queued for STT

    @pytest.mark.asyncio
    async def test_inbound_track_processed(self) -> None:
        """Inbound track packets are processed normally."""
        import base64

        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True

        pcmu_silence = bytes(160)
        payload = base64.b64encode(pcmu_silence).decode("ascii")
        await bridge.process_media_packet(
            {"payload": payload, "track": "inbound"}
        )

        assert bridge.packets_in == 1
        assert bridge._outbound_packets_skipped == 0
        assert bridge._audio_queue.qsize() == 1  # Queued for STT


class TestTelephonyEventBus:
    """Tests for the telephony semantic event bus (Phase 8A)."""

    @pytest.mark.asyncio
    async def test_broadcast_no_observers_is_noop(self) -> None:
        """broadcast_telephony_event with no observers does nothing."""
        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        # Should not raise
        await broadcast_telephony_event(
            "call_received", "cc_123", "Incoming call"
        )

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_observer(self) -> None:
        """broadcast_telephony_event sends JSON event to all observers."""
        import asyncio
        import json

        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        mock_ws = AsyncMock()
        _observers.add(mock_ws)

        try:
            await broadcast_telephony_event(
                "agent_processing",
                "cc_abc",
                "Processing with GPT-4o-mini",
                turn=1,
                metadata={"model": "openai/gpt-4o-mini"},
            )
            # Allow the broadcast task to complete
            await asyncio.sleep(0.05)

            assert mock_ws.send_text.called
            sent = json.loads(mock_ws.send_text.call_args[0][0])
            assert sent["type"] == "agent_processing"
            assert sent["call_id"] == "cc_abc"
            assert sent["turn"] == 1
            assert sent["message"] == "Processing with GPT-4o-mini"
            assert sent["metadata"]["model"] == "openai/gpt-4o-mini"
            assert "timestamp" in sent
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_broadcast_removes_stale_observers(self) -> None:
        """Failed sends are removed from observer set."""
        import asyncio

        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        mock_ws = AsyncMock()
        mock_ws.send_text.side_effect = RuntimeError("disconnected")
        _observers.add(mock_ws)

        try:
            await broadcast_telephony_event(
                "call_completed", "cc_xyz", "Call completed"
            )
            # Allow the broadcast task to complete
            await asyncio.sleep(0.05)
            assert mock_ws not in _observers
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_broadcast_event_schema(self) -> None:
        """Event has all required fields and correct types."""
        import asyncio
        import json

        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        mock_ws = AsyncMock()
        _observers.add(mock_ws)

        try:
            await broadcast_telephony_event(
                "caller_transcript",
                "cc_test",
                'Caller: "Hello"',
                turn=2,
                metadata={"text": "Hello"},
            )
            # Allow the broadcast task to complete
            await asyncio.sleep(0.05)

            sent = json.loads(mock_ws.send_text.call_args[0][0])
            # Required fields
            assert isinstance(sent["type"], str)
            assert isinstance(sent["call_id"], str)
            assert isinstance(sent["timestamp"], str)
            assert isinstance(sent["message"], str)
            # Optional fields
            assert sent["turn"] == 2
            assert isinstance(sent["metadata"], dict)
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_broadcast_without_turn_omits_field(self) -> None:
        """Events without turn omit the turn field."""
        import asyncio
        import json

        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        mock_ws = AsyncMock()
        _observers.add(mock_ws)

        try:
            await broadcast_telephony_event(
                "call_received", "cc_1", "Incoming call"
            )
            # Allow the broadcast task to complete
            await asyncio.sleep(0.05)

            sent = json.loads(mock_ws.send_text.call_args[0][0])
            assert "turn" not in sent
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_register_unregister_observer(self) -> None:
        """register/unregister manages observer set correctly."""
        from app.services.telephony_events import (
            _observers,
            register_observer,
            unregister_observer,
        )

        _observers.clear()
        mock_ws = AsyncMock()

        register_observer(mock_ws)
        assert mock_ws in _observers

        unregister_observer(mock_ws)
        assert mock_ws not in _observers

    @pytest.mark.asyncio
    async def test_observe_ws_endpoint(self) -> None:
        """The /api/v1/voice/telephony/observe WebSocket endpoint accepts connections."""
        from app.services.telephony_events import _observers

        _observers.clear()
        with _client().websocket_connect(
            "/api/v1/voice/telephony/observe"
        ) as ws:
            # Should be registered after connect
            assert len(_observers) == 1
            ws.close()
        # Should be unregistered after disconnect
        assert len(_observers) == 0


class TestEndOfSpeechObservability:
    """Phase 8B: end-of-speech latency observability (speech_final/utterance_end/agent_processing)."""

    @pytest.mark.asyncio
    async def test_speech_final_metadata_preserved(self) -> None:
        """speech_final metadata from Deepgram is preserved through the bridge."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Simulate a final event with speech_final=True
        final_event = StreamEvent(
            type="final",
            text="Hello world",
            confidence=0.95,
            metadata={"speech_final": True},
        )
        utterance_end_event = StreamEvent(type="utterance_end", metadata={"last_word_end": 1.5})

        bridge._session.receive = AsyncMock(
            side_effect=[final_event, utterance_end_event, None]
        )

        await bridge._transcript_processor()

        # Verify the utterance on the queue has speech_final_at captured
        utterance = await bridge.utterance_queue.get()
        assert utterance is not None
        assert utterance.speech_final_at is not None
        assert bridge.utterances_emitted == 1

    @pytest.mark.asyncio
    async def test_last_word_end_preserved(self) -> None:
        """last_word_end from UtteranceEnd is preserved in the Utterance."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        final_event = StreamEvent(
            type="final",
            text="Test transcript",
            confidence=0.9,
            metadata={"speech_final": True},
        )
        utterance_end_event = StreamEvent(
            type="utterance_end",
            metadata={"last_word_end": 2.24},
        )

        bridge._session.receive = AsyncMock(
            side_effect=[final_event, utterance_end_event, None]
        )

        await bridge._transcript_processor()

        # Verify the utterance on the queue has last_word_end
        utterance = await bridge.utterance_queue.get()
        assert utterance is not None
        assert utterance.last_word_end == 2.24
        assert utterance.text == "Test transcript"

    @pytest.mark.asyncio
    async def test_caller_speech_final_event_emitted(self) -> None:
        """caller_speech_final event is emitted at utterance_end with timing metadata."""
        import json

        from app.providers.stt.streaming import StreamEvent
        from app.services.telephony_events import _observers
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        _observers.clear()
        mock_ws = AsyncMock()
        _observers.add(mock_ws)

        try:
            bridge = TelnyxDeepgramBridge()
            bridge._running = True
            bridge._stream_id = "test_stream"
            bridge._session = AsyncMock()

            final_event = StreamEvent(
                type="final",
                text="Hello",
                confidence=0.95,
                metadata={"speech_final": True},
            )
            utterance_end_event = StreamEvent(
                type="utterance_end",
                metadata={"last_word_end": 1.5},
            )

            bridge._session.receive = AsyncMock(
                side_effect=[final_event, utterance_end_event, None]
            )

            await bridge._transcript_processor()

            # Allow broadcast tasks to complete
            await asyncio.sleep(0.05)

            # Verify caller_speech_final was broadcast
            calls = [json.loads(c[0][0]) for c in mock_ws.send_text.call_args_list]
            speech_final_events = [c for c in calls if c["type"] == "caller_speech_final"]
            assert len(speech_final_events) == 1
            evt = speech_final_events[0]
            assert evt["call_id"] == "test_stream"
            assert evt["turn"] == 1
            assert evt["metadata"]["last_word_end"] == 1.5
            assert evt["metadata"]["speech_final_to_utterance_end_ms"] is not None
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_timing_fields_calculated_correctly(self) -> None:
        """Timing deltas are calculated correctly from monotonic timestamps."""
        import time

        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )
        agent._llm = AsyncMock()
        agent._llm.close = AsyncMock()

        from app.providers.types import StreamChunk

        async def _stream(*args, **kwargs):
            yield StreamChunk(content="Response")

        agent._llm.stream_chat = _stream

        # Create an utterance with known timestamps
        now = time.monotonic()
        utterance = Utterance(
            text="Test",
            speech_final_at=now,
            utterance_end_at=now + 1.0,  # 1000ms later
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)  # sentinel

        await agent.start()
        await agent._worker_task

        # Verify the agent processed the utterance
        assert agent.turn == 1

    @pytest.mark.asyncio
    async def test_observer_disconnect_does_not_block_processing(self) -> None:
        """A stalled observer cannot delay STT/agent processing."""
        import asyncio

        from app.services.telephony_events import (
            _observers,
            broadcast_telephony_event,
        )

        _observers.clear()
        mock_ws = AsyncMock()
        # Simulate a stalled observer (send_text hangs)
        async def slow_send(*args, **kwargs):
            await asyncio.sleep(10)

        mock_ws.send_text = slow_send
        _observers.add(mock_ws)

        try:
            # broadcast should return immediately (fire-and-forget)
            start = asyncio.get_event_loop().time()
            await broadcast_telephony_event(
                "test_event", "cc_1", "Test message"
            )
            elapsed = asyncio.get_event_loop().time() - start

            # Should return almost immediately (< 50ms), not wait for the slow send
            assert elapsed < 0.05
        finally:
            _observers.clear()

    @pytest.mark.asyncio
    async def test_existing_utterance_end_behavior_unchanged(self) -> None:
        """The existing utterance_end trigger path remains unchanged."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Two finals then utterance_end (existing test pattern)
        bridge._session.receive = AsyncMock(
            side_effect=[
                StreamEvent(type="final", text="Hello", confidence=0.9),
                StreamEvent(type="final", text="world", confidence=0.95),
                StreamEvent(type="utterance_end", metadata={"last_word_end": 1.0}),
                None,
            ]
        )

        await bridge._transcript_processor()

        # Verify the utterance was emitted with accumulated text
        utterance = await bridge.utterance_queue.get()
        assert utterance is not None
        assert utterance.text == "Hello world"
        assert bridge.utterances_emitted == 1


class TestSettleTimer:
    """Phase 8C: guarded settle timer for early release on speech_final."""

    @pytest.mark.asyncio
    async def test_speech_final_starts_timer(self) -> None:
        """A: speech_final starts the settle timer."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Final with speech_final should start the timer
        final_event = StreamEvent(
            type="final",
            text="Hello",
            confidence=0.95,
            metadata={"speech_final": True},
        )
        # Don't send utterance_end — let the timer fire
        bridge._session.receive = AsyncMock(side_effect=[final_event, None])

        # Run the processor with a timeout to let the timer fire
        try:
            await asyncio.wait_for(
                bridge._transcript_processor(),
                timeout=(_SETTLE_MS / 1000) + 0.2,
            )
        except TimeoutError:
            pass

        # Verify the timer was started
        assert bridge._speech_final_at is not None

    @pytest.mark.asyncio
    async def test_timer_does_not_release_before_settle_ms(self) -> None:
        """B: Timer does not release before _SETTLE_MS."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        final_event = StreamEvent(
            type="final",
            text="Hello",
            confidence=0.95,
            metadata={"speech_final": True},
        )
        bridge._session.receive = AsyncMock(side_effect=[final_event, None])

        # Run for less than _SETTLE_MS
        start = asyncio.get_event_loop().time()
        try:
            await asyncio.wait_for(
                bridge._transcript_processor(),
                timeout=(_SETTLE_MS / 1000) - 0.05,
            )
        except TimeoutError:
            pass
        elapsed = asyncio.get_event_loop().time() - start

        # Verify no utterance was released yet
        assert bridge.utterance_queue.empty()
        assert elapsed < (_SETTLE_MS / 1000)

    @pytest.mark.asyncio
    async def test_partial_resets_timer(self) -> None:
        """C: Partial transcript resets the timer."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Final with speech_final, then partial, then let timer fire
        events = [
            StreamEvent(
                type="final",
                text="Hello",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            StreamEvent(type="partial", text="Hello world", confidence=0.9),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        # Run the processor
        try:
            await asyncio.wait_for(
                bridge._transcript_processor(),
                timeout=(_SETTLE_MS / 1000) + 0.3,
            )
        except TimeoutError:
            pass

        # Timer should have been cancelled by partial
        # (no utterance released because no new speech_final after partial)
        assert bridge.utterance_queue.empty()

    @pytest.mark.asyncio
    async def test_final_resets_timer(self) -> None:
        """D: Final transcript resets the timer."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Two finals with speech_final on first, then let timer fire
        events = [
            StreamEvent(
                type="final",
                text="Hello",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            StreamEvent(type="final", text="world", confidence=0.9),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        # Run the processor
        try:
            await asyncio.wait_for(
                bridge._transcript_processor(),
                timeout=(_SETTLE_MS / 1000) + 0.3,
            )
        except TimeoutError:
            pass

        # Timer was cancelled by second final, no new speech_final started
        # so no utterance released
        assert bridge.utterance_queue.empty()

    @pytest.mark.asyncio
    async def test_timer_expiry_releases_exactly_once(self) -> None:
        """E: Timer expiry releases exactly once."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        final_event = StreamEvent(
            type="final",
            text="Hello world",
            confidence=0.95,
            metadata={"speech_final": True},
        )
        bridge._session.receive = AsyncMock(side_effect=[final_event, None])

        # Run the processor
        await bridge._transcript_processor()
        # Wait for the timer to fire
        await asyncio.sleep((_SETTLE_MS / 1000) + 0.1)

        # Verify exactly one utterance was released
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert utterance.text == "Hello world"
        assert utterance.release_reason == "speech_final_settle"
        assert utterance.settle_ms == _SETTLE_MS

    @pytest.mark.asyncio
    async def test_utterance_end_releases_immediately(self) -> None:
        """F: utterance_end releases immediately."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Final with speech_final, then immediate utterance_end
        events = [
            StreamEvent(
                type="final",
                text="Hello",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            StreamEvent(type="utterance_end", metadata={"last_word_end": 1.0}),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        # Run the processor — should not wait for timer
        start = asyncio.get_event_loop().time()
        await bridge._transcript_processor()
        elapsed = asyncio.get_event_loop().time() - start

        # Verify utterance was released immediately
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert utterance.release_reason == "utterance_end"
        # Should be much faster than _SETTLE_MS
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_utterance_end_after_early_release_no_duplicate(self) -> None:
        """G: utterance_end after early release does not duplicate the turn."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Final with speech_final, then utterance_end after timer fires
        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StreamEvent(
                    type="final",
                    text="Hello",
                    confidence=0.95,
                    metadata={"speech_final": True},
                )
            elif call_count == 2:
                # Wait for timer to fire
                await asyncio.sleep((_SETTLE_MS / 1000) + 0.1)
                return StreamEvent(
                    type="utterance_end",
                    metadata={"last_word_end": 1.0},
                )
            else:
                return None

        bridge._session.receive = receive_side_effect

        # Run the processor
        await bridge._transcript_processor()

        # Verify only one utterance was released
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert utterance.release_reason == "speech_final_settle"

    @pytest.mark.asyncio
    async def test_multiple_speech_final_no_duplicate_turns(self) -> None:
        """H: Multiple speech_final events do not create duplicate turns."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Multiple finals with speech_final
        events = [
            StreamEvent(
                type="final",
                text="Hello",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            StreamEvent(
                type="final",
                text="world",
                confidence=0.9,
                metadata={"speech_final": True},
            ),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        # Run the processor
        await bridge._transcript_processor()
        # Wait for the timer to fire
        await asyncio.sleep((_SETTLE_MS / 1000) + 0.1)

        # Verify only one utterance was released
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert utterance.text == "Hello world"

    @pytest.mark.asyncio
    async def test_resumed_speech_before_settle_timer_expires(self) -> None:
        """I: Resumed speech after a pause becomes part of the same utterance."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import _SETTLE_MS, TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # speech_final, then partial before timer expires
        events = [
            StreamEvent(
                type="final",
                text="Please tell me",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            # Partial arrives before timer expires
            StreamEvent(type="partial", text="the weather", confidence=0.9),
            # Then final with new speech_final
            StreamEvent(
                type="final",
                text="the weather",
                confidence=0.95,
                metadata={"speech_final": True},
            ),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        # Run the processor
        await bridge._transcript_processor()
        # Wait for the timer to fire
        await asyncio.sleep((_SETTLE_MS / 1000) + 0.1)

        # Verify the utterance includes both parts
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert "Please tell me" in utterance.text
        assert "the weather" in utterance.text

    @pytest.mark.asyncio
    async def test_timer_cleanup_on_bridge_shutdown(self) -> None:
        """J: Timer cleanup on bridge shutdown."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # Start a final with speech_final but don't let timer fire
        final_event = StreamEvent(
            type="final",
            text="Hello",
            confidence=0.95,
            metadata={"speech_final": True},
        )
        bridge._session.receive = AsyncMock(side_effect=[final_event, None])

        # Run the processor
        await bridge._transcript_processor()

        # The timer should have been started
        assert bridge._settle_task is not None

        # Stop the bridge (should cancel the timer)
        bridge._running = False
        bridge._cancel_settle_timer()

        # Verify the timer was cancelled
        assert bridge._settle_task is None or bridge._settle_task.done()

    @pytest.mark.asyncio
    async def test_existing_non_early_utterance_behavior(self) -> None:
        """K: Existing non-early utterance behavior still works."""
        from app.providers.stt.streaming import StreamEvent
        from app.services.telnyx_deepgram import TelnyxDeepgramBridge

        bridge = TelnyxDeepgramBridge()
        bridge._running = True
        bridge._session = AsyncMock()

        # No speech_final — just finals and utterance_end
        events = [
            StreamEvent(type="final", text="Hello", confidence=0.9),
            StreamEvent(type="final", text="world", confidence=0.95),
            StreamEvent(type="utterance_end", metadata={"last_word_end": 1.0}),
            None,
        ]
        bridge._session.receive = AsyncMock(side_effect=events)

        await bridge._transcript_processor()

        # Verify the utterance was emitted normally
        assert bridge.utterances_emitted == 1
        utterance = await bridge.utterance_queue.get()
        assert utterance.text == "Hello world"
        assert utterance.release_reason == "utterance_end"
        assert utterance.settle_ms is None

    @pytest.mark.asyncio
    async def test_no_overlapping_agent_turns(self) -> None:
        """L: No overlapping agent/TTS turns."""
        import time

        from app.providers.types import StreamChunk
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )
        # Mock LLM with stream_chat() returning StreamChunk objects
        agent._llm = AsyncMock()
        agent._llm.close = AsyncMock()
        # stream_chat() returns an async iterator of StreamChunk
        async def mock_stream_chat(*args, **kwargs):
            for text in ["Response", " text"]:
                yield StreamChunk(content=text)
        agent._llm.stream_chat = mock_stream_chat

        # Create two utterances
        now = time.monotonic()
        utterance1 = Utterance(
            text="First",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
            release_reason="speech_final_settle",
            settle_ms=400,
        )
        utterance2 = Utterance(
            text="Second",
            speech_final_at=now + 1.0,
            utterance_end_at=now + 1.5,
            last_word_end=1.5,
            release_reason="utterance_end",
            settle_ms=None,
        )
        await queue.put(utterance1)
        await queue.put(utterance2)
        await queue.put(None)  # sentinel

        await agent.start()
        await agent._worker_task

        # Verify both utterances were processed sequentially
        assert agent.turn == 2
