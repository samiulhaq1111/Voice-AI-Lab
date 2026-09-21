"""Focused tests for the Telnyx telephony webhook endpoint."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

TELEPHONY_WEBHOOK_PATH = "/api/v1/voice/telephony/webhook"


def _client() -> TestClient:
    return TestClient(app)


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
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
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
        assert "evt_456" in caplog.text
        assert "cc_def" in caplog.text

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
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
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
        with (
            patch(
                "app.services.telnyx_client.TelnyxCallControlClient",
            ) as mock_client_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_log_ok"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once_with("cc_log_ok")
        # Webhook logs the call_control_id from the payload
        assert "cc_log_ok" in caplog.text

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
        mock_close = AsyncMock()
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = mock_close
            _client().post(
                TELEPHONY_WEBHOOK_PATH, json=_initiated_payload()
            )
        mock_close.assert_awaited_once()


class TestTelnyxCallControlClient:
    """Unit tests for TelnyxCallControlClient with mocked httpx."""

    def test_answer_call_success(self) -> None:
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

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            client.answer_call("cc_test")
        )
        assert result == {"data": {"result": "ok"}}
        mock_client.post.assert_awaited_once_with(
            "/calls/cc_test/actions/answer", json={}
        )

    def test_answer_call_http_error(self) -> None:
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

        import asyncio

        try:
            asyncio.get_event_loop().run_until_complete(
                client.answer_call("cc_bad")
            )
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

    def test_speak_call_success(self) -> None:
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

        import asyncio

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            result = asyncio.get_event_loop().run_until_complete(
                client.speak("cc_speak", "Hello, this is the Voice AI Lab.")
            )
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

    def test_speak_call_http_error(self) -> None:
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

        import asyncio

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            try:
                asyncio.get_event_loop().run_until_complete(
                    client.speak("cc_bad_speak", "Hello")
                )
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "400" in str(e)
                assert "Invalid call_control_id" in str(e)

    def test_speak_custom_voice_override(self) -> None:
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

        import asyncio

        with patch("app.services.telnyx_client.settings") as mock_settings:
            mock_settings.telnyx_speak_voice = "female"
            asyncio.get_event_loop().run_until_complete(
                client.speak(
                    "cc_voice", "Hi", voice="Telnyx.KokoroTTS.af"
                )
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
        with patch(
            "app.services.telnyx_client.TelnyxCallControlClient",
        ) as mock_client_cls:
            mock_client_cls.return_value.answer_call = mock_answer
            mock_client_cls.return_value.speak = mock_speak
            mock_client_cls.return_value.close = AsyncMock()
            resp = _client().post(
                TELEPHONY_WEBHOOK_PATH,
                json=_initiated_payload(call_control_id="cc_speak_test"),
            )
        assert resp.status_code == 200
        mock_answer.assert_awaited_once_with("cc_speak_test")
        mock_speak.assert_awaited_once_with(
            "cc_speak_test", "Hello, this is the Voice AI Lab."
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
