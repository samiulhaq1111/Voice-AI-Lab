"""Tests for the ElevenLabs TTS adapter with mocked HTTP."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.providers.tts.elevenlabs import ElevenLabsAdapter
from app.providers.types import TTSResult


@pytest.fixture
def adapter() -> ElevenLabsAdapter:
    return ElevenLabsAdapter(
        api_key="test-key",
        default_voice="test-voice-id",
        default_model="eleven_flash_v2_5",
    )


class TestElevenLabsAdapter:
    """Tests for ElevenLabsAdapter."""

    def test_provider_name(self, adapter: ElevenLabsAdapter) -> None:
        assert adapter.provider_name == "elevenlabs"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        from app.core.config import settings

        settings.elevenlabs_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            ElevenLabsAdapter(api_key="")

    @pytest.mark.asyncio
    async def test_synthesize_success(self, adapter: ElevenLabsAdapter) -> None:
        fake_audio = b"\x00\x01\x02fake-audio-data"
        mock_response = httpx.Response(
            200,
            content=fake_audio,
            headers={"content-type": "audio/mpeg"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.synthesize("Hello world")

        assert isinstance(result, TTSResult)
        assert result.audio_data == fake_audio
        assert result.content_type == "audio/mpeg"
        assert result.metadata["provider"] == "elevenlabs"
        assert result.metadata["voice"] == "test-voice-id"

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_override(self, adapter: ElevenLabsAdapter) -> None:
        mock_response = httpx.Response(
            200,
            content=b"audio",
            headers={"content-type": "audio/mpeg"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/custom-voice"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await adapter.synthesize("Test", voice="custom-voice")

        assert result.audio_data == b"audio"
        call_url = mock_post.call_args.args[0]
        assert "custom-voice" in call_url

    @pytest.mark.asyncio
    async def test_synthesize_auth_failure(self, adapter: ElevenLabsAdapter) -> None:
        mock_response = httpx.Response(
            401,
            json={"detail": "Unauthorized"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="authentication failed.*401"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_payment_required_402(self, adapter: ElevenLabsAdapter) -> None:
        """HTTP 402 should raise payment required error."""
        mock_response = httpx.Response(
            402,
            json={"detail": {"message": "Insufficient credits", "status": "payment_required"}},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="billing/entitlement"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_invalid_input(self, adapter: ElevenLabsAdapter) -> None:
        mock_response = httpx.Response(
            422,
            json={"detail": "Invalid input"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="invalid input"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_timeout(self, adapter: ElevenLabsAdapter) -> None:
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_server_error(self, adapter: ElevenLabsAdapter) -> None:
        mock_response = httpx.Response(
            500,
            json={"error": "Internal error"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="server error.*500"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_bad_request_400(self, adapter: ElevenLabsAdapter) -> None:
        """HTTP 400 should log error detail and raise RuntimeError."""
        mock_response = httpx.Response(
            400,
            json={"detail": {"message": "Invalid model", "status": "invalid_model"}},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="API error.*400"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_forbidden_403(self, adapter: ElevenLabsAdapter) -> None:
        """HTTP 403 should raise authorization error."""
        mock_response = httpx.Response(
            403,
            json={"detail": "Forbidden"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="authorization failed"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_rate_limit_429(self, adapter: ElevenLabsAdapter) -> None:
        """HTTP 429 should raise rate limit error."""
        mock_response = httpx.Response(
            429,
            json={"detail": "Rate limit exceeded"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="rate limit"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_model_override(self, adapter: ElevenLabsAdapter) -> None:
        """Model override should be used in request payload."""
        mock_response = httpx.Response(
            200,
            content=b"audio",
            headers={"content-type": "audio/mpeg"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await adapter.synthesize("Test", model="eleven_multilingual_v2")

        call_payload = mock_post.call_args.kwargs["json"]
        assert call_payload["model_id"] == "eleven_multilingual_v2"

    @pytest.mark.asyncio
    async def test_synthesize_voice_override(self, adapter: ElevenLabsAdapter) -> None:
        """Voice override should appear in URL."""
        mock_response = httpx.Response(
            200,
            content=b"audio",
            headers={"content-type": "audio/mpeg"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/override-voice"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await adapter.synthesize("Test", voice="override-voice")

        call_url = mock_post.call_args.args[0]
        assert "override-voice" in call_url
        assert result.metadata["voice"] == "override-voice"

    @pytest.mark.asyncio
    async def test_close(self, adapter: ElevenLabsAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_format_sent_as_query_param(self, adapter: ElevenLabsAdapter) -> None:
        """output_format must be a query parameter, not in the JSON body."""
        mock_response = httpx.Response(
            200,
            content=b"audio",
            headers={"content-type": "audio/mpeg"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await adapter.synthesize("Test")

        # Verify output_format is in query params, NOT in JSON body
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["params"]["output_format"] == "mp3_44100_128"
        assert "output_format" not in call_kwargs["json"]


class TestElevenLabsErrorDiagnostics:
    """Focused tests for structured ElevenLabs error detail preservation."""

    @pytest.mark.asyncio
    async def test_401_structured_error_preserves_details(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 401 with structured ElevenLabs error body preserves all fields."""
        mock_response = httpx.Response(
            401,
            json={
                "detail": {
                    "message": "Invalid API key",
                    "status": "invalid_api_key",
                    "request_id": "req_abc123",
                },
                "request_id": "req_abc123",
            },
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            # Must contain status code
            assert "401" in error_msg
            # Must contain error code from detail.status
            assert "invalid_api_key" in error_msg
            # Must contain error message from detail.message
            assert "Invalid API key" in error_msg
            # Must indicate authentication category
            assert "authentication failed" in error_msg

    @pytest.mark.asyncio
    async def test_402_not_reported_as_authentication(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 402 must NOT be reported as authentication failure."""
        mock_response = httpx.Response(
            402,
            json={
                "detail": {
                    "message": "Insufficient credits",
                    "status": "payment_required",
                }
            },
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            # Must indicate billing, NOT authentication
            assert "billing" in error_msg or "entitlement" in error_msg
            assert "402" in error_msg
            assert "authentication" not in error_msg
            assert "Insufficient credits" in error_msg

    @pytest.mark.asyncio
    async def test_429_not_reported_as_authentication(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 429 must NOT be reported as authentication failure."""
        mock_response = httpx.Response(
            429,
            json={
                "detail": {
                    "message": "Rate limit exceeded",
                    "status": "rate_limit_error",
                }
            },
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            # Must indicate rate limit, NOT authentication
            assert "rate limit" in error_msg
            assert "429" in error_msg
            assert "authentication" not in error_msg

    @pytest.mark.asyncio
    async def test_401_with_string_detail(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 401 with a plain string detail (not dict) is handled safely."""
        mock_response = httpx.Response(
            401,
            json={"detail": "Unauthorized"},
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            assert "401" in error_msg
            assert "authentication failed" in error_msg
            assert "Unauthorized" in error_msg

    @pytest.mark.asyncio
    async def test_401_with_non_json_body(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 401 with non-JSON body is handled safely."""
        mock_response = httpx.Response(
            401,
            text="<html>Unauthorized</html>",
            headers={"content-type": "text/html"},
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            assert "401" in error_msg
            assert "authentication failed" in error_msg

    @pytest.mark.asyncio
    async def test_500_server_error_distinct(
        self, adapter: ElevenLabsAdapter
    ) -> None:
        """A 500 must be reported as server error, not authentication."""
        mock_response = httpx.Response(
            500,
            json={"detail": {"message": "Internal server error"}},
            request=httpx.Request(
                "POST", "https://api.elevenlabs.io/v1/text-to-speech/test-voice-id"
            ),
        )

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await adapter.synthesize("Hello")

            error_msg = str(exc_info.value)
            assert "server error" in error_msg
            assert "500" in error_msg
            assert "authentication" not in error_msg
