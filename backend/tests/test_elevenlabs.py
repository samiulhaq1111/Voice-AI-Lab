"""Tests for the ElevenLabs TTS adapter with mocked HTTP."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.providers.tts.elevenlabs import ElevenLabsAdapter
from app.providers.types import TTSResult


@pytest.fixture
def adapter() -> ElevenLabsAdapter:
    return ElevenLabsAdapter(
        api_key="test-key",
        default_voice="test-voice-id",
        default_model="eleven_monolingual_v1",
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
            with pytest.raises(RuntimeError, match="authentication failed"):
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
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await adapter.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_close(self, adapter: ElevenLabsAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
        mock_close.assert_called_once()
