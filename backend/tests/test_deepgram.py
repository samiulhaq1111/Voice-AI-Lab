"""Tests for the Deepgram STT adapter with mocked HTTP."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.providers.stt.deepgram import DeepgramAdapter
from app.providers.types import STTResult


@pytest.fixture
def adapter() -> DeepgramAdapter:
    return DeepgramAdapter(api_key="test-key", default_model="nova-3")


class TestDeepgramAdapter:
    """Tests for DeepgramAdapter."""

    def test_provider_name(self, adapter: DeepgramAdapter) -> None:
        assert adapter.provider_name == "deepgram"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        from app.core.config import settings

        settings.deepgram_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            DeepgramAdapter(api_key="")

    @pytest.mark.asyncio
    async def test_transcribe_success(self, adapter: DeepgramAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "Hello world",
                                    "confidence": 0.95,
                                }
                            ]
                        }
                    ],
                    "language": "en",
                    "metadata": {"duration": 2.5},
                }
            },
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.transcribe(b"fake-audio-data")

        assert isinstance(result, STTResult)
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.is_final is True
        assert result.duration_seconds == 2.5

    @pytest.mark.asyncio
    async def test_transcribe_empty_response(self, adapter: DeepgramAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={"results": {"channels": []}},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.transcribe(b"fake-audio")

        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_auth_failure(self, adapter: DeepgramAdapter) -> None:
        mock_response = httpx.Response(
            401,
            json={"error": "Unauthorized"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="authentication failed"):
                await adapter.transcribe(b"fake-audio")

    @pytest.mark.asyncio
    async def test_transcribe_server_error(self, adapter: DeepgramAdapter) -> None:
        mock_response = httpx.Response(
            500,
            json={"error": "Internal server error"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await adapter.transcribe(b"fake-audio")

    @pytest.mark.asyncio
    async def test_transcribe_timeout(self, adapter: DeepgramAdapter) -> None:
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await adapter.transcribe(b"fake-audio")

    @pytest.mark.asyncio
    async def test_transcribe_with_model_override(self, adapter: DeepgramAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "test", "confidence": 0.9}]}
                    ],
                    "metadata": {},
                }
            },
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await adapter.transcribe(b"audio", model="nova-2")

        assert result.text == "test"
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["params"]["model"] == "nova-2"

    @pytest.mark.asyncio
    async def test_close(self, adapter: DeepgramAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
        mock_close.assert_called_once()
