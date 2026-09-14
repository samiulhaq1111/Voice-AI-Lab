"""Tests for the Deepgram STT adapter with mocked HTTP."""

import logging

import httpx
import pytest
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


# ---------------------------------------------------------------------------
# Tests: Deepgram Configuration Safety
# ---------------------------------------------------------------------------


class TestDeepgramConfigSafety:
    """Verify API key is never exposed and auth header is correct."""

    def test_missing_key_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter with empty key raises ValueError when settings also empty."""
        from app.core.config import settings

        monkeypatch.setattr(settings, "deepgram_api_key", "")
        with pytest.raises(ValueError, match="API key not configured"):
            DeepgramAdapter(api_key="")

    def test_adapter_stores_key_internally(self) -> None:
        """Adapter can be instantiated with a key."""
        adapter = DeepgramAdapter(api_key="my-secret-key")
        assert adapter.provider_name == "deepgram"
        # Key is stored internally, not exposed via any public property
        assert adapter._api_key == "my-secret-key"

    def test_no_public_key_property(self) -> None:
        """Adapter has no public attribute exposing the API key."""
        adapter = DeepgramAdapter(api_key="my-secret-key")
        # Only internal _api_key should hold it
        public_attrs = {
            k: v for k, v in vars(adapter).items() if not k.startswith("_")
        }
        for val in public_attrs.values():
            assert "my-secret-key" not in str(val)

    def test_authorization_header_format(self) -> None:
        """Authorization header uses 'Token <key>' format."""
        adapter = DeepgramAdapter(api_key="test-token-123")
        # The httpx client default headers contain the auth
        auth_header = adapter._client.headers.get("Authorization", "")
        assert auth_header == "Token test-token-123"

    @pytest.mark.asyncio
    async def test_api_key_never_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """API key value must never appear in log output."""
        secret = "super-secret-dg-key-9999"
        adapter = DeepgramAdapter(api_key=secret)

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

        with caplog.at_level(logging.DEBUG, logger="voice_ai_lab"):
            with patch.object(
                adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
            ):
                await adapter.transcribe(b"fake-audio")

        # The secret must NOT appear in any log record
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in str(record.__dict__)

    @pytest.mark.asyncio
    async def test_log_contains_api_key_configured_true(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log should contain api_key_configured=true but not the key itself."""
        adapter = DeepgramAdapter(api_key="some-key")

        mock_response = httpx.Response(
            200,
            json={
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "hi", "confidence": 0.8}]}
                    ],
                    "metadata": {},
                }
            },
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )

        with caplog.at_level(logging.INFO, logger="voice_ai_lab"):
            with patch.object(
                adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
            ):
                await adapter.transcribe(b"audio-data")

        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "api_key_configured=True" in log_text
        assert "some-key" not in log_text


# ---------------------------------------------------------------------------
# Tests: HTTP Response Diagnostics
# ---------------------------------------------------------------------------


class TestDeepgramHTTPDiagnostics:
    """Verify safe error handling for all HTTP status codes."""

    @pytest.fixture
    def adapter(self) -> DeepgramAdapter:
        return DeepgramAdapter(api_key="test-key", default_model="nova-3")

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self, adapter: DeepgramAdapter) -> None:
        mock_resp = httpx.Response(
            401,
            json={"err_msg": "Authentication error"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="authentication failed"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_403_raises_error(self, adapter: DeepgramAdapter) -> None:
        mock_resp = httpx.Response(
            403,
            json={"err_msg": "Forbidden"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 403"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_400_raises_bad_request(self, adapter: DeepgramAdapter) -> None:
        mock_resp = httpx.Response(
            400,
            json={"err_msg": "Bad request"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit(self, adapter: DeepgramAdapter) -> None:
        mock_resp = httpx.Response(
            429,
            json={"err_msg": "Rate limit exceeded"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 429"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_500_raises_server_error(self, adapter: DeepgramAdapter) -> None:
        mock_resp = httpx.Response(
            500,
            json={"error": "Internal server error"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, adapter: DeepgramAdapter) -> None:
        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("connection timeout"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await adapter.transcribe(b"audio")

    @pytest.mark.asyncio
    async def test_error_messages_do_not_contain_key(self, adapter: DeepgramAdapter) -> None:
        """Error messages must never contain the API key."""
        mock_resp = httpx.Response(
            401,
            json={"err_msg": "Unauthorized"},
            request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            try:
                await adapter.transcribe(b"audio")
                pytest.fail("Expected RuntimeError")
            except RuntimeError as e:
                assert "test-key" not in str(e)
                assert "Token" not in str(e)
