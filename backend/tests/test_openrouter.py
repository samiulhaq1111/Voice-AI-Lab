"""Tests for the OpenRouter LLM adapter with mocked HTTP."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.providers.llm.openrouter import OpenRouterAdapter
from app.providers.types import LLMMessage, LLMResponse, ToolSchema


@pytest.fixture
def adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(api_key="test-key", default_model="test-model")


class TestOpenRouterAdapter:
    """Tests for OpenRouterAdapter."""

    def test_provider_name(self, adapter: OpenRouterAdapter) -> None:
        assert adapter.provider_name == "openrouter"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from app.core.config import settings

        settings.openrouter_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            OpenRouterAdapter(api_key="")

    @pytest.mark.asyncio
    async def test_chat_simple_response(self, adapter: OpenRouterAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        messages = [LLMMessage(role="user", content="Hi")]

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.chat(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        assert result.usage["total_tokens"] == 15
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, adapter: OpenRouterAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "NYC"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        messages = [LLMMessage(role="user", content="Weather in NYC?")]
        tools = [
            ToolSchema(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {}},
            )
        ]

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.chat(messages, tools=tools)

        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_chat_auth_failure(self, adapter: OpenRouterAdapter) -> None:
        mock_response = httpx.Response(
            401,
            json={"error": "Unauthorized"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="authentication failed"):
                await adapter.chat([LLMMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_chat_timeout(self, adapter: OpenRouterAdapter) -> None:
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await adapter.chat([LLMMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_chat_server_error(self, adapter: OpenRouterAdapter) -> None:
        mock_response = httpx.Response(
            500,
            json={"error": "Internal error"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await adapter.chat([LLMMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_chat_empty_choices(self, adapter: OpenRouterAdapter) -> None:
        mock_response = httpx.Response(
            200,
            json={"choices": []},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.chat([LLMMessage(role="user", content="hi")])

        assert result.content is None
        assert result.finish_reason == "empty"

    @pytest.mark.asyncio
    async def test_serialize_tool_message(self, adapter: OpenRouterAdapter) -> None:
        msg = LLMMessage(
            role="tool",
            content='{"temp": "72F"}',
            tool_call_id="call_1",
            name="get_weather",
        )
        serialized = adapter._serialize_message(msg)
        assert serialized["role"] == "tool"
        assert serialized["content"] == '{"temp": "72F"}'
        assert serialized["tool_call_id"] == "call_1"
        assert serialized["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_close(self, adapter: OpenRouterAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
        mock_close.assert_called_once()
