"""Tests for the /api/v1/chat endpoint and chat service."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.providers.types import LLMMessage, LLMResponse, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_tool_call(tc_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests: Chat API
# ---------------------------------------------------------------------------


class TestChatSimpleResponse:
    """Test: simple user → LLM → final response via API."""

    @pytest.mark.asyncio
    async def test_simple_chat(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            return_value=_mock_llm_response(content="Hi there! How can I help?")
        )
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hi there! How can I help?"
        assert "session_id" in data
        assert data["iterations"] >= 1
        assert "usage" in data
        assert data["usage"]["total_tokens"] == 15


class TestChatToolCall:
    """Test: LLM → tool call → tool result → final response via API."""

    @pytest.mark.asyncio
    async def test_chat_with_tool_call(self, client: AsyncClient) -> None:
        tc = _make_tool_call("tc1", "get_weather", {"location": "Islamabad"})
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            side_effect=[
                _mock_llm_response(content=None, tool_calls=[tc]),
                _mock_llm_response(content="The weather in Islamabad is sunny, 78F."),
            ]
        )
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "What is the weather in Islamabad?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Islamabad" in data["response"]
        assert len(data["tool_calls"]) == 1
        assert data["iterations"] == 2


class TestChatSessionCreation:
    """Test: session is created when session_id is null."""

    @pytest.mark.asyncio
    async def test_new_session_created(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="New session!"))
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello", "session_id": None},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"]  # not empty


class TestChatSessionReuse:
    """Test: existing session_id is reused."""

    @pytest.mark.asyncio
    async def test_existing_session(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Welcome back!"))
        mock_llm.close = AsyncMock()

        # First create a session
        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            r1 = await client.post("/api/v1/chat", json={"message": "Hi"})
            session_id = r1.json()["session_id"]

            # Reuse it
            mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Second message!"))
            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "Hello again", "session_id": session_id},
            )

        assert r2.status_code == 200
        assert r2.json()["session_id"] == session_id


class TestChatInvalidSession:
    """Test: non-existent session_id returns 404."""

    @pytest.mark.asyncio
    async def test_invalid_session(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello", "session_id": "nonexistent-id"},
            )

        assert response.status_code == 404


class TestChatInvalidRequest:
    """Test: invalid request body returns 422."""

    @pytest.mark.asyncio
    async def test_empty_message(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_message(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/chat",
            json={},
        )
        assert response.status_code == 422


class TestChatMissingProviderConfig:
    """Test: missing provider configuration returns 400."""

    @pytest.mark.asyncio
    async def test_missing_provider(self, client: AsyncClient) -> None:
        from app.providers.factory import ProviderError

        with patch(
            "app.services.chat_service.get_llm_provider",
            side_effect=ProviderError("Unsupported LLM provider: 'unknown'"),
        ):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello", "provider": "unknown"},
            )

        assert response.status_code == 400


class TestChatLLMFailure:
    """Test: LLM provider failure returns 502."""

    @pytest.mark.asyncio
    async def test_llm_failure(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM is down"))
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
            )

        assert response.status_code == 502


class TestChatResponseSchema:
    """Test: API response matches expected schema."""

    @pytest.mark.asyncio
    async def test_response_schema(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Test response"))
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Test"},
            )

        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "tool_calls" in data
        assert "usage" in data
        assert "iterations" in data
        assert "latency_ms" in data
        assert isinstance(data["tool_calls"], list)
        assert isinstance(data["usage"], dict)
        assert isinstance(data["iterations"], int)


class TestChatNoAPIKeysExposed:
    """Test: API keys are never in the response."""

    @pytest.mark.asyncio
    async def test_no_keys_in_response(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="OK"))
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Test"},
            )

        response_text = response.text
        assert "test-deepgram-key" not in response_text
        assert "test-openrouter-key" not in response_text
        assert "test-elevenlabs-key" not in response_text


class TestChatProviderOverride:
    """Test: provider/model can be overridden per request."""

    @pytest.mark.asyncio
    async def test_provider_override(self, client: AsyncClient) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_mock_llm_response(content="Custom provider!"))
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm) as mock_factory:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello", "provider": "openrouter", "model": "custom-model"},
            )

        assert response.status_code == 200
        mock_factory.assert_called_once_with("openrouter", model="custom-model")


class TestChatMultipleToolCalls:
    """Test: multiple tool calls in one response."""

    @pytest.mark.asyncio
    async def test_multiple_tools(self, client: AsyncClient) -> None:
        tc1 = _make_tool_call("tc1", "get_weather", {"location": "Islamabad"})
        tc2 = _make_tool_call("tc2", "get_employee", {"employee_id": "E001"})
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(
            side_effect=[
                _mock_llm_response(content=None, tool_calls=[tc1, tc2]),
                _mock_llm_response(content="Weather is sunny and Alice is in Engineering."),
            ]
        )
        mock_llm.close = AsyncMock()

        with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Get weather and employee info"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["tool_calls"]) == 2
        assert data["iterations"] == 2
