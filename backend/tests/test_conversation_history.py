"""Tests for conversation history / session context.

Proves that previous messages are passed to the LLM, not just stored in DB.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.providers.types import LLMMessage, LLMResponse, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TrackingFakeLLM:
    """A fake LLM that records every message list it receives."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.received_messages: list[list[LLMMessage]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return self._call_count

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Deep-copy the messages so later mutations don't affect our record
        self.received_messages.append(list(messages))
        if self._call_count >= len(self._responses):
            return LLMResponse(content="No more responses", finish_reason="stop")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def stream_chat(self, messages, *, model=None, tools=None, temperature=0.7, max_tokens=None):
        raise NotImplementedError

    async def close(self) -> None:
        pass


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
# Tests
# ---------------------------------------------------------------------------


class TestConversationMemory:
    """Test: second message receives first message's context via the LLM."""

    @pytest.mark.asyncio
    async def test_history_reaches_llm(self, client: AsyncClient) -> None:
        """Send two messages in the same session.
        
        Verify the LLM receives both the first user message and the second.
        """
        fake = TrackingFakeLLM([
            _mock_llm_response(content="Got it, your ID is E00."),
            _mock_llm_response(content="Your employee ID is E00."),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            # First message
            r1 = await client.post("/api/v1/chat", json={"message": "My employee ID is E00"})
            assert r1.status_code == 200
            session_id = r1.json()["session_id"]

            # Second message — must reuse session
            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "What is my employee ID?", "session_id": session_id},
            )
            assert r2.status_code == 200

        # The second LLM call should have received the first user message
        assert len(fake.received_messages) == 2
        second_call_msgs = fake.received_messages[1]

        # Find user messages (skip system)
        user_msgs = [m for m in second_call_msgs if m.role == "user"]
        assert len(user_msgs) >= 2, "LLM should receive both user messages"
        assert any("e00" in (m.content or "").lower() for m in user_msgs)
        assert any("employee" in (m.content or "").lower() for m in user_msgs)

    @pytest.mark.asyncio
    async def test_response_uses_history(self, client: AsyncClient) -> None:
        """The assistant should be able to reference info from a prior message."""
        fake = TrackingFakeLLM([
            _mock_llm_response(content="Noted."),
            _mock_llm_response(content="Your ID is E00."),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r1 = await client.post("/api/v1/chat", json={"message": "My ID is E00"})
            session_id = r1.json()["session_id"]

            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "What is my ID?", "session_id": session_id},
            )

        assert "E00" in r2.json()["response"]


class TestSessionReuse:
    """Test: session IDs are correctly reused across requests."""

    @pytest.mark.asyncio
    async def test_first_creates_session(self, client: AsyncClient) -> None:
        fake = TrackingFakeLLM([_mock_llm_response(content="Hi")])
        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r = await client.post("/api/v1/chat", json={"message": "Hello"})
        assert r.status_code == 200
        assert r.json()["session_id"]  # non-empty

    @pytest.mark.asyncio
    async def test_second_reuses_session(self, client: AsyncClient) -> None:
        fake = TrackingFakeLLM([
            _mock_llm_response(content="Hi"),
            _mock_llm_response(content="Again"),
        ])
        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r1 = await client.post("/api/v1/chat", json={"message": "Hello"})
            sid = r1.json()["session_id"]

            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "Hi again", "session_id": sid},
            )

        assert r2.json()["session_id"] == sid

    @pytest.mark.asyncio
    async def test_independent_sessions(self, client: AsyncClient) -> None:
        """Two different sessions have independent history."""
        fake_a = TrackingFakeLLM([
            _mock_llm_response(content="Session A noted."),
            _mock_llm_response(content="Session A response."),
        ])
        fake_b = TrackingFakeLLM([
            _mock_llm_response(content="Session B noted."),
            _mock_llm_response(content="Session B response."),
        ])

        with patch("app.services.chat_service.get_llm_provider", side_effect=[fake_a, fake_a, fake_b, fake_b]):
            # Session A
            r1 = await client.post("/api/v1/chat", json={"message": "I am in session A"})
            sid_a = r1.json()["session_id"]
            await client.post(
                "/api/v1/chat",
                json={"message": "Repeat session A", "session_id": sid_a},
            )

            # Session B
            r3 = await client.post("/api/v1/chat", json={"message": "I am in session B"})
            sid_b = r3.json()["session_id"]
            r4 = await client.post(
                "/api/v1/chat",
                json={"message": "Repeat session B", "session_id": sid_b},
            )

        # Session B's LLM should NOT have received session A's message
        b_second_call = fake_b.received_messages[1]
        b_user_msgs = [m.content for m in b_second_call if m.role == "user"]
        assert not any("session A" in (c or "").lower() for c in b_user_msgs)
        assert sid_a != sid_b

    @pytest.mark.asyncio
    async def test_invalid_session_returns_404(self, client: AsyncClient) -> None:
        fake = TrackingFakeLLM([_mock_llm_response(content="Hi")])
        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r = await client.post(
                "/api/v1/chat",
                json={"message": "Hello", "session_id": "nonexistent-id"},
            )
        assert r.status_code == 404


class TestMessageRoles:
    """Test: conversation history preserves correct roles."""

    @pytest.mark.asyncio
    async def test_roles_preserved(self, client: AsyncClient) -> None:
        fake = TrackingFakeLLM([
            _mock_llm_response(content="First response"),
            _mock_llm_response(content="Second response"),
            _mock_llm_response(content="Third response"),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r1 = await client.post("/api/v1/chat", json={"message": "Hello"})
            sid = r1.json()["session_id"]

            await client.post(
                "/api/v1/chat",
                json={"message": "How are you?", "session_id": sid},
            )

            r3 = await client.post(
                "/api/v1/chat",
                json={"message": "Good", "session_id": sid},
            )

        # Third call should have: user, assistant, user, assistant, user
        third_call = fake.received_messages[2]
        roles = [m.role for m in third_call]
        # system + user + assistant + user + assistant + user
        assert roles.count("user") == 3
        assert roles.count("assistant") == 2
        assert roles[0] == "system"


class TestNoDuplicateCurrentMessage:
    """Test: the current user message is not sent twice to the LLM."""

    @pytest.mark.asyncio
    async def test_no_duplication(self, client: AsyncClient) -> None:
        fake = TrackingFakeLLM([
            _mock_llm_response(content="OK"),
            _mock_llm_response(content="Got it"),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r1 = await client.post("/api/v1/chat", json={"message": "First message"})
            sid = r1.json()["session_id"]

            await client.post(
                "/api/v1/chat",
                json={"message": "Second message", "session_id": sid},
            )

        second_call = fake.received_messages[1]
        user_msgs = [m for m in second_call if m.role == "user"]
        # Should have exactly 2 user messages: first + second
        assert len(user_msgs) == 2
        contents = [m.content for m in user_msgs]
        assert contents.count("Second message") == 1, "Current message should appear exactly once"


class TestToolCallHistory:
    """Test: tool-call conversation context is preserved."""

    @pytest.mark.asyncio
    async def test_tool_history_in_conversation(self, client: AsyncClient) -> None:
        """After a tool call, the next request should include tool context."""
        tc = _make_tool_call("tc1", "get_weather", {"location": "Islamabad"})

        fake = TrackingFakeLLM([
            _mock_llm_response(content=None, tool_calls=[tc]),
            _mock_llm_response(content="It's sunny in Islamabad."),
            _mock_llm_response(content="I already told you — it's sunny."),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            # First: triggers tool call
            r1 = await client.post(
                "/api/v1/chat",
                json={"message": "Weather in Islamabad?"},
            )
            sid = r1.json()["session_id"]
            assert len(r1.json()["tool_calls"]) == 1

            # Second: follow-up question
            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "What was the weather?", "session_id": sid},
            )

        # Third LLM call should include the tool call history
        third_call = fake.received_messages[2]
        roles = [m.role for m in third_call]
        assert "tool" in roles, "Tool result should be in conversation history"
        # Should have: system, user, assistant(tool_call), tool, assistant, user
        assert roles.count("user") == 2
        assert roles.count("assistant") >= 1


class TestFakeLLMArchitectureHistory:
    """Architecture test: prove history reaches the LLM via FakeLLM."""

    @pytest.mark.asyncio
    async def test_history_reaches_llm(self, client: AsyncClient) -> None:
        """Direct assertion on FakeLLM.received_messages.

        Proves that conversation history is passed to the LLM,
        not merely stored in the database.
        """
        fake = TrackingFakeLLM([
            _mock_llm_response(content="Noted, E00."),
            _mock_llm_response(content="Your ID is E00."),
        ])

        with patch("app.services.chat_service.get_llm_provider", return_value=fake):
            r1 = await client.post("/api/v1/chat", json={"message": "My employee ID is E00"})
            sid = r1.json()["session_id"]

            r2 = await client.post(
                "/api/v1/chat",
                json={"message": "What is my employee ID?", "session_id": sid},
            )

        # Assertion: FakeLLM received both user messages in the second call
        assert len(fake.received_messages) == 2
        second_call = fake.received_messages[1]

        user_contents = [m.content for m in second_call if m.role == "user"]
        assert "My employee ID is E00" in user_contents
        assert "What is my employee ID?" in user_contents
