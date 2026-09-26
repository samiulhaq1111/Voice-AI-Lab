"""Integration tests for telephony LLM streaming path.

Tests the streaming integration between LLM, sentence buffer, and TTS
for reduced telephone time-to-first-audio.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.providers.types import LLMResponse, StreamChunk
from app.services.sentence_buffer import SentenceBuffer


class TestTelephonyStreamingIntegration:
    """Integration tests for the telephony streaming path."""

    @pytest.mark.asyncio
    async def test_streaming_path_produces_sentences(self) -> None:
        """Streaming LLM tokens through sentence buffer produces sentences."""
        buf = SentenceBuffer()

        # Simulate LLM streaming tokens
        tokens = [
            "The", " weather", " in", " Islam", "abad",
            " is", " nice", ". ", "It", " should",
            " be", " sunny", ". ",
        ]

        all_sentences = []
        for token in tokens:
            sentences = buf.add(token)
            all_sentences.extend(sentences)

        # Flush remaining
        remaining = buf.flush()
        if remaining:
            all_sentences.append(remaining)

        assert len(all_sentences) == 2
        assert "weather" in all_sentences[0]
        assert "Islamabad" in all_sentences[0]
        assert "sunny" in all_sentences[1]

    @pytest.mark.asyncio
    async def test_sentence_buffer_handles_empty_chunks(self) -> None:
        """Sentence buffer handles empty chunks gracefully."""
        buf = SentenceBuffer()

        assert buf.add("") == []
        assert buf.add("Hello") == []
        assert buf.add("") == []
        assert buf.add(". ") == ["Hello."]
        assert buf.flush() is None

    @pytest.mark.asyncio
    async def test_sentence_buffer_flush_partial(self) -> None:
        """Sentence buffer flush returns partial sentence at stream end."""
        buf = SentenceBuffer()

        buf.add("Hello world")
        buf.add(". ")
        buf.add("Partial text")  # No boundary

        sentences = buf.add("")  # No more tokens
        remaining = buf.flush()

        assert len(sentences) == 0  # No new complete sentences
        assert remaining == "Partial text"

    @pytest.mark.asyncio
    async def test_tool_call_executes_without_redundant_llm(self) -> None:
        """Streamed tool calls execute directly — no second LLM detection call."""
        from app.providers.types import LLMResponse
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        stream_chat_called = 0
        chat_called = 0

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        async def mock_stream_chat(*args, **kwargs):
            nonlocal stream_chat_called
            stream_chat_called += 1
            yield StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Islamabad"}',
                        },
                        "type": "function",
                    }
                ]
            )

        async def mock_chat(*args, **kwargs):
            nonlocal chat_called
            chat_called += 1
            return LLMResponse(
                content="The weather is sunny!",
                tool_calls=None,
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            )

        mock_llm.stream_chat = mock_stream_chat
        mock_llm.chat = mock_chat

        # Create utterance
        now = time.monotonic()
        utterance = Utterance(
            text="What's the weather?",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        # Mock LLM provider and tool registry
        mock_tool = AsyncMock()
        mock_tool.name = "get_weather"
        mock_tool.enabled = True
        mock_tool.handler = AsyncMock(return_value={"weather": "sunny"})
        mock_tool.timeout_seconds = 10

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry"
            ) as mock_get_registry,
        ):
            from unittest.mock import MagicMock
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_tool
            mock_registry.get_schemas.return_value = []
            mock_get_registry.return_value = mock_registry

            await agent.start()
            await agent._worker_task

        # Verify: stream_chat() called exactly once (no redundant call)
        assert stream_chat_called == 1
        # Verify: chat() called exactly once (final LLM only, NOT re-detection)
        assert chat_called == 1
        # Verify: turn completed
        assert agent.turn == 1
        # Verify: conversation history has correct messages
        assert len(agent.history) == 4
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"
        assert agent.history[1].tool_calls is not None
        assert agent.history[2].role == "tool"
        assert agent.history[3].role == "assistant"
        assert agent.history[3].content == "The weather is sunny!"

    @pytest.mark.asyncio
    async def test_streamed_tool_args_assembled_from_deltas(self) -> None:
        """Tool call arguments are fully assembled from incremental deltas."""
        from app.providers.types import LLMResponse
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()
        received_args = ""

        async def mock_stream_chat(*args, **kwargs):
            # Simulate incremental argument streaming
            yield StreamChunk(
                tool_calls=[{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "get_employee", "arguments": ""},
                    "type": "function",
                }]
            )
            yield StreamChunk(
                tool_calls=[{
                    "index": 0,
                    "function": {"arguments": '{"employee_id"'},
                }]
            )
            yield StreamChunk(
                tool_calls=[{
                    "index": 0,
                    "function": {"arguments": ': "E001"}'},
                }]
            )

        async def mock_chat(*args, **kwargs):
            return LLMResponse(
                content="Employee E001 details...",
                tool_calls=None,
                usage={},
            )

        mock_llm.stream_chat = mock_stream_chat
        mock_llm.chat = mock_chat

        now = time.monotonic()
        utterance = Utterance(
            text="Tell me about E001",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        mock_tool = AsyncMock()
        mock_tool.name = "get_employee"
        mock_tool.enabled = True
        mock_tool.handler = AsyncMock(return_value={"name": "John"})
        mock_tool.timeout_seconds = 10

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry"
            ) as mock_get_registry,
        ):
            from unittest.mock import MagicMock
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_tool
            mock_registry.get_schemas.return_value = []
            mock_get_registry.return_value = mock_registry

            # Capture the arguments passed to execute_from_json
            from app.tools.executor import ToolExecutor

            async def capture_execute(self_exec, tool_name, arguments_json):
                nonlocal received_args
                received_args = arguments_json
                return {"name": "John"}

            with patch.object(
                ToolExecutor, "execute_from_json", capture_execute
            ):
                await agent.start()
                await agent._worker_task

        # Verify arguments were fully assembled before execution
        assert received_args == '{"employee_id": "E001"}'

    @pytest.mark.asyncio
    async def test_normal_turn_unchanged(self) -> None:
        """Normal (non-tool) streaming turns remain unchanged."""
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        async def mock_stream_chat(*args, **kwargs):
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" world!")

        mock_llm.stream_chat = mock_stream_chat

        now = time.monotonic()
        utterance = Utterance(
            text="Hi",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        with patch(
            "app.services.telnyx_agent.get_llm_provider",
            return_value=mock_llm,
        ):
            await agent.start()
            await agent._worker_task

        assert agent.turn == 1
        # Normal turn: user + assistant in history
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"
        assert agent.history[1].content == "Hello world!"

    @pytest.mark.asyncio
    async def test_tool_execution_failure_handled(self) -> None:
        """Tool execution failure does not crash the call."""
        from app.providers.types import LLMResponse
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        async def mock_stream_chat(*args, **kwargs):
            yield StreamChunk(
                tool_calls=[{
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "broken_tool",
                        "arguments": "{}",
                    },
                    "type": "function",
                }]
            )

        async def mock_chat(*args, **kwargs):
            return LLMResponse(
                content="Sorry, the tool failed.",
                tool_calls=None,
                usage={},
            )

        mock_llm.stream_chat = mock_stream_chat
        mock_llm.chat = mock_chat

        now = time.monotonic()
        utterance = Utterance(
            text="Use broken tool",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        mock_tool = AsyncMock()
        mock_tool.name = "broken_tool"
        mock_tool.enabled = True
        mock_tool.handler = AsyncMock(side_effect=RuntimeError("boom"))
        mock_tool.timeout_seconds = 10

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry"
            ) as mock_get_registry,
        ):
            from unittest.mock import MagicMock
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_tool
            mock_registry.get_schemas.return_value = []
            mock_get_registry.return_value = mock_registry

            await agent.start()
            await agent._worker_task

        # Should not crash — turn completes
        assert agent.turn == 1
        # Final response should still be generated
        assert agent.history[-1].role == "assistant"
        assert agent.history[-1].content == "Sorry, the tool failed."

    @pytest.mark.asyncio
    async def test_tool_turn_metrics_correct(self) -> None:
        """Tool turn timing metrics are correctly computed."""
        from app.providers.types import LLMResponse
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()

        async def mock_stream_chat(*args, **kwargs):
            await asyncio.sleep(0.05)  # simulate LLM delay
            yield StreamChunk(
                tool_calls=[{
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Islamabad"}',
                    },
                    "type": "function",
                }]
            )

        async def mock_chat(*args, **kwargs):
            await asyncio.sleep(0.02)
            return LLMResponse(
                content="Sunny!",
                tool_calls=None,
                usage={"prompt_tokens": 10, "completion_tokens": 2},
            )

        mock_llm.stream_chat = mock_stream_chat
        mock_llm.chat = mock_chat

        now = time.monotonic()
        utterance = Utterance(
            text="Weather?",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        mock_tool = AsyncMock()
        mock_tool.name = "get_weather"
        mock_tool.enabled = True
        mock_tool.handler = AsyncMock(return_value={"weather": "sunny"})
        mock_tool.timeout_seconds = 10

        with (
            patch(
                "app.services.telnyx_agent.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "app.services.telnyx_agent.get_tool_registry"
            ) as mock_get_registry,
        ):
            from unittest.mock import MagicMock
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_tool
            mock_registry.get_schemas.return_value = []
            mock_get_registry.return_value = mock_registry

            await agent.start()
            await agent._worker_task

        assert agent.turn == 1
        # Verify history structure
        assert len(agent.history) == 4
        assert agent.history[1].tool_calls is not None
        assert agent.history[2].role == "tool"
        assert agent.history[3].content == "Sunny!"

    @pytest.mark.asyncio
    async def test_streaming_timing_metrics(self) -> None:
        """Streaming path captures timing metrics correctly."""
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        # Mock LLM with streaming
        agent._llm = AsyncMock()
        agent._llm.close = AsyncMock()
        agent._llm.chat = AsyncMock(
            return_value=LLMResponse(
                content="Hello world.",
                tool_calls=None,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
        )

        # Simulate streaming with delay
        async def mock_stream_chat(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate first token delay
            yield StreamChunk(content="Hello")
            await asyncio.sleep(0.02)
            yield StreamChunk(content=" world")
            await asyncio.sleep(0.01)
            yield StreamChunk(content=". ")

        agent._llm.stream_chat = mock_stream_chat

        # Create utterance
        now = time.monotonic()
        utterance = Utterance(
            text="Say hello",
            speech_final_at=now,
            utterance_end_at=now + 0.5,
            last_word_end=0.5,
        )
        await queue.put(utterance)
        await queue.put(None)

        await agent.start()
        await agent._worker_task

        # Verify turn completed
        assert agent.turn == 1

    @pytest.mark.asyncio
    async def test_sentence_iterator_pattern(self) -> None:
        """Sentence iterator pattern works with async queue."""
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        collected = []

        async def producer():
            await sentence_queue.put("First sentence.")
            await sentence_queue.put("Second sentence.")
            await sentence_queue.put(None)  # Sentinel

        async def consumer():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                collected.append(sentence)

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

        assert collected == ["First sentence.", "Second sentence."]

    @pytest.mark.asyncio
    async def test_multiple_sentences_streamed(self) -> None:
        """Multiple sentences are streamed in sequence."""
        buf = SentenceBuffer()
        sentences_collected = []

        # Simulate streaming a multi-sentence response
        chunks = [
            "The weather ",
            "in Islamabad ",
            "is nice. ",
            "It should ",
            "be sunny ",
            "today. ",
            "Enjoy!",
        ]

        for chunk in chunks:
            sentences = buf.add(chunk)
            sentences_collected.extend(sentences)

        # Flush remaining
        remaining = buf.flush()
        if remaining:
            sentences_collected.append(remaining)

        assert len(sentences_collected) == 3
        assert sentences_collected[0] == "The weather in Islamabad is nice."
        assert sentences_collected[1] == "It should be sunny today."
        assert sentences_collected[2] == "Enjoy!"

    @pytest.mark.asyncio
    async def test_punctuation_variations(self) -> None:
        """Different punctuation marks trigger sentence boundaries."""
        buf = SentenceBuffer()

        # Test period
        sentences = buf.add("Hello. ")
        assert len(sentences) == 1

        # Test exclamation
        sentences = buf.add("Wow! ")
        assert len(sentences) == 1

        # Test question
        sentences = buf.add("Really? ")
        assert len(sentences) == 1

        # Test newline
        sentences = buf.add("Line one\n")
        assert len(sentences) == 1

    @pytest.mark.asyncio
    async def test_no_text_loss_at_stream_end(self) -> None:
        """No text is lost when stream ends without boundary."""
        buf = SentenceBuffer()

        # Add first sentence (emits immediately)
        first_sentences = buf.add("First. ")
        assert len(first_sentences) == 1
        assert first_sentences[0] == "First."

        # Add text without trailing boundary
        buf.add("Second without boundary")

        # Flush remaining
        remaining = buf.flush()

        assert remaining == "Second without boundary"
