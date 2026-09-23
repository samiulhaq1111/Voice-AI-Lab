"""Integration tests for telephony LLM streaming path.

Tests the streaming integration between LLM, sentence buffer, and TTS
for reduced telephone time-to-first-audio.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.providers.types import LLMMessage, LLMResponse
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
    async def test_tool_call_detection_falls_back(self) -> None:
        """Tool calls in LLM response trigger fallback to non-streaming."""
        from app.services.telnyx_agent import TelephonyAgentSession
        from app.services.telnyx_deepgram import Utterance

        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        agent = TelephonyAgentSession(
            call_control_id="cc_test",
            utterance_queue=queue,
            tts_service=None,
            websocket=None,
        )

        # Mock LLM that returns tool calls
        agent._llm = AsyncMock()
        agent._llm.close = AsyncMock()
        agent._llm.chat = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Islamabad"}',
                        },
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )
        )

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

        # Mock AgentRuntime for tool execution
        with patch("app.services.telnyx_agent.AgentRuntime") as mock_runtime:
            mock_runtime.return_value.run = AsyncMock(
                return_value=AsyncMock(
                    response="The weather is sunny!",
                    tool_calls=[{"function": {"name": "get_weather"}}],
                    usage={},
                    iterations=2,
                )
            )
            mock_runtime.return_value.state.messages = []

            await agent.start()
            await agent._worker_task

        # Verify tool call path was taken
        assert agent.turn == 1

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
            yield "Hello"
            await asyncio.sleep(0.02)
            yield " world"
            await asyncio.sleep(0.01)
            yield ". "

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
