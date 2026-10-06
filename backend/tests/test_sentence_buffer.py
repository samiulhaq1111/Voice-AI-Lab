"""Unit tests for SentenceBuffer.

Tests the deterministic sentence boundary detection used for feeding
the TTS pipeline incrementally during LLM streaming.
"""

import pytest

from app.services.sentence_buffer import SentenceBuffer


class TestSentenceBuffer:
    """SentenceBuffer unit tests."""

    def test_single_sentence_with_period(self) -> None:
        """A single sentence ending with period is emitted."""
        buf = SentenceBuffer()
        sentences = buf.add("Hello world. ")
        assert sentences == ["Hello world."]

    def test_single_sentence_with_exclamation(self) -> None:
        """A single sentence ending with exclamation is emitted."""
        buf = SentenceBuffer()
        sentences = buf.add("Wow! ")
        assert sentences == ["Wow!"]

    def test_single_sentence_with_question(self) -> None:
        """A single sentence ending with question mark is emitted."""
        buf = SentenceBuffer()
        sentences = buf.add("How are you? ")
        assert sentences == ["How are you?"]

    def test_multiple_sentences_in_one_chunk(self) -> None:
        """Multiple sentences in a single chunk are all emitted."""
        buf = SentenceBuffer()
        sentences = buf.add("Hello. How are you? I'm fine. ")
        assert sentences == ["Hello.", "How are you?", "I'm fine."]

    def test_incremental_accumulation(self) -> None:
        """Sentences are emitted as boundaries arrive incrementally."""
        buf = SentenceBuffer()

        # Partial word — no sentence yet
        assert buf.add("Hello") == []
        assert buf.add(" world") == []

        # Sentence boundary arrives
        assert buf.add(". ") == ["Hello world."]

        # Next sentence builds up
        assert buf.add("How ") == []
        assert buf.add("are ") == []
        assert buf.add("you? ") == ["How are you?"]

    def test_flush_returns_remaining_text(self) -> None:
        """Flush returns any text without a trailing boundary."""
        buf = SentenceBuffer()
        buf.add("Hello world. ")
        buf.add("I'm fine")  # No boundary

        remaining = buf.flush()
        assert remaining == "I'm fine"

    def test_flush_returns_none_when_empty(self) -> None:
        """Flush returns None when buffer is empty."""
        buf = SentenceBuffer()
        buf.add("Hello. ")  # Complete sentence
        remaining = buf.flush()
        assert remaining is None

    def test_flush_clears_buffer(self) -> None:
        """Flush clears the buffer after returning content."""
        buf = SentenceBuffer()
        buf.add("Partial text")

        assert buf.flush() == "Partial text"
        assert buf.flush() is None  # Buffer is now empty

    def test_empty_chunks_ignored(self) -> None:
        """Empty chunks do not affect the buffer."""
        buf = SentenceBuffer()
        assert buf.add("") == []
        buf.add("Hello")
        assert buf.add("") == []
        assert buf.add(". ") == ["Hello."]

    def test_whitespace_stripping(self) -> None:
        """Sentences have leading/trailing whitespace stripped."""
        buf = SentenceBuffer()
        sentences = buf.add("  Hello world.  ")
        assert sentences == ["Hello world."]

    def test_newline_as_boundary(self) -> None:
        """Newline characters act as sentence boundaries."""
        buf = SentenceBuffer()
        sentences = buf.add("Line one\nLine two\n")
        assert sentences == ["Line one", "Line two"]

    def test_mixed_boundaries(self) -> None:
        """Mixed punctuation and newline boundaries work correctly."""
        buf = SentenceBuffer()
        sentences = buf.add("Hello. How?\nFine!\n")
        assert sentences == ["Hello.", "How?", "Fine!"]

    def test_no_false_boundary_mid_word(self) -> None:
        """Period in middle of text without space is not a boundary."""
        buf = SentenceBuffer()
        # "Dr.Smith" — no space after period
        sentences = buf.add("Dr.Smith")
        assert sentences == []  # No boundary yet

        # Now add space — should NOT split "Dr.Smith"
        sentences = buf.add(" is here. ")
        assert sentences == ["Dr.Smith is here."]

    def test_punctuation_at_end_without_space(self) -> None:
        """Punctuation at end without trailing space waits for more."""
        buf = SentenceBuffer()
        sentences = buf.add("Hello world.")
        assert sentences == []  # No space after period yet

        # Now add space
        sentences = buf.add(" ")
        assert sentences == ["Hello world."]

    def test_multiple_spaces_after_punctuation(self) -> None:
        """Multiple spaces after punctuation still triggers boundary."""
        buf = SentenceBuffer()
        sentences = buf.add("Hello.   World")
        assert sentences == ["Hello."]

    def test_streaming_token_by_token(self) -> None:
        """Simulates real LLM streaming token by token."""
        buf = SentenceBuffer()
        tokens = ["The", " weather", " in", " Islam", "abad", " is", " nice", ". ",
                  "It", " should", " be", " sunny", ". "]

        all_sentences = []
        for token in tokens:
            sentences = buf.add(token)
            all_sentences.extend(sentences)

        assert all_sentences == [
            "The weather in Islamabad is nice.",
            "It should be sunny.",
        ]

    def test_long_text_multiple_sentences(self) -> None:
        """Long text with many sentences is handled correctly."""
        buf = SentenceBuffer()
        text = (
            "First sentence. Second sentence! Third sentence? "
            "Fourth sentence. Fifth sentence! "
        )
        sentences = buf.add(text)
        assert len(sentences) == 5
        assert sentences[0] == "First sentence."
        assert sentences[4] == "Fifth sentence!"

    def test_buffer_len_and_bool(self) -> None:
        """Buffer length and truthiness work for debugging."""
        buf = SentenceBuffer()
        assert len(buf) == 0
        assert not buf

        buf.add("Hello")
        assert len(buf) == 5
        assert buf

        buf.add(". ")
        # After sentence emission, buffer should be empty
        assert len(buf) == 0
        assert not buf


class TestSentenceBufferEarlyChunking:
    """Phase 1: opt-in early chunking for the realtime TTS gateway."""

    def test_trailing_question_emits_immediately(self) -> None:
        """A long-enough buffer ending in '?' emits without a space."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("Hello, how are you?") == ["Hello, how are you?"]

    def test_trailing_period_emits_immediately(self) -> None:
        """A single sentence ending in '.' emits before the stream ends."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("The best approach is WebSockets.") == [
            "The best approach is WebSockets."
        ]

    def test_short_trailing_punctuation_still_waits(self) -> None:
        """Fragments below the minimum keep accumulating."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("Bye.") == []
        assert buf.flush() == "Bye."

    def test_decimal_not_split(self) -> None:
        """"3." waits for its decimal continuation instead of emitting."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("The current version 3.") == []
        assert buf.add("14 is available.") == [
            "The current version 3.14 is available."
        ]

    def test_max_chunk_splits_at_word_boundary(self) -> None:
        """Over-long boundary-less text soft-splits at the last space."""
        buf = SentenceBuffer(min_first_chunk_chars=15, max_chunk_chars=40)
        text = "The quick brown fox jumps over the lazy dog and keeps running"
        assert buf.add(text) == ["The quick brown fox jumps over the lazy"]
        assert buf.flush() == "dog and keeps running"

    def test_max_chunk_never_splits_words(self) -> None:
        """A single word longer than the max is never split."""
        buf = SentenceBuffer(min_first_chunk_chars=15, max_chunk_chars=20)
        assert buf.add("Supercalifragilisticexpialidocious") == []
        assert buf.add(" word") == []
        assert buf.flush() == "Supercalifragilisticexpialidocious word"

    def test_timeout_flush_at_word_boundary(self) -> None:
        """flush_on_timeout emits complete words and keeps the partial word."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("I think the best approach is WebSock") == []
        assert buf.flush_on_timeout() == ["I think the best approach is"]
        assert buf.flush() == "WebSock"

    def test_timeout_flush_too_short_is_noop(self) -> None:
        """A buffer below the minimum is left untouched by the timeout."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("Almost") == []
        assert buf.flush_on_timeout() == []
        assert buf.flush() == "Almost"

    def test_timeout_flush_requires_early_chunking(self) -> None:
        """Legacy buffers keep the original behaviour (no timeout flush)."""
        buf = SentenceBuffer()
        assert buf.add("Hello world without a boundary") == []
        assert buf.flush_on_timeout() == []

    def test_timeout_flush_holds_possible_decimal(self) -> None:
        """A trailing digit-dot is not flushed by the timeout either."""
        buf = SentenceBuffer(min_first_chunk_chars=15)
        assert buf.add("The total is 42.") == []
        assert buf.flush_on_timeout() == []
        assert buf.flush() == "The total is 42."

    def test_remaining_text_emitted_at_stream_end(self) -> None:
        """The trailing remainder is still returned by flush()."""
        buf = SentenceBuffer(min_first_chunk_chars=15, max_chunk_chars=160)
        assert buf.add("Hello, how are you? I'm doing") == ["Hello, how are you?"]
        assert buf.flush() == "I'm doing"

    def test_legacy_mode_unchanged(self) -> None:
        """Default construction keeps the boundary-only semantics."""
        buf = SentenceBuffer()
        assert buf.add("Hello world.") == []
        assert buf.add(" ") == ["Hello world."]
