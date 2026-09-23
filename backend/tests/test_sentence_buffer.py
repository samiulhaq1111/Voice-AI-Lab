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
