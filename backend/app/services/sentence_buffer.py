"""Sentence buffer for LLM streaming to TTS.

Accumulates streamed text tokens and emits complete sentences based on
deterministic sentence boundary detection. Used to feed the TTS pipeline
incrementally during LLM streaming, reducing time-to-first-audio.

Sentence boundaries recognized:
    - ". " (period followed by space)
    - "! " (exclamation followed by space)
    - "? " (question mark followed by space)
    - Newline characters

The buffer strips whitespace from emitted sentences and handles edge cases
like empty chunks, trailing punctuation, and partial sentences at stream end.
"""


class SentenceBuffer:
    """Accumulates streamed text and emits complete sentences.

    Usage:
        buffer = SentenceBuffer()

        # As LLM tokens arrive
        sentences = buffer.add("Hello world. ")
        # sentences = ["Hello world."]

        sentences = buffer.add("How are you? ")
        # sentences = ["How are you?"]

        sentences = buffer.add("I'm fine")
        # sentences = [] (no boundary yet)

        # At stream end, flush remaining
        remaining = buffer.flush()
        # remaining = "I'm fine"
    """

    def __init__(self) -> None:
        """Initialize an empty sentence buffer."""
        self._buffer: str = ""

    def add(self, text: str) -> list[str]:
        """Add text chunk to buffer and return any complete sentences.

        Args:
            text: Text chunk from LLM stream (may be partial word/sentence).

        Returns:
            List of complete sentences found (may be empty if no boundary yet).
        """
        if not text:
            return []

        self._buffer += text
        sentences: list[str] = []

        # Extract all complete sentences from buffer
        while True:
            boundary = self._find_boundary()
            if boundary == -1:
                break
            sentence = self._buffer[:boundary].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[boundary:]

        return sentences

    def flush(self) -> str | None:
        """Return any remaining buffered text at stream end.

        Call this after the LLM stream completes to ensure no text is lost.
        The buffer is cleared after flushing.

        Returns:
            Remaining text if any, None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _find_boundary(self) -> int:
        """Find the index after the first sentence boundary.

        Recognizes:
            - ". " (period + space)
            - "! " (exclamation + space)
            - "? " (question mark + space)
            - Newline character

        Returns:
            Index after the boundary, or -1 if no boundary found.
        """
        for i, char in enumerate(self._buffer):
            # Check for punctuation followed by space
            if char in ".!?" and i + 1 < len(self._buffer):
                if self._buffer[i + 1] == " ":
                    return i + 2  # Include punctuation and space
            # Check for newline
            if char == "\n":
                return i + 1  # Include newline

        return -1

    def __len__(self) -> int:
        """Return the current buffer length (for debugging)."""
        return len(self._buffer)

    def __bool__(self) -> bool:
        """Return True if buffer has content (for debugging)."""
        return bool(self._buffer)
