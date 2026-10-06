"""Sentence buffer for LLM streaming to TTS.

Accumulates streamed text tokens and emits complete sentences based on
deterministic sentence boundary detection. Used to feed the TTS pipeline
incrementally during LLM streaming, reducing time-to-first-audio.

Sentence boundaries recognized:
    - ". " (period followed by space)
    - "! " (exclamation followed by space)
    - "? " (question mark followed by space)
    - Newline characters

Opt-in early chunking (Phase 1 realtime latency): when the constructor
receives ``min_first_chunk_chars`` / ``max_chunk_chars`` the buffer also
emits useful TTS chunks BEFORE a ". " boundary arrives:

    - a buffer ENDING in ".", "!" or "?" emits immediately once it is at
      least ``min_first_chunk_chars`` long (a trailing "." directly after a
      digit is held back — it may open a decimal like "3.14");
    - a boundary-less run longer than ``max_chunk_chars`` is soft-split at
      the last word boundary that fits, so words are never split;
    - ``flush_on_timeout()`` performs the same safe split after a stall.

Without those arguments the buffer behaves exactly as before — telephony
and all legacy callers keep the original ". "-only semantics.

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

    Early chunking (opt-in, used by the realtime gateway):
        buffer = SentenceBuffer(min_first_chunk_chars=15, max_chunk_chars=160)

        # Trailing punctuation without a following space still emits.
        sentences = buffer.add("Hello, how are you?")
        # sentences = ["Hello, how are you?"]

        # A stalled stream can be flushed at the last word boundary.
        buffer.add("I think the best approach is WebSockets")
        buffer.flush_on_timeout()
        # ["I think the best approach is"]
    """

    def __init__(
        self,
        *,
        min_first_chunk_chars: int | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        """Initialize an empty sentence buffer.

        Args:
            min_first_chunk_chars: Enables early chunking. Chunks emitted
                off a trailing punctuation mark or a timeout flush are
                never shorter than this, so tiny fragments ("e.g.",
                "Bye.") keep accumulating. None disables early chunking
                entirely (legacy boundary-only behaviour).
            max_chunk_chars: Soft maximum chunk length. A boundary-less run
                longer than this is split at the last word boundary that
                fits. None disables soft splitting.
        """
        self._buffer: str = ""
        self._min_first_chunk_chars = min_first_chunk_chars
        self._max_chunk_chars = max_chunk_chars
        # Early chunking is opt-in: legacy callers (telephony, old tests)
        # pass no arguments and keep the original boundary-only behaviour.
        self._early_chunking = min_first_chunk_chars is not None

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

        if self._early_chunking:
            # Phase 1: no ". " boundary yet — still emit early chunks.
            # Over-long runs are soft-split at a word boundary first, then
            # a buffer that already ends with sentence punctuation emits.
            while self._take_max_chunk(sentences):
                pass
            self._emit_terminated_buffer(sentences)

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

    def flush_on_timeout(self) -> list[str]:
        """Flush a stalled buffer at a safe word boundary (early chunking).

        Called by the realtime gateway's chunk watchdog when the LLM stream
        stalls. Emits the whole buffer when it already ends with sentence
        punctuation; otherwise emits everything up to the last word boundary
        and keeps the trailing partial word buffered for the next token.

        Returns:
            Zero or one TTS-ready chunks; [] when early chunking is
            disabled, the buffer is too short, or emitting would require
            splitting a word.
        """
        if not self._early_chunking or not self._buffer:
            return []

        sentences: list[str] = []
        if self._emit_terminated_buffer(sentences):
            return sentences

        # Soft split: everything before the last space is complete words.
        split_at = self._buffer.rfind(" ")
        if split_at == -1:
            return []
        chunk = self._buffer[:split_at].strip()
        if len(chunk) < (self._min_first_chunk_chars or 0):
            return []
        sentences.append(chunk)
        self._buffer = self._buffer[split_at + 1 :]
        return sentences

    def _emit_terminated_buffer(self, sentences: list[str]) -> bool:
        """Emit the whole buffer when it ends with sentence punctuation.

        Early chunking: the LLM often ends a sentence on a token whose
        trailing space has not streamed yet, so the ". "/"! "/"? " boundary
        never fires mid-stream. A buffer ending in ".", "!" or "?" is a
        safe TTS chunk once it is at least ``min_first_chunk_chars`` long.

        A trailing "." directly preceded by a digit is held back — it may
        open a decimal number ("3." waiting for "14").

        Returns:
            True when a chunk was appended (the buffer is then empty).
        """
        stripped = self._buffer.rstrip()
        if not stripped or stripped[-1] not in ".!?":
            return False
        if len(stripped) < (self._min_first_chunk_chars or 0):
            return False
        if stripped.endswith(".") and len(stripped) >= 2 and stripped[-2].isdigit():
            return False
        sentences.append(stripped)
        self._buffer = ""
        return True

    def _take_max_chunk(self, sentences: list[str]) -> bool:
        """Soft-split the buffer at the configured maximum length.

        Emits everything up to the last word boundary that fits inside
        ``max_chunk_chars`` and keeps the remainder buffered. Words are
        never split: when the candidate boundary would emit a chunk
        shorter than ``min_first_chunk_chars`` (or there is no space at
        all) nothing is emitted and more text is awaited.

        Returns:
            True when a chunk was appended.
        """
        max_chars = self._max_chunk_chars
        if max_chars is None:
            return False
        if len(self._buffer.strip()) <= max_chars:
            return False
        split_at = self._buffer.rfind(" ", 0, max_chars + 1)
        if split_at == -1:
            return False
        chunk = self._buffer[:split_at].strip()
        if len(chunk) < (self._min_first_chunk_chars or 0):
            return False
        sentences.append(chunk)
        self._buffer = self._buffer[split_at + 1 :]
        return True

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
