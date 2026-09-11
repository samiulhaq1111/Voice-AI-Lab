"""Abstract base interface for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.providers.types import LLMMessage, LLMResponse, ToolSchema


class LLMInterface(ABC):
    """Abstract interface that all LLM providers must implement.

    The Agent Runtime depends on this interface, NOT on concrete providers.
    Adding a new LLM provider requires implementing this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of this LLM provider (e.g. 'openrouter')."""

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history.
            model: Model identifier override.
            tools: Available tool schemas for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with content and/or tool calls.
        """

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding text chunks.

        Args:
            messages: Conversation history.
            model: Model identifier override.
            tools: Available tool schemas for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Text chunks of the response.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this provider."""
