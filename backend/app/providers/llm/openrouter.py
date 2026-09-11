"""OpenRouter LLM adapter implementing LLMInterface."""

from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage, LLMResponse, ToolSchema

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAdapter(LLMInterface):
    """OpenRouter adapter for LLM chat completions.

    Uses the OpenAI-compatible OpenRouter REST API.
    Requires OPENROUTER_API_KEY in environment configuration.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_key = api_key or settings.openrouter_api_key
        if not resolved_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY.")

        self._api_key = resolved_key
        self._default_model = default_model or settings.default_llm_model or ""
        self._timeout = timeout or settings.llm_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "OpenRouter adapter initialized (model=%s)",
            self._default_model or "default",
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to OpenRouter."""
        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages,
            resolved_model,
            tools,
            temperature,
            max_tokens,
        )

        logger.info(
            "OpenRouter chat request (model=%s, messages=%d)",
            resolved_model,
            len(messages),
        )

        try:
            response = await self._client.post(_OPENROUTER_CHAT_URL, json=payload)
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            try:
                error_body = e.response.json()
                detail = error_body.get("error", {}).get("message", "")
            except Exception:
                detail = e.response.text[:200]
            if status == 401:
                logger.error("OpenRouter authentication failed")
                raise RuntimeError("OpenRouter authentication failed") from e
            logger.error("OpenRouter API error: HTTP %d — %s", status, detail)
            raise RuntimeError(f"OpenRouter API error: HTTP {status} — {detail}") from e

        except httpx.TimeoutException:
            logger.error("OpenRouter request timed out after %.1fs", self._timeout)
            raise RuntimeError("OpenRouter request timed out")

        except httpx.HTTPError as e:
            logger.error("OpenRouter HTTP error: %s", e)
            raise RuntimeError(f"OpenRouter HTTP error: {e}") from e

        data = response.json()
        return self._parse_response(data)

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion chunks from OpenRouter.

        Yields text content chunks. Tool calls during streaming
        are not yielded as text; use chat() for tool-call support.
        """
        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages,
            resolved_model,
            tools,
            temperature,
            max_tokens,
        )
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST",
                _OPENROUTER_CHAT_URL,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    import json

                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

        except httpx.HTTPStatusError as e:
            logger.error("OpenRouter stream error: HTTP %d", e.response.status_code)
            raise RuntimeError(f"OpenRouter stream error: HTTP {e.response.status_code}") from e

        except httpx.HTTPError as e:
            logger.error("OpenRouter stream HTTP error: %s", e)
            raise RuntimeError(f"OpenRouter stream error: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()

    def _build_payload(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict:
        """Build the OpenAI-compatible request payload."""
        payload: dict = {
            "model": model,
            "messages": [self._serialize_message(m) for m in messages],
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        return payload

    @staticmethod
    def _serialize_message(msg: LLMMessage) -> dict:
        """Convert a generic LLMMessage to OpenAI-compatible dict."""
        result: dict = {"role": msg.role}

        if msg.content is not None:
            result["content"] = msg.content

        if msg.tool_calls is not None:
            result["tool_calls"] = msg.tool_calls

        if msg.tool_call_id is not None:
            result["tool_call_id"] = msg.tool_call_id

        if msg.name is not None:
            result["name"] = msg.name

        return result

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        """Parse OpenRouter/OpenAI-compatible response into LLMResponse."""
        choices = data.get("choices", [])

        if not choices:
            return LLMResponse(
                content=None,
                finish_reason="empty",
                metadata={"raw": data},
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        finish_reason = choice.get("finish_reason", "stop")

        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        logger.info(
            "OpenRouter response (tokens=%s, finish=%s)",
            usage.get("total_tokens"),
            finish_reason,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "openrouter", "model": data.get("model", "")},
        )
