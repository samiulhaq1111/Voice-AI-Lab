"""Focused tests for the shared Chat/Voice OpenRouter model catalogue.

Scope: adding current paid OpenRouter model IDs to the Chat/Voice selectable
list, removing the stale ``anthropic/claude-3.5-sonnet`` entry, preserving
free models / defaults / provider identity, and verifying that a selected
model ID reaches the OpenRouter adapter unchanged.

The Benchmark catalogue (app/benchmarks/configurations.py) is a separate
catalogue and is intentionally NOT covered here.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import settings
from app.providers.factory import (
    LLM_FREE_MODELS,
    LLM_PAID_MODELS,
    get_available_providers,
)
from app.providers.llm.openrouter import OpenRouterAdapter
from app.providers.types import LLMMessage
from app.services.voice_service import VoiceConfig

TARGET_PAID_MODEL_IDS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-5-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-chat-v3.1",
]

STALE_MODEL_ID = "anthropic/claude-3.5-sonnet"


def _llm_entry() -> dict:
    return get_available_providers()["llm"][0]


class TestProviderCatalogue:
    """Tests for the shared Chat/Voice model catalogue."""

    def test_target_paid_models_present(self) -> None:
        entry = _llm_entry()
        for model_id in TARGET_PAID_MODEL_IDS:
            assert model_id in entry["models"], f"missing from models: {model_id}"
            assert model_id in entry["paid_models"], f"missing from paid_models: {model_id}"

    def test_stale_claude_sonnet_removed(self) -> None:
        entry = _llm_entry()
        assert STALE_MODEL_ID not in entry["models"]
        assert STALE_MODEL_ID not in entry["paid_models"]

    def test_free_models_preserved(self) -> None:
        entry = _llm_entry()
        for model_id in LLM_FREE_MODELS:
            assert model_id in entry["models"], f"free model dropped: {model_id}"
            assert model_id not in entry["paid_models"]

    def test_models_equals_free_plus_paid(self) -> None:
        entry = _llm_entry()
        assert entry["models"] == [*LLM_FREE_MODELS, *LLM_PAID_MODELS]
        assert entry["paid_models"] == list(LLM_PAID_MODELS)

    def test_provider_identity_unchanged(self) -> None:
        info = get_available_providers()
        assert info["llm"][0]["provider"] == "openrouter"
        assert info["stt"][0]["provider"] == "deepgram"
        assert info["tts"][0]["provider"] == "elevenlabs"

    def test_default_model_preserved(self) -> None:
        entry = _llm_entry()
        expected = settings.default_llm_model or "nvidia/nemotron-3.5-lightning:free"
        assert entry["default_model"] == expected
        # The default must remain a selectable, working model.
        assert "openai/gpt-4o-mini" in entry["models"]

    def test_no_secrets_in_catalogue_payload(self) -> None:
        serialized = json.dumps(get_available_providers())
        assert "api_key" not in serialized
        assert "API_KEY" not in serialized
        for secret in (
            settings.openrouter_api_key,
            settings.deepgram_api_key,
            settings.elevenlabs_api_key,
        ):
            if secret:
                assert secret not in serialized


class TestVoiceConfigModelSelection:
    """Selected model IDs must pass through the voice config untouched."""

    def test_selected_paid_model_passthrough(self) -> None:
        cfg = VoiceConfig.from_dict(
            {"llm_provider": "openrouter", "llm_model": "openai/gpt-4.1-mini"}
        )
        assert cfg.llm_provider == "openrouter"
        assert cfg.llm_model == "openai/gpt-4.1-mini"

    def test_each_target_model_accepted(self) -> None:
        for model_id in TARGET_PAID_MODEL_IDS:
            cfg = VoiceConfig.from_dict({"llm_model": model_id})
            assert cfg.llm_model == model_id

    def test_empty_model_falls_back_to_settings_default(self) -> None:
        cfg = VoiceConfig.from_dict({"llm_model": ""})
        assert cfg.llm_model == settings.default_llm_model


class TestAdapterModelPassthrough:
    """Selected model IDs must reach the OpenRouter adapter payload unchanged."""

    @pytest.mark.asyncio
    async def test_new_paid_id_as_default_model(self) -> None:
        adapter = OpenRouterAdapter(api_key="test-key", default_model="openai/gpt-4.1-mini")
        mock_response = httpx.Response(
            200,
            json={
                "model": "openai/gpt-4.1-mini",
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock) as mocked:
            mocked.return_value = mock_response
            await adapter.chat([LLMMessage(role="user", content="Hi")])

        # The adapter passes the model ID through unchanged (no allowlist).
        sent = mocked.await_args.kwargs["json"]
        assert sent["model"] == "openai/gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_per_request_model_override(self) -> None:
        adapter = OpenRouterAdapter(api_key="test-key", default_model="openai/gpt-4o-mini")
        mock_response = httpx.Response(
            200,
            json={
                "model": "google/gemini-2.5-flash",
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
                ],
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        with patch.object(
            adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mocked:
            await adapter.chat(
                [LLMMessage(role="user", content="Hi")],
                model="google/gemini-2.5-flash",
            )

        assert mocked.await_args.kwargs["json"]["model"] == "google/gemini-2.5-flash"
