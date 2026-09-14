"""Provider availability and diagnostic API endpoint."""

import httpx
from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger
from app.providers.factory import get_available_providers

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
async def list_providers() -> dict:
    """List available providers and their configuration status.

    Returns safe metadata only — no API keys or secrets.
    """
    return get_available_providers()


@router.get("/diagnostics")
async def provider_diagnostics() -> dict:
    """Development-only diagnostic: test actual provider connectivity.

    Makes a minimal authenticated request to each configured provider
    to distinguish configuration problems from credential problems.

    NEVER returns API keys or secrets.
    """
    result: dict = {
        "stt": {},
        "llm": {},
        "tts": {},
    }

    # --- Deepgram STT diagnostic ---
    dg_key = settings.deepgram_api_key
    dg_configured = bool(dg_key)
    dg_info: dict = {
        "configured": dg_configured,
        "default_model": settings.default_stt_model or "nova-3",
    }

    if dg_configured:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.get(
                    "https://api.deepgram.com/v1/auth/token",
                    headers={"Authorization": f"Token {dg_key}"},
                )
                dg_info["authenticated"] = resp.status_code == 200
                dg_info["status"] = resp.status_code
                if resp.status_code == 401:
                    dg_info["error"] = "Invalid credentials"
                elif resp.status_code == 403:
                    dg_info["error"] = "Forbidden — check API key permissions"
        except httpx.TimeoutException:
            dg_info["authenticated"] = False
            dg_info["error"] = "Connection timed out"
        except Exception as e:
            dg_info["authenticated"] = False
            dg_info["error"] = f"Connection failed: {type(e).__name__}"

    result["stt"]["deepgram"] = dg_info

    # --- OpenRouter LLM diagnostic ---
    or_key = settings.openrouter_api_key
    result["llm"]["openrouter"] = {
        "configured": bool(or_key),
        "default_model": settings.default_llm_model or "(not set)",
    }

    # --- ElevenLabs TTS diagnostic ---
    el_key = settings.elevenlabs_api_key
    result["tts"]["elevenlabs"] = {
        "configured": bool(el_key),
        "default_model": settings.default_tts_model or "eleven_flash_v2_5",
        "default_voice_configured": bool(settings.default_tts_voice),
    }

    logger.info(
        "[CONFIG] Provider diagnostics: deepgram=%s openrouter=%s elevenlabs=%s",
        dg_configured,
        bool(or_key),
        bool(el_key),
    )

    return result
