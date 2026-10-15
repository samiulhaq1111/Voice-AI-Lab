"""Provider availability and diagnostic API endpoint."""

import time
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger
from app.providers.factory import get_available_providers

router = APIRouter(prefix="/providers", tags=["providers"])

# ---------------------------------------------------------------------------
# Dynamic OpenRouter LLM catalogue (Chat/Voice model dropdowns).
#
# The curated factory lists (LLM_FREE_MODELS / LLM_PAID_MODELS) remain the
# authoritative base — existing models, defaults and the free/paid split are
# never altered. This endpoint additionally merges live model IDs fetched
# from OpenRouter so newly released models appear without manual edits.
# ---------------------------------------------------------------------------

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Bounded so the catalogue endpoint can never hang on a slow upstream.
_OPENROUTER_MODELS_TIMEOUT_S = 4.0
# In-process cache; the dropdowns refresh at most this often.
_OPENROUTER_MODELS_CACHE_TTL_S = 600.0

# Cache slots: (dynamic_free_ids, dynamic_paid_ids) + monotonic fetch marker.
_dynamic_models_cache: tuple[list[str], list[str]] | None = None
_dynamic_models_cached_at: float = 0.0
_dynamic_models_fetched_at: str = ""


def _price_is_zero(value: object) -> bool:
    """True only when a price is explicitly present and exactly zero."""
    if value is None:
        return False
    try:
        return float(str(value).strip()) == 0.0
    except ValueError:
        return False


async def _fetch_dynamic_model_groups() -> tuple[list[str], list[str]] | None:
    """Fetch ``(free_ids, paid_ids)`` from OpenRouter's models API.

    Returns None on missing credentials, timeout, or any upstream failure —
    callers then fall back to the static curated catalogue unchanged.
    Successful results are cached in-process for ~10 minutes. Model IDs are
    preserved exactly as returned by OpenRouter.
    """
    global _dynamic_models_cache, _dynamic_models_cached_at
    global _dynamic_models_fetched_at

    now = time.monotonic()
    if (
        _dynamic_models_cache is not None
        and now - _dynamic_models_cached_at < _OPENROUTER_MODELS_CACHE_TTL_S
    ):
        return _dynamic_models_cache

    api_key = settings.openrouter_api_key
    if not api_key:
        logger.warning(
            "[PROVIDERS] OPENROUTER_API_KEY not set; using static LLM catalogue"
        )
        return None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_OPENROUTER_MODELS_TIMEOUT_S)
        ) as client:
            response = await client.get(
                _OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                # Server-side filter: only models that can generate text.
                params={"output_modalities": "text"},
            )
            response.raise_for_status()
        payload = response.json()
        free_ids: list[str] = []
        paid_ids: list[str] = []
        for model in payload.get("data", []):
            model_id = model.get("id")
            if not model_id or not isinstance(model_id, str):
                continue
            # Defensive re-check of the documented output modality field.
            output_modalities = (model.get("architecture") or {}).get(
                "output_modalities"
            ) or []
            if "text" not in output_modalities:
                continue
            pricing = model.get("pricing") or {}
            if _price_is_zero(pricing.get("prompt")) and _price_is_zero(
                pricing.get("completion")
            ):
                free_ids.append(model_id)
            else:
                paid_ids.append(model_id)
    except httpx.TimeoutException:
        logger.warning(
            "[PROVIDERS] OpenRouter model catalogue timed out; "
            "using static LLM catalogue"
        )
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "[PROVIDERS] OpenRouter model catalogue HTTP %d; "
            "using static LLM catalogue",
            e.response.status_code,
        )
        return None
    except Exception as e:
        logger.warning(
            "[PROVIDERS] OpenRouter model catalogue fetch failed (%s); "
            "using static LLM catalogue",
            type(e).__name__,
        )
        return None

    _dynamic_models_cache = (free_ids, paid_ids)
    _dynamic_models_cached_at = now
    _dynamic_models_fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    logger.info(
        "[PROVIDERS] OpenRouter catalogue refreshed free=%d paid=%d",
        len(free_ids),
        len(paid_ids),
    )
    return _dynamic_models_cache


def _merge_dynamic_models(
    entry: dict,
    dynamic_free: list[str],
    dynamic_paid: list[str],
) -> dict:
    """Merge dynamic OpenRouter model IDs into a catalogue entry in place.

    Curated models keep their order and free/paid classification and are
    never duplicated or reclassified; dynamic free models extend the free
    group, dynamic paid models extend ``paid_models``. Upstream model IDs
    are preserved exactly.
    """
    curated = list(entry.get("models", []))
    seen = set(curated)
    new_free = [m for m in dynamic_free if m not in seen]
    new_paid = [m for m in dynamic_paid if m not in seen]
    entry["models"] = [*curated, *new_free, *new_paid]
    entry["paid_models"] = [*entry.get("paid_models", []), *new_paid]
    entry["dynamic"] = True
    entry["dynamic_fetched_at"] = _dynamic_models_fetched_at or None
    return entry


@router.get("")
async def list_providers() -> dict:
    """List available providers and their configuration status.

    Returns safe metadata only — no API keys or secrets. The OpenRouter LLM
    entry additionally merges dynamically fetched model IDs (text-output
    models only, ~10 min cache); on any fetch failure the static curated
    catalogue is returned unchanged.
    """
    result = get_available_providers()
    groups = await _fetch_dynamic_model_groups()
    if groups:
        dynamic_free, dynamic_paid = groups
        for entry in result.get("llm", []):
            if entry.get("provider") == "openrouter":
                _merge_dynamic_models(entry, dynamic_free, dynamic_paid)
    return result


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
