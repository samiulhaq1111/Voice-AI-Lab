"""Telnyx Call Control V2 webhook endpoint (Phase 6C — telephony).

Receives Telnyx webhook events, answers inbound calls, and issues a
fixed greeting speak action. No AI/media/STT/TTS yet.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.logging import logger

router = APIRouter()


@router.post("/api/v1/voice/telephony/webhook")
async def telnyx_webhook(request: Request) -> JSONResponse:
    """Receive Telnyx Call Control V2 webhook events.

    On call.initiated: answers the inbound call via Telnyx Call Control.
    Returns 200 to Telnyx promptly.
    """
    try:
        body = await request.json()
    except Exception:
        logger.warning("[VOICE:TELNYX] Webhook received with non-JSON body")
        return JSONResponse(status_code=200, content={"status": "ok"})

    logger.info("[VOICE:TELNYX] Webhook received")

    # Extract fields defensively — Telnyx envelope may vary
    data = body.get("data", {}) if isinstance(body, dict) else {}
    event_type = data.get("event_type", "") if isinstance(data, dict) else ""
    event_id = data.get("id", "") if isinstance(data, dict) else ""

    payload = data.get("payload", {}) if isinstance(data, dict) else {}
    call_control_id = (
        payload.get("call_control_id", "") if isinstance(payload, dict) else ""
    )

    if event_type:
        logger.info("[VOICE:TELNYX] event_type=%s", event_type)
    if event_id:
        logger.info("[VOICE:TELNYX] event_id=%s", event_id)
    if call_control_id:
        logger.info("[VOICE:TELNYX] call_control_id=%s", call_control_id)

    # Answer inbound call on call.initiated, then speak greeting
    if event_type == "call.initiated" and call_control_id:
        await _handle_inbound_call(call_control_id)

    return JSONResponse(status_code=200, content={"status": "ok"})


async def _handle_inbound_call(call_control_id: str) -> None:
    """Answer an inbound call and speak a fixed greeting."""
    from app.services.telnyx_client import TelnyxCallControlClient

    greeting = "Hello, this is the Voice AI Lab."
    client = None
    try:
        client = TelnyxCallControlClient()
        await client.answer_call(call_control_id)
        await client.speak(call_control_id, greeting)
    except ValueError as e:
        logger.error("[VOICE:TELNYX] Cannot handle call — %s", e)
    except RuntimeError as e:
        logger.error("[VOICE:TELNYX] Call handling failed — %s", e)
    except Exception as e:
        logger.error(
            "[VOICE:TELNYX] Call handling unexpected error — %s: %s",
            type(e).__name__,
            e,
        )
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
