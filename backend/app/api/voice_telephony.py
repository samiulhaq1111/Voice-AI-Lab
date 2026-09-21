"""Telnyx Call Control V2 webhook endpoint (Phase 6C — telephony).

Receives Telnyx webhook events, answers inbound calls, speaks a greeting,
and starts bidirectional media streaming. Includes a WebSocket endpoint
for Telnyx media stream connections with Deepgram realtime STT,
AgentRuntime integration, and ElevenLabs TTS → telephone audio output.

Milestone 7 flow:
    Telnyx PCMU 8kHz → Deepgram PCM 16kHz → transcript
    → utterance_end → AgentRuntime → text response
    → ElevenLabs TTS → MP3 → PCMU 8kHz → Telnyx media WS → caller hears AI
"""

import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.services.telnyx_media import MediaStreamSession

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

    # Extract fields defensively — Telnyx envelope may vary
    data = body.get("data", {}) if isinstance(body, dict) else {}
    event_type = data.get("event_type", "") if isinstance(data, dict) else ""

    payload = data.get("payload", {}) if isinstance(data, dict) else {}
    call_control_id = (
        payload.get("call_control_id", "") if isinstance(payload, dict) else ""
    )

    logger.info("[VOICE:TELNYX] %s", event_type or "webhook")

    # Answer inbound call on call.initiated, speak greeting, start media stream
    if event_type == "call.initiated" and call_control_id:
        await _handle_inbound_call(call_control_id)

    return JSONResponse(status_code=200, content={"status": "ok"})


async def _handle_inbound_call(call_control_id: str) -> None:
    """Answer an inbound call, speak greeting, then start media streaming."""
    from app.services.telnyx_client import TelnyxCallControlClient

    greeting = "Hello, this is the Voice AI Lab."
    client = None
    try:
        client = TelnyxCallControlClient()
        await client.answer_call(call_control_id)
        await client.speak(call_control_id, greeting)
        # Start bidirectional media streaming after speak
        ws_url = settings.telnyx_media_ws_url
        if ws_url:
            await client.streaming_start(call_control_id, ws_url)
        else:
            logger.warning(
                "[VOICE:TELNYX] TELNYX_MEDIA_WS_URL not configured — "
                "skipping media stream"
            )
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


@router.websocket("/api/v1/voice/telephony/media")
async def telnyx_media_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for Telnyx bidirectional media streaming.

    Telnyx connects here after streaming_start. We receive media events,
    forward audio to Deepgram for realtime STT, and on utterance_end
    process the transcript through AgentRuntime.

    Milestone 6: Telnyx → Deepgram → AgentRuntime → text response (logged).
    """
    from app.services.telnyx_agent import TelephonyAgentSession
    from app.services.telnyx_deepgram import TelnyxDeepgramBridge
    from app.services.telnyx_tts import TelnyxTTSService

    await websocket.accept()
    session = MediaStreamSession()
    bridge = TelnyxDeepgramBridge()
    agent: TelephonyAgentSession | None = None
    tts_service = TelnyxTTSService()

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse message to extract event type
            try:
                msg = json.loads(raw)
                event = msg.get("event", "") if isinstance(msg, dict) else ""
            except json.JSONDecodeError:
                event = ""

            # Handle start event — initialize Deepgram bridge + agent
            if event == "start":
                stream_id = msg.get("stream_id", "")
                call_control_id = (
                    msg.get("start", {}).get("call_control_id", "")
                )
                session.handle_message(raw)
                try:
                    await bridge.start(stream_id)
                    await tts_service.start()
                    # Start agent session (consumes from bridge.utterance_queue)
                    agent = TelephonyAgentSession(
                        call_control_id=call_control_id or stream_id,
                        utterance_queue=bridge.utterance_queue,
                        tts_service=tts_service,
                        websocket=websocket,
                    )
                    await agent.start()
                except Exception as e:
                    logger.error(
                        "[VOICE:TELNYX:MEDIA] Session start failed "
                        "error=%s",
                        e,
                    )
                continue

            # Handle media event — forward to bridge (no test audio when TTS active)
            if event == "media":
                session.handle_message(raw)
                media_data = msg.get("media", {})

                # Forward to Deepgram bridge (non-blocking)
                if bridge._running:
                    await bridge.process_media_packet(media_data)

                # Only send test audio if TTS is not active
                if not tts_service._tts:
                    outbound = session.next_outbound_media()
                    if outbound is not None:
                        await websocket.send_text(json.dumps(outbound))
                continue

            # Handle stop event — stop bridge, agent, and TTS
            if event == "stop":
                session.handle_message(raw)
                if agent:
                    await agent.stop()
                await bridge.stop()
                await tts_service.stop()
                continue

            # Handle other events (connected, unknown)
            session.handle_message(raw)

    except WebSocketDisconnect:
        logger.info(
            "[VOICE:TELNYX] call.completed turns=%d "
            "stt_turns=%d inbound_packets=%d",
            agent.turn if agent else 0,
            bridge.utterances_emitted,
            session.media_packets_in,
        )
    except Exception as e:
        logger.error(
            "[VOICE:TELNYX:MEDIA] WebSocket error type=%s: %s",
            type(e).__name__,
            e,
        )
    finally:
        # Ensure cleanup on any exit
        if agent:
            await agent.stop()
        await bridge.stop()
        await tts_service.stop()
