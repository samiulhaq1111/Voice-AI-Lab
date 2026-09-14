"""WebSocket endpoint for the turn-based voice pipeline."""

import base64
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.logging import logger
from app.services.voice_service import (
    VoiceConfig,
    VoiceError,
    VoiceEvents,
    _create_session,
    _validate_config,
    process_voice_turn,
)

router = APIRouter()


def _send_json(ws: WebSocket, event: dict[str, Any]) -> Any:
    """Send a JSON event to the WebSocket client (async)."""
    return ws.send_json(event)


async def _handle_voice_session(ws: WebSocket) -> None:
    """Handle a complete voice session over WebSocket."""
    db: Session | None = None
    audio_chunks: list[bytes] = []
    session = None
    cfg: VoiceConfig | None = None
    started = False

    try:
        while True:
            # Receive either JSON text or binary audio
            raw = await ws.receive()

            if "text" in raw:
                # JSON control message
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type", "")
                logger.info("[VOICE] Client event received event_type=%s", msg_type)

                if msg_type == "start":
                    if started:
                        await ws.send_json({"type": "error", "message": "Session already started"})
                        continue

                    # Parse configuration
                    config_data = msg.get("configuration", {})
                    session_id = msg.get("session_id")
                    cfg = VoiceConfig.from_dict(config_data)

                    # Validate providers
                    try:
                        _validate_config(cfg)
                    except VoiceError as e:
                        await ws.send_json({"type": "error", "message": e.message})
                        continue

                    # Create or reuse session
                    db = SessionLocal()
                    try:
                        session = _create_session(db, cfg, session_id)
                    except VoiceError as e:
                        await ws.send_json({"type": "error", "message": e.message})
                        db.close()
                        db = None
                        continue

                    started = True
                    logger.info(
                        "[VOICE] Session created/reused session_id=%s",
                        session.id,
                    )
                    logger.info(
                        "[VOICE] Configuration received STT=%s/%s LLM=%s/%s TTS=%s/%s/%s",
                        cfg.stt_provider,
                        cfg.stt_model,
                        cfg.llm_provider,
                        cfg.llm_model,
                        cfg.tts_provider,
                        cfg.tts_model,
                        cfg.tts_voice or "(default)",
                    )
                    await ws.send_json({"type": "session_started", "session_id": session.id})
                    audio_chunks = []

                elif msg_type == "stop":
                    if not started or session is None or db is None or cfg is None:
                        await ws.send_json({"type": "error", "message": "No active session"})
                        continue

                    if not audio_chunks:
                        await ws.send_json({"type": "error", "message": "No audio received"})
                        continue

                    # Combine audio chunks
                    audio_data = b"".join(audio_chunks)
                    audio_chunks = []
                    logger.info(
                        "[VOICE] Voice turn stopped audio_bytes=%d",
                        len(audio_data),
                    )

                    # Build event callbacks
                    async def _on_processing():
                        await ws.send_json({"type": "processing"})

                    async def _on_transcript(text: str, final: bool):
                        await ws.send_json({"type": "transcript", "text": text, "final": final})

                    async def _on_agent_response(text: str):
                        await ws.send_json({"type": "agent_response", "text": text})

                    async def _on_tool_call(tc: dict, result: dict):
                        await ws.send_json(
                            {
                                "type": "tool_call",
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": json.loads(
                                    tc.get("function", {}).get("arguments", "{}")
                                ),
                            }
                        )

                    async def _on_audio(audio_bytes: bytes, ct: str):
                        await ws.send_json(
                            {
                                "type": "audio",
                                "format": ct,
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        )

                    async def _on_completed():
                        await ws.send_json({"type": "completed"})

                    async def _on_error(msg: str):
                        await ws.send_json({"type": "error", "message": msg})

                    events = VoiceEvents(
                        on_processing=_on_processing,
                        on_transcript=_on_transcript,
                        on_agent_response=_on_agent_response,
                        on_tool_call=_on_tool_call,
                        on_audio=_on_audio,
                        on_completed=_on_completed,
                        on_error=_on_error,
                    )

                    try:
                        await process_voice_turn(
                            db=db,
                            audio_data=audio_data,
                            session=session,
                            cfg=cfg,
                            events=events,
                            audio_content_type="audio/webm",
                        )
                    except VoiceError as e:
                        await ws.send_json({"type": "error", "message": e.message})
                    except Exception as e:
                        logger.error("Voice turn failed: %s", e)
                        await ws.send_json({"type": "error", "message": f"Internal error: {e}"})

                    # Mark session ended
                    from datetime import UTC, datetime

                    session.ended_at = datetime.now(UTC)
                    session.status = "completed"
                    db.commit()

                    # Reset for next turn
                    started = False
                    session = None
                    cfg = None
                    db.close()
                    db = None

                else:
                    msg = f"Unknown message type: {msg_type}"
                    await ws.send_json({"type": "error", "message": msg})

            elif "bytes" in raw:
                # Binary audio data
                if not started:
                    continue  # Ignore audio before START
                audio_chunks.append(raw["bytes"])
                logger.info(
                    "[VOICE] Audio received bytes=%d total_chunks=%d",
                    len(raw["bytes"]),
                    len(audio_chunks),
                )

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
    except Exception as e:
        logger.error("Voice WebSocket error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if db is not None:
            db.close()


@router.websocket("/api/v1/voice/ws")
async def voice_websocket(ws: WebSocket) -> None:
    """Turn-based voice pipeline over WebSocket.

    Protocol:
        Client → Server:
            {"type": "start", "session_id": null, "configuration": {...}}
            <binary audio chunks>
            {"type": "stop"}

        Server → Client:
            {"type": "session_started", "session_id": "..."}
            {"type": "processing"}
            {"type": "transcript", "text": "...", "final": true}
            {"type": "agent_response", "text": "..."}
            {"type": "tool_call", "name": "...", "arguments": {...}}
            {"type": "audio", "format": "audio/mpeg", "data": "<base64>"}
            {"type": "completed"}
            {"type": "error", "message": "..."}
    """
    await ws.accept()
    logger.info("[VOICE] WebSocket connection accepted")
    await _handle_voice_session(ws)
