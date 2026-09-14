"""Voice service: orchestrates the turn-based voice pipeline.

Flow:
    audio bytes → STT → transcript → AgentRuntime → text response → TTS → audio bytes

This service is provider-independent. It obtains providers through the factory
and uses interfaces (STTInterface, LLMInterface, TTSInterface).
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.core.config import settings
from app.core.logging import logger
from app.models.message import Message
from app.models.tool_call import ToolCall
from app.models.usage_record import UsageRecord
from app.models.voice_session import VoiceSession
from app.providers.factory import (
    ProviderError,
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
)
from app.providers.types import STTResult, TTSResult
from app.services.chat_service import _load_history
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor


@dataclass
class VoiceConfig:
    """Provider configuration for a voice session."""

    stt_provider: str = ""
    stt_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    tts_provider: str = ""
    tts_model: str = ""
    tts_voice: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, str] | None) -> "VoiceConfig":
        """Build config from optional browser-supplied dict, filling defaults."""
        if not data:
            data = {}
        return cls(
            stt_provider=data.get("stt_provider", "") or settings.default_stt_provider,
            stt_model=data.get("stt_model", "") or settings.default_stt_model,
            llm_provider=data.get("llm_provider", "") or settings.default_llm_provider,
            llm_model=data.get("llm_model", "") or settings.default_llm_model,
            tts_provider=data.get("tts_provider", "") or settings.default_tts_provider,
            tts_model=data.get("tts_model", "") or settings.default_tts_model,
            tts_voice=data.get("tts_voice", "") or settings.default_tts_voice,
        )


@dataclass
class VoiceEvents:
    """Callbacks for emitting events to the WebSocket client.

    All callbacks are async callables.
    """

    on_processing: Callable[[], Any] | None = None
    on_transcript: Callable[[str, bool], Any] | None = None
    on_agent_response: Callable[[str], Any] | None = None
    on_tool_call: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None
    on_audio: Callable[[bytes, str], Any] | None = None  # (audio_bytes, content_type)
    on_completed: Callable[[], Any] | None = None
    on_error: Callable[[str], Any] | None = None


class VoiceError(Exception):
    """Raised when a voice request cannot be processed."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _validate_config(cfg: VoiceConfig) -> None:
    """Validate that requested providers are supported and configured."""
    from app.providers.factory import (
        SUPPORTED_LLM_PROVIDERS,
        SUPPORTED_STT_PROVIDERS,
        SUPPORTED_TTS_PROVIDERS,
    )

    if cfg.stt_provider not in SUPPORTED_STT_PROVIDERS:
        raise VoiceError(f"Unsupported STT provider: '{cfg.stt_provider}'")
    if cfg.llm_provider not in SUPPORTED_LLM_PROVIDERS:
        raise VoiceError(f"Unsupported LLM provider: '{cfg.llm_provider}'")
    if cfg.tts_provider not in SUPPORTED_TTS_PROVIDERS:
        raise VoiceError(f"Unsupported TTS provider: '{cfg.tts_provider}'")

    if not settings.is_provider_configured(cfg.stt_provider):
        raise VoiceError(f"STT provider '{cfg.stt_provider}' is not configured (missing API key)")
    if not settings.is_provider_configured(cfg.llm_provider):
        raise VoiceError(f"LLM provider '{cfg.llm_provider}' is not configured (missing API key)")
    if not settings.is_provider_configured(cfg.tts_provider):
        raise VoiceError(f"TTS provider '{cfg.tts_provider}' is not configured (missing API key)")


def _create_session(db: Session, cfg: VoiceConfig, session_id: str | None) -> VoiceSession:
    """Create or reuse a voice session."""
    if session_id:
        session = db.query(VoiceSession).filter(VoiceSession.id == session_id).first()
        if session is None:
            raise VoiceError(f"Session '{session_id}' not found")
        # Update provider config on the existing session
        session.stt_provider = cfg.stt_provider
        session.stt_model = cfg.stt_model
        session.llm_provider = cfg.llm_provider
        session.llm_model = cfg.llm_model
        session.tts_provider = cfg.tts_provider
        session.tts_model = cfg.tts_model
        session.tts_voice = cfg.tts_voice
        session.status = "active"
        session.ended_at = None
        db.commit()
        db.refresh(session)
        return session

    session = VoiceSession(
        status="active",
        stt_provider=cfg.stt_provider,
        stt_model=cfg.stt_model,
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
        tts_provider=cfg.tts_provider,
        tts_model=cfg.tts_model,
        tts_voice=cfg.tts_voice,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info("Created voice session %s", session.id)
    return session


def _save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str | None,
    sequence: int,
    token_count: int | None = None,
    llm_data: dict[str, Any] | None = None,
) -> Message:
    """Persist a message to the database."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        sequence=sequence,
        token_count=token_count,
        llm_data=json.dumps(llm_data) if llm_data else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def _save_tool_call(
    db: Session,
    session_id: str,
    tool_name: str,
    tool_input: str | None,
    tool_output: str | None,
    status: str,
    error_message: str | None = None,
    duration_ms: float | None = None,
) -> ToolCall:
    """Persist a tool call record."""
    tc = ToolCall(
        session_id=session_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    db.add(tc)
    db.commit()
    db.refresh(tc)
    return tc


def _save_usage(
    db: Session,
    session_id: str,
    provider_type: str,
    provider_name: str,
    model: str | None,
    usage: dict[str, int] | None = None,
    duration_seconds: float | None = None,
) -> UsageRecord:
    """Persist a usage record."""
    record = UsageRecord(
        session_id=session_id,
        provider_type=provider_type,
        provider_name=provider_name,
        model=model,
        input_tokens=usage.get("prompt_tokens") if usage else None,
        output_tokens=usage.get("completion_tokens") if usage else None,
        duration_seconds=duration_seconds,
        metadata_json=json.dumps({"total_tokens": usage.get("total_tokens", 0)}) if usage else None,
    )
    db.add(record)
    db.commit()
    return record


async def process_voice_turn(
    db: Session,
    audio_data: bytes,
    session: VoiceSession,
    cfg: VoiceConfig,
    events: VoiceEvents,
    audio_content_type: str = "audio/webm",
) -> dict[str, Any]:
    """Process a single voice turn: STT → Agent → TTS.

    Args:
        db: Database session.
        audio_data: Raw audio bytes from the browser.
        session: The voice session (already created/reused).
        cfg: Resolved provider configuration.
        events: Event callbacks for WebSocket emission.
        audio_content_type: MIME type of the incoming audio.

    Returns:
        Dict with transcript, response, session_id, tool_calls.
    """
    start_time = time.monotonic()
    sid = session.id
    logger.info("[VOICE] Turn processing started session_id=%s", sid)

    # --- STT ---
    stt_start = time.monotonic()
    logger.info("[VOICE] STT started provider=%s model=%s", cfg.stt_provider, cfg.stt_model)
    try:
        stt = get_stt_provider(cfg.stt_provider, model=cfg.stt_model)
        stt_result: STTResult = await stt.transcribe(
            audio_data,
            model=cfg.stt_model,
            content_type=audio_content_type,
        )
    except Exception as e:
        stt_duration = time.monotonic() - stt_start
        logger.error(
            "[VOICE] Turn failed stage=stt error_type=%s error=%s",
            type(e).__name__,
            str(e),
        )
        if events.on_error:
            await events.on_error(f"STT error: {e}")
        raise VoiceError(f"STT failed: {e}") from e
    finally:
        await stt.close()

    stt_duration = time.monotonic() - stt_start
    transcript = stt_result.text.strip()
    logger.info(
        "[VOICE] STT completed transcript_length=%d duration_seconds=%.2f",
        len(transcript),
        stt_duration,
    )

    if not transcript:
        logger.warning(
            "[VOICE] Empty transcript from STT "
            "audio_bytes=%d stt_provider=%s action=skip_agent_skip_tts",
            len(audio_data),
            cfg.stt_provider,
        )
        if events.on_transcript:
            await events.on_transcript("", True)
        if events.on_completed:
            await events.on_completed()
        return {
            "transcript": "",
            "response": "",
            "session_id": sid,
            "tool_calls": [],
        }

    # Emit transcript
    if events.on_transcript:
        await events.on_transcript(transcript, True)

    # Save STT usage
    _save_usage(db, sid, "stt", cfg.stt_provider, cfg.stt_model, duration_seconds=stt_duration)

    # --- Agent ---
    if events.on_processing:
        await events.on_processing()

    logger.info("[VOICE] Agent started provider=%s model=%s", cfg.llm_provider, cfg.llm_model)

    try:
        llm = get_llm_provider(cfg.llm_provider, model=cfg.llm_model)
    except (ProviderError, ValueError) as e:
        raise VoiceError(str(e)) from e

    # Load conversation history
    history = _load_history(db, sid)

    # Build agent
    tool_registry = get_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    config = AgentConfig(
        llm_model=cfg.llm_model,
        max_tool_rounds=settings.max_agent_iterations,
    )
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        config=config,
    )

    # Tool call callback → emit events
    async def _on_tool_call(tc: dict[str, Any], result: dict[str, Any]) -> None:
        if events.on_tool_call:
            await events.on_tool_call(tc, result)

    try:
        agent_result: AgentResult = await runtime.run(
            transcript,
            initial_messages=history,
            on_tool_call=_on_tool_call,
        )
    except Exception as e:
        logger.error(
            "[VOICE] Turn failed stage=agent error_type=%s error=%s",
            type(e).__name__,
            str(e),
        )
        if events.on_error:
            await events.on_error(f"Agent error: {e}")
        raise VoiceError(f"Agent failed: {e}") from e

    response_text = agent_result.response
    logger.info(
        "[VOICE] Agent completed iterations=%d tool_calls=%d",
        agent_result.iterations,
        len(agent_result.tool_calls),
    )

    # Save agent messages
    seq = session.message_count
    new_start = len(history)
    new_msgs = runtime.state.messages[new_start:]
    for i, llm_msg in enumerate(new_msgs):
        save_seq = seq + i
        token_count: int | None = None
        if i == len(new_msgs) - 1 and llm_msg.role == "assistant":
            token_count = agent_result.usage.get("total_tokens")
        _save_message(
            db,
            sid,
            llm_msg.role,
            llm_msg.content,
            save_seq,
            token_count=token_count,
            llm_data={
                "role": llm_msg.role,
                "content": llm_msg.content,
                "tool_calls": llm_msg.tool_calls,
                "tool_call_id": llm_msg.tool_call_id,
                "name": llm_msg.name,
            },
        )
    session.message_count = seq + len(new_msgs)

    # Save tool call records
    for tc_dict in agent_result.tool_calls:
        func_data = tc_dict.get("function", {})
        tool_name = func_data.get("name", "unknown")
        tool_input = func_data.get("arguments", "{}")
        _save_tool_call(db, sid, tool_name, tool_input, None, "success")

    # Save LLM usage
    _save_usage(db, sid, "llm", cfg.llm_provider, cfg.llm_model, usage=agent_result.usage)

    # Emit agent response
    if events.on_agent_response:
        await events.on_agent_response(response_text)

    # --- TTS ---
    tts_start = time.monotonic()
    audio_bytes: bytes = b""
    audio_ct: str = "audio/mpeg"
    tts_failed = False

    if response_text:
        logger.info("[VOICE] TTS started provider=%s model=%s", cfg.tts_provider, cfg.tts_model)
        try:
            tts = get_tts_provider(
                cfg.tts_provider,
                voice=cfg.tts_voice,
                model=cfg.tts_model,
            )
            tts_result: TTSResult = await tts.synthesize(
                response_text,
                voice=cfg.tts_voice,
                model=cfg.tts_model,
            )
            audio_bytes = tts_result.audio_data
            audio_ct = tts_result.content_type
            logger.info(
                "[VOICE] TTS completed audio_bytes=%d",
                len(audio_bytes),
            )
        except Exception as e:
            tts_failed = True
            logger.error(
                "[VOICE] Turn failed stage=tts error_type=%s error=%s",
                type(e).__name__,
                str(e),
            )
            if events.on_error:
                await events.on_error(f"TTS error: {e}")
        finally:
            try:
                await tts.close()
            except Exception:
                pass

    tts_duration = time.monotonic() - tts_start

    # Emit audio only if TTS succeeded
    if audio_bytes and events.on_audio:
        await events.on_audio(audio_bytes, audio_ct)

    # Save TTS usage
    _save_usage(db, sid, "tts", cfg.tts_provider, cfg.tts_model, duration_seconds=tts_duration)

    # --- Complete (only if TTS did not fail) ---
    if not tts_failed:
        if events.on_completed:
            await events.on_completed()

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "[VOICE] Voice turn completed session_id=%s latency_ms=%.0f "
        "transcript_length=%d response_length=%d",
        sid,
        latency_ms,
        len(transcript),
        len(response_text),
    )

    return {
        "transcript": transcript,
        "response": response_text,
        "session_id": sid,
        "tool_calls": agent_result.tool_calls,
        "latency_ms": round(latency_ms, 2),
    }
