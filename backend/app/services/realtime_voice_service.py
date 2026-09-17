"""Realtime voice service: orchestrates the realtime voice pipeline (Phase 6B).

Flow:
    streaming STT → final utterance → AgentRuntime → text response → browser

This service manages:
    - Realtime voice session state
    - Conversation history persistence
    - AgentRuntime invocation on final utterances
    - Message persistence

The WebSocket endpoint remains responsible for:
    - WebSocket communication
    - Realtime STT event forwarding
    - Receiving final utterances
    - Invoking this service
    - Sending agent_response/error events
"""

import json
from typing import Any

from sqlalchemy.orm import Session

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.core.config import settings
from app.core.logging import logger
from app.models.message import Message
from app.models.tool_call import ToolCall
from app.models.usage_record import UsageRecord
from app.models.voice_session import VoiceSession
from app.providers.factory import ProviderError, get_llm_provider
from app.providers.llm.interface import LLMInterface
from app.providers.types import LLMMessage
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor


class RealtimeVoiceError(Exception):
    """Raised when a realtime voice request cannot be processed."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _message_to_llm(msg: Message) -> LLMMessage:
    """Convert a persisted Message to a generic LLMMessage.

    Uses llm_data (full JSON) when available, falls back to
    role/content for backward compatibility.
    """
    if msg.llm_data:
        try:
            data = json.loads(msg.llm_data)
            return LLMMessage(
                role=data.get("role", msg.role),
                content=data.get("content"),
                tool_calls=data.get("tool_calls"),
                tool_call_id=data.get("tool_call_id"),
                name=data.get("name"),
            )
        except (json.JSONDecodeError, TypeError):
            pass
    return LLMMessage(role=msg.role, content=msg.content)


def _load_history(db: Session, session_id: str) -> list[LLMMessage]:
    """Load recent conversation messages from the database.

    Returns the latest N messages ordered by sequence, converted
    to generic LLMMessage objects suitable for AgentRuntime.
    """
    limit = settings.max_conversation_messages
    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.sequence.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()  # oldest first
    return [_message_to_llm(m) for m in rows]


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
    """Persist a tool call record to the database."""
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
    provider_name: str,
    model: str | None,
    usage: dict[str, int],
) -> UsageRecord:
    """Persist usage record to the database."""
    record = UsageRecord(
        session_id=session_id,
        provider_type="llm",
        provider_name=provider_name,
        model=model,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        metadata_json=json.dumps({"total_tokens": usage.get("total_tokens", 0)}),
    )
    db.add(record)
    db.commit()
    return record


def create_realtime_session(
    db: Session,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> VoiceSession:
    """Create a new realtime voice session.

    Args:
        db: Database session.
        llm_provider: LLM provider override.
        llm_model: LLM model override.

    Returns:
        The created VoiceSession.
    """
    resolved_provider = llm_provider or settings.default_llm_provider
    resolved_model = llm_model or settings.default_llm_model or None

    session = VoiceSession(
        status="active",
        llm_provider=resolved_provider,
        llm_model=resolved_model,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(
        "Created realtime voice session %s (provider=%s model=%s)",
        session.id,
        resolved_provider,
        resolved_model,
    )
    return session


async def process_realtime_utterance(
    db: Session,
    session: VoiceSession,
    transcript: str,
    llm: LLMInterface | None = None,
    on_tool_call: Any | None = None,
) -> dict[str, Any]:
    """Process a single utterance through AgentRuntime.

    Args:
        db: Database session.
        session: The voice session (already created).
        transcript: The final transcript text.
        llm: Optional pre-built LLM interface (for testing).
        on_tool_call: Optional callback for tool execution events.

    Returns:
        Dict with response, tool_calls, usage, iterations.

    Raises:
        RealtimeVoiceError: If processing fails.
    """
    sid = session.id
    resolved_provider = session.llm_provider or settings.default_llm_provider
    resolved_model = session.llm_model or settings.default_llm_model or None

    logger.info(
        "[REALTIME] Utterance processing started session_id=%s transcript_length=%d",
        sid,
        len(transcript),
    )

    # Get or create LLM
    if llm is None:
        try:
            llm = get_llm_provider(resolved_provider, model=resolved_model)
        except ProviderError as e:
            raise RealtimeVoiceError(str(e)) from e
        except ValueError as e:
            raise RealtimeVoiceError(str(e)) from e

    # Load conversation history BEFORE the agent loop
    history = _load_history(db, sid)
    logger.info("[REALTIME] Loaded %d history messages for session %s", len(history), sid)

    # Build agent
    tool_registry = get_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    config = AgentConfig(
        llm_model=resolved_model,
        max_tool_rounds=settings.max_agent_iterations,
    )
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        config=config,
    )

    # Run agent loop
    try:
        result: AgentResult = await runtime.run(
            transcript,
            initial_messages=history,
            on_tool_call=on_tool_call,
        )
    except Exception as e:
        logger.error(
            "[REALTIME] Utterance processing failed session_id=%s error_type=%s error=%s",
            sid,
            type(e).__name__,
            str(e),
        )
        raise RealtimeVoiceError(f"Agent error: {e}") from e

    response_text = result.response
    logger.info(
        "[REALTIME] Utterance processing completed session_id=%s iterations=%d tool_calls=%d",
        sid,
        result.iterations,
        len(result.tool_calls),
    )

    # Save all new messages from runtime state.
    # History messages are already persisted; save only the new ones
    # (user message + assistant response + any intermediate tool messages)
    seq = session.message_count
    new_start = len(history)
    new_msgs = runtime.state.messages[new_start:]
    for i, llm_msg in enumerate(new_msgs):
        save_seq = seq + i
        # Attach token count to the final assistant message
        token_count: int | None = None
        if i == len(new_msgs) - 1 and llm_msg.role == "assistant":
            token_count = result.usage.get("total_tokens")
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

    # Save tool calls
    for tc_dict in result.tool_calls:
        func_data = tc_dict.get("function", {})
        tool_name = func_data.get("name", "unknown")
        tool_input = func_data.get("arguments", "{}")
        _save_tool_call(db, sid, tool_name, tool_input, None, "success")

    # Save usage
    if result.usage:
        _save_usage(db, sid, resolved_provider, resolved_model, result.usage)

    # Update session
    session.status = "active"
    db.commit()

    # NOTE: Do NOT call runtime.close() here.
    # The LLM client is owned by the session (_handle_realtime_session) and
    # must survive across multiple utterances. runtime.close() would close
    # the shared LLM client, breaking subsequent utterances.

    return {
        "response": response_text,
        "tool_calls": result.tool_calls,
        "usage": result.usage,
        "iterations": result.iterations,
    }
