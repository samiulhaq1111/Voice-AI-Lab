"""Chat API endpoint for text-based agent interaction."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging import logger
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatError, process_chat

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Send a text message to the agent and receive a response.

    The agent will process the message through the LLM, handle any tool calls,
    and return the final response.

    - If session_id is null, a new session is created.
    - Provider and model default to configured values if not specified.
    - API keys are never exposed in the response.
    """
    logger.info("Chat request (session=%s)", request.session_id or "new")

    try:
        result = await process_chat(
            db=db,
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
            model=request.model,
        )
    except ChatError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    return ChatResponse(**result)
