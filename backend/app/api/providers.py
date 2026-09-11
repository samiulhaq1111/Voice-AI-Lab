"""Provider availability API endpoint."""

from fastapi import APIRouter

from app.providers.factory import get_available_providers

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
async def list_providers() -> dict:
    """List available providers and their configuration status.

    Returns safe metadata only — no API keys or secrets.
    """
    return get_available_providers()
