"""Retell AI custom-function endpoints (Phase 9 — Retell POC).

Thin HTTP wrappers around existing tool handlers so Retell agents
can invoke business logic via webhook.

This module is intentionally minimal — no authentication yet (development
only, behind ngrok).  Retell signature verification will be added in a
later phase.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.tools.demo_tools import get_employee_handler

router = APIRouter(prefix="/retell", tags=["retell"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class GetEmployeeRequest(BaseModel):
    """Request body for the get_employee custom function."""

    employee_id: str = Field(
        ...,
        min_length=1,
        description="Employee ID (e.g. E001)",
    )


class ToolSuccessResponse(BaseModel):
    """Standard success response for Retell custom functions."""

    success: bool = True
    result: dict


class ToolErrorResponse(BaseModel):
    """Standard error response for Retell custom functions."""

    success: bool = False
    error: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/tools/get_employee")
async def retell_get_employee(request: Request) -> dict:
    """TEMPORARY DEBUG ENDPOINT — logs raw Retell payload before parsing.

    This bypasses Pydantic validation so we can see exactly what Retell
    sends.  Will be reverted to the validated version once the payload
    shape is confirmed.
    """
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "(not set)")

    # Attempt JSON parse (may fail — that's fine, we log the raw bytes)
    try:
        import json

        parsed_json = json.loads(raw_body)
    except Exception:
        parsed_json = "(not valid JSON)"

    logger.info(
        "[RETELL-DEBUG] raw request: method=%s content_type=%s raw_body=%s parsed=%s",
        request.method,
        content_type,
        raw_body.decode("utf-8", errors="replace"),
        parsed_json,
    )

    # Try to extract employee_id from whatever shape Retell sends
    employee_id: str | None = None
    if isinstance(parsed_json, dict):
        # Direct shape: {"employee_id": "E001"}
        if "employee_id" in parsed_json:
            employee_id = parsed_json["employee_id"]
        # Wrapped shape: {"arguments": {"employee_id": "E001"}}
        elif "arguments" in parsed_json and isinstance(parsed_json["arguments"], dict):
            employee_id = parsed_json["arguments"].get("employee_id")
        # Alternative wrap: {"args": {"employee_id": "E001"}}
        elif "args" in parsed_json and isinstance(parsed_json["args"], dict):
            employee_id = parsed_json["args"].get("employee_id")

    if not employee_id:
        logger.warning(
            "[RETELL-DEBUG] could not extract employee_id from payload: %s",
            parsed_json,
        )
        return {
            "debug": True,
            "success": False,
            "error": "employee_id not found in payload",
            "raw_payload": parsed_json,
        }

    logger.info("[RETELL-DEBUG] extracted employee_id=%s", employee_id)

    try:
        result = await get_employee_handler(employee_id=employee_id)
    except Exception as e:
        logger.error(
            "[RETELL] get_employee failed employee_id=%s error=%s",
            employee_id,
            e,
        )
        return {"success": False, "error": "Employee lookup failed"}

    return {"success": True, "result": result}
