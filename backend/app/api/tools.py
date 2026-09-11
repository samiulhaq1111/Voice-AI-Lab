"""Tools API endpoints."""

from fastapi import APIRouter, Depends

from app.schemas import ToolInfoResponse, ToolListResponse
from app.services.tool_service import get_tool_registry
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("", response_model=ToolListResponse)
async def list_tools(registry: ToolRegistry = Depends(get_tool_registry)) -> ToolListResponse:
    """List all registered tools."""
    tools = registry.list_tools()
    return ToolListResponse(
        tools=[
            ToolInfoResponse(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                enabled=t.enabled,
            )
            for t in tools
        ],
        count=len(tools),
    )
