"""Tests for the Retell custom-function endpoints (Phase 9 POC)."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_employee_known_id(client: AsyncClient) -> None:
    """POST with a known employee_id returns 200 with correct data."""
    response = await client.post(
        "/api/v1/retell/tools/get_employee",
        json={"employee_id": "E001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    result = data["result"]
    assert result["employee_id"] == "E001"
    assert result["name"] == "Alice Johnson"
    assert result["department"] == "Engineering"
    assert result["title"] == "Senior Developer"


@pytest.mark.asyncio
async def test_get_employee_second_id(client: AsyncClient) -> None:
    """POST with E002 returns Bob Smith."""
    response = await client.post(
        "/api/v1/retell/tools/get_employee",
        json={"employee_id": "E002"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["name"] == "Bob Smith"
    assert data["result"]["department"] == "Marketing"


@pytest.mark.asyncio
async def test_get_employee_unknown_id(client: AsyncClient) -> None:
    """POST with an unknown employee_id returns the default 'Unknown' record."""
    response = await client.post(
        "/api/v1/retell/tools/get_employee",
        json={"employee_id": "E999"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    result = data["result"]
    assert result["employee_id"] == "E999"
    assert result["name"] == "Unknown"
    assert result["department"] == "Unknown"


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_employee_missing_field(client: AsyncClient) -> None:
    """POST with no employee_id returns 200 with success=false (debug mode)."""
    response = await client.post(
        "/api/v1/retell/tools/get_employee",
        json={},
    )
    # Debug endpoint returns 200 + success=false instead of 422
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "raw_payload" in data


@pytest.mark.asyncio
async def test_get_employee_empty_string(client: AsyncClient) -> None:
    """POST with empty employee_id returns 200 with success=false (debug mode)."""
    response = await client.post(
        "/api/v1/retell/tools/get_employee",
        json={"employee_id": ""},
    )
    # Debug endpoint: empty string is falsy, so success=false
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False


@pytest.mark.asyncio
async def test_get_employee_handler_exception(client: AsyncClient) -> None:
    """If the handler raises, the endpoint returns success=false (debug mode)."""
    with patch(
        "app.api.retell.get_employee_handler",
        side_effect=RuntimeError("boom"),
    ):
        response = await client.post(
            "/api/v1/retell/tools/get_employee",
            json={"employee_id": "E001"},
        )
    # Debug endpoint returns 200 + success=false instead of 500
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "error" in data


# ---------------------------------------------------------------------------
# Verify no duplication — the real handler is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_employee_calls_existing_handler(client: AsyncClient) -> None:
    """Verify the endpoint delegates to the real get_employee_handler."""
    from app.tools.demo_tools import get_employee_handler

    with patch(
        "app.api.retell.get_employee_handler",
        wraps=get_employee_handler,
    ) as mock_handler:
        response = await client.post(
            "/api/v1/retell/tools/get_employee",
            json={"employee_id": "E001"},
        )
    assert response.status_code == 200
    mock_handler.assert_awaited_once_with(employee_id="E001")
