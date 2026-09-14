"""Basic tests for the Voice AI Lab backend."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Test the health check endpoint returns ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_tools_endpoint(client: AsyncClient) -> None:
    """Test the tools list endpoint returns demo tools."""
    response = await client.get("/api/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    tool_names = {t["name"] for t in data["tools"]}
    assert "get_weather" in tool_names
    assert "get_employee" in tool_names
    assert "get_leave_balance" in tool_names


@pytest.mark.asyncio
async def test_providers_endpoint(client: AsyncClient) -> None:
    """Test the providers endpoint returns provider metadata."""
    response = await client.get("/api/v1/providers")
    assert response.status_code == 200
    data = response.json()
    assert "stt" in data
    assert "llm" in data
    assert "tts" in data
    # Verify no API keys are exposed
    response_text = response.text
    assert "test-deepgram-key" not in response_text
    assert "test-openrouter-key" not in response_text
    assert "test-elevenlabs-key" not in response_text
    # Verify provider structure
    assert data["stt"][0]["provider"] == "deepgram"
    assert data["llm"][0]["provider"] == "openrouter"
    assert data["tts"][0]["provider"] == "elevenlabs"


@pytest.mark.asyncio
async def test_providers_diagnostics_endpoint(client: AsyncClient) -> None:
    """Test the diagnostics endpoint returns safe configuration status."""
    response = await client.get("/api/v1/providers/diagnostics")
    assert response.status_code == 200
    data = response.json()

    # Structure check
    assert "stt" in data
    assert "llm" in data
    assert "tts" in data
    assert "deepgram" in data["stt"]
    assert "openrouter" in data["llm"]
    assert "elevenlabs" in data["tts"]

    # Deepgram diagnostic fields
    dg = data["stt"]["deepgram"]
    assert "configured" in dg
    assert "default_model" in dg
    assert dg["configured"] is True  # conftest sets a fake key

    # No API keys must ever appear in the response
    response_text = response.text
    assert "test-deepgram-key" not in response_text
    assert "test-openrouter-key" not in response_text
    assert "test-elevenlabs-key" not in response_text
