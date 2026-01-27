"""Tests for token generation endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create async test client."""
    # Set required env vars for testing
    import os
    os.environ.setdefault("LIVEKIT_URL", "wss://test.livekit.cloud")
    os.environ.setdefault("LIVEKIT_API_KEY", "test-api-key")
    os.environ.setdefault("LIVEKIT_API_SECRET", "test-api-secret-that-is-long-enough")
    
    from main import app
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    """Test health endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_root(client: AsyncClient):
    """Test root endpoint returns API info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


@pytest.mark.anyio
async def test_generate_token_post(client: AsyncClient):
    """Test token generation via POST."""
    response = await client.post(
        "/token",
        json={
            "room_name": "test-room",
            "participant_name": "test-user",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert data["room_name"] == "test-room"
    assert data["participant_identity"] == "test-user"


@pytest.mark.anyio
async def test_generate_token_get(client: AsyncClient):
    """Test token generation via GET."""
    response = await client.get(
        "/token",
        params={
            "room_name": "test-room",
            "participant_name": "test-user",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
