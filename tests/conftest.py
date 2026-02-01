import os
import pytest
from typing import AsyncGenerator
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

# Set environment variables BEFORE importing app
os.environ.setdefault("LIVEKIT_URL", "wss://test.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-api-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-api-secret-that-is-long-enough")
os.environ.setdefault("APP_ENV", "development")

from src.main import app
from src.core.security import AuthUser
from src.api.deps import get_current_user

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with unauthenticated access by default."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def mock_auth_user() -> AuthUser:
    """Create a mock authenticated user."""
    return AuthUser({
        "sub": "test-user-id",
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated"
    })

@pytest.fixture
async def auth_client(mock_auth_user) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client that is authenticated."""
    # Override the auth dependency
    app.dependency_overrides[get_current_user] = lambda: mock_auth_user
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
    
    # Clean up override
    app.dependency_overrides.clear()
