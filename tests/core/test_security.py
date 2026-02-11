import pytest
import jwt
from unittest.mock import patch, MagicMock

from src.core.security import check_user_access, AuthUser, verify_token
from src.core.config import settings


def test_auth_user_model():
    """Test AuthUser model initialization."""
    payload = {"sub": "123", "email": "test@example.com", "role": "admin"}
    user = AuthUser(payload)
    assert user.id == "123"
    assert user.email == "test@example.com"
    assert user.role == "admin"


def test_check_user_access():
    """Test access control logic."""
    user = AuthUser({"sub": "user123"})
    
    # Can access own data
    assert check_user_access(user, "user123") is True
    assert check_user_access(user, "kwami_user123") is True
    
    # Cannot access others
    assert check_user_access(user, "user456") is False
    assert check_user_access(user, "kwami_user456") is False


@pytest.mark.asyncio
async def test_verify_token_no_jwks():
    """Test verification fails if JWKS not configured."""
    original_url = settings.supabase_url
    settings.supabase_url = None

    try:
        with pytest.raises(jwt.InvalidTokenError, match="JWKS not configured"):
            await verify_token("fake-token")
    finally:
        settings.supabase_url = original_url
