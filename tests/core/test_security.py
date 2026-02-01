import pytest
import jwt
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from src.core.security import check_user_access, AuthUser, verify_with_secret
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

@patch("jwt.decode")
def test_verify_with_secret(mock_decode):
    """Test symmetric token verification."""
    # Temporarily set secret
    original_secret = settings.supabase_jwt_secret
    settings.supabase_jwt_secret = "test-secret"
    
    try:
        mock_decode.return_value = {"sub": "123"}
        
        result = verify_with_secret("fake-token")
        
        assert result["sub"] == "123"
        mock_decode.assert_called_with(
            "fake-token",
            "test-secret",
            algorithms=["HS256"],
            audience="authenticated"
        )
    finally:
        settings.supabase_jwt_secret = original_secret

def test_verify_with_secret_no_config():
    """Test verification fails if secret not configured."""
    original_secret = settings.supabase_jwt_secret
    settings.supabase_jwt_secret = None
    
    try:
        with pytest.raises(jwt.InvalidTokenError):
            verify_with_secret("token")
    finally:
        settings.supabase_jwt_secret = original_secret
