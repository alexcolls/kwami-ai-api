"""Security and authentication logic."""

import logging
from typing import Optional

import jwt
from jwt import PyJWKClient

from src.core.config import settings

logger = logging.getLogger("kwami-api.security")

# JWKS client for asymmetric key verification (cached)
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> Optional[PyJWKClient]:
    """Get or create JWKS client for Supabase."""
    global _jwks_client
    if _jwks_client is None and settings.supabase_jwks_url:
        _jwks_client = PyJWKClient(settings.supabase_jwks_url, cache_keys=True)
        logger.info(f"JWKS client initialized: {settings.supabase_jwks_url}")
    return _jwks_client


class AuthUser:
    """Authenticated user information extracted from JWT."""

    def __init__(self, payload: dict):
        self.id: str = payload.get("sub", "")
        self.email: Optional[str] = payload.get("email")
        self.role: str = payload.get("role", "authenticated")
        self.aud: str = payload.get("aud", "")
        self.raw_payload: dict = payload

    def __repr__(self) -> str:
        return f"AuthUser(id={self.id}, email={self.email})"


async def verify_token(token: str) -> dict:
    """Verify a Supabase JWT using JWKS (asymmetric key verification).

    Args:
        token: The JWT string from the Authorization header.

    Returns:
        Decoded JWT payload.

    Raises:
        jwt.InvalidTokenError: If verification fails.
    """
    jwks_client = get_jwks_client()
    if not jwks_client:
        raise jwt.InvalidTokenError(
            "JWKS not configured. Set SUPABASE_URL to enable authentication."
        )

    # Read algorithm from token header
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "RS256")
    except Exception:
        alg = "RS256"

    signing_key = jwks_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=[alg],
        audience="authenticated",
    )


def check_user_access(user: AuthUser, user_id: str) -> bool:
    """
    Check if the authenticated user has access to the requested user_id.
    
    Returns True if:
    - User is authenticated and the user_id matches their ID
    
    Returns False if:
    - User ID doesn't match the authenticated user
    """
    # User can only access their own data
    # Note: user_id in the request may have a "kwami_" prefix
    clean_user_id = user_id.replace("kwami_", "")
    return user.id == clean_user_id or user.id == user_id
