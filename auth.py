"""Authentication middleware for Supabase JWT validation."""

import logging
from typing import Annotated

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

logger = logging.getLogger("kwami-api.auth")

# Optional auth - auto_error=False allows unauthenticated requests
security = HTTPBearer(auto_error=False)

# JWKS client for asymmetric key verification (cached)
_jwks_client: PyJWKClient | None = None


def get_jwks_client() -> PyJWKClient | None:
    """Get or create JWKS client for Supabase."""
    global _jwks_client
    if _jwks_client is None and settings.supabase_jwks_url:
        _jwks_client = PyJWKClient(settings.supabase_jwks_url, cache_keys=True)
        logger.info(f"🔑 JWKS client initialized: {settings.supabase_jwks_url}")
    return _jwks_client


class AuthUser:
    """Authenticated user information extracted from JWT."""

    def __init__(self, payload: dict):
        self.id: str = payload.get("sub", "")
        self.email: str | None = payload.get("email")
        self.role: str = payload.get("role", "authenticated")
        self.aud: str = payload.get("aud", "")
        self.raw_payload: dict = payload

    def __repr__(self) -> str:
        return f"AuthUser(id={self.id}, email={self.email})"


async def _verify_with_jwks(token: str, alg: str) -> dict:
    """Verify JWT using JWKS (for asymmetric algorithms)."""
    jwks_client = get_jwks_client()
    if not jwks_client:
        raise jwt.InvalidTokenError(
            "JWKS not configured. Set SUPABASE_URL for asymmetric key verification."
        )
    
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=[alg],
        audience="authenticated",
    )


def _verify_with_secret(token: str) -> dict:
    """Verify JWT using shared secret (for HS256)."""
    if not settings.supabase_jwt_secret:
        raise jwt.InvalidTokenError(
            "JWT secret not configured. Set SUPABASE_JWT_SECRET for HS256 verification."
        )
    
    return jwt.decode(
        token,
        settings.supabase_jwt_secret,
        algorithms=["HS256"],
        audience="authenticated",
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]
) -> AuthUser | None:
    """
    Decode and validate Supabase JWT, return user info.
    
    Supports both:
    - Asymmetric keys (RS256, ES256) via JWKS endpoint
    - Shared secret (HS256) via JWT secret
    
    Returns None if:
    - No credentials provided
    - Auth is not configured
    
    Raises HTTPException 401 if:
    - Token is invalid or expired
    """
    # If no credentials provided, return None (anonymous)
    if not credentials:
        return None

    # If auth is not configured, allow anonymous access
    if not settings.auth_enabled:
        logger.debug("Auth not configured, allowing anonymous access")
        return None

    try:
        # Get token algorithm from header
        try:
            header = jwt.get_unverified_header(credentials.credentials)
            alg = header.get("alg", "")
            logger.info(f"🔐 Token header - alg: {alg}, typ: {header.get('typ')}")
        except Exception as e:
            logger.warning(f"Could not read token header: {e}")
            alg = ""
        
        # Choose verification method based on algorithm
        if alg in ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512"):
            # Asymmetric algorithm - use JWKS
            payload = await _verify_with_jwks(credentials.credentials, alg)
        else:
            # Symmetric algorithm (HS256) - use JWT secret
            payload = _verify_with_secret(credentials.credentials)
        
        user = AuthUser(payload)
        logger.debug(f"Authenticated user: {user}")
        return user

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        logger.warning("Invalid token audience")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_auth(
    user: Annotated[AuthUser | None, Depends(get_current_user)]
) -> AuthUser:
    """
    Dependency that requires authentication.
    
    Use this for endpoints that must have a valid authenticated user.
    Raises 401 if user is not authenticated.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


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
