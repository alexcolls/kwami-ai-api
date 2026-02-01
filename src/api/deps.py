"""API dependencies."""

import logging
from typing import Annotated, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import settings
from src.core.security import (
    AuthUser,
    verify_with_jwks,
    verify_with_secret,
)

logger = logging.getLogger("kwami-api.deps")

# Optional auth - auto_error=False allows unauthenticated requests
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security_scheme)]
) -> Optional[AuthUser]:
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
            payload = await verify_with_jwks(credentials.credentials, alg)
        else:
            # Symmetric algorithm (HS256) - use JWT secret
            payload = verify_with_secret(credentials.credentials)
        
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
    user: Annotated[Optional[AuthUser], Depends(get_current_user)]
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
