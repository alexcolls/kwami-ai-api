"""Token generation endpoints.

Note: Agent dispatching is handled by LiveKit Cloud's auto-dispatch feature.
Do NOT manually dispatch agents here to avoid duplicate agents in rooms.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.services.livekit import create_token
from src.core.config import settings
from src.api.deps import require_auth
from src.core.security import AuthUser

logger = logging.getLogger("kwami-api.token")
router = APIRouter()


class TokenRequest(BaseModel):
    """Request body for token generation."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(
        ..., min_length=1, max_length=128, alias="roomName", description="Room name to join"
    )
    participant_name: Optional[str] = Field(
        None, min_length=1, max_length=128, alias="participantName", description="Display name for participant"
    )
    participant_identity: Optional[str] = Field(
        None, max_length=128, alias="participantIdentity", description="Unique identity (defaults to participant_name)"
    )

    # Permissions
    can_publish: bool = Field(True, alias="canPublish", description="Allow publishing audio/video tracks")
    can_subscribe: bool = Field(True, alias="canSubscribe", description="Allow subscribing to tracks")
    can_publish_data: bool = Field(True, alias="canPublishData", description="Allow publishing data messages")

    # Kwami-specific metadata
    kwami_id: Optional[str] = Field(None, alias="kwamiId", description="Kwami instance ID for agent matching")


class TokenResponse(BaseModel):
    """Response containing the generated token."""

    token: str = Field(..., description="JWT access token")
    room_name: str = Field(..., description="Room name")
    participant_identity: str = Field(..., description="Participant identity")
    livekit_url: str = Field(..., description="LiveKit server URL to connect to")


@router.post("", response_model=TokenResponse)
async def generate_token(
    request: TokenRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    """
    Generate a LiveKit access token for a participant.

    This endpoint creates a JWT token that allows a client to connect
    to a LiveKit room with the specified permissions.
    
    Note: Agent dispatching is handled automatically by LiveKit Cloud's
    auto-dispatch feature when a user joins the room.
    
    Requires authentication - user's Supabase ID is used as participant identity.
    """
    # Use Supabase user ID as the identity (for memory persistence)
    identity = user.id
    participant_name = user.email or request.participant_name or identity
    logger.info(f"📥 Token request: room={request.room_name}, user={user.id}, email={user.email}")

    try:
        token = create_token(
            room_name=request.room_name,
            participant_name=participant_name,
            participant_identity=identity,
            can_publish=request.can_publish,
            can_subscribe=request.can_subscribe,
            can_publish_data=request.can_publish_data,
            kwami_id=request.kwami_id,
        )

        logger.info(f"🎫 Token generated for '{identity}' in room '{request.room_name}'")

        return TokenResponse(
            token=token,
            room_name=request.room_name,
            participant_identity=identity,
            livekit_url=settings.livekit_url,
        )

    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate token")


@router.get("", response_model=TokenResponse)
async def generate_token_get(
    user: Annotated[AuthUser, Depends(require_auth)],
    room_name: Annotated[str, Query(min_length=1, max_length=128, description="Room name")],
    participant_name: Annotated[str, Query(min_length=1, max_length=128, description="Participant name")],
    participant_identity: Annotated[Optional[str], Query(max_length=128)] = None,
):
    """
    Generate a LiveKit access token (GET method for simple integrations).

    For production use, prefer the POST endpoint with full options.
    Requires authentication.
    """
    request = TokenRequest(
        room_name=room_name,
        participant_name=participant_name,
        participant_identity=participant_identity,
    )
    return await generate_token(request, user)
