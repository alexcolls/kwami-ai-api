"""Token generation endpoints."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from livekit import api
from pydantic import BaseModel, ConfigDict, Field

from token_utils import create_token, DEFAULT_AGENT_NAME
from config import settings
from auth import require_auth, AuthUser

logger = logging.getLogger("kwami-api.token")
router = APIRouter()


async def dispatch_agent_to_room(room_name: str, agent_name: str = DEFAULT_AGENT_NAME):
    """Dispatch an agent to a room (called in background after token is generated).
    
    Checks if an agent is already in the room before dispatching to prevent duplicates.
    """
    lk_api = None
    try:
        # Convert wss:// to https:// for the API endpoint if needed
        api_url = settings.livekit_url.replace("wss://", "https://").replace("ws://", "http://")
        
        lk_api = api.LiveKitAPI(
            api_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        
        # Check if agent is already in the room
        try:
            participants = await lk_api.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            # Check if any participant is an agent (agents typically have "agent" in metadata or specific identity pattern)
            agent_present = any(
                p.kind == api.ParticipantInfo.Kind.AGENT or 
                (p.identity and "agent" in p.identity.lower())
                for p in participants.participants
            )
            if agent_present:
                logger.info(f"🤖 Agent already in room '{room_name}', skipping dispatch")
                return
        except Exception as e:
            # Room might not exist yet, continue with dispatch
            logger.debug(f"Could not check room participants: {e}")
        
        # Dispatch the agent
        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=agent_name,
            )
        )
        dispatch_id = getattr(dispatch, 'dispatch_id', None) or getattr(dispatch, 'id', 'unknown')
        logger.info(f"🤖 Agent '{agent_name}' dispatched to room '{room_name}' (dispatch_id: {dispatch_id})")
        
    except Exception as e:
        logger.error(f"Failed to dispatch agent to room '{room_name}': {type(e).__name__}: {e}")
    finally:
        if lk_api:
            await lk_api.aclose()


class TokenRequest(BaseModel):
    """Request body for token generation."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(
        ..., min_length=1, max_length=128, alias="roomName", description="Room name to join"
    )
    participant_name: str | None = Field(
        None, min_length=1, max_length=128, alias="participantName", description="Display name for participant"
    )
    participant_identity: str | None = Field(
        None, max_length=128, alias="participantIdentity", description="Unique identity (defaults to participant_name)"
    )

    # Permissions
    can_publish: bool = Field(True, alias="canPublish", description="Allow publishing audio/video tracks")
    can_subscribe: bool = Field(True, alias="canSubscribe", description="Allow subscribing to tracks")
    can_publish_data: bool = Field(True, alias="canPublishData", description="Allow publishing data messages")

    # Kwami-specific metadata
    kwami_id: str | None = Field(None, alias="kwamiId", description="Kwami instance ID for agent matching")


class TokenResponse(BaseModel):
    """Response containing the generated token."""

    token: str = Field(..., description="JWT access token")
    room_name: str = Field(..., description="Room name")
    participant_identity: str = Field(..., description="Participant identity")
    livekit_url: str = Field(..., description="LiveKit server URL to connect to")


@router.post("", response_model=TokenResponse)
async def generate_token(
    request: TokenRequest,
    background_tasks: BackgroundTasks,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    """
    Generate a LiveKit access token for a participant.

    This endpoint creates a JWT token that allows a client to connect
    to a LiveKit room with the specified permissions. It also dispatches
    the Kwami agent to the room.
    
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
        
        # Dispatch agent to room in background (ensures agent joins even for existing rooms)
        background_tasks.add_task(dispatch_agent_to_room, request.room_name)

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
    background_tasks: BackgroundTasks,
    user: Annotated[AuthUser, Depends(require_auth)],
    room_name: Annotated[str, Query(min_length=1, max_length=128, description="Room name")],
    participant_name: Annotated[str, Query(min_length=1, max_length=128, description="Participant name")],
    participant_identity: Annotated[str | None, Query(max_length=128)] = None,
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
    return await generate_token(request, background_tasks, user)
