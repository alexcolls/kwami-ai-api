"""LiveKit SIP helpers for outbound calling."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import HTTPException
from livekit import api as livekit_api
from livekit.protocol.sip import CreateSIPParticipantRequest

from src.core.config import settings

logger = logging.getLogger("kwami-api.telephony")


def build_call_room_name(kwami_id: str) -> str:
    return f"kwami-call-{kwami_id[:8]}-{uuid4().hex[:10]}"


async def create_outbound_call(
    *,
    kwami_id: str,
    phone_number: str,
    caller_id: str,
    participant_name: str,
    wait_until_answered: bool = False,
) -> dict[str, str]:
    if not settings.livekit_sip_outbound_trunk_id:
        raise HTTPException(status_code=503, detail="LiveKit SIP outbound trunk is not configured")

    room_name = build_call_room_name(kwami_id)
    participant_identity = f"sip_{uuid4().hex[:12]}"
    participant_metadata = f'{{"kwami_id":"{kwami_id}","channel":"voice_phone"}}'
    participant_attributes = {
        settings.livekit_sip_participant_attribute_key: kwami_id,
        "channel": "voice_phone",
    }

    request = CreateSIPParticipantRequest(
        sip_trunk_id=settings.livekit_sip_outbound_trunk_id,
        sip_call_to=phone_number,
        sip_number=caller_id,
        room_name=room_name,
        participant_identity=participant_identity,
        participant_name=participant_name,
        participant_metadata=participant_metadata,
        participant_attributes=participant_attributes,
        wait_until_answered=wait_until_answered,
        play_dialtone=True,
    )

    async with livekit_api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    ) as lkapi:
        participant = await lkapi.sip.create_sip_participant(request)

    logger.info(
        "Created outbound SIP participant for kwami=%s room=%s to=%s",
        kwami_id,
        room_name,
        phone_number,
    )
    return {
        "room_name": room_name,
        "participant_identity": participant.participant_identity,
        "provider_call_sid": participant.sip_call_id,
    }
