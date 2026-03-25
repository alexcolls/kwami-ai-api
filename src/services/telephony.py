"""LiveKit SIP helpers for outbound calling."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import HTTPException
from livekit import api as livekit_api
from livekit.api.sip_service import (
    CreateSIPParticipantRequest,
    ListSIPInboundTrunkRequest,
    ListSIPOutboundTrunkRequest,
)

from src.core.config import settings

logger = logging.getLogger("kwami-api.telephony")


def build_call_room_name(kwami_id: str) -> str:
    return f"kwami-call-{kwami_id[:8]}-{uuid4().hex[:10]}"


def _response_items(response: object) -> list[object]:
    for field_name in ("items", "trunks"):
        value = getattr(response, field_name, None)
        if value:
            return list(value)
    return []


async def _find_outbound_trunk(lkapi: livekit_api.LiveKitAPI, trunk_id: str) -> object | None:
    response = await lkapi.sip.list_outbound_trunk(ListSIPOutboundTrunkRequest())
    for trunk in _response_items(response):
        if getattr(trunk, "sip_trunk_id", None) == trunk_id:
            return trunk
    return None


async def _find_inbound_trunk(lkapi: livekit_api.LiveKitAPI, trunk_id: str) -> object | None:
    response = await lkapi.sip.list_inbound_trunk(ListSIPInboundTrunkRequest())
    for trunk in _response_items(response):
        if getattr(trunk, "sip_trunk_id", None) == trunk_id:
            return trunk
    return None


def _merged_numbers(existing: object, phone_number: str) -> list[str]:
    current = list(getattr(existing, "numbers", None) or [])
    merged = list(dict.fromkeys([*current, phone_number]))
    return merged


async def sync_shared_livekit_trunks(phone_number: str) -> dict[str, object]:
    """Ensure a purchased number is usable on shared LiveKit SIP infrastructure."""
    sync: dict[str, object] = {
        "outbound": {"configured": bool(settings.livekit_sip_outbound_trunk_id), "synced": False},
        "inbound": {"configured": bool(settings.livekit_sip_inbound_trunk_id), "synced": False},
        "strategy": "shared_trunks",
        "notes": [],
    }

    async with livekit_api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    ) as lkapi:
        if settings.livekit_sip_outbound_trunk_id:
            outbound = await _find_outbound_trunk(lkapi, settings.livekit_sip_outbound_trunk_id)
            if not outbound:
                raise HTTPException(status_code=503, detail="Configured LiveKit outbound SIP trunk was not found")
            outbound_numbers = _merged_numbers(outbound, phone_number)
            await lkapi.sip.update_outbound_trunk_fields(
                settings.livekit_sip_outbound_trunk_id,
                numbers=outbound_numbers,
            )
            sync["outbound"] = {
                "configured": True,
                "synced": True,
                "numberCount": len(outbound_numbers),
            }
        else:
            sync["notes"].append(
                "No shared LiveKit outbound trunk is configured yet, so this number cannot be used as caller ID for outbound calls."
            )

        if settings.livekit_sip_inbound_trunk_id:
            inbound = await _find_inbound_trunk(lkapi, settings.livekit_sip_inbound_trunk_id)
            if not inbound:
                raise HTTPException(status_code=503, detail="Configured LiveKit inbound SIP trunk was not found")
            inbound_numbers = _merged_numbers(inbound, phone_number)
            await lkapi.sip.update_inbound_trunk_fields(
                settings.livekit_sip_inbound_trunk_id,
                numbers=inbound_numbers,
            )
            sync["inbound"] = {
                "configured": True,
                "synced": True,
                "numberCount": len(inbound_numbers),
            }
        else:
            sync["inbound"] = {
                "configured": False,
                "synced": True,
                "strategy": "shared_voice_webhook",
            }
            sync["notes"].append(
                "Inbound routing uses the shared Twilio voice webhook, so the LiveKit inbound trunk does not need an explicit per-number allowlist."
            )

    return sync


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
