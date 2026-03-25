"""Per-kwami telephony and messaging APIs."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.deps import require_auth
from src.core.config import settings
from src.core.security import AuthUser
from src.services.channels import (
    build_agent_bootstrap_payload,
    create_call_event,
    create_message_event,
    ensure_contact,
    ensure_conversation,
    get_channel,
    get_channel_by_kind,
    get_owned_kwami,
    list_channels_for_kwami,
    normalize_phone_number,
    recent_events_for_kwami,
    update_channel,
    upsert_channel,
)
from src.services.telephony import create_outbound_call, sync_shared_livekit_trunks
from src.services.twilio_service import (
    attach_phone_number_to_sip_trunk,
    extract_twilio_error,
    purchase_phone_number,
    search_available_numbers,
    send_whatsapp_message,
)

router = APIRouter()
logger = logging.getLogger("kwami-api.channels")


class NumberSearchResponse(BaseModel):
    results: list[dict[str, Any]]


class PhonePurchaseRequest(BaseModel):
    kwami_id: str = Field(alias="kwamiId")
    phone_number: str = Field(alias="phoneNumber")
    display_name: str | None = Field(default=None, alias="displayName")
    country_code: str = Field(default="US", alias="countryCode")


class ChannelUpdateRequest(BaseModel):
    channel_id: str = Field(alias="channelId")
    status: str | None = None
    provider_sender: str | None = Field(default=None, alias="providerSender")
    metadata: dict[str, Any] | None = None


class OutboundCallRequest(BaseModel):
    kwami_id: str = Field(alias="kwamiId")
    to_number: str = Field(alias="toNumber")
    channel_id: str | None = Field(default=None, alias="channelId")
    wait_until_answered: bool = Field(default=False, alias="waitUntilAnswered")


class OutboundMessageRequest(BaseModel):
    kwami_id: str = Field(alias="kwamiId")
    to_number: str = Field(alias="toNumber")
    body: str
    channel_id: str | None = Field(default=None, alias="channelId")


@router.get("/kwamis/{kwami_id}")
async def get_kwami_channels(
    kwami_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    kwami = get_owned_kwami(user.id, kwami_id)
    return {
        "kwami": {
            "id": kwami["id"],
            "name": kwami.get("name") or "Kwami",
            "runtimeConfig": build_agent_bootstrap_payload(kwami),
        },
        "channels": list_channels_for_kwami(user.id, kwami_id),
        "events": recent_events_for_kwami(user.id, kwami_id),
    }


@router.get("/phone/search", response_model=NumberSearchResponse)
async def search_phone_numbers(
    user: Annotated[AuthUser, Depends(require_auth)],
    kwami_id: Annotated[str, Query(alias="kwamiId")],
    country_code: Annotated[str, Query(alias="countryCode")] = settings.twilio_phone_country,
    area_code: Annotated[str | None, Query(alias="areaCode")] = None,
    contains: Annotated[str | None, Query(alias="contains")] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
):
    get_owned_kwami(user.id, kwami_id)
    return NumberSearchResponse(
        results=search_available_numbers(
            country_code=country_code,
            area_code=area_code,
            contains=contains,
            limit=limit,
        )
    )


@router.post("/phone/purchase")
async def purchase_kwami_phone_number(
    request: PhonePurchaseRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    kwami = get_owned_kwami(user.id, request.kwami_id)
    phone_number = normalize_phone_number(request.phone_number, request.country_code)

    purchase = purchase_phone_number(
        phone_number=phone_number,
        friendly_name=request.display_name or f"{kwami.get('name') or 'Kwami'} Line",
    )

    trunk_phone_sid = None
    shared_infra_sync: dict[str, Any] | None = None
    voice_status = "active"
    voice_outbound_ready = bool(settings.livekit_sip_outbound_trunk_id)
    if purchase.get("sid"):
        try:
            trunk_phone_sid = attach_phone_number_to_sip_trunk(str(purchase["sid"]))
        except Exception as exc:  # pragma: no cover - provider failure
            logger.warning("Failed to attach number to Twilio SIP trunk: %s", exc)
        try:
            shared_infra_sync = await sync_shared_livekit_trunks(phone_number)
            outbound_state = shared_infra_sync.get("outbound") if isinstance(shared_infra_sync, dict) else {}
            voice_outbound_ready = bool(
                isinstance(outbound_state, dict) and outbound_state.get("synced")
            )
            if not voice_outbound_ready and settings.livekit_sip_outbound_trunk_id:
                voice_status = "routing_pending"
        except Exception as exc:  # pragma: no cover - provider failure
            voice_status = "routing_pending"
            voice_outbound_ready = False
            shared_infra_sync = {"error": str(exc), "strategy": "shared_trunks"}
            logger.warning("Failed to sync purchased number to shared LiveKit trunks: %s", exc)

    voice_channel = upsert_channel(
        user_id=user.id,
        kwami_id=request.kwami_id,
        kind="voice_phone",
        phone_number=phone_number,
        display_name=request.display_name or kwami.get("name"),
        country_code=request.country_code,
        status=voice_status,
        capabilities={"voice": True, "outbound": voice_outbound_ready},
        metadata={
            "twilioSipTrunkPhoneSid": trunk_phone_sid,
            "sharedInfrastructure": shared_infra_sync,
        },
        provider_channel_sid=str(purchase.get("sid") or ""),
        livekit_outbound_trunk_id=settings.livekit_sip_outbound_trunk_id,
    )
    whatsapp_channel = upsert_channel(
        user_id=user.id,
        kwami_id=request.kwami_id,
        kind="whatsapp",
        phone_number=phone_number,
        display_name=request.display_name or kwami.get("name"),
        country_code=request.country_code,
        status="pending_setup",
        capabilities={"whatsapp": True, "requiresApproval": True},
        metadata={
            "note": "Enable this sender in Twilio/Meta before using production WhatsApp.",
            "sharedInfrastructure": {
                "phoneProvisionedByPlatform": True,
                "routingStrategy": "provider_webhooks",
            },
        },
        provider_channel_sid=str(purchase.get("sid") or ""),
        provider_sender=f"whatsapp:{phone_number}",
        livekit_outbound_trunk_id=settings.livekit_sip_outbound_trunk_id,
    )
    return {
        "voiceChannel": voice_channel,
        "whatsappChannel": whatsapp_channel,
        "sharedInfrastructure": shared_infra_sync,
    }


@router.post("/whatsapp/configure")
async def configure_whatsapp_channel(
    request: ChannelUpdateRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    channel = get_channel(user.id, request.channel_id)
    if channel["kind"] != "whatsapp":
        raise HTTPException(status_code=400, detail="Channel is not a WhatsApp sender")
    updated = update_channel(
        channel["id"],
        user_id=user.id,
        updates={
            "status": request.status or channel["status"],
            "provider_sender": request.provider_sender or channel.get("provider_sender"),
            "metadata": {**(channel.get("metadata") or {}), **(request.metadata or {})},
        },
    )
    return {"channel": updated}


@router.post("/calls/outbound")
async def start_outbound_call(
    request: OutboundCallRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    kwami = get_owned_kwami(user.id, request.kwami_id)
    to_number = normalize_phone_number(request.to_number, settings.twilio_phone_country)
    channel = (
        get_channel(user.id, request.channel_id)
        if request.channel_id
        else get_channel_by_kind(user.id, request.kwami_id, "voice_phone")
    )
    if not channel:
        raise HTTPException(status_code=404, detail="No phone channel configured for this kwami")

    contact = ensure_contact(
        user_id=user.id,
        kwami_id=request.kwami_id,
        phone_number=to_number,
    )
    conversation = ensure_conversation(
        user_id=user.id,
        kwami_id=request.kwami_id,
        channel_id=channel["id"],
        kind="call",
        contact_id=contact["id"],
        metadata={"channelKind": "voice_phone"},
    )

    try:
        outbound = await create_outbound_call(
            kwami_id=request.kwami_id,
            phone_number=to_number,
            caller_id=channel["phone_number"],
            participant_name=kwami.get("name") or "Kwami",
            wait_until_answered=request.wait_until_answered,
        )
        event = create_call_event(
            conversation_id=conversation["id"],
            channel_id=channel["id"],
            user_id=user.id,
            kwami_id=request.kwami_id,
            direction="outbound",
            provider_call_sid=outbound["provider_call_sid"],
            livekit_room_name=outbound["room_name"],
            participant_identity=outbound["participant_identity"],
            from_number=channel["phone_number"],
            to_number=to_number,
            status="queued",
            provider_payload={"waitUntilAnswered": request.wait_until_answered},
        )
    except Exception as exc:
        error_code, error_message = extract_twilio_error(exc)
        create_call_event(
            conversation_id=conversation["id"],
            channel_id=channel["id"],
            user_id=user.id,
            kwami_id=request.kwami_id,
            direction="outbound",
            provider_call_sid=None,
            livekit_room_name=None,
            participant_identity=None,
            from_number=channel["phone_number"],
            to_number=to_number,
            status="failed",
            error_code=error_code,
            error_message=error_message,
            provider_payload={"error": str(exc)},
        )
        raise

    return {"call": event, "conversationId": conversation["id"]}


@router.post("/messages/outbound")
async def send_outbound_message(
    request: OutboundMessageRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    if not request.body.strip():
        raise HTTPException(status_code=400, detail="Message body is required")

    get_owned_kwami(user.id, request.kwami_id)
    to_number = normalize_phone_number(request.to_number, settings.twilio_phone_country)
    channel = (
        get_channel(user.id, request.channel_id)
        if request.channel_id
        else get_channel_by_kind(user.id, request.kwami_id, "whatsapp")
    )
    if not channel:
        raise HTTPException(status_code=404, detail="No WhatsApp channel configured for this kwami")

    from_address = channel.get("provider_sender") or settings.twilio_whatsapp_from
    if not from_address:
        raise HTTPException(status_code=503, detail="No WhatsApp sender is configured")

    if channel["status"] not in {"active", "sandbox"}:
        raise HTTPException(
            status_code=409,
            detail="WhatsApp sender is not ready. Complete Twilio/Meta setup first.",
        )

    contact = ensure_contact(
        user_id=user.id,
        kwami_id=request.kwami_id,
        phone_number=to_number,
        whatsapp_address=f"whatsapp:{to_number}",
    )
    conversation = ensure_conversation(
        user_id=user.id,
        kwami_id=request.kwami_id,
        channel_id=channel["id"],
        kind="whatsapp",
        contact_id=contact["id"],
        metadata={"channelKind": "whatsapp"},
    )

    message = send_whatsapp_message(
        from_address=from_address,
        to_address=f"whatsapp:{to_number}",
        body=request.body.strip(),
    )
    event = create_message_event(
        conversation_id=conversation["id"],
        channel_id=channel["id"],
        contact_id=contact["id"],
        user_id=user.id,
        kwami_id=request.kwami_id,
        direction="outbound",
        provider_message_sid=message["sid"],
        provider_status=message["status"],
        from_address=message["from"],
        to_address=message["to"],
        body=request.body.strip(),
    )
    return {"message": event, "conversationId": conversation["id"]}
