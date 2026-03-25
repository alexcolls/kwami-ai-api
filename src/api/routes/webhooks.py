"""Twilio voice and WhatsApp webhooks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from src.core.config import settings
from src.services.channels import (
    create_call_event,
    create_message_event,
    ensure_contact,
    ensure_conversation,
    find_channel_by_address,
    get_owned_kwami,
    maybe_normalize_phone_number,
    update_call_event_status,
    update_message_event_status,
)
from src.services.twilio_service import (
    build_message_ack_response,
    build_voice_bridge_response,
    validate_twilio_request,
)

router = APIRouter()
logger = logging.getLogger("kwami-api.webhooks")


async def _form_payload(request: Request) -> dict[str, str]:
    form = await request.form()
    return {str(key): str(value) for key, value in form.multi_items()}


@router.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    payload = await _form_payload(request)
    await validate_twilio_request(request, payload)

    called = maybe_normalize_phone_number(
        payload.get("To") or payload.get("Called"),
        settings.twilio_phone_country,
    )
    caller = maybe_normalize_phone_number(
        payload.get("From") or payload.get("Caller"),
        settings.twilio_phone_country,
    )

    if not called:
        raise HTTPException(status_code=400, detail="Missing destination number")

    channel = find_channel_by_address(called)
    if not channel:
        logger.warning("Inbound voice call to unknown number: %s", called)
        return Response("<Response><Reject/></Response>", media_type="application/xml")

    contact = ensure_contact(
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        phone_number=caller or "unknown",
        display_name=payload.get("CallerName"),
    )
    conversation = ensure_conversation(
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        channel_id=channel["id"],
        kind="call",
        contact_id=contact["id"],
        external_thread_id=payload.get("CallSid"),
        metadata={"source": "twilio_voice"},
    )
    create_call_event(
        conversation_id=conversation["id"],
        channel_id=channel["id"],
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        direction="inbound",
        provider_call_sid=payload.get("CallSid"),
        livekit_room_name=None,
        participant_identity=None,
        from_number=caller,
        to_number=called,
        status=payload.get("CallStatus") or "ringing",
        provider_payload=payload,
    )

    if not settings.livekit_sip_inbound_uri:
        return Response(
            "<Response><Say>Voice is not configured for this number yet.</Say></Response>",
            media_type="application/xml",
        )

    xml = build_voice_bridge_response(
        livekit_uri=settings.livekit_sip_inbound_uri,
        headers={
            "X-Kwami-Id": channel["kwami_id"],
            "X-Kwami-Channel-Id": channel["id"],
            "X-Kwami-User-Id": channel["user_id"],
            "X-Contact-Number": caller or "",
        },
    )
    return Response(xml, media_type="application/xml")


@router.post("/twilio/voice/status")
async def twilio_voice_status_webhook(request: Request):
    payload = await _form_payload(request)
    await validate_twilio_request(request, payload)
    if payload.get("CallSid"):
        update_call_event_status(
            payload["CallSid"],
            status=payload.get("CallStatus") or "completed",
            duration_seconds=int(payload["CallDuration"]) if payload.get("CallDuration") else None,
            error_code=payload.get("ErrorCode"),
            error_message=payload.get("SipResponseCode") or payload.get("ErrorMessage"),
            provider_payload=payload,
        )
    return JSONResponse({"ok": True})


@router.post("/twilio/whatsapp")
async def twilio_whatsapp_webhook(request: Request):
    payload = await _form_payload(request)
    await validate_twilio_request(request, payload)

    to_address = payload.get("To") or ""
    from_address = payload.get("From") or ""
    body = payload.get("Body") or ""
    channel = find_channel_by_address(to_address)
    if not channel:
        logger.warning("Inbound WhatsApp message for unknown sender: %s", to_address)
        return Response(build_message_ack_response("This sender is not configured."), media_type="application/xml")

    contact_number = maybe_normalize_phone_number(from_address.replace("whatsapp:", ""), settings.twilio_phone_country)
    contact = ensure_contact(
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        phone_number=contact_number or from_address,
        display_name=payload.get("ProfileName"),
        whatsapp_address=from_address,
    )
    conversation = ensure_conversation(
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        channel_id=channel["id"],
        kind="whatsapp",
        contact_id=contact["id"],
        external_thread_id=from_address,
        metadata={"source": "twilio_whatsapp"},
    )
    create_message_event(
        conversation_id=conversation["id"],
        channel_id=channel["id"],
        contact_id=contact["id"],
        user_id=channel["user_id"],
        kwami_id=channel["kwami_id"],
        direction="inbound",
        provider_message_sid=payload.get("MessageSid"),
        provider_status=payload.get("SmsStatus") or "received",
        from_address=from_address,
        to_address=to_address,
        body=body,
        requires_followup=True,
        provider_payload=payload,
    )

    kwami = get_owned_kwami(channel["user_id"], channel["kwami_id"])
    ack_text = (
        f"{kwami.get('name') or 'Kwami'} received your message. "
        "WhatsApp is connected, and an automated reply pipeline can now be layered on top of this thread."
    )
    return Response(build_message_ack_response(ack_text), media_type="application/xml")


@router.post("/twilio/whatsapp/status")
async def twilio_whatsapp_status_webhook(request: Request):
    payload = await _form_payload(request)
    await validate_twilio_request(request, payload)
    if payload.get("MessageSid"):
        update_message_event_status(
            payload["MessageSid"],
            provider_status=payload.get("MessageStatus") or payload.get("SmsStatus") or "sent",
            error_code=payload.get("ErrorCode"),
            error_message=payload.get("ErrorMessage"),
            provider_payload=payload,
        )
    return JSONResponse({"ok": True})
