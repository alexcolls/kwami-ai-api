"""Twilio provisioning, webhook validation, and messaging helpers."""

from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import HTTPException, Request
from twilio.base.exceptions import TwilioRestException
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import Dial, VoiceResponse

from src.core.config import settings

logger = logging.getLogger("kwami-api.twilio")

_twilio_client: Client | None = None


def get_twilio_client() -> Client:
    global _twilio_client
    if _twilio_client is None:
        if not settings.twilio_enabled:
            raise RuntimeError("Platform phone provisioning is not configured")
        _twilio_client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    return _twilio_client


def ensure_twilio_enabled() -> None:
    if not settings.twilio_enabled:
        raise HTTPException(
            status_code=503,
            detail="Platform phone provisioning is not configured on the server",
        )


def webhook_url(path: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    if not settings.app_public_url:
        return None
    return f"{settings.app_public_url.rstrip('/')}{path}"


async def validate_twilio_request(request: Request, form_payload: dict[str, str]) -> None:
    """Validate webhook signatures when the auth token is configured."""
    if not settings.twilio_auth_token:
        return

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing Twilio signature")

    validator = RequestValidator(settings.twilio_auth_token)
    url = str(request.url)
    if not validator.validate(url, form_payload, signature):
        raise HTTPException(status_code=401, detail="Invalid Twilio signature")


def search_available_numbers(
    *,
    country_code: str,
    area_code: str | None,
    contains: str | None,
    limit: int,
) -> list[dict[str, str]]:
    ensure_twilio_enabled()
    client = get_twilio_client()
    search = client.available_phone_numbers(country_code.upper()).local
    items = search.list(area_code=area_code, contains=contains, limit=limit)
    return [
        {
            "phoneNumber": item.phone_number,
            "friendlyName": item.friendly_name,
            "region": item.region,
            "locality": item.locality,
            "postalCode": item.postal_code,
            "capabilities": {
                "voice": bool(getattr(item, "capabilities", {}).get("voice", False)),
                "sms": bool(getattr(item, "capabilities", {}).get("SMS", False))
                or bool(getattr(item, "capabilities", {}).get("sms", False)),
                "mms": bool(getattr(item, "capabilities", {}).get("MMS", False))
                or bool(getattr(item, "capabilities", {}).get("mms", False)),
            },
        }
        for item in items
    ]


def purchase_phone_number(
    *,
    phone_number: str,
    friendly_name: str,
) -> dict[str, str | None]:
    ensure_twilio_enabled()
    client = get_twilio_client()
    incoming = client.incoming_phone_numbers.create(
        phone_number=phone_number,
        friendly_name=friendly_name,
        voice_url=webhook_url("/webhooks/twilio/voice"),
        voice_method="POST",
        status_callback=webhook_url(
            "/webhooks/twilio/voice/status",
            explicit=settings.twilio_voice_status_callback_url,
        ),
        status_callback_method="POST",
        sms_url=webhook_url("/webhooks/twilio/whatsapp"),
        sms_method="POST",
        sms_status_callback=webhook_url(
            "/webhooks/twilio/whatsapp/status",
            explicit=settings.twilio_messaging_status_callback_url,
        ),
    )
    return {
        "sid": incoming.sid,
        "phone_number": incoming.phone_number,
        "friendly_name": incoming.friendly_name,
        "status": getattr(incoming, "status", None),
    }


def attach_phone_number_to_sip_trunk(phone_number_sid: str) -> str | None:
    """Attach a purchased number to the shared Twilio SIP trunk."""
    if not settings.twilio_sip_trunk_sid:
        return None

    ensure_twilio_enabled()
    client = get_twilio_client()
    resource = client.trunking.v1.trunks(settings.twilio_sip_trunk_sid).phone_numbers.create(
        phone_number_sid=phone_number_sid
    )
    return resource.sid


def send_whatsapp_message(
    *,
    from_address: str,
    to_address: str,
    body: str,
) -> dict[str, str | None]:
    ensure_twilio_enabled()
    client = get_twilio_client()
    message = client.messages.create(
        from_=from_address,
        to=to_address,
        body=body,
        status_callback=webhook_url(
            "/webhooks/twilio/whatsapp/status",
            explicit=settings.twilio_messaging_status_callback_url,
        ),
    )
    return {
        "sid": message.sid,
        "status": message.status,
        "from": message.from_,
        "to": message.to,
    }


def build_voice_bridge_response(*, livekit_uri: str, headers: dict[str, str]) -> str:
    """Return TwiML that bridges the PSTN call into LiveKit SIP."""
    response = VoiceResponse()
    dial = Dial(answer_on_bridge=True)
    dial.sip(_append_sip_headers(livekit_uri, headers))
    response.append(dial)
    return str(response)


def build_message_ack_response(text: str) -> str:
    response = MessagingResponse()
    response.message(text)
    return str(response)


def _append_sip_headers(uri: str, headers: dict[str, str]) -> str:
    """Twilio accepts SIP custom headers as URI query parameters."""
    parsed = urlparse(uri)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(headers)
    return urlunparse(parsed._replace(query=urlencode(query)))


def extract_twilio_error(exc: Exception) -> tuple[str | None, str | None]:
    if isinstance(exc, TwilioRestException):
        return str(exc.code) if exc.code else None, exc.msg
    return None, str(exc)
