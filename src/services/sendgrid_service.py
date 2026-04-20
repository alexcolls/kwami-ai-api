"""SendGrid helpers for outbound email and inbound webhook verification."""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

import httpx

from src.core.config import settings

logger = logging.getLogger("kwami-api.sendgrid")

SENDGRID_SEND_URL = "https://api.sendgrid.com/v3/mail/send"


async def send_email(
    *,
    from_address: str,
    to_addresses: list[str],
    subject: str,
    body_text: str = "",
    body_html: str = "",
    cc_addresses: list[str] | None = None,
    reply_to: str | None = None,
) -> str | None:
    """Send an email via the SendGrid v3 Mail Send API.

    Returns the SendGrid ``X-Message-Id`` on success or ``None`` on failure.
    """
    if not settings.sendgrid_api_key:
        raise RuntimeError("SENDGRID_API_KEY is not configured")

    personalizations: dict[str, Any] = {
        "to": [{"email": addr} for addr in to_addresses],
    }
    if cc_addresses:
        personalizations["cc"] = [{"email": addr} for addr in cc_addresses]

    payload: dict[str, Any] = {
        "personalizations": [personalizations],
        "from": {"email": from_address},
        "subject": subject,
        "content": [],
    }

    if body_text:
        payload["content"].append({"type": "text/plain", "value": body_text})
    if body_html:
        payload["content"].append({"type": "text/html", "value": body_html})
    if not payload["content"]:
        payload["content"].append({"type": "text/plain", "value": ""})

    if reply_to:
        payload["reply_to"] = {"email": reply_to}

    headers = {
        "Authorization": f"Bearer {settings.sendgrid_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(SENDGRID_SEND_URL, json=payload, headers=headers)

    if resp.status_code not in (200, 201, 202):
        logger.error(
            "SendGrid send failed status=%s body=%s",
            resp.status_code,
            resp.text[:500],
        )
        return None

    message_id = resp.headers.get("X-Message-Id")
    logger.info("Email sent via SendGrid message_id=%s", message_id)
    return message_id


def verify_inbound_webhook(
    token: str,
    timestamp: str,
    signature: str,
) -> bool:
    """Verify a SendGrid Inbound Parse webhook signature.

    If no secret is configured the check is skipped (development mode).
    """
    secret = settings.sendgrid_inbound_webhook_secret
    if not secret:
        return True

    payload = timestamp + token
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
