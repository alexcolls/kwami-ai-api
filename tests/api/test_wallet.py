import hashlib
import hmac
import json

import pytest

from src.core.config import settings
from src.services import wallet_service


@pytest.mark.anyio
async def test_create_wallet_endpoint(auth_client, monkeypatch):
    async def fake_create_kwami_wallet(user_id: str, kwami_id: str):
        return {
            "id": "wallet-1",
            "user_id": user_id,
            "kwami_id": kwami_id,
            "public_key": "mock_pubkey",
            "status": "active",
        }

    monkeypatch.setattr("src.api.routes.wallet.create_kwami_wallet", fake_create_kwami_wallet)

    response = await auth_client.post("/wallets/kwamis/kwami-1")
    assert response.status_code == 200
    assert response.json()["wallet"]["id"] == "wallet-1"


@pytest.mark.anyio
async def test_get_wallet_overview_endpoint(auth_client, monkeypatch):
    async def fake_get_overview(user_id: str, kwami_id: str):
        return {
            "wallet": {
                "id": "wallet-1",
                "kwami_id": kwami_id,
                "public_key": "mock_pubkey",
                "status": "active",
                "custody_type": "custodial_hsm_mpc",
                "network": "mainnet-beta",
            },
            "balances": [],
            "allowlist": [],
            "transactions": [],
            "funding_intents": [],
        }

    monkeypatch.setattr("src.api.routes.wallet.get_kwami_wallet_overview", fake_get_overview)

    response = await auth_client.get("/wallets/kwamis/kwami-1")
    assert response.status_code == 200
    assert response.json()["wallet"]["kwami_id"] == "kwami-1"


@pytest.mark.anyio
async def test_create_funding_intent_passes_idempotency(auth_client, monkeypatch):
    captured = {}

    async def fake_create_funding_intent(
        user_id: str,
        *,
        kwami_id: str,
        provider: str,
        asset_mint: str,
        asset_symbol: str,
        amount,
        amount_usd,
        sender_wallet_pubkey,
        idempotency_key=None,
    ):
        captured["idempotency_key"] = idempotency_key
        return {
            "id": "intent-1",
            "provider": provider,
            "status": "pending",
            "asset_symbol": asset_symbol,
            "expected_amount": str(amount),
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr("src.api.routes.wallet.create_funding_intent", fake_create_funding_intent)

    response = await auth_client.post(
        "/wallets/kwamis/kwami-1/fund/phantom-intent",
        json={
            "assetMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "assetSymbol": "USDC",
            "amount": 10,
            "idempotencyKey": "idem-123",
        },
    )
    assert response.status_code == 200
    assert captured["idempotency_key"] == "idem-123"


def test_wallet_webhook_signature_verification():
    payload = json.dumps({"intent_id": "intent-1"}).encode("utf-8")
    secret = "wallet-secret-test"
    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    old_secret = settings.wallet_webhook_secret
    settings.wallet_webhook_secret = secret
    try:
        assert wallet_service.verify_wallet_webhook_signature(payload, signature) is True
        assert wallet_service.verify_wallet_webhook_signature(payload, "invalid") is False
    finally:
        settings.wallet_webhook_secret = old_secret
