"""Credits system endpoints.

Provides endpoints for:
- Viewing credit balance and transaction history
- Purchasing credits via Stripe Checkout
- Stripe webhook for payment confirmation
- Agent usage reporting (internal API key auth)
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from src.api.deps import require_auth
from src.core.config import settings
from src.core.security import AuthUser
from src.services.credits import (
    CREDIT_PACKS,
    MICRO_CREDITS_PER_CREDIT,
    get_balance,
    get_transactions,
    get_usage_logs,
    process_usage_report,
)
from src.services.stripe_service import create_checkout_session, handle_webhook_event

logger = logging.getLogger("kwami-api.credits")
router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class CreditBalanceResponse(BaseModel):
    """User credit balance."""

    balance: int = Field(description="Current balance in micro-credits")
    balance_credits: float = Field(description="Current balance in display credits")
    lifetime_purchased: int = Field(description="Total purchased in micro-credits")
    lifetime_used: int = Field(description="Total used in micro-credits")


class CreditPackResponse(BaseModel):
    """A purchasable credit pack."""

    id: str
    name: str
    credits: int
    price_cents: int
    price_display: str
    popular: bool


class CreditPacksResponse(BaseModel):
    """All available credit packs."""

    packs: list[CreditPackResponse]


class PurchaseRequest(BaseModel):
    """Request to initiate a credit purchase."""

    pack_id: str = Field(..., description="Credit pack ID: starter, standard, or pro")
    success_url: str = Field(..., description="URL to redirect on successful payment")
    cancel_url: str = Field(..., description="URL to redirect on cancelled payment")


class PurchaseResponse(BaseModel):
    """Response with Stripe Checkout URL."""

    checkout_url: str


class TransactionItem(BaseModel):
    """A credit transaction record."""

    id: str
    type: str
    amount: int
    balance_after: int
    description: str | None
    metadata: dict | None
    created_at: str


class TransactionsResponse(BaseModel):
    """Paginated transaction history."""

    transactions: list[TransactionItem]
    count: int


class UsageLogItem(BaseModel):
    """A usage log record."""

    id: str
    session_id: str
    model_type: str
    model_id: str
    units_used: float
    cost_usd: float
    credits_charged: int
    created_at: str


class UsageLogsResponse(BaseModel):
    """Paginated usage logs."""

    logs: list[UsageLogItem]
    count: int


class UsageReportItem(BaseModel):
    """A single usage item in an agent report."""

    model_type: str = Field(..., description="stt, llm, tts, or realtime")
    model_id: str = Field(..., description="Model identifier")
    units_used: float = Field(..., description="Tokens, minutes, or characters")


class UsageReportRequest(BaseModel):
    """Agent usage report request."""

    user_id: str = Field(..., description="Supabase user ID")
    session_id: str = Field(..., description="LiveKit room name")
    usage: list[UsageReportItem] = Field(..., description="Usage items")


class UsageReportResponse(BaseModel):
    """Response from processing a usage report."""

    total_credits_charged: int
    new_balance: int
    items_processed: int


# =============================================================================
# Endpoints - User-facing (require auth)
# =============================================================================


@router.get("/balance", response_model=CreditBalanceResponse)
async def get_credit_balance(
    user: Annotated[AuthUser, Depends(require_auth)],
):
    """Get the current user's credit balance."""
    data = await get_balance(user.id)
    balance = data["balance"]
    return CreditBalanceResponse(
        balance=balance,
        balance_credits=balance / MICRO_CREDITS_PER_CREDIT,
        lifetime_purchased=data["lifetime_purchased"],
        lifetime_used=data["lifetime_used"],
    )


@router.get("/packs", response_model=CreditPacksResponse)
async def get_credit_packs():
    """Get available credit packs for purchase."""
    packs = []
    for pack in CREDIT_PACKS.values():
        packs.append(CreditPackResponse(
            id=pack["id"],
            name=pack["name"],
            credits=pack["credits"],
            price_cents=pack["price_cents"],
            price_display=f"${pack['price_cents'] / 100:.2f}",
            popular=pack["popular"],
        ))
    return CreditPacksResponse(packs=packs)


@router.post("/purchase", response_model=PurchaseResponse)
async def purchase_credits(
    request: PurchaseRequest,
    user: Annotated[AuthUser, Depends(require_auth)],
):
    """Create a Stripe Checkout Session to purchase credits."""
    try:
        checkout_url = await create_checkout_session(
            user_id=user.id,
            pack_id=request.pack_id,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
        )
        return PurchaseResponse(checkout_url=checkout_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Stripe not configured: {e}")
        raise HTTPException(
            status_code=503,
            detail="Payment processing is not currently available",
        )


@router.get("/transactions", response_model=TransactionsResponse)
async def get_credit_transactions(
    user: Annotated[AuthUser, Depends(require_auth)],
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get the user's credit transaction history."""
    transactions = await get_transactions(user.id, limit=limit, offset=offset)
    items = [
        TransactionItem(
            id=t["id"],
            type=t["type"],
            amount=t["amount"],
            balance_after=t["balance_after"],
            description=t.get("description"),
            metadata=t.get("metadata"),
            created_at=t["created_at"],
        )
        for t in transactions
    ]
    return TransactionsResponse(transactions=items, count=len(items))


@router.get("/usage", response_model=UsageLogsResponse)
async def get_credit_usage(
    user: Annotated[AuthUser, Depends(require_auth)],
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session_id: Optional[str] = Query(None),
):
    """Get the user's credit usage logs."""
    logs = await get_usage_logs(
        user.id,
        limit=limit,
        offset=offset,
        session_id=session_id,
    )
    items = [
        UsageLogItem(
            id=l["id"],
            session_id=l["session_id"],
            model_type=l["model_type"],
            model_id=l["model_id"],
            units_used=l["units_used"],
            cost_usd=l["cost_usd"],
            credits_charged=l["credits_charged"],
            created_at=l["created_at"],
        )
        for l in logs
    ]
    return UsageLogsResponse(logs=items, count=len(items))


# =============================================================================
# Endpoints - Stripe Webhook (no user auth, verified by Stripe signature)
# =============================================================================


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events.

    Verified using the Stripe-Signature header, not user auth.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    try:
        result = await handle_webhook_event(payload, sig_header)
        return result
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except RuntimeError as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


# =============================================================================
# Endpoints - Agent Usage Report (internal API key auth)
# =============================================================================


def _verify_internal_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> None:
    """Verify the internal API key used by the agent to report usage."""
    if not settings.internal_api_key:
        raise HTTPException(
            status_code=503,
            detail="Internal API not configured",
        )
    if x_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@router.post("/usage/report", response_model=UsageReportResponse)
async def report_usage(
    request: UsageReportRequest,
    _: Annotated[None, Depends(_verify_internal_api_key)],
):
    """Report AI usage from the agent after a session ends.

    This endpoint is called by the LiveKit agent (not the frontend).
    Authenticated via an internal API key (X-API-Key header).
    """
    logger.info(
        f"Usage report received: user={request.user_id}, "
        f"session={request.session_id}, items={len(request.usage)}"
    )

    usage_items = [
        {
            "model_type": item.model_type,
            "model_id": item.model_id,
            "units_used": item.units_used,
        }
        for item in request.usage
    ]

    result = await process_usage_report(
        user_id=request.user_id,
        session_id=request.session_id,
        usage_items=usage_items,
    )

    return UsageReportResponse(
        total_credits_charged=result["total_credits_charged"],
        new_balance=result["new_balance"],
        items_processed=len(result["items"]),
    )
