"""Credit system service.

Handles credit balance operations, cost-to-credit conversion,
and Supabase database interactions for the credits system.
"""

import logging
from typing import Any

from supabase import create_client, Client

from src.core.config import settings
from src.services.pricing import (
    ALL_PRICING,
    TokenPricing,
    AudioPricing,
    RealtimePricing,
)

logger = logging.getLogger("kwami-api.credits")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MICRO_CREDITS_PER_CREDIT = 1000
USD_PER_CREDIT = 0.001  # 1 credit = $0.001
MARKUP_MULTIPLIER = 2.0  # 2x markup on raw provider costs

# Credit pack definitions: (pack_id, display_name, credits, price_cents)
CREDIT_PACKS = {
    "starter": {
        "id": "starter",
        "name": "Starter",
        "credits": 5_000,
        "price_cents": 500,  # $5.00
        "popular": False,
    },
    "standard": {
        "id": "standard",
        "name": "Standard",
        "credits": 25_000,
        "price_cents": 2500,  # $25.00
        "popular": True,
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "credits": 100_000,
        "price_cents": 10000,  # $100.00
        "popular": False,
    },
}

# ---------------------------------------------------------------------------
# Supabase admin client (singleton)
# ---------------------------------------------------------------------------

_supabase_client: Client | None = None


def get_supabase_admin() -> Client:
    """Get the Supabase admin client using the secret API key."""
    global _supabase_client
    if _supabase_client is None:
        if not settings.supabase_url or not settings.supabase_secret_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY must be set for the credits system"
            )
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_secret_key,
        )
    return _supabase_client


# ---------------------------------------------------------------------------
# Cost to credits conversion
# ---------------------------------------------------------------------------


def usd_to_micro_credits(cost_usd: float) -> int:
    """Convert a USD cost to micro-credits with markup.

    Args:
        cost_usd: Raw provider cost in USD.

    Returns:
        Amount in micro-credits (rounded up).
    """
    marked_up = cost_usd * MARKUP_MULTIPLIER
    credits = marked_up / USD_PER_CREDIT
    micro = int(credits * MICRO_CREDITS_PER_CREDIT)
    return max(micro, 1)  # minimum 1 micro-credit per operation


def calculate_usage_cost(
    model_id: str,
    model_type: str,
    units_used: float,
) -> tuple[float, int]:
    """Calculate raw USD cost and micro-credits for a usage event.

    Args:
        model_id: The model identifier (e.g. 'openai/gpt-4o-mini').
        model_type: One of 'llm', 'stt', 'tts', 'realtime'.
        units_used: Tokens for LLM, minutes for STT, characters for TTS.

    Returns:
        Tuple of (cost_usd, micro_credits).
    """
    pricing_entry = ALL_PRICING.get(model_id)

    if pricing_entry is None:
        # Unknown model: fallback to a conservative estimate
        logger.warning(f"No pricing for model {model_id}, using fallback")
        cost_usd = units_used * 0.000002  # ~$2/1M tokens fallback
        return cost_usd, usd_to_micro_credits(cost_usd)

    pricing = pricing_entry.pricing
    cost_usd = 0.0

    if isinstance(pricing, TokenPricing):
        # units_used = total_tokens (prompt + completion combined for simplicity)
        cost_usd = (units_used / 1_000_000) * (
            (pricing.input_per_1m + pricing.output_per_1m) / 2
        )
    elif isinstance(pricing, AudioPricing):
        if pricing.per_minute:
            cost_usd = units_used * pricing.per_minute
        elif pricing.per_1m_characters:
            cost_usd = (units_used / 1_000_000) * pricing.per_1m_characters
    elif isinstance(pricing, RealtimePricing):
        # units_used = minutes of audio
        cost_usd = units_used * (
            (pricing.audio_input_per_minute + pricing.audio_output_per_minute) / 2
        )

    return cost_usd, usd_to_micro_credits(cost_usd)


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


async def get_balance(user_id: str) -> dict[str, Any]:
    """Get a user's credit balance.

    Returns dict with balance, lifetime_purchased, lifetime_used (all in micro-credits).
    Creates a row with 0 balance if the user has no record.
    """
    sb = get_supabase_admin()
    result = sb.table("user_credits").select("*").eq("user_id", user_id).execute()

    if result.data:
        row = result.data[0]
        return {
            "balance": row["balance"],
            "lifetime_purchased": row["lifetime_purchased"],
            "lifetime_used": row["lifetime_used"],
            "updated_at": row["updated_at"],
        }

    # User has no row yet (shouldn't happen with trigger, but handle gracefully)
    sb.table("user_credits").insert({
        "user_id": user_id,
        "balance": 0,
        "lifetime_purchased": 0,
        "lifetime_used": 0,
    }).execute()

    return {
        "balance": 0,
        "lifetime_purchased": 0,
        "lifetime_used": 0,
        "updated_at": None,
    }


async def add_credits(
    user_id: str,
    amount_micro: int,
    transaction_type: str = "purchase",
    description: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Add credits to a user's balance using the DB function.

    Returns the new balance in micro-credits.
    """
    sb = get_supabase_admin()
    result = sb.rpc(
        "add_credits",
        {
            "p_user_id": user_id,
            "p_amount": amount_micro,
            "p_type": transaction_type,
            "p_description": description or "",
            "p_metadata": metadata or {},
        },
    ).execute()

    new_balance = result.data
    logger.info(
        f"Added {amount_micro} micro-credits to user {user_id}, "
        f"new balance: {new_balance}"
    )
    return new_balance


async def deduct_credits(
    user_id: str,
    amount_micro: int,
    description: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Deduct credits from a user's balance using the DB function.

    Returns the new balance. Raises if insufficient funds.
    """
    sb = get_supabase_admin()
    try:
        result = sb.rpc(
            "deduct_credits",
            {
                "p_user_id": user_id,
                "p_amount": amount_micro,
                "p_description": description or "",
                "p_metadata": metadata or {},
            },
        ).execute()

        new_balance = result.data
        logger.info(
            f"Deducted {amount_micro} micro-credits from user {user_id}, "
            f"new balance: {new_balance}"
        )
        return new_balance
    except Exception as e:
        if "Insufficient credits" in str(e):
            raise ValueError("Insufficient credits") from e
        raise


async def log_usage(
    user_id: str,
    session_id: str,
    model_type: str,
    model_id: str,
    units_used: float,
    cost_usd: float,
    credits_charged: int,
) -> None:
    """Insert a usage log row."""
    sb = get_supabase_admin()
    sb.table("credit_usage_logs").insert({
        "user_id": user_id,
        "session_id": session_id,
        "model_type": model_type,
        "model_id": model_id,
        "units_used": units_used,
        "cost_usd": cost_usd,
        "credits_charged": credits_charged,
    }).execute()


async def get_transactions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get paginated transaction history for a user."""
    sb = get_supabase_admin()
    result = (
        sb.table("credit_transactions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


async def get_usage_logs(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
) -> list[dict]:
    """Get paginated usage logs for a user."""
    sb = get_supabase_admin()
    query = (
        sb.table("credit_usage_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
    )

    if session_id:
        query = query.eq("session_id", session_id)

    result = query.range(offset, offset + limit - 1).execute()
    return result.data or []


async def process_usage_report(
    user_id: str,
    session_id: str,
    usage_items: list[dict],
) -> dict:
    """Process a batch usage report from the agent.

    Each item in usage_items: {model_type, model_id, units_used}

    Returns summary with total_credits_charged and new_balance.
    """
    total_micro_credits = 0
    logged_items = []

    for item in usage_items:
        model_type = item["model_type"]
        model_id = item["model_id"]
        units_used = item["units_used"]

        cost_usd, micro_credits = calculate_usage_cost(model_id, model_type, units_used)

        await log_usage(
            user_id=user_id,
            session_id=session_id,
            model_type=model_type,
            model_id=model_id,
            units_used=units_used,
            cost_usd=cost_usd,
            credits_charged=micro_credits,
        )

        total_micro_credits += micro_credits
        logged_items.append({
            "model_type": model_type,
            "model_id": model_id,
            "units_used": units_used,
            "cost_usd": round(cost_usd, 6),
            "credits_charged": micro_credits,
        })

    # Deduct total from user balance
    new_balance = 0
    if total_micro_credits > 0:
        try:
            new_balance = await deduct_credits(
                user_id=user_id,
                amount_micro=total_micro_credits,
                description=f"Session usage: {session_id}",
                metadata={
                    "session_id": session_id,
                    "items_count": len(logged_items),
                },
            )
        except ValueError:
            logger.warning(
                f"Insufficient credits for user {user_id}, "
                f"session {session_id}. Usage logged but not fully charged."
            )

    return {
        "total_credits_charged": total_micro_credits,
        "new_balance": new_balance,
        "items": logged_items,
    }
