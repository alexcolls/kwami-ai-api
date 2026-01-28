"""Models information endpoints.

Provides endpoints for:
- Available models from LiveKit SDK (dynamically extracted)
- Model pricing information
- LLM metrics structure
"""

import logging
from typing import Literal, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pricing import (
    ALL_PRICING,
    LLM_PRICING,
    STT_PRICING,
    TTS_PRICING,
    REALTIME_PRICING,
    ModelPricing,
    TokenPricing,
    AudioPricing,
    RealtimePricing,
    get_model_pricing,
)

logger = logging.getLogger("kwami-api.models")
router = APIRouter()


# =============================================================================
# Model Type Definitions (extracted from LiveKit SDK)
# =============================================================================

def _extract_literal_values(type_hint: Any) -> list[str]:
    """Extract string values from a Literal type hint."""
    origin = getattr(type_hint, "__origin__", None)
    if origin is Literal:
        return list(type_hint.__args__)
    return []


def _get_sdk_models() -> dict[str, list[str]]:
    """
    Dynamically extract available models from LiveKit Agents SDK.
    Falls back to cached values if SDK import fails.
    """
    try:
        # Try to import from livekit.agents.inference
        from livekit.agents.inference.llm import (
            OpenAIModels,
            GoogleModels,
            KimiModels,
            DeepSeekModels,
            LLMModels,
        )
        
        return {
            "openai": _extract_literal_values(OpenAIModels),
            "google": _extract_literal_values(GoogleModels),
            "kimi": _extract_literal_values(KimiModels),
            "deepseek": _extract_literal_values(DeepSeekModels),
            "all_llm": _extract_literal_values(LLMModels) if hasattr(LLMModels, "__args__") else [],
        }
    except ImportError as e:
        logger.warning(f"Could not import LiveKit SDK models: {e}. Using fallback.")
        return _get_fallback_models()
    except Exception as e:
        logger.error(f"Error extracting SDK models: {e}")
        return _get_fallback_models()


def _get_fallback_models() -> dict[str, list[str]]:
    """Fallback model list when SDK import fails."""
    return {
        "openai": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "openai/gpt-5-nano",
            "openai/gpt-5.1",
            "openai/gpt-5.1-chat-latest",
            "openai/gpt-5.2",
            "openai/gpt-5.2-chat-latest",
            "openai/gpt-oss-120b",
        ],
        "google": [
            "google/gemini-3-pro",
            "google/gemini-3-flash",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
            "google/gemini-2.0-flash",
            "google/gemini-2.0-flash-lite",
        ],
        "kimi": ["moonshotai/kimi-k2-instruct"],
        "deepseek": [
            "deepseek-ai/deepseek-v3",
            "deepseek-ai/deepseek-v3.2",
        ],
        "all_llm": [],  # Will be populated from above
    }


# =============================================================================
# Response Models
# =============================================================================

class ModelTypeEnum(str, Enum):
    llm = "llm"
    stt = "stt"
    tts = "tts"
    realtime = "realtime"


class AvailableModelsResponse(BaseModel):
    """Response containing available models from LiveKit SDK."""
    source: Literal["sdk", "fallback"] = Field(
        description="Whether models were extracted from SDK or using fallback"
    )
    llm: dict[str, list[str]] = Field(description="LLM models by provider")
    stt: list[str] = Field(description="Available STT models")
    tts: list[str] = Field(description="Available TTS models")
    realtime: list[str] = Field(description="Available realtime models")


class PricingResponse(BaseModel):
    """Response containing model pricing information."""
    last_updated: str = Field(description="When pricing was last updated")
    models: dict[str, ModelPricing] = Field(description="Pricing by model ID")


class SingleModelPricingResponse(BaseModel):
    """Response for a single model's pricing."""
    model_id: str
    provider: str
    model_type: str
    display_name: str
    pricing: TokenPricing | AudioPricing | RealtimePricing
    notes: str | None = None


class LLMMetricsSchema(BaseModel):
    """Schema describing the LLMMetrics structure from LiveKit."""
    description: str = Field(
        default="LLMMetrics are emitted by the LLM after each generation. "
        "Subscribe to 'metrics_collected' event on session.llm to receive these."
    )
    fields: dict[str, str] = Field(
        default={
            "timestamp": "float - Unix timestamp when metrics were collected",
            "type": "str - Always 'llm' for LLM metrics",
            "label": "str - Label identifying the metric source",
            "request_id": "str - Unique identifier for the request",
            "duration": "float - Total duration of the request in seconds",
            "ttft": "float - Time to first token in seconds",
            "cancelled": "bool - Whether the request was cancelled",
            "completion_tokens": "int - Number of tokens in the completion",
            "prompt_tokens": "int - Number of tokens in the prompt",
            "total_tokens": "int - Total tokens (prompt + completion)",
            "tokens_per_second": "float - Token generation rate",
        }
    )
    example_usage: str = Field(
        default="""
from livekit.agents.metrics import LLMMetrics

def on_metrics(metrics: LLMMetrics):
    print(f"Tokens: {metrics.total_tokens}")
    print(f"TTFT: {metrics.ttft:.3f}s")
    print(f"Speed: {metrics.tokens_per_second:.1f} tok/s")

session.llm.on("metrics_collected", on_metrics)
"""
    )


class CostEstimate(BaseModel):
    """Estimated cost based on token usage."""
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float
    cached_tokens: int | None = None
    cached_cost_usd: float | None = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/available", response_model=AvailableModelsResponse)
async def get_available_models():
    """
    Get all available models from LiveKit SDK.
    
    This endpoint dynamically extracts model definitions from the LiveKit Agents SDK,
    ensuring you always have the latest models available. Falls back to a cached list
    if the SDK import fails.
    """
    sdk_models = _get_sdk_models()
    
    # Check if we got SDK models or fallback
    source: Literal["sdk", "fallback"] = "sdk"
    try:
        from livekit.agents.inference.llm import OpenAIModels
        source = "sdk"
    except ImportError:
        source = "fallback"
        # Populate all_llm from individual providers for fallback
        sdk_models["all_llm"] = (
            sdk_models["openai"] +
            sdk_models["google"] +
            sdk_models["kimi"] +
            sdk_models["deepseek"]
        )
    
    return AvailableModelsResponse(
        source=source,
        llm=sdk_models,
        stt=list(STT_PRICING.keys()),
        tts=list(TTS_PRICING.keys()),
        realtime=list(REALTIME_PRICING.keys()),
    )


@router.get("/pricing", response_model=PricingResponse)
async def get_all_pricing(
    model_type: ModelTypeEnum | None = Query(
        None,
        description="Filter by model type (llm, stt, tts, realtime)"
    ),
):
    """
    Get pricing information for all models.
    
    Optionally filter by model type. Prices are in USD.
    - LLM: per 1M tokens
    - STT: per minute of audio
    - TTS: per 1M characters
    - Realtime: per minute of audio I/O
    """
    if model_type:
        match model_type:
            case ModelTypeEnum.llm:
                models = LLM_PRICING
            case ModelTypeEnum.stt:
                models = STT_PRICING
            case ModelTypeEnum.tts:
                models = TTS_PRICING
            case ModelTypeEnum.realtime:
                models = REALTIME_PRICING
    else:
        models = ALL_PRICING
    
    return PricingResponse(
        last_updated="2026-01-28",
        models=models,
    )


@router.get("/pricing/{model_id:path}", response_model=SingleModelPricingResponse)
async def get_model_pricing_endpoint(model_id: str):
    """
    Get pricing for a specific model.
    
    Model ID should be in format 'provider/model-name', e.g. 'openai/gpt-4o-mini'.
    """
    pricing = get_model_pricing(model_id)
    if not pricing:
        raise HTTPException(
            status_code=404,
            detail=f"Pricing not found for model: {model_id}. "
            "Check /models/available for valid model IDs."
        )
    
    return SingleModelPricingResponse(
        model_id=pricing.model_id,
        provider=pricing.provider,
        model_type=pricing.model_type,
        display_name=pricing.display_name,
        pricing=pricing.pricing,
        notes=pricing.notes,
    )


@router.get("/metrics/llm", response_model=LLMMetricsSchema)
async def get_llm_metrics_schema():
    """
    Get the schema and usage information for LLMMetrics.
    
    LLMMetrics are emitted by the LLM after each generation and contain
    token counts, latency information, and other useful metrics for
    monitoring and cost estimation.
    """
    return LLMMetricsSchema()


@router.post("/estimate-cost", response_model=CostEstimate)
async def estimate_cost(
    model_id: str = Query(..., description="Model ID (e.g., 'openai/gpt-4o-mini')"),
    prompt_tokens: int = Query(..., ge=0, description="Number of prompt tokens"),
    completion_tokens: int = Query(..., ge=0, description="Number of completion tokens"),
    cached_tokens: int = Query(0, ge=0, description="Number of cached prompt tokens"),
):
    """
    Estimate the cost for a given token usage.
    
    Useful for calculating costs based on LLMMetrics data.
    Only works for LLM models with token-based pricing.
    """
    pricing = get_model_pricing(model_id)
    if not pricing:
        raise HTTPException(
            status_code=404,
            detail=f"Pricing not found for model: {model_id}"
        )
    
    if pricing.model_type != "llm":
        raise HTTPException(
            status_code=400,
            detail=f"Cost estimation only available for LLM models. "
            f"Model {model_id} is type '{pricing.model_type}'."
        )
    
    token_pricing = pricing.pricing
    if not isinstance(token_pricing, TokenPricing):
        raise HTTPException(
            status_code=500,
            detail="Invalid pricing configuration for model"
        )
    
    # Calculate costs (prices are per 1M tokens)
    non_cached_prompt = prompt_tokens - cached_tokens
    prompt_cost = (non_cached_prompt / 1_000_000) * token_pricing.input_per_1m
    completion_cost = (completion_tokens / 1_000_000) * token_pricing.output_per_1m
    
    cached_cost = None
    if cached_tokens > 0 and token_pricing.cached_input_per_1m:
        cached_cost = (cached_tokens / 1_000_000) * token_pricing.cached_input_per_1m
        prompt_cost += cached_cost
    
    return CostEstimate(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cost_usd=round(prompt_cost, 6),
        completion_cost_usd=round(completion_cost, 6),
        total_cost_usd=round(prompt_cost + completion_cost, 6),
        cached_tokens=cached_tokens if cached_tokens > 0 else None,
        cached_cost_usd=round(cached_cost, 6) if cached_cost else None,
    )
