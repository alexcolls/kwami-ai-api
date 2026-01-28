"""Models information endpoints.

Provides endpoints for:
- Available models from LiveKit SDK (dynamically extracted from inference + plugins)
- Model pricing information
- LLM metrics structure
"""

import logging
from typing import Literal, Any, Union, get_args, get_origin
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
# Helpers for extracting model definitions from LiveKit SDK
# =============================================================================

def _extract_literal_values(type_hint: Any) -> list[str]:
    """Extract string values from a Literal or Union type hint."""
    if type_hint is None:
        return []
    
    origin = get_origin(type_hint)
    
    if origin is Literal:
        return [str(arg) for arg in get_args(type_hint)]
    
    if origin is Union:
        values = []
        for arg in get_args(type_hint):
            values.extend(_extract_literal_values(arg))
        return values
    
    if hasattr(type_hint, "__args__"):
        return [str(arg) for arg in type_hint.__args__ if isinstance(arg, str)]
    
    return []


def _safe_import_and_extract(module_path: str, type_name: str) -> list[str]:
    """Safely import a module and extract Literal values from a type."""
    try:
        import importlib
        module = importlib.import_module(module_path)
        type_hint = getattr(module, type_name, None)
        if type_hint:
            return _extract_literal_values(type_hint)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import {module_path}.{type_name}: {e}")
    return []


# =============================================================================
# Model extraction functions (one per type)
# =============================================================================

def _get_llm_models() -> dict[str, Any]:
    """Extract all LLM models from LiveKit SDK and plugins."""
    inference = {}
    plugins = {}
    source = "sdk"
    
    # LiveKit Inference LLM
    try:
        from livekit.agents.inference.llm import (
            OpenAIModels, GoogleModels, KimiModels, DeepSeekModels,
        )
        inference = {
            "openai": _extract_literal_values(OpenAIModels),
            "google": _extract_literal_values(GoogleModels),
            "kimi": _extract_literal_values(KimiModels),
            "deepseek": _extract_literal_values(DeepSeekModels),
        }
    except ImportError as e:
        logger.warning(f"Could not import inference LLM models: {e}")
        inference = {
            "openai": [
                "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
                "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
                "openai/gpt-4o", "openai/gpt-4o-mini",
            ],
            "google": [
                "google/gemini-2.5-pro", "google/gemini-2.5-flash",
                "google/gemini-2.0-flash", "google/gemini-2.0-flash-lite",
            ],
            "kimi": ["moonshotai/kimi-k2-instruct"],
            "deepseek": ["deepseek-ai/deepseek-v3"],
        }
        source = "fallback"
    
    # Plugin LLMs
    plugin_extractions = [
        ("openai", "livekit.plugins.openai.models", "ChatModels"),
        ("anthropic", "livekit.plugins.anthropic.models", "ChatModels"),
        ("google", "livekit.plugins.google.models", "ChatModels"),
        ("groq", "livekit.plugins.groq.models", "ChatModels"),
        ("mistralai", "livekit.plugins.mistralai.models", "ChatModels"),
        # OpenAI-compatible providers
        ("cerebras", "livekit.plugins.openai.models", "CerebrasChatModels"),
        ("perplexity", "livekit.plugins.openai.models", "PerplexityChatModels"),
        ("xai", "livekit.plugins.openai.models", "XAIChatModels"),
        ("deepseek", "livekit.plugins.openai.models", "DeepSeekChatModels"),
        ("together", "livekit.plugins.openai.models", "TogetherChatModels"),
        ("telnyx", "livekit.plugins.openai.models", "TelnyxChatModels"),
        ("nebius", "livekit.plugins.openai.models", "NebiusChatModels"),
        ("octoai", "livekit.plugins.openai.models", "OctoChatModels"),
    ]
    
    for provider, module, type_name in plugin_extractions:
        models = _safe_import_and_extract(module, type_name)
        if models:
            plugins[provider] = models
    
    return {"inference": inference, "plugins": plugins, "source": source}


def _get_stt_models() -> dict[str, Any]:
    """Extract all STT models from LiveKit SDK and plugins."""
    inference = []
    plugins = {}
    source = "sdk"
    
    # LiveKit Inference STT
    try:
        from livekit.agents.inference.stt import STTModels
        inference = _extract_literal_values(STTModels)
    except ImportError:
        inference = []
        source = "fallback"
    
    # Plugin STTs
    plugin_extractions = [
        ("deepgram", "livekit.plugins.deepgram.models", "DeepgramModels"),
        ("google", "livekit.plugins.google.models", "SpeechModels"),
        ("groq", "livekit.plugins.groq.models", "STTModels"),
        ("elevenlabs", "livekit.plugins.elevenlabs.models", "STTModels"),
        ("cartesia", "livekit.plugins.cartesia.models", "STTModels"),
        ("mistralai", "livekit.plugins.mistralai.models", "STTModels"),
    ]
    
    for provider, module, type_name in plugin_extractions:
        models = _safe_import_and_extract(module, type_name)
        if models:
            plugins[provider] = models
    
    return {"inference": inference, "plugins": plugins, "source": source}


def _get_tts_models() -> dict[str, Any]:
    """Extract all TTS models from LiveKit SDK and plugins."""
    inference = []
    plugins = {}
    source = "sdk"
    
    # LiveKit Inference TTS
    try:
        from livekit.agents.inference.tts import TTSModels
        inference = _extract_literal_values(TTSModels)
    except ImportError:
        inference = []
        source = "fallback"
    
    # Plugin TTSs
    plugin_extractions = [
        ("openai", "livekit.plugins.openai.models", "TTSModels"),
        ("deepgram", "livekit.plugins.deepgram.models", "TTSModels"),
        ("elevenlabs", "livekit.plugins.elevenlabs.models", "TTSModels"),
        ("cartesia", "livekit.plugins.cartesia.models", "TTSModels"),
        ("google", "livekit.plugins.google.models", "GeminiTTSModels"),
        ("groq", "livekit.plugins.groq.models", "TTSModels"),
        ("rime", "livekit.plugins.rime.models", "TTSModels"),
    ]
    
    for provider, module, type_name in plugin_extractions:
        models = _safe_import_and_extract(module, type_name)
        if models:
            plugins[provider] = models
    
    return {"inference": inference, "plugins": plugins, "source": source}


def _get_realtime_models() -> dict[str, Any]:
    """Extract all Realtime models from LiveKit SDK and plugins."""
    inference = []
    plugins = {}
    source = "sdk"
    
    # Plugin Realtime models
    openai_rt = _safe_import_and_extract("livekit.plugins.openai.models", "RealtimeModels")
    if openai_rt:
        plugins["openai"] = openai_rt
    
    google_rt = _safe_import_and_extract("livekit.plugins.google.realtime.api_proto", "LiveAPIModels")
    if google_rt:
        plugins["google"] = google_rt
    
    # AWS Nova Sonic (fallback since SDK needs extra dependencies)
    aws_rt = _safe_import_and_extract(
        "livekit.plugins.aws.experimental.realtime.types", "REALTIME_MODELS"
    )
    if not aws_rt:
        aws_rt = ["amazon.nova-sonic-v1:0", "amazon.nova-2-sonic-v1:0"]
    plugins["aws"] = aws_rt
    
    return {"inference": inference, "plugins": plugins, "source": source}


# =============================================================================
# Response Models
# =============================================================================

class ModelTypeEnum(str, Enum):
    llm = "llm"
    stt = "stt"
    tts = "tts"
    realtime = "realtime"


class ModelsResponse(BaseModel):
    """Response for a single model type."""
    source: Literal["sdk", "fallback"] = Field(
        description="Whether models were extracted from SDK or using fallback"
    )
    inference: dict[str, list[str]] | list[str] = Field(
        description="Models available via LiveKit Inference (hosted)"
    )
    plugins: dict[str, list[str]] = Field(
        description="Models available via plugins (direct provider access)"
    )


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


class ModelTypeCapabilities(BaseModel):
    """Capabilities for a model type."""
    description: str
    input: list[str] = Field(description="Supported input types")
    output: list[str] = Field(description="Supported output types")
    features: list[str] = Field(description="Available features")
    optional_input: list[str] | None = Field(
        default=None, description="Optional input types (model-dependent)"
    )


class VisionCapableModels(BaseModel):
    """Models that support vision/image input."""
    description: str = "LLM models that can process images and video frames"
    models: dict[str, list[str]] = Field(description="Vision-capable models by provider")
    usage_notes: str


class VideoInputModels(BaseModel):
    """Models that support live video input."""
    description: str = "Realtime models with native video stream support"
    models: dict[str, list[str]] = Field(description="Video-capable models by provider")
    usage_notes: str


class CapabilitiesResponse(BaseModel):
    """Complete capabilities overview for all model types."""
    llm: ModelTypeCapabilities
    stt: ModelTypeCapabilities
    tts: ModelTypeCapabilities
    realtime: ModelTypeCapabilities
    vision: VisionCapableModels
    video_input: VideoInputModels


# =============================================================================
# Capabilities Data
# =============================================================================

def _get_capabilities() -> CapabilitiesResponse:
    """Build the capabilities response with all model type info."""
    return CapabilitiesResponse(
        llm=ModelTypeCapabilities(
            description="Large Language Models for text generation and reasoning",
            input=["text", "images"],
            output=["text"],
            features=[
                "streaming",
                "function_calling",
                "vision",
                "structured_output",
                "context_caching",
            ],
            optional_input=["images (vision-capable models only)"],
        ),
        stt=ModelTypeCapabilities(
            description="Speech-to-Text models for transcription",
            input=["audio"],
            output=["text"],
            features=[
                "streaming",
                "word_timestamps",
                "language_detection",
                "punctuation",
                "speaker_diarization",
            ],
        ),
        tts=ModelTypeCapabilities(
            description="Text-to-Speech models for voice synthesis",
            input=["text"],
            output=["audio"],
            features=[
                "streaming",
                "multiple_voices",
                "voice_cloning",
                "ssml_support",
                "emotion_control",
            ],
        ),
        realtime=ModelTypeCapabilities(
            description="Realtime multimodal models for bidirectional voice conversations",
            input=["audio", "text"],
            output=["audio", "text"],
            features=[
                "bidirectional_streaming",
                "low_latency",
                "turn_detection",
                "interruption_handling",
                "function_calling",
            ],
            optional_input=["video (some models)"],
        ),
        vision=VisionCapableModels(
            description="LLM models that can process images and video frames",
            models={
                "openai": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4-vision-preview",
                ],
                "anthropic": [
                    "claude-3-5-sonnet-latest",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-latest",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ],
                "google": [
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ],
            },
            usage_notes=(
                "Vision models can process static images added to chat context. "
                "For video, sample frames at suitable intervals (e.g., on each user turn). "
                "Use ImageContent to add images from URLs, base64, or video frames."
            ),
        ),
        video_input=VideoInputModels(
            description="Realtime models with native video stream support",
            models={
                "openai": [
                    "gpt-4o-realtime-preview",
                ],
                "google": [
                    "gemini-live-2.5-flash-native-audio",
                    "gemini-2.0-flash-exp",
                ],
            },
            usage_notes=(
                "These models can process continuous video streams natively. "
                "Enable with video_input=True in RoomOptions. "
                "The agent automatically receives frames from the user's camera or screen share. "
                "Default: 1 FPS while speaking, 0.33 FPS otherwise."
            ),
        ),
    )


# =============================================================================
# Endpoints - Models by Type
# =============================================================================

@router.get("/llm", response_model=ModelsResponse)
async def get_llm_models():
    """
    Get all available LLM (Large Language Model) models.
    
    Returns models from:
    - **LiveKit Inference**: Hosted models (no API keys needed)
    - **Plugins**: Direct provider integrations (requires your own API keys)
    
    Providers include: OpenAI, Anthropic, Google, Groq, Mistral, and more.
    """
    result = _get_llm_models()
    return ModelsResponse(
        source=result["source"],
        inference=result["inference"],
        plugins=result["plugins"],
    )


@router.get("/stt", response_model=ModelsResponse)
async def get_stt_models():
    """
    Get all available STT (Speech-to-Text) models.
    
    Returns models from:
    - **LiveKit Inference**: Hosted models (no API keys needed)
    - **Plugins**: Direct provider integrations (requires your own API keys)
    
    Providers include: Deepgram, Google, Groq, ElevenLabs, Cartesia, and more.
    """
    result = _get_stt_models()
    return ModelsResponse(
        source=result["source"],
        inference=result["inference"],
        plugins=result["plugins"],
    )


@router.get("/tts", response_model=ModelsResponse)
async def get_tts_models():
    """
    Get all available TTS (Text-to-Speech) models.
    
    Returns models from:
    - **LiveKit Inference**: Hosted models (no API keys needed)
    - **Plugins**: Direct provider integrations (requires your own API keys)
    
    Providers include: OpenAI, ElevenLabs, Cartesia, Deepgram, Rime, and more.
    """
    result = _get_tts_models()
    return ModelsResponse(
        source=result["source"],
        inference=result["inference"],
        plugins=result["plugins"],
    )


@router.get("/realtime", response_model=ModelsResponse)
async def get_realtime_models():
    """
    Get all available Realtime (multimodal voice) models.
    
    Realtime models support bidirectional audio streaming with optional video.
    These models handle STT, LLM, and TTS in a single integrated pipeline.
    
    Providers include: OpenAI (GPT-4o Realtime), Google (Gemini Live), AWS (Nova Sonic).
    """
    result = _get_realtime_models()
    return ModelsResponse(
        source=result["source"],
        inference=result["inference"],
        plugins=result["plugins"],
    )


# =============================================================================
# Endpoints - Capabilities
# =============================================================================

@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_model_capabilities():
    """
    Get capabilities overview for all model types.
    
    Returns detailed information about:
    - **Input/Output types**: What each model type accepts and produces
    - **Features**: Available capabilities (streaming, function calling, etc.)
    - **Vision models**: LLMs that can process images and video frames
    - **Video input models**: Realtime models with native video stream support
    
    Use this endpoint to understand what's possible with each model type
    and which specific models support advanced features like vision.
    """
    return _get_capabilities()


# =============================================================================
# Endpoints - Pricing
# =============================================================================

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
            "Check /models/llm, /models/stt, /models/tts, or /models/realtime for valid model IDs."
        )
    
    return SingleModelPricingResponse(
        model_id=pricing.model_id,
        provider=pricing.provider,
        model_type=pricing.model_type,
        display_name=pricing.display_name,
        pricing=pricing.pricing,
        notes=pricing.notes,
    )


# =============================================================================
# Endpoints - Metrics & Cost Estimation
# =============================================================================

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
