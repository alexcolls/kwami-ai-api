"""Model pricing configuration.

This file contains pricing information for AI models used with LiveKit.
Prices are in USD per 1M tokens (or per minute for audio models).

Update this file when provider pricing changes.
Last updated: 2026-01-28
"""

from typing import Literal
from pydantic import BaseModel


class TokenPricing(BaseModel):
    """Pricing for token-based models (LLM)."""
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    cached_input_per_1m: float | None = None  # USD per 1M cached input tokens (if supported)


class AudioPricing(BaseModel):
    """Pricing for audio-based models (STT/TTS)."""
    per_minute: float | None = None  # USD per minute
    per_1m_characters: float | None = None  # USD per 1M characters (for TTS)


class RealtimePricing(BaseModel):
    """Pricing for realtime models (audio in/out)."""
    audio_input_per_minute: float  # USD per minute of audio input
    audio_output_per_minute: float  # USD per minute of audio output
    text_input_per_1m: float | None = None  # USD per 1M text input tokens
    text_output_per_1m: float | None = None  # USD per 1M text output tokens


class ModelPricing(BaseModel):
    """Complete pricing info for a model."""
    model_id: str
    provider: str
    model_type: Literal["llm", "stt", "tts", "realtime"]
    display_name: str
    pricing: TokenPricing | AudioPricing | RealtimePricing
    notes: str | None = None


# =============================================================================
# LLM Models Pricing
# =============================================================================

LLM_PRICING: dict[str, ModelPricing] = {
    # OpenAI Models (via LiveKit Inference)
    "openai/gpt-4o": ModelPricing(
        model_id="openai/gpt-4o",
        provider="openai",
        model_type="llm",
        display_name="GPT-4o",
        pricing=TokenPricing(
            input_per_1m=2.50,
            output_per_1m=10.00,
            cached_input_per_1m=1.25,
        ),
    ),
    "openai/gpt-4o-mini": ModelPricing(
        model_id="openai/gpt-4o-mini",
        provider="openai",
        model_type="llm",
        display_name="GPT-4o Mini",
        pricing=TokenPricing(
            input_per_1m=0.15,
            output_per_1m=0.60,
            cached_input_per_1m=0.075,
        ),
    ),
    "openai/gpt-4.1": ModelPricing(
        model_id="openai/gpt-4.1",
        provider="openai",
        model_type="llm",
        display_name="GPT-4.1",
        pricing=TokenPricing(
            input_per_1m=2.00,
            output_per_1m=8.00,
            cached_input_per_1m=0.50,
        ),
    ),
    "openai/gpt-4.1-mini": ModelPricing(
        model_id="openai/gpt-4.1-mini",
        provider="openai",
        model_type="llm",
        display_name="GPT-4.1 Mini",
        pricing=TokenPricing(
            input_per_1m=0.40,
            output_per_1m=1.60,
            cached_input_per_1m=0.10,
        ),
    ),
    "openai/gpt-4.1-nano": ModelPricing(
        model_id="openai/gpt-4.1-nano",
        provider="openai",
        model_type="llm",
        display_name="GPT-4.1 Nano",
        pricing=TokenPricing(
            input_per_1m=0.10,
            output_per_1m=0.40,
            cached_input_per_1m=0.025,
        ),
    ),
    
    # Google Models (via LiveKit Inference)
    "google/gemini-2.0-flash": ModelPricing(
        model_id="google/gemini-2.0-flash",
        provider="google",
        model_type="llm",
        display_name="Gemini 2.0 Flash",
        pricing=TokenPricing(
            input_per_1m=0.10,
            output_per_1m=0.40,
            cached_input_per_1m=0.025,
        ),
    ),
    "google/gemini-2.0-flash-lite": ModelPricing(
        model_id="google/gemini-2.0-flash-lite",
        provider="google",
        model_type="llm",
        display_name="Gemini 2.0 Flash Lite",
        pricing=TokenPricing(
            input_per_1m=0.075,
            output_per_1m=0.30,
            cached_input_per_1m=0.01875,
        ),
    ),
    "google/gemini-2.5-flash": ModelPricing(
        model_id="google/gemini-2.5-flash",
        provider="google",
        model_type="llm",
        display_name="Gemini 2.5 Flash",
        pricing=TokenPricing(
            input_per_1m=0.15,
            output_per_1m=0.60,
            cached_input_per_1m=0.0375,
        ),
    ),
    "google/gemini-2.5-pro": ModelPricing(
        model_id="google/gemini-2.5-pro",
        provider="google",
        model_type="llm",
        display_name="Gemini 2.5 Pro",
        pricing=TokenPricing(
            input_per_1m=1.25,
            output_per_1m=10.00,
            cached_input_per_1m=0.3125,
        ),
    ),
    
    # DeepSeek Models (via LiveKit Inference)
    "deepseek-ai/deepseek-v3": ModelPricing(
        model_id="deepseek-ai/deepseek-v3",
        provider="deepseek",
        model_type="llm",
        display_name="DeepSeek V3",
        pricing=TokenPricing(
            input_per_1m=0.27,
            output_per_1m=1.10,
            cached_input_per_1m=0.07,
        ),
    ),
    
    # Kimi Models (via LiveKit Inference)
    "moonshotai/kimi-k2-instruct": ModelPricing(
        model_id="moonshotai/kimi-k2-instruct",
        provider="moonshot",
        model_type="llm",
        display_name="Kimi K2 Instruct",
        pricing=TokenPricing(
            input_per_1m=0.60,
            output_per_1m=2.40,
        ),
    ),
}

# =============================================================================
# STT Models Pricing
# =============================================================================

STT_PRICING: dict[str, ModelPricing] = {
    "deepgram/nova-3-general": ModelPricing(
        model_id="deepgram/nova-3-general",
        provider="deepgram",
        model_type="stt",
        display_name="Deepgram Nova 3",
        pricing=AudioPricing(per_minute=0.0043),
    ),
    "deepgram/nova-2-general": ModelPricing(
        model_id="deepgram/nova-2-general",
        provider="deepgram",
        model_type="stt",
        display_name="Deepgram Nova 2",
        pricing=AudioPricing(per_minute=0.0043),
    ),
    "assemblyai/universal": ModelPricing(
        model_id="assemblyai/universal",
        provider="assemblyai",
        model_type="stt",
        display_name="AssemblyAI Universal",
        pricing=AudioPricing(per_minute=0.0037),
    ),
}

# =============================================================================
# TTS Models Pricing
# =============================================================================

TTS_PRICING: dict[str, ModelPricing] = {
    "cartesia/sonic-3": ModelPricing(
        model_id="cartesia/sonic-3",
        provider="cartesia",
        model_type="tts",
        display_name="Cartesia Sonic 3",
        pricing=AudioPricing(per_1m_characters=15.00),
    ),
    "cartesia/sonic-2": ModelPricing(
        model_id="cartesia/sonic-2",
        provider="cartesia",
        model_type="tts",
        display_name="Cartesia Sonic 2",
        pricing=AudioPricing(per_1m_characters=15.00),
    ),
    "elevenlabs/eleven_turbo_v2_5": ModelPricing(
        model_id="elevenlabs/eleven_turbo_v2_5",
        provider="elevenlabs",
        model_type="tts",
        display_name="ElevenLabs Turbo v2.5",
        pricing=AudioPricing(per_1m_characters=30.00),
    ),
    "openai/tts-1": ModelPricing(
        model_id="openai/tts-1",
        provider="openai",
        model_type="tts",
        display_name="OpenAI TTS-1",
        pricing=AudioPricing(per_1m_characters=15.00),
    ),
    "openai/tts-1-hd": ModelPricing(
        model_id="openai/tts-1-hd",
        provider="openai",
        model_type="tts",
        display_name="OpenAI TTS-1 HD",
        pricing=AudioPricing(per_1m_characters=30.00),
    ),
}

# =============================================================================
# Realtime Models Pricing
# =============================================================================

REALTIME_PRICING: dict[str, ModelPricing] = {
    "openai/gpt-4o-realtime-preview": ModelPricing(
        model_id="openai/gpt-4o-realtime-preview",
        provider="openai",
        model_type="realtime",
        display_name="GPT-4o Realtime",
        pricing=RealtimePricing(
            audio_input_per_minute=0.06,
            audio_output_per_minute=0.24,
            text_input_per_1m=5.00,
            text_output_per_1m=20.00,
        ),
    ),
    "openai/gpt-4o-mini-realtime-preview": ModelPricing(
        model_id="openai/gpt-4o-mini-realtime-preview",
        provider="openai",
        model_type="realtime",
        display_name="GPT-4o Mini Realtime",
        pricing=RealtimePricing(
            audio_input_per_minute=0.01,
            audio_output_per_minute=0.04,
            text_input_per_1m=0.60,
            text_output_per_1m=2.40,
        ),
    ),
}

# =============================================================================
# Combined pricing dictionary
# =============================================================================

ALL_PRICING: dict[str, ModelPricing] = {
    **LLM_PRICING,
    **STT_PRICING,
    **TTS_PRICING,
    **REALTIME_PRICING,
}


def get_pricing_by_type(model_type: Literal["llm", "stt", "tts", "realtime"]) -> dict[str, ModelPricing]:
    """Get all pricing for a specific model type."""
    match model_type:
        case "llm":
            return LLM_PRICING
        case "stt":
            return STT_PRICING
        case "tts":
            return TTS_PRICING
        case "realtime":
            return REALTIME_PRICING


def get_model_pricing(model_id: str) -> ModelPricing | None:
    """Get pricing for a specific model."""
    return ALL_PRICING.get(model_id)
