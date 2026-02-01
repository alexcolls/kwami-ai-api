"""Application settings using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "kwami-lk-api"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # CORS - stored as comma-separated string, accessed via property
    cors_origins_str: str = Field(default="*", alias="CORS_ORIGINS")

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]

    # LiveKit
    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: str = Field(alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(alias="LIVEKIT_API_SECRET")

    # Memory
    zep_api_key: str | None = Field(default=None, alias="ZEP_API_KEY")

    # Supabase Auth
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_jwt_secret: str | None = Field(default=None, alias="SUPABASE_JWT_SECRET")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    @property
    def auth_enabled(self) -> bool:
        """Check if authentication is configured."""
        return self.supabase_url is not None or self.supabase_jwt_secret is not None
    
    @computed_field
    @property
    def supabase_jwks_url(self) -> str | None:
        """Get JWKS URL for Supabase project."""
        if self.supabase_url:
            return f"{self.supabase_url}/auth/v1/.well-known/jwks.json"
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
