"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import (
    admin_reconciliation,
    calendar,
    channels,
    contacts,
    credits,
    email,
    health,
    internal,
    languages,
    memory,
    models,
    token,
    wallet,
    voices,
    webhooks,
)
from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("kwami-api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log incoming request bodies for debugging."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/token" and request.method == "POST":
            body = await request.body()
            logger.info(f"📨 Raw request body: {body.decode()[:500]}")
            # Recreate the request with the body since we consumed it
            from starlette.requests import Request as StarletteRequest
            scope = request.scope
            async def receive():
                return {"type": "http.request", "body": body}
            request = StarletteRequest(scope, receive)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"🚀 Starting {settings.app_name} v0.1.0")
    logger.info(f"🌐 Listening on {settings.api_host}:{settings.api_port}")
    logger.info(f"📡 LiveKit URL: {settings.livekit_url}")
    logger.info(f"🔑 API Key: {settings.livekit_api_key[:8]}...")
    logger.info(f"🌍 Environment: {settings.app_env}")
    if settings.kwami_api_key and settings.kwami_api_key.strip():
        logger.info("📊 Kwami API key for usage report: set")
    else:
        logger.warning("📊 Kwami API key for usage report: NOT SET (agent usage reports will get 503)")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Kwami AI LiveKit API",
    description="Token endpoint and configuration API for Kwami AI agents",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.show_docs else None,
    redoc_url="/redoc" if settings.show_docs else None,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(token.router, prefix="/token", tags=["Token"])
app.include_router(memory.router, prefix="/memory", tags=["Memory"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(voices.router, prefix="/voices", tags=["Voices"])
app.include_router(languages.router, prefix="/languages", tags=["Languages"])
app.include_router(credits.router, prefix="/credits", tags=["Credits"])
app.include_router(channels.router, prefix="/channels", tags=["Channels"])
app.include_router(contacts.router, prefix="/contacts", tags=["Contacts"])
app.include_router(wallet.router, prefix="/wallets", tags=["Wallets"])
app.include_router(email.router, prefix="/email", tags=["Email"])
app.include_router(calendar.router, prefix="/calendar", tags=["Calendar"])
app.include_router(internal.router, prefix="/internal", tags=["Internal"])
app.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])
app.include_router(
    admin_reconciliation.router,
    prefix="/admin/reconciliation",
    tags=["Admin Reconciliation"],
)


def run():
    """Run the API server."""
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
