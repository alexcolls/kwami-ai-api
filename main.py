"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from routes import health, token, memory
from config import settings

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
    logger.info(f"📡 LiveKit URL: {settings.livekit_url}")
    logger.info(f"🔑 API Key: {settings.livekit_api_key[:8]}...")
    logger.info(f"🌍 Environment: {settings.app_env}")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Kwami AI LiveKit API",
    description="Token endpoint and configuration API for Kwami AI agents",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
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


def run():
    """Run the API server."""
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
