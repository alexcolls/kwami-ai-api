"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "kwami-lk-api"}


@router.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Kwami AI LiveKit API",
        "version": "0.1.0",
        "docs": "/docs",
    }
