"""
Health Check Endpoint

Provides a simple health check to verify the API is running.
This is the first endpoint created — used to validate the scaffold works.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        JSON with status, timestamp, and version info.
    """
    return {
        "status": "healthy",
        "service": "ai-risk-manager",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_mode": "SYNTHETIC / TEST DATA",
    }
