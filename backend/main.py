"""
AI Risk Manager — FastAPI Application Entry Point
==================================================

This is the main application file. It creates the FastAPI app,
initializes shared application services (such as the single-process
in-memory AuditTrailManager), and registers all API route modules.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from audit_trail import AuditTrailManager
from api.routes import health
import windows
import pipeline
import evaluation
import audit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs on startup and shutdown."""
    print(f"🚀 AI Risk Manager starting in {settings.APP_ENV} mode")
    print(f"📊 Database: {settings.DATABASE_URL}")
    print("⚠️  Using SYNTHETIC / TEST DATA only")

    # Initialize shared in-memory AuditTrailManager instance
    app.state.audit_manager = AuditTrailManager()

    yield

    print("👋 AI Risk Manager shutting down")


# Create the FastAPI application
app: FastAPI = FastAPI(
    title="AI Risk Manager",
    description=(
        "Fraud-Spike Detection System — Detects unusual increases in "
        "suspicious transaction activity for merchants. "
        "STRICTLY DEFENSE-ONLY. Uses SYNTHETIC / TEST DATA only."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0].get("msg", "Invalid request parameter.") if errors else "Invalid request parameter."
    # Clean up error message prefix if from pydantic custom validator
    if "Value error, " in msg:
        msg = msg.replace("Value error, ", "")
    return JSONResponse(
        status_code=400,
        content={"detail": msg},
    )

# CORS middleware — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router)
app.include_router(windows.router)
app.include_router(pipeline.router)
app.include_router(evaluation.router)
app.include_router(audit.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.APP_PORT, reload=True)
