"""
OliveGuard API - Main Application
==================================
FastAPI entry point. Stateless, CORS-enabled, with logging middleware.

Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api import __version__
from api.routers import analyze, research

# -----------------------------------------------------------------------------
# Logging Setup (Structured, per SKILL.md)
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("oliveguard.api")


# -----------------------------------------------------------------------------
# Lifespan (Startup / Shutdown)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks. No heavy init here (lazy GEE)."""
    logger.info("OliveGuard API starting")
    yield
    logger.info("OliveGuard API shutting down")


# -----------------------------------------------------------------------------
# App Factory
# -----------------------------------------------------------------------------

app = FastAPI(
    title="OliveGuard API",
    description="AI-Powered Water Stress Detection for Olive Orchards (Ghafsai, Morocco)",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -----------------------------------------------------------------------------
# CORS (Allow Frontend)
# -----------------------------------------------------------------------------

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------

app.include_router(analyze.router)
app.include_router(research.router)


# -----------------------------------------------------------------------------
# Logging Middleware (per SKILL.md)
# -----------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        "%s %s - %d - %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        process_time,
    )
    return response


# -----------------------------------------------------------------------------
# Health & Root Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """API info and links."""
    return {
        "name": "OliveGuard API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """Health check for load balancers and Docker."""
    return {"status": "ok"}
