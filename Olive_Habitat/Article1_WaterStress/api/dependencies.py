"""
OliveGuard API Dependencies
==========================
Dependency injection for FastAPI. Uses lazy imports to minimize
cold start times (serverless optimization per SKILL.md).
"""

from typing import Annotated

from fastapi import Depends

from api.config import Settings, get_settings


# -----------------------------------------------------------------------------
# Config Dependency
# -----------------------------------------------------------------------------

SettingsDep = Annotated[Settings, Depends(get_settings)]


# -----------------------------------------------------------------------------
# GEE Initialization (Lazy)
# -----------------------------------------------------------------------------

def get_gee_initialized() -> bool:
    """
    Lazy GEE initialization. Import only when needed to reduce cold start.
    Returns True if GEE is ready for use.
    """
    from src.gee_pipeline import initialize_gee

    return initialize_gee()
