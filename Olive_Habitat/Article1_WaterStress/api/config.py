"""
OliveGuard API Configuration
============================
Central settings for the FastAPI backend. Uses environment variables
for deployment flexibility (serverless, Docker, VPS).

Follows: Stateless design, env-based config for 12-factor compliance.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


@dataclass
class Settings:
    """
    API settings loaded from environment variables.
    All values have sensible defaults for local development.
    """

    # Server
    host: str = field(default_factory=lambda: os.getenv("OLIVEGUARD_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("OLIVEGUARD_PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("OLIVEGUARD_DEBUG", "false").lower() == "true")

    # CORS - Allow frontend (Next.js typically on 3000)
    cors_origins: List[str] = field(
        default_factory=lambda: (
            os.getenv("OLIVEGUARD_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        )
    )

    # GEE
    gee_project_id: str = field(default_factory=lambda: os.getenv("OLIVEGUARD_GEE_PROJECT_ID", "phd1-481917"))
    gee_timeout: int = field(default_factory=lambda: int(os.getenv("OLIVEGUARD_GEE_TIMEOUT", "60")))
    use_mock_analyze: bool = field(
        default_factory=lambda: os.getenv("OLIVEGUARD_USE_MOCK_ANALYZE", "false").lower() == "true"
    )

    # Optional: Redis (for future caching)
    redis_url: str | None = field(default_factory=lambda: os.getenv("OLIVEGUARD_REDIS_URL") or None)

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("OLIVEGUARD_LOG_LEVEL", "INFO"))


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance. Avoids re-reading env on every request.
    """
    return Settings()
