#!/usr/bin/env python3
"""
Run the OliveGuard API server.
Usage: python run_api.py
"""
import uvicorn

from api.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "api.main:app",
        host=s.host,
        port=s.port,
        reload=s.debug,
    )
