"""
Analyze Router
=============
POST /analyze - Orchard water stress analysis using GEE + anomaly detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException

from api.config import get_settings
from api.dependencies import SettingsDep
from api.schemas.requests import AnalyzeRequest
from api.schemas.responses import AnalysisResult, TimeSeriesPoint
from api.services.gee_adapter import extract_ndvi_timeseries
from api.services.anomaly import compute_anomaly_from_df

logger = logging.getLogger("oliveguard.api")
router = APIRouter(prefix="/analyze", tags=["analyze"])


def _run_gee_extraction(coordinates: list):
    """Blocking GEE extraction - runs in thread pool."""
    return extract_ndvi_timeseries(coordinates)


def _mock_result() -> AnalysisResult:
    """Fallback when GEE unavailable (OLIVEGUARD_USE_MOCK_ANALYZE=true)."""
    now = datetime.now()
    ndvi_series = []
    ideal_series = []
    for i in range(12):
        d = now - timedelta(days=30 * (11 - i))
        date_str = d.strftime("%Y-%m-%d")
        ideal_val = 0.5 + 0.2 * (d.month / 12 - 0.5)
        ndvi_series.append(TimeSeriesPoint(date=date_str, value=round(ideal_val + 0.02, 3)))
        ideal_series.append(TimeSeriesPoint(date=date_str, value=round(ideal_val, 3)))
    return AnalysisResult(
        anomalyScore=0.25,
        ndviSeries=ndvi_series,
        idealSeries=ideal_series,
        stressStatus="Healthy",
        waterContent=0.8,
        dtwDistance=12.5,
    )


@router.post("", response_model=AnalysisResult)
async def analyze_orchard(
    request: AnalyzeRequest,
    settings: SettingsDep,
) -> AnalysisResult:
    """
    Analyze water stress for a drawn orchard polygon.
    Fetches Sentinel-2 + ERA5 data via GEE, computes anomaly score.
    Set OLIVEGUARD_USE_MOCK_ANALYZE=true for mock data (no GEE).
    """
    coords = request.geometry.coordinates
    if not coords or not coords[0] or len(coords[0]) < 3:
        raise HTTPException(status_code=400, detail="Invalid polygon: need at least 3 points")

    if settings.use_mock_analyze:
        return _mock_result()

    try:
        df = await asyncio.wait_for(
            asyncio.to_thread(_run_gee_extraction, coords),
            timeout=float(settings.gee_timeout),
        )
    except asyncio.TimeoutError:
        logger.warning("GEE extraction timed out for polygon")
        raise HTTPException(
            status_code=504,
            detail="Analysis timed out. Try a smaller area or try again later.",
        )
    except RuntimeError as e:
        logger.exception("GEE extraction failed")
        raise HTTPException(status_code=503, detail=str(e))

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="No satellite data found for this area. Check cloud cover or date range.",
        )

    return compute_anomaly_from_df(df)
