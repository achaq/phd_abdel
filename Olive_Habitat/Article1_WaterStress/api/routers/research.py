"""
Research Router
==============
Endpoints for academic comparative analysis.
Runs multiple models side-by-side to validate the proposed LSTM approach.
"""

import time
import asyncio
import numpy as np
import pandas as pd
from collections import Counter
from fastapi import APIRouter, HTTPException, Depends

from api.schemas.research import CompareRequest, ComparisonResponse, ModelResult
from api.dependencies import SettingsDep
from api.services.gee_adapter import extract_ndvi_timeseries
from api.services.model_inference import get_model_service
from api.services.anomaly import compute_anomaly_from_df

router = APIRouter(prefix="/research", tags=["research"])

def _run_gee_extraction(coordinates: list):
    """Blocking GEE extraction."""
    return extract_ndvi_timeseries(coordinates)

def _get_status(score: float) -> str:
    if score < 0.4: return "Healthy"
    if score < 0.7: return "Warning"
    return "Critical"

@router.post("/compare", response_model=ComparisonResponse)
async def compare_models(
    request: CompareRequest,
    settings: SettingsDep,
) -> ComparisonResponse:
    """
    Run comparative analysis: Proposed LSTM vs Baselines.
    """
    coords = request.geometry.coordinates
    if not coords or not coords[0] or len(coords[0]) < 3:
        raise HTTPException(status_code=400, detail="Invalid polygon")

    # 1. Fetch Data (Once)
    start_time = time.time()
    try:
        if settings.use_mock_analyze:
            # Create dummy DF
            dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
            df = pd.DataFrame({
                'date': dates,
                'NDVI': np.random.uniform(0.3, 0.8, 30),
                'NDWI': np.random.uniform(-0.2, 0.1, 30),
                'Temperature_C': np.random.uniform(15, 35, 30),
                'Precipitation_mm': np.random.uniform(0, 5, 30)
            })
        else:
            df = await asyncio.wait_for(
                asyncio.to_thread(_run_gee_extraction, coords),
                timeout=float(settings.gee_timeout),
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data extraction failed: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=422, detail="No data found")

    results = []

    # 2. Run Models
    
    # Model A: Proposed LSTM Autoencoder
    if "LSTM-AE" in request.models:
        t0 = time.time()
        try:
            ai_service = get_model_service()
            score = ai_service.predict_anomaly(df)
            status = _get_status(score)
            details = "Reconstruction Error (MSE)"
        except Exception as e:
            score = 0.0
            status = "Error"
            details = str(e)
        
        results.append(ModelResult(
            model_name="LSTM Autoencoder (Proposed)",
            anomaly_score=round(score, 4),
            stress_status=status,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
            details=details
        ))

    # Model B: Heuristic (NDWI Threshold)
    if "Heuristic-NDWI" in request.models:
        t0 = time.time()
        ndwi_mean = df["NDWI"].mean() if "NDWI" in df.columns else 0
        # Simple inversion: -0.5 is stressed (1.0), 0.2 is healthy (0.0)
        # 0.2 - (-0.5) = 0.7 range
        score = max(0, min(1, (0.2 - ndwi_mean) / 0.7))
        results.append(ModelResult(
            model_name="Heuristic (NDWI Baseline)",
            anomaly_score=round(score, 4),
            stress_status=_get_status(score),
            execution_time_ms=round((time.time() - t0) * 1000, 2),
            details=f"Mean NDWI: {ndwi_mean:.3f}"
        ))

    # Model C: Statistical Z-Score
    if "Statistical-ZScore" in request.models:
        t0 = time.time()
        ndvi = df["NDVI"].fillna(0)
        mean = ndvi.mean()
        std = ndvi.std() + 1e-6
        # Last value z-score
        last_val = ndvi.iloc[-1]
        z_score = (mean - last_val) / std # Positive if dropped below mean
        # Sigmoid to 0-1
        score = 1 / (1 + np.exp(-(z_score - 2))) # Shift so z=2 -> 0.5
        results.append(ModelResult(
            model_name="Statistical Z-Score",
            anomaly_score=round(score, 4),
            stress_status=_get_status(score),
            execution_time_ms=round((time.time() - t0) * 1000, 2),
            details=f"Z-Score: {z_score:.2f}"
        ))

    # 3. Compute Consensus
    statuses = [r.stress_status for r in results if r.stress_status != "Error"]
    if not statuses:
        consensus = "Unknown"
        agreement = 0.0
    else:
        counts = Counter(statuses)
        consensus = counts.most_common(1)[0][0]
        agreement = counts[consensus] / len(statuses)

    return ComparisonResponse(
        results=results,
        agreement_score=round(agreement, 2),
        consensus_status=consensus
    )
