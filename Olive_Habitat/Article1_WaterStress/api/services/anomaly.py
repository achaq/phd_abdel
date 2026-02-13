"""
Anomaly Detection Service
========================
Computes stress score, ideal curve, and DTW distance from extracted time series.
"""

import pandas as pd
from datetime import datetime
from typing import Literal

from api.schemas.responses import AnalysisResult, TimeSeriesPoint
from api.services.model_inference import get_model_service


def _ideal_ndvi(month: float) -> float:
    """Generic Mediterranean olive NDVI seasonal curve (0-indexed month)."""
    return 0.5 + 0.2 * (month / 12 - 0.5)


def compute_anomaly_from_df(df: pd.DataFrame) -> AnalysisResult:
    """
    Convert extracted DataFrame to AnalysisResult.
    Uses NDWI for water stress (lower = more stressed).
    """
    if df.empty or "NDVI" not in df.columns:
        return _empty_result()

    df = df.sort_values("date")
    dates = pd.to_datetime(df["date"])

    # Build series (sample to ~12 points for chart if too many)
    ndvi_vals = df["NDVI"].ffill().bfill().tolist()
    ndvi_series = [
        TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(float(v), 4))
        for d, v in zip(dates, ndvi_vals)
    ]

    # Ideal curve: seasonal baseline
    ideal_series = [
        TimeSeriesPoint(
            date=d.strftime("%Y-%m-%d"),
            value=round(_ideal_ndvi(d.month), 4),
        )
        for d in dates
    ]

    # Water stress from NDWI (primary) and NDVI (secondary)
    ndwi_mean = df["NDWI"].mean() if "NDWI" in df.columns else 0
    ndvi_mean = df["NDVI"].mean()

    # Heuristic Score (Baseline)
    water_content = max(0.1, min(1.0, 0.5 + ndwi_mean))
    heuristic_score = max(0, min(1, 1 - water_content))

    # AI Model Score (LSTM Autoencoder)
    try:
        model_service = get_model_service()
        ai_score = model_service.predict_anomaly(df)
    except Exception as e:
        print(f"⚠️ AI Inference failed: {e}")
        ai_score = 0.0

    # Ensemble: 70% AI, 30% Heuristic (if AI is active)
    # If AI is 0 (model missing), falls back to heuristic via max/weights
    if model_service.model is not None:
        anomaly_score = (ai_score * 0.7) + (heuristic_score * 0.3)
    else:
        anomaly_score = heuristic_score

    # DTW distance (simplified: mean absolute deviation from ideal)
    if len(ndvi_vals) == len(ideal_series):
        dtw_distance = sum(
            abs(a - b) for a, b in zip(ndvi_vals, [p.value for p in ideal_series])
        ) / max(len(ndvi_vals), 1) * 100
    else:
        dtw_distance = abs(ndvi_mean - 0.5) * 50

    # Stress status
    if anomaly_score < 0.4:
        stress_status: Literal["Healthy", "Warning", "Critical"] = "Healthy"
    elif anomaly_score < 0.7:
        stress_status = "Warning"
    else:
        stress_status = "Critical"

    return AnalysisResult(
        anomalyScore=round(anomaly_score, 4),
        ndviSeries=ndvi_series,
        idealSeries=ideal_series,
        stressStatus=stress_status,
        waterContent=round(water_content, 4),
        dtwDistance=round(dtw_distance, 2),
    )


def _empty_result() -> AnalysisResult:
    """Return empty result when no data."""
    return AnalysisResult(
        anomalyScore=0.5,
        ndviSeries=[],
        idealSeries=[],
        stressStatus="Warning",
        waterContent=0.5,
        dtwDistance=None,
    )
