"""
Response Schemas
===============
Output models for API endpoints. Matches frontend AnalysisResult interface.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TimeSeriesPoint(BaseModel):
    """Single point in a time series (NDVI, ideal curve, etc.)."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: float = Field(..., description="Index value (e.g. NDVI 0-1)")


class AnalysisResult(BaseModel):
    """
    Analysis result for orchard water stress. Matches frontend AnalysisResult.
    """

    anomalyScore: float = Field(
        ...,
        ge=0,
        le=1,
        description="Stress score 0-1 (higher = more stressed)",
    )
    ndviSeries: list[TimeSeriesPoint] = Field(
        ...,
        description="Actual NDVI time series",
    )
    idealSeries: list[TimeSeriesPoint] = Field(
        ...,
        description="Ideal/baseline NDVI curve",
    )
    stressStatus: Literal["Healthy", "Warning", "Critical"] = Field(
        ...,
        description="Human-readable stress level",
    )
    waterContent: float = Field(
        ...,
        ge=0,
        le=1,
        description="Water content estimate 0-1",
    )
    dtwDistance: float | None = Field(
        default=None,
        description="Dynamic Time Warping distance to baseline",
    )
