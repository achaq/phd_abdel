"""
Research Schemas
===============
Models for comparative study endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

from api.schemas.requests import OrchardGeometry

class CompareRequest(BaseModel):
    """Request to compare multiple models on a polygon."""
    geometry: OrchardGeometry = Field(..., description="GeoJSON polygon")
    models: List[str] = Field(
        default=["LSTM-AE", "Heuristic-NDWI", "Statistical-ZScore"],
        description="List of models to run"
    )

class ModelResult(BaseModel):
    """Result from a single model."""
    model_name: str
    anomaly_score: float
    stress_status: Literal["Healthy", "Warning", "Critical"]
    execution_time_ms: float
    details: str | None = None

class ComparisonResponse(BaseModel):
    """Comparative analysis result."""
    results: List[ModelResult]
    agreement_score: float = Field(..., description="Percentage of models that agree on the status (0-1)")
    consensus_status: str = Field(..., description="Majority vote status")
