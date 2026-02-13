"""
Request Schemas
==============
Input models for API endpoints. Matches frontend Geometry (GeoJSON Polygon).
"""

from pydantic import BaseModel, Field


class OrchardGeometry(BaseModel):
    """
    GeoJSON Polygon geometry. Matches frontend Geometry interface.
    coordinates: [[[lon, lat], [lon, lat], ...]] - outer ring of polygon
    """

    type: str = Field(default="Polygon", description="GeoJSON type")
    coordinates: list[list[list[float]]] = Field(
        ...,
        description="Polygon coordinates: [[[lon, lat], ...]]",
        min_length=1,
    )


class AnalyzeRequest(BaseModel):
    """
    Request body for POST /analyze.
    Wraps geometry to allow optional params (e.g. date range) later.
    """

    geometry: OrchardGeometry = Field(..., description="GeoJSON polygon of the orchard")
