"""
GEE Adapter Service
==================
Wraps src/gee_pipeline for dynamic polygons. Converts GeoJSON to ee.Geometry
and runs lightweight extraction for the API.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any

# Lazy import to reduce cold start (GEE is heavy)
def _get_ee():
    import ee
    return ee

def _get_pipeline():
    from src.gee_pipeline import (
        initialize_gee,
        get_sentinel2_timeseries,
        get_era5_timeseries,
    )
    from src.config import gee_config
    return initialize_gee, get_sentinel2_timeseries, get_era5_timeseries, gee_config


def geo_json_to_ee_geometry(coordinates: list[list[list[float]]]) -> Any:
    """
    Convert GeoJSON polygon coordinates to ee.Geometry.Polygon.
    coordinates: [[[lon, lat], [lon, lat], ...]] - outer ring
    """
    ee = _get_ee()
    # GeoJSON uses [lon, lat]; ee.Geometry.Polygon expects list of [lon, lat]
    ring = coordinates[0]
    return ee.Geometry.Polygon(ring)


def extract_ndvi_timeseries(
    coordinates: list[list[list[float]]],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Extract NDVI/NDWI time series for a polygon. Runs GEE extraction.
    Returns DataFrame with columns: date, NDVI, NDWI, Temperature_C, Precipitation_mm
    """
    ee = _get_ee()
    init_gee, get_s2, get_era5, gee_config = _get_pipeline()

    if not init_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    roi = geo_json_to_ee_geometry(coordinates)

    if start_date is None:
        end = datetime.now()
        start = end - timedelta(days=365)
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
    elif end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Get Sentinel-2 (NDVI, NDWI)
    s2_collection = get_s2(roi, start_date, end_date, apply_veg_mask=True)

    def collection_to_df(collection, properties: list[str]) -> pd.DataFrame:
        try:
            data = collection.reduceColumns(
                ee.Reducer.toList(len(properties)),
                properties,
            ).get("list").getInfo()
            return pd.DataFrame(data, columns=properties) if data else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"GEE extraction failed: {e}") from e

    s2_props = ["date", "NDVI", "NDWI", "SAVI", "EVI"]
    df_s2 = collection_to_df(s2_collection, s2_props)

    if df_s2.empty:
        return df_s2

    # Get ERA5 for context
    era5_collection = get_era5(roi, start_date, end_date)
    era5_props = ["date", "Temperature_C", "Precipitation_mm"]
    df_era5 = collection_to_df(era5_collection, era5_props)

    # Merge
    df_s2["date"] = pd.to_datetime(df_s2["date"])
    df_era5["date"] = pd.to_datetime(df_era5["date"])
    df_s2.set_index("date", inplace=True)
    df_era5.set_index("date", inplace=True)
    df_s2_daily = df_s2.resample("D").mean().interpolate(method="linear")
    df_merged = df_era5.join(df_s2_daily, how="inner")
    df_merged = df_merged.dropna(subset=["NDVI", "NDWI"])
    df_merged.reset_index(inplace=True)

    return df_merged
