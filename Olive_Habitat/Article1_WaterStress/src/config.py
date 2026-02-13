"""
OliveGuard Configuration Module
===============================
Central configuration for the OliveGuard water stress detection system.

Author: Abdelkarim ACHAQ
Project: AI-Based Olive Stress Detection (Ghafsai, Morocco)
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base directory (relative to workspace root)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
PLOTS_DIR = NOTEBOOKS_DIR / "plots"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GOOGLE EARTH ENGINE CONFIGURATION
# =============================================================================

@dataclass
class GEEConfig:
    """Google Earth Engine configuration settings."""
    
    # GEE Project ID
    project_id: str = "phd1-481917"
    
    # Date range for data extraction (last 3 years)
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    
    # Sentinel-2 settings
    s2_collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    s2_cloud_threshold: int = 20  # Scene-level cloud percentage threshold
    s2_cloudless_threshold: int = 40  # Pixel-level cloud probability threshold
    
    # Sentinel-1 settings
    s1_collection: str = "COPERNICUS/S1_GRD"
    s1_instrument_mode: str = "IW"  # Interferometric Wide swath
    s1_polarization: List[str] = field(default_factory=lambda: ["VV", "VH"])
    s1_orbit_pass: str = "DESCENDING"  # Morning pass for diurnal analysis
    s1_resolution: int = 10  # meters
    
    # Landsat 8/9 settings (for thermal - Phase 5)
    landsat_collection: str = "LANDSAT/LC08/C02/T1_L2"
    landsat_cloud_threshold: int = 20
    
    # ERA5-Land settings
    era5_collection: str = "ECMWF/ERA5_LAND/DAILY_AGGR"
    
    # s2cloudless collection
    s2_cloudless_collection: str = "COPERNICUS/S2_CLOUD_PROBABILITY"
    
    # Processing settings
    scale: int = 20  # Output resolution in meters
    max_pixels: int = 1e9


# =============================================================================
# STUDY AREA CONFIGURATION (GHAFSAI, PRE-RIF)
# =============================================================================

@dataclass
class StudyAreaConfig:
    """Configuration for the Ghafsai study area."""
    
    # Study area name
    name: str = "Ghafsai"
    region: str = "Pre-Rif"
    country: str = "Morocco"
    
    # Bounding box coordinates [lon, lat]
    # Format: [[west, south], [east, south], [east, north], [west, north], [west, south]]
    polygon_coords: List[List[float]] = field(default_factory=lambda: [
        [-4.947147546412382, 34.620655908492395],
        [-4.689655480982695, 34.620655908492395],
        [-4.689655480982695, 34.72964124033107],
        [-4.947147546412382, 34.72964124033107],
        [-4.947147546412382, 34.620655908492395]  # Closing the polygon
    ])
    
    # Center point for map visualization
    center_lat: float = 34.675
    center_lon: float = -4.818
    
    # Map zoom level
    zoom: int = 12


# =============================================================================
# VEGETATION INDICES CONFIGURATION
# =============================================================================

@dataclass
class IndicesConfig:
    """Configuration for vegetation and water indices."""
    
    # NDVI: Normalized Difference Vegetation Index
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi_bands: Tuple[str, str] = ("B8", "B4")
    
    # NDWI: Normalized Difference Water Index (Gao's formulation)
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi_bands: Tuple[str, str] = ("B3", "B8")
    
    # SAVI: Soil-Adjusted Vegetation Index
    # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    savi_bands: Tuple[str, str] = ("B8", "B4")
    savi_l: float = 0.5  # Soil brightness correction factor
    
    # EVI: Enhanced Vegetation Index
    # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    evi_bands: Tuple[str, str, str] = ("B8", "B4", "B2")
    evi_g: float = 2.5
    evi_c1: float = 6.0
    evi_c2: float = 7.5
    evi_l: float = 1.0
    
    # Vegetation mask threshold
    # Pixels with NDVI < this value are considered non-vegetated (soil/rock)
    vegetation_mask_threshold: float = 0.2


# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    
    # Training data
    n_samples: int = 1000  # Number of random sample points
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    
    # Time series settings for LSTM
    sequence_length: int = 30  # Days
    
    # Features to use
    optical_features: List[str] = field(default_factory=lambda: [
        "B2", "B3", "B4", "B8", "NDVI", "NDWI", "SAVI", "EVI"
    ])
    radar_features: List[str] = field(default_factory=lambda: ["VV", "VH"])
    climate_features: List[str] = field(default_factory=lambda: [
        "Temperature_C", "Precipitation_mm"
    ])
    
    # Stress label definition
    # Stress = 1 if (NDWI < mean - n_std * std) AND (Temp > temp_threshold)
    # OR if NDWI is very low regardless of temperature
    stress_ndwi_std_multiplier: float = 0.5  # More sensitive (was 1.0)
    stress_temp_threshold: float = 25.0  # Lower threshold for Mediterranean (was 30.0)
    stress_ndwi_absolute_threshold: float = -0.55  # Absolute minimum NDWI
    
    # Multi-class stress levels (Phase 5)
    stress_levels: Dict[int, str] = field(default_factory=lambda: {
        0: "Healthy",
        1: "Mild Stress",
        2: "Moderate Stress",
        3: "Severe Stress"
    })
    
    # Model paths
    rf_model_path: str = str(MODELS_DIR / "rf_stress_model.pkl")
    xgb_model_path: str = str(MODELS_DIR / "xgb_stress_model.pkl")
    cnn_lstm_model_path: str = str(MODELS_DIR / "cnn_lstm_stress.h5")


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

@dataclass
class OutputConfig:
    """Configuration for output files."""
    
    # Training data CSV
    training_data_csv: str = str(DATA_DIR / "ghafsai_training_data.csv")
    
    # Comparative Study Datasets (Phase 3)
    training_data_climate_driven_csv: str = str(DATA_DIR / "training_data_climate_driven.csv")
    training_data_data_driven_csv: str = str(DATA_DIR / "training_data_data_driven.csv")
    
    # Processed features CSV
    processed_features_csv: str = str(DATA_DIR / "processed_features.csv")
    
    # SHAP plots
    shap_summary_rf: str = str(PLOTS_DIR / "shap_summary_rf.png")
    shap_summary_cnn: str = str(PLOTS_DIR / "shap_summary_cnn.png")
    
    # Feature importance
    feature_importance_plot: str = str(PLOTS_DIR / "feature_importance.png")
    
    # Model comparison
    model_comparison_plot: str = str(PLOTS_DIR / "model_comparison.png")
    
    # Stress timeline
    stress_timeline_plot: str = str(PLOTS_DIR / "stress_timeline.png")


# =============================================================================
# API CONFIGURATION (Week 2)
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for the FastAPI backend."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Redis cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 86400  # 24 hours in seconds
    
    # GEE query timeout
    gee_timeout: int = 60  # seconds


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Create default configuration instances
gee_config = GEEConfig()
study_area_config = StudyAreaConfig()
indices_config = IndicesConfig()
ml_config = MLConfig()
output_config = OutputConfig()
api_config = APIConfig()


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("OLIVEGUARD CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\n📍 Study Area: {study_area_config.name}, {study_area_config.region}, {study_area_config.country}")
    print(f"   Center: ({study_area_config.center_lat:.4f}, {study_area_config.center_lon:.4f})")
    print(f"\n🛰️  Data Sources:")
    print(f"   - Sentinel-2: {gee_config.s2_collection}")
    print(f"   - Sentinel-1: {gee_config.s1_collection}")
    print(f"   - ERA5-Land: {gee_config.era5_collection}")
    print(f"   - Date Range: {gee_config.start_date} to {gee_config.end_date}")
    print(f"\n📊 Indices: NDVI, NDWI, SAVI (L={indices_config.savi_l}), EVI")
    print(f"   Vegetation Mask: NDVI > {indices_config.vegetation_mask_threshold}")
    print(f"\n🤖 ML Settings:")
    print(f"   - Training Samples: {ml_config.n_samples}")
    print(f"   - Sequence Length: {ml_config.sequence_length} days")
    print(f"   - Features: {len(ml_config.optical_features) + len(ml_config.radar_features) + len(ml_config.climate_features)} total")
    print(f"\n📁 Output Paths:")
    print(f"   - Training Data: {output_config.training_data_csv}")
    print(f"   - Models: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
