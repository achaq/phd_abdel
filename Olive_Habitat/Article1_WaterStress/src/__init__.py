"""
OliveGuard: AI-Powered Digital Twin for Water Stress Detection
===============================================================

A comprehensive system for detecting water stress in olive orchards
using multi-sensor satellite data and machine learning.

Modules:
--------
- config: Central configuration settings
- gee_pipeline: Google Earth Engine data extraction
- feature_engineering: Feature processing and creation
- train_model: Classical ML model training
- train_deep_model: Deep learning model training (Week 5)
- explainability: SHAP-based model interpretation (Week 6)

Author: OliveGuard Research Team
Project: AI-Based Olive Stress Detection (Ghafsai, Morocco)
"""

__version__ = "0.1.0"
__author__ = "OliveGuard Research Team"

# Import main components for easy access
from .config import (
    gee_config,
    study_area_config,
    indices_config,
    ml_config,
    output_config,
    api_config,
    print_config_summary
)

# GEE Pipeline functions will be available after Day 1 implementation
try:
    from .gee_pipeline import (
        initialize_gee,
        get_study_area,
        mask_clouds_qa60,
        compute_all_indices,
        extract_training_data,
        visualize_study_area
    )
except ImportError:
    pass  # Will be available after gee_pipeline.py is created

__all__ = [
    # Configuration
    'gee_config',
    'study_area_config', 
    'indices_config',
    'ml_config',
    'output_config',
    'api_config',
    'print_config_summary',
    
    # GEE Pipeline
    'initialize_gee',
    'get_study_area',
    'mask_clouds_qa60',
    'compute_all_indices',
    'extract_training_data',
    'visualize_study_area',
]
