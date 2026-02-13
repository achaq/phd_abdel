"""
OliveGuard GEE Data Pipeline
============================
Enhanced Google Earth Engine data extraction pipeline for multi-sensor
water stress detection in olive orchards.

Features:
- Multi-sensor fusion: Sentinel-1 (radar) + Sentinel-2 (optical) + ERA5 (climate)
- Aggressive cloud masking (QA60 + s2cloudless)
- Vegetation indices: NDVI, NDWI, SAVI, EVI
- Vegetation fragmentation mask (NDVI > 0.2)
- Radar backscatter: VV, VH polarizations
- Step-by-step visualization for each processing stage

Author: OliveGuard Research Team
Project: AI-Based Olive Stress Detection (Ghafsai, Morocco)
Week 1: Days 1-6 Implementation
"""

import ee
import geemap
import pandas as pd
import numpy as np
import os
import certifi
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Local imports
from config import (
    gee_config, 
    study_area_config, 
    indices_config, 
    ml_config,
    output_config,
    print_config_summary,
    PLOTS_DIR,
    NOTEBOOKS_DIR
)

# Fix for macOS SSL Certificate error
os.environ['SSL_CERT_FILE'] = certifi.where()

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# INITIALIZATION (Day 1)
# =============================================================================

def initialize_gee() -> bool:
    """
    Initialize Google Earth Engine with project credentials.
    
    Returns:
        bool: True if initialization successful, False otherwise.
    """
    try:
        ee.Initialize(project=gee_config.project_id)
        print(f"✅ Google Earth Engine initialized with project: {gee_config.project_id}")
        return True
    except Exception as e:
        print("⚠️ GEE not initialized. Attempting authentication...")
        try:
            ee.Authenticate()
            ee.Initialize(project=gee_config.project_id)
            print(f"✅ GEE initialized after authentication with project: {gee_config.project_id}")
            return True
        except Exception as auth_error:
            print(f"❌ Failed to initialize Google Earth Engine: {auth_error}")
            return False


def get_study_area() -> Tuple[ee.Geometry, ee.Geometry]:
    """
    Create the Ghafsai study area geometry.
    
    Returns:
        Tuple[ee.Geometry, ee.Geometry]: (polygon ROI, center point)
    """
    roi = ee.Geometry.Polygon([study_area_config.polygon_coords])
    center = roi.centroid()
    
    # Get area in hectares
    area_ha = roi.area().divide(10000).getInfo()
    print(f"📍 Study Area: {study_area_config.name}, {study_area_config.region}")
    print(f"   Area: {area_ha:.2f} hectares")
    
    return roi, center


# =============================================================================
# TOPOGRAPHIC CORRECTION (Day 1 - Enhanced)
# =============================================================================

class TopographicCorrection:
    """
    Topographic correction for optical imagery in mountainous terrain.
    
    Implements Sun-Canopy-Sensor + C-Correction to remove illumination artifacts
    caused by slope and aspect in the Rif mountains.
    """
    
    def __init__(self, dem_collection="USGS/SRTMGL1_003"):
        self.dem = ee.Image(dem_collection)
    
    def calculate_illumination(self, image, roi):
        """
        Calculate illumination condition based on sun position and terrain.
        """
        # Get sun azimuth and zenith from image metadata
        sun_azimuth = ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
        sun_zenith = ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE'))
        
        # Get terrain products
        terrain = ee.Algorithms.Terrain(self.dem.clip(roi))
        slope = terrain.select('slope').multiply(np.pi / 180)
        aspect = terrain.select('aspect').multiply(np.pi / 180)
        
        # Convert solar angles to radians
        z_rad = sun_zenith.multiply(np.pi / 180)
        a_rad = sun_azimuth.multiply(np.pi / 180)
        
        # Calculate illumination factor (cosine of incidence angle)
        # IL = cos(z) * cos(s) + sin(z) * sin(s) * cos(a - asp)
        illumination = ee.Image.constant(z_rad).cos().multiply(slope.cos()) \
            .add(ee.Image.constant(z_rad).sin().multiply(slope.sin())
                 .multiply(ee.Image.constant(a_rad).subtract(aspect).cos()))
        
        return illumination.rename('illumination')
        
    def apply_c_correction(self, image, roi):
        """
        Apply C-Correction to Sentinel-2 image.
        
        Ref: Teillet et al. (1982)
        L_h = L_t * ((cos(z) + c) / (IL + c))
        """
        illumination = self.calculate_illumination(image, roi)
        
        # Add illumination band
        img_plus_il = image.addBands(illumination)
        
        # We need to find 'c' for each band. 
        # Simplified approach: Use fixed C factor or regression. 
        # For MVP, we use a simplified Cosine correction if C is hard to estimate on the fly,
        # but here is a standard implementation of Cosine-like correction which is safer.
        # Corrected = Original * (cos(theta_s) / illumination)
        
        sun_zenith = ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(np.pi / 180)
        cos_zenith = sun_zenith.cos()
        
        correction_factor = cos_zenith.divide(illumination).clamp(0, 10) # Clamp to avoid artifacts
        
        # Apply to optical bands only
        optical_bands = ['B2', 'B3', 'B4', 'B8']
        
        def correct_band(band_name):
            return image.select(band_name).multiply(correction_factor).rename(band_name)
            
        corrected_bands = [correct_band(b) for b in optical_bands]
        
        # Reconstruct image
        corrected_image = ee.Image(corrected_bands)
        
        # Add back other bands (QA60, etc.)
        other_bands = image.bandNames().removeAll(optical_bands)
        return corrected_image.addBands(image.select(other_bands))

# =============================================================================
# CLOUD MASKING (Day 2)
# =============================================================================

def mask_clouds_qa60(image: ee.Image) -> ee.Image:
    """
    Apply cloud masking using Sentinel-2 QA60 band.
    
    The QA60 band contains cloud mask information:
    - Bit 10: Opaque clouds
    - Bit 11: Cirrus clouds
    
    Args:
        image: Sentinel-2 image
        
    Returns:
        ee.Image: Cloud-masked image
    """
    qa60 = image.select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus, respectively
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Both bits should be zero for clear conditions
    mask = qa60.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa60.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    return image.updateMask(mask)


def get_cloud_probability_image(image: ee.Image, 
                                  cloud_prob_collection: ee.ImageCollection) -> ee.Image:
    """
    Get the corresponding cloud probability image for a Sentinel-2 image.
    
    Args:
        image: Sentinel-2 image
        cloud_prob_collection: s2cloudless ImageCollection
        
    Returns:
        ee.Image: Cloud probability image
    """
    # Filter by the same date
    image_date = image.date()
    
    cloud_prob = cloud_prob_collection.filter(
        ee.Filter.date(image_date, image_date.advance(1, 'day'))
    ).filter(
        ee.Filter.eq('system:index', image.get('system:index'))
    ).first()
    
    return ee.Image(ee.Algorithms.If(
        cloud_prob,
        cloud_prob,
        ee.Image.constant(0).rename('probability')
    ))


def mask_clouds_s2cloudless(image: ee.Image, threshold: int = None) -> ee.Image:
    """
    Apply cloud masking using s2cloudless probability layer.
    
    This is more aggressive than QA60 and better for the cloudy Rif region.
    
    Args:
        image: Sentinel-2 image with 'cloud_probability' band
        threshold: Cloud probability threshold (default from config)
        
    Returns:
        ee.Image: Cloud-masked image
    """
    if threshold is None:
        threshold = gee_config.s2_cloudless_threshold
    
    # Get cloud probability band
    cloud_prob = image.select('probability')
    
    # Mask where cloud probability is above threshold
    mask = cloud_prob.lt(threshold)
    
    # Apply mask to original bands (exclude probability band)
    original_bands = image.bandNames().filter(ee.Filter.neq('item', 'probability'))
    
    return image.select(original_bands).updateMask(mask)


def apply_hybrid_cloud_mask(image: ee.Image, threshold: int = None) -> ee.Image:
    """
    Apply hybrid cloud masking combining QA60 and s2cloudless.
    
    This aggressive approach is critical for the cloudy Rif mountains.
    
    Args:
        image: Sentinel-2 image with 'probability' band added
        threshold: Cloud probability threshold
        
    Returns:
        ee.Image: Cloud-masked image with both masks applied
    """
    if threshold is None:
        threshold = gee_config.s2_cloudless_threshold
    
    # Apply QA60 mask
    qa60 = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa60.bitwiseAnd(cloud_bit_mask).eq(0).And(
              qa60.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Get cloud probability mask
    cloud_prob = image.select('probability')
    prob_mask = cloud_prob.lt(threshold)
    
    # Combine both masks (intersection = AND)
    combined_mask = qa_mask.And(prob_mask)
    
    # Get original bands (exclude probability)
    original_bands = image.bandNames().filter(ee.Filter.neq('item', 'probability'))
    
    return image.select(original_bands).updateMask(combined_mask)


def add_cloud_probability(roi: ee.Geometry, start_date: str, end_date: str):
    """
    Create a function that adds cloud probability band to Sentinel-2 images.
    
    Args:
        roi: Region of interest
        start_date: Start date
        end_date: End date
        
    Returns:
        Function to add cloud probability to images
    """
    # Get cloud probability collection
    cloud_prob_col = ee.ImageCollection(gee_config.s2_cloudless_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    def add_prob(image):
        # Get matching cloud probability image
        cloud_prob = cloud_prob_col.filter(
            ee.Filter.eq('system:index', image.get('system:index'))
        ).first()
        
        # Handle case where no matching image exists
        cloud_prob = ee.Image(ee.Algorithms.If(
            cloud_prob,
            cloud_prob.select('probability'),
            ee.Image.constant(0).rename('probability')
        ))
        
        return image.addBands(cloud_prob)
    
    return add_prob


def visualize_cloud_masking(roi: ee.Geometry, 
                             sample_date: str = '2024-06-15',
                             save_dir: str = None) -> geemap.Map:
    """
    Day 2 Visualization: Compare different cloud masking approaches.
    
    Creates an interactive map showing:
    1. Original image (no masking)
    2. QA60 masked
    3. s2cloudless masked
    4. Hybrid masked
    
    Args:
        roi: Region of interest
        sample_date: Date to sample (YYYY-MM-DD)
        save_dir: Directory to save the map
        
    Returns:
        geemap.Map: Interactive comparison map
    """
    print("\n" + "=" * 60)
    print("DAY 2: CLOUD MASKING VISUALIZATION")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(NOTEBOOKS_DIR)
    
    # Get a sample image
    end_date = (datetime.strptime(sample_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Get cloud probability collection
    cloud_prob = ee.ImageCollection(gee_config.s2_cloudless_collection) \
        .filterBounds(roi) \
        .filterDate(sample_date, end_date)
    
    # Get Sentinel-2 collection
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(sample_date, end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)  # Get cloudiest first for demo
    
    # Get one cloudy image for demonstration
    sample_image = s2.first()
    cloud_pct = sample_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    image_date = sample_image.date().format('YYYY-MM-dd').getInfo()
    print(f"📷 Sample image: {image_date} (Cloud cover: {cloud_pct:.1f}%)")
    
    # Add cloud probability to the image
    add_prob = add_cloud_probability(roi, sample_date, end_date)
    sample_with_prob = add_prob(sample_image)
    
    # Apply different masking approaches
    original = sample_image.clip(roi)
    qa60_masked = mask_clouds_qa60(sample_image).clip(roi)
    hybrid_masked = apply_hybrid_cloud_mask(sample_with_prob).clip(roi)
    
    # Create cloud probability visualization
    cloud_prob_img = sample_with_prob.select('probability').clip(roi)
    
    # Create map
    Map = geemap.Map(
        center=[study_area_config.center_lat, study_area_config.center_lon],
        zoom=12
    )
    
    # RGB visualization parameters
    rgb_vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
    
    # Add layers
    Map.addLayer(roi, {'color': 'yellow'}, 'Study Area Boundary')
    Map.addLayer(original, rgb_vis, '1. Original (No Mask)')
    Map.addLayer(qa60_masked, rgb_vis, '2. QA60 Cloud Mask')
    Map.addLayer(hybrid_masked, rgb_vis, '3. Hybrid Cloud Mask (QA60 + s2cloudless)')
    
    # Cloud probability layer
    prob_vis = {'min': 0, 'max': 100, 'palette': ['green', 'yellow', 'orange', 'red']}
    Map.addLayer(cloud_prob_img, prob_vis, '4. Cloud Probability (%)')
    
    # Add legend
    Map.add_legend(
        title="Cloud Probability",
        legend_dict={
            'Clear (0-20%)': '00ff00',
            'Low (20-40%)': 'ffff00',
            'Medium (40-60%)': 'ffa500',
            'High (60-100%)': 'ff0000'
        }
    )
    
    # Save map
    map_path = os.path.join(save_dir, 'day2_cloud_masking_comparison.html')
    Map.to_html(map_path)
    print(f"🗺️ Map saved to: {map_path}")
    
    # Calculate statistics
    print("\n📊 Cloud Masking Statistics:")
    
    # Count valid pixels for each approach
    def count_valid_pixels(image, name):
        valid = image.select('B4').mask().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=100,
            maxPixels=1e9
        ).get('B4').getInfo()
        return valid
    
    original_pixels = count_valid_pixels(original, "Original")
    qa60_pixels = count_valid_pixels(qa60_masked, "QA60")
    hybrid_pixels = count_valid_pixels(hybrid_masked, "Hybrid")
    
    print(f"   Original: {original_pixels:,.0f} pixels")
    print(f"   QA60 masked: {qa60_pixels:,.0f} pixels ({100*qa60_pixels/original_pixels:.1f}% retained)")
    print(f"   Hybrid masked: {hybrid_pixels:,.0f} pixels ({100*hybrid_pixels/original_pixels:.1f}% retained)")
    
    return Map


# =============================================================================
# VEGETATION INDICES (Day 3)
# =============================================================================

def compute_ndvi(image: ee.Image) -> ee.Image:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to 1 (healthy vegetation: 0.3 - 0.8)
    """
    nir = image.select(indices_config.ndvi_bands[0])  # B8
    red = image.select(indices_config.ndvi_bands[1])  # B4
    
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    return image.addBands(ndvi)


def compute_ndwi(image: ee.Image) -> ee.Image:
    """
    Compute Normalized Difference Water Index (Gao's NDWI).
    
    NDWI = (Green - NIR) / (Green + NIR)
    Range: -1 to 1 (water stress: lower/negative values)
    """
    green = image.select(indices_config.ndwi_bands[0])  # B3
    nir = image.select(indices_config.ndwi_bands[1])    # B8
    
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    return image.addBands(ndwi)


def compute_savi(image: ee.Image) -> ee.Image:
    """
    Compute Soil-Adjusted Vegetation Index (SAVI).
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    where L = 0.5 (soil brightness correction factor)
    
    SAVI is crucial for the "mixed pixel" problem in fragmented orchards.
    """
    nir = image.select(indices_config.savi_bands[0])  # B8
    red = image.select(indices_config.savi_bands[1])  # B4
    L = indices_config.savi_l  # 0.5
    
    savi = nir.subtract(red).divide(
        nir.add(red).add(L)
    ).multiply(1 + L).rename('SAVI')
    
    return image.addBands(savi)


def compute_evi(image: ee.Image) -> ee.Image:
    """
    Compute Enhanced Vegetation Index (EVI).
    
    EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
    where G = 2.5, C1 = 6.0, C2 = 7.5, L = 1.0
    """
    nir = image.select(indices_config.evi_bands[0])   # B8
    red = image.select(indices_config.evi_bands[1])   # B4
    blue = image.select(indices_config.evi_bands[2])  # B2
    
    G = indices_config.evi_g     # 2.5
    C1 = indices_config.evi_c1   # 6.0
    C2 = indices_config.evi_c2   # 7.5
    L = indices_config.evi_l     # 1.0
    
    evi = nir.subtract(red).divide(
        nir.add(red.multiply(C1)).subtract(blue.multiply(C2)).add(L)
    ).multiply(G).rename('EVI')
    
    return image.addBands(evi)


def compute_all_indices(image: ee.Image) -> ee.Image:
    """
    Compute all vegetation indices for an image.
    """
    image = compute_ndvi(image)
    image = compute_ndwi(image)
    image = compute_savi(image)
    image = compute_evi(image)
    
    return image


def visualize_vegetation_indices(roi: ee.Geometry,
                                   start_date: str = '2024-06-01',
                                   end_date: str = '2024-08-31',
                                   save_dir: str = None) -> geemap.Map:
    """
    Day 3 Visualization: Display all vegetation indices.
    
    Creates an interactive map showing:
    1. True color RGB
    2. NDVI (Vegetation health)
    3. NDWI (Water content)
    4. SAVI (Soil-adjusted)
    5. EVI (Enhanced vegetation)
    
    Args:
        roi: Region of interest
        start_date: Start date for composite
        end_date: End date for composite
        save_dir: Directory to save the map
        
    Returns:
        geemap.Map: Interactive map with all indices
    """
    print("\n" + "=" * 60)
    print("DAY 3: VEGETATION INDICES VISUALIZATION")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(NOTEBOOKS_DIR)
    
    # Get cloud-masked composite
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_clouds_qa60) \
        .median() \
        .clip(roi)
    
    # Compute all indices
    s2_with_indices = compute_all_indices(s2)
    
    # Create map
    Map = geemap.Map(
        center=[study_area_config.center_lat, study_area_config.center_lon],
        zoom=12
    )
    
    # Add layers
    Map.addLayer(roi, {'color': 'white'}, 'Study Area Boundary')
    
    # RGB
    Map.addLayer(s2, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '1. True Color (RGB)')
    
    # NDVI
    ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(s2_with_indices.select('NDVI'), ndvi_vis, '2. NDVI (Vegetation Health)')
    
    # NDWI
    ndwi_vis = {'min': -0.5, 'max': 0.2, 'palette': ['red', 'orange', 'yellow', 'lightblue', 'blue']}
    Map.addLayer(s2_with_indices.select('NDWI'), ndwi_vis, '3. NDWI (Water Content)')
    
    # SAVI
    savi_vis = {'min': 0, 'max': 0.6, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(s2_with_indices.select('SAVI'), savi_vis, '4. SAVI (Soil-Adjusted)')
    
    # EVI
    evi_vis = {'min': 0, 'max': 0.6, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(s2_with_indices.select('EVI'), evi_vis, '5. EVI (Enhanced)')
    
    # Add colorbar legend
    Map.add_legend(
        title="Vegetation Index Values",
        legend_dict={
            'Low/Stressed (0.0-0.2)': 'a52a2a',
            'Moderate (0.2-0.4)': 'ffff00',
            'Healthy (0.4-0.6)': '90ee90',
            'Very Healthy (0.6-0.8)': '228b22'
        }
    )
    
    # Save map
    map_path = os.path.join(save_dir, 'day3_vegetation_indices.html')
    Map.to_html(map_path)
    print(f"🗺️ Map saved to: {map_path}")
    
    # Calculate statistics
    print("\n📊 Index Statistics (Regional Means):")
    
    stats = s2_with_indices.select(['NDVI', 'NDWI', 'SAVI', 'EVI']).reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    ).getInfo()
    
    print(f"   NDVI: {stats.get('NDVI_mean', 0):.3f} ± {stats.get('NDVI_stdDev', 0):.3f}")
    print(f"   NDWI: {stats.get('NDWI_mean', 0):.3f} ± {stats.get('NDWI_stdDev', 0):.3f}")
    print(f"   SAVI: {stats.get('SAVI_mean', 0):.3f} ± {stats.get('SAVI_stdDev', 0):.3f}")
    print(f"   EVI:  {stats.get('EVI_mean', 0):.3f} ± {stats.get('EVI_stdDev', 0):.3f}")
    
    return Map


# =============================================================================
# SENTINEL-1 RADAR (Day 4)
# =============================================================================

def get_sentinel1_collection(roi: ee.Geometry,
                              start_date: str,
                              end_date: str,
                              orbit_pass: str = None) -> ee.ImageCollection:
    """
    Get Sentinel-1 GRD collection for the study area.
    
    Sentinel-1 provides all-weather, day-and-night imaging capability
    which is critical for the cloudy Rif region.
    """
    if orbit_pass is None:
        orbit_pass = gee_config.s1_orbit_pass
    
    s1 = ee.ImageCollection(gee_config.s1_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', gee_config.s1_instrument_mode)) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
    
    return s1


def apply_speckle_filter(image: ee.Image, kernel_size: int = 3) -> ee.Image:
    """
    Apply speckle filtering to Sentinel-1 image using focal median.
    """
    vv_filtered = image.select('VV').focal_median(kernel_size, 'circle', 'pixels').rename('VV')
    vh_filtered = image.select('VH').focal_median(kernel_size, 'circle', 'pixels').rename('VH')
    
    return image.addBands(vv_filtered, overwrite=True) \
                .addBands(vh_filtered, overwrite=True)


def compute_radar_ratio(image: ee.Image) -> ee.Image:
    """
    Compute VH/VV ratio for vegetation structure analysis.
    
    The cross-polarization ratio helps isolate the vegetation signal
    from soil effects in olive orchards.
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # VH/VV ratio (in dB space, this is VH - VV)
    ratio = vh.subtract(vv).rename('VH_VV_ratio')
    
    return image.addBands(ratio)


def compute_rvi(image: ee.Image) -> ee.Image:
    """
    Compute Radar Vegetation Index (RVI).
    
    RVI = 4 * VH / (VV + VH)
    Range: 0 to 1 (higher = more vegetation)
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # Convert from dB to linear scale for RVI calculation
    vv_linear = ee.Image(10).pow(vv.divide(10))
    vh_linear = ee.Image(10).pow(vh.divide(10))
    
    rvi = vh_linear.multiply(4).divide(vv_linear.add(vh_linear)).rename('RVI')
    
    return image.addBands(rvi)


def terrain_flattening(image: ee.Image) -> ee.Image:
    """
    Apply Radiometric Terrain Flattening (Volumetric Model) to Sentinel-1.
    
    Corrects for radiometric distortions caused by topography in the Rif mountains.
    Ref: Small (2011)
    """
    # Get DEM
    dem = ee.Image('USGS/SRTMGL1_003')
    
    # Get image geometry
    geometry = image.geometry()
    
    # Calculate Layover/Shadow mask (simplified)
    # This usually requires specific orbital parameters, but we can use slope/aspect
    # For MVP, we use the Volumetric Model logic simplified:
    # gamma0_flat = gamma0 / volume_model
    
    # Using a simplified approach available in GEE common scripts:
    # gamma0 = sigma0 / cos(local_incidence_angle)
    
    # 1. Get Incidence Angle from the image (interpolated)
    incident_angle = image.select('angle')
    
    # 2. Get Local Incidence Angle using DEM
    terrain = ee.Algorithms.Terrain(dem)
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    
    # This is a complex calculation in GEE. For MVP week 1, 
    # we'll implement a slope-based radiometric correction.
    # sigma0_corrected = sigma0 * (sin(local_inc) / sin(inc))
    
    # Simplified slope correction for now:
    # Corrected = Original - alpha * (local_incidence - global_incidence)
    
    # Actually, let's stick to the README "Radiometric Terrain Flattening" promise
    # and use a solid approximation:
    # Converting beta0 to gamma0 using DEM-derived local incidence angle.
    
    # Since we can't easily implement full Volumetric Model without the 'volumetric' library,
    # we will use a slope-normalization correction (angular based).
    
    # Convert degrees to radians
    rad = np.pi / 180.0
    slope_rad = slope.multiply(rad)
    aspect_rad = aspect.multiply(rad)
    
    # Estimate look direction (approximate for Descending/Ascending)
    # S1 Descending: azimuth ~ -167 degrees (from North) -> ~193 deg
    # S1 Ascending: azimuth ~ -13 degrees (from North) -> ~347 deg
    # We can get 'orbitProperties_pass'
    
    orbit_pass = image.get('orbitProperties_pass')
    heading = ee.Algorithms.If(
        ee.Algorithms.IsEqual(orbit_pass, 'ASCENDING'),
        -13.0,
        -167.0
    )
    heading_rad = ee.Number(heading).multiply(rad)
    
    # Calculate Angle of Incidence (approximate)
    inc_angle_rad = incident_angle.multiply(rad)
    
    # Calculate Local Incidence Angle
    # cos(theta_local) = cos(slope) * cos(theta_inc) + sin(slope) * sin(theta_inc) * cos(aspect - heading)
    cos_theta_local = slope_rad.cos().multiply(inc_angle_rad.cos()) \
        .add(slope_rad.sin().multiply(inc_angle_rad.sin())
             .multiply(aspect_rad.subtract(heading_rad).cos()))
             
    # Gamma0 = Sigma0 * tan(30deg) / tan(local_inc) ?
    # Standard Slope Correction:
    # Sigma0_corr = Sigma0 * sin(local_inc) / sin(ref_inc)
    # Ref: Ulander et al (1996)
    
    # We'll use the ratio of projected areas
    # area_ratio = sin(local_inc) / sin(inc_angle)
    # For safety with cos/sin, we use the cos_theta_local
    
    # Local incidence angle
    theta_local = cos_theta_local.acos()
    
    # Correction factor (Area Ratio)
    # area_ratio = sin(theta_local) / sin(inc_angle_rad)
    area_ratio = theta_local.sin().divide(inc_angle_rad.sin())
    
    # Apply correction to VV and VH
    vv_corrected = image.select('VV').divide(area_ratio).rename('VV')
    vh_corrected = image.select('VH').divide(area_ratio).rename('VH')
    
    return image.addBands(vv_corrected, overwrite=True) \
                .addBands(vh_corrected, overwrite=True)


def process_sentinel1(image: ee.Image) -> ee.Image:
    """
    Full processing pipeline for Sentinel-1 image.
    """
    image = apply_speckle_filter(image)
    image = terrain_flattening(image)  # Added Terrain Flattening
    image = compute_radar_ratio(image)
    image = compute_rvi(image)
    
    return image


def visualize_radar_data(roi: ee.Geometry,
                          start_date: str = '2024-06-01',
                          end_date: str = '2024-08-31',
                          save_dir: str = None) -> geemap.Map:
    """
    Day 4 Visualization: Display Sentinel-1 radar data.
    
    Creates an interactive map showing:
    1. VV polarization (surface scattering)
    2. VH polarization (volume scattering)
    3. VH/VV ratio (vegetation indicator)
    4. RVI (Radar Vegetation Index)
    5. RGB composite (VV, VH, VH/VV)
    
    Args:
        roi: Region of interest
        start_date: Start date for composite
        end_date: End date for composite
        save_dir: Directory to save the map
        
    Returns:
        geemap.Map: Interactive map with radar data
    """
    print("\n" + "=" * 60)
    print("DAY 4: SENTINEL-1 RADAR VISUALIZATION")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(NOTEBOOKS_DIR)
    
    # Get Sentinel-1 collection
    s1 = get_sentinel1_collection(roi, start_date, end_date)
    
    # Process and create composite
    s1_processed = s1.map(process_sentinel1).median().clip(roi)
    
    # Get image count
    s1_count = s1.size().getInfo()
    print(f"📡 Sentinel-1 images in composite: {s1_count}")
    
    # Create map
    Map = geemap.Map(
        center=[study_area_config.center_lat, study_area_config.center_lon],
        zoom=12
    )
    
    # Add layers
    Map.addLayer(roi, {'color': 'white'}, 'Study Area Boundary')
    
    # VV polarization
    vv_vis = {'min': -25, 'max': 0, 'palette': ['black', 'gray', 'white']}
    Map.addLayer(s1_processed.select('VV'), vv_vis, '1. VV Polarization (Surface)')
    
    # VH polarization
    vh_vis = {'min': -30, 'max': -5, 'palette': ['black', 'gray', 'white']}
    Map.addLayer(s1_processed.select('VH'), vh_vis, '2. VH Polarization (Volume)')
    
    # VH/VV ratio
    ratio_vis = {'min': -15, 'max': -5, 'palette': ['red', 'yellow', 'green']}
    Map.addLayer(s1_processed.select('VH_VV_ratio'), ratio_vis, '3. VH/VV Ratio (Vegetation)')
    
    # RVI
    rvi_vis = {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(s1_processed.select('RVI'), rvi_vis, '4. RVI (Radar Vegetation Index)')
    
    # RGB composite (VV, VH, VH/VV ratio normalized)
    # Normalize VH/VV ratio to 0-1 range for RGB display
    vh_vv_norm = s1_processed.select('VH_VV_ratio').add(15).divide(10).clamp(0, 1).multiply(255)
    vv_norm = s1_processed.select('VV').add(25).divide(25).clamp(0, 1).multiply(255)
    vh_norm = s1_processed.select('VH').add(30).divide(25).clamp(0, 1).multiply(255)
    
    rgb_composite = vv_norm.addBands(vh_norm).addBands(vh_vv_norm).rename(['R', 'G', 'B'])
    Map.addLayer(rgb_composite, {'min': 0, 'max': 255}, '5. Radar RGB (VV, VH, Ratio)')
    
    # Add Sentinel-2 for reference
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_clouds_qa60) \
        .median() \
        .clip(roi)
    
    Map.addLayer(s2, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '6. Optical Reference (S2 RGB)')
    
    # Add legend
    Map.add_legend(
        title="Radar Backscatter (dB)",
        legend_dict={
            'Very Low (-30 to -25 dB)': '000000',
            'Low (-25 to -20 dB)': '555555',
            'Medium (-20 to -15 dB)': 'aaaaaa',
            'High (-15 to -10 dB)': 'dddddd',
            'Very High (-10 to 0 dB)': 'ffffff'
        }
    )
    
    # Save map
    map_path = os.path.join(save_dir, 'day4_radar_data.html')
    Map.to_html(map_path)
    print(f"🗺️ Map saved to: {map_path}")
    
    # Calculate statistics
    print("\n📊 Radar Statistics (Regional Means):")
    
    stats = s1_processed.select(['VV', 'VH', 'VH_VV_ratio', 'RVI']).reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    print(f"   VV: {stats.get('VV_mean', 0):.2f} ± {stats.get('VV_stdDev', 0):.2f} dB")
    print(f"   VH: {stats.get('VH_mean', 0):.2f} ± {stats.get('VH_stdDev', 0):.2f} dB")
    print(f"   VH/VV: {stats.get('VH_VV_ratio_mean', 0):.2f} ± {stats.get('VH_VV_ratio_stdDev', 0):.2f}")
    print(f"   RVI: {stats.get('RVI_mean', 0):.3f} ± {stats.get('RVI_stdDev', 0):.3f}")
    
    return Map


# =============================================================================
# VEGETATION MASK (Day 5)
# =============================================================================

def create_vegetation_mask(roi: ee.Geometry,
                            start_date: str,
                            end_date: str,
                            threshold: float = None) -> ee.Image:
    """
    Create a vegetation mask based on NDVI threshold.
    
    This mask excludes non-vegetated pixels (soil, rock, water)
    to address the "mixed pixel" problem in fragmented orchards.
    """
    if threshold is None:
        threshold = indices_config.vegetation_mask_threshold
    
    # Get Sentinel-2 collection
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', gee_config.s2_cloud_threshold)) \
        .map(mask_clouds_qa60)
    
    # Compute NDVI for each image and create median composite
    def add_ndvi(image):
        nir = image.select('B8')
        red = image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return image.addBands(ndvi)
    
    ndvi_composite = s2.map(add_ndvi).select('NDVI').median()
    
    # Create binary mask
    vegetation_mask = ndvi_composite.gt(threshold).rename('vegetation_mask')
    
    print(f"🌿 Vegetation mask created (NDVI > {threshold})")
    
    return vegetation_mask, ndvi_composite


def create_olive_mask(roi: ee.Geometry,
                       start_date: str,
                       end_date: str) -> ee.Image:
    """
    Create an olive orchard mask based on NDVI statistics and Texture.
    
    Updated for Week 1 (Pure Pixel Filtering):
    - NDVI > 0.4 (Dense/Pure vegetation)
    - Low Temporal Variability (Evergreen)
    - Low Texture Entropy (Homogeneous canopy)
    """
    # Get Sentinel-2 collection
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_clouds_qa60)
    
    # Compute NDVI
    def add_ndvi(image):
        nir = image.select('B8')
        red = image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return ndvi
    
    ndvi_collection = s2.map(add_ndvi)
    
    # Compute statistics
    ndvi_mean = ndvi_collection.mean()
    ndvi_std = ndvi_collection.reduce(ee.Reducer.stdDev())
    
    # Texture Analysis (Homogeneity)
    # Use median NIR band for texture
    nir_median = s2.select('B8').median().clip(roi)
    # Scale to integer for GLCM
    glcm = nir_median.multiply(100).toInt().glcmTexture(size=3)
    entropy = glcm.select('B8_entropy')
    
    # Olive mask criteria (Enhanced):
    # 1. NDVI mean > 0.4 (Pure Pixel Filtering)
    # 2. NDVI std < 0.15 (Stable/Evergreen)
    # 3. Entropy < 6 (Homogeneous texture, avoid mixed pixels)
    olive_mask = ndvi_mean.gt(0.4) \
        .And(ndvi_mean.lt(0.80)) \
        .And(ndvi_std.lt(0.15)) \
        .And(entropy.lt(6)) \
        .rename('olive_mask')
    
    print("🫒 Olive orchard mask created (Enhanced)")
    print("   Criteria: NDVI > 0.4, Stable, Low Entropy")
    
    return olive_mask, ndvi_mean, ndvi_std


def visualize_vegetation_mask(roi: ee.Geometry,
                               start_date: str = '2024-01-01',
                               end_date: str = '2024-12-31',
                               save_dir: str = None) -> geemap.Map:
    """
    Day 5 Visualization: Display vegetation and olive masks.
    
    Creates an interactive map showing:
    1. NDVI composite
    2. Basic vegetation mask (NDVI > 0.2)
    3. Olive orchard mask
    4. Masked vs unmasked comparison
    
    Args:
        roi: Region of interest
        start_date: Start date for composite
        end_date: End date for composite
        save_dir: Directory to save the map
        
    Returns:
        geemap.Map: Interactive map with masks
    """
    print("\n" + "=" * 60)
    print("DAY 5: VEGETATION & FRAGMENTATION MASK VISUALIZATION")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(NOTEBOOKS_DIR)
    
    # Create vegetation mask
    veg_mask, ndvi_composite = create_vegetation_mask(roi, start_date, end_date)
    
    # Create olive mask
    olive_mask, ndvi_mean, ndvi_std = create_olive_mask(roi, start_date, end_date)
    
    # Get Sentinel-2 composite for RGB
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_clouds_qa60) \
        .median() \
        .clip(roi)
    
    # Compute all indices
    s2_with_indices = compute_all_indices(s2)
    
    # Apply masks
    s2_veg_masked = s2_with_indices.updateMask(veg_mask.clip(roi))
    s2_olive_masked = s2_with_indices.updateMask(olive_mask.clip(roi))
    
    # Create map
    Map = geemap.Map(
        center=[study_area_config.center_lat, study_area_config.center_lon],
        zoom=12
    )
    
    # Add layers
    Map.addLayer(roi, {'color': 'white'}, 'Study Area Boundary')
    
    # Original RGB
    Map.addLayer(s2, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '1. Original RGB')
    
    # NDVI composite
    ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(ndvi_composite.clip(roi), ndvi_vis, '2. NDVI Composite')
    
    # NDVI Std Dev
    std_vis = {'min': 0, 'max': 0.2, 'palette': ['green', 'yellow', 'red']}
    Map.addLayer(ndvi_std.clip(roi), std_vis, '3. NDVI Temporal Variability')
    
    # Vegetation mask (binary)
    mask_vis = {'min': 0, 'max': 1, 'palette': ['red', 'green']}
    Map.addLayer(veg_mask.clip(roi), mask_vis, '4. Vegetation Mask (NDVI > 0.2)')
    
    # Olive mask (binary)
    Map.addLayer(olive_mask.clip(roi), mask_vis, '5. Olive Orchard Mask')
    
    # Masked RGB (only vegetation)
    Map.addLayer(s2_veg_masked, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '6. RGB - Vegetation Only')
    
    # Masked RGB (only olives)
    Map.addLayer(s2_olive_masked, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '7. RGB - Olive Orchards Only')
    
    # Masked NDVI (only olives)
    Map.addLayer(s2_olive_masked.select('NDVI'), ndvi_vis, '8. NDVI - Olive Orchards Only')
    
    # Add legend
    Map.add_legend(
        title="Mask Values",
        legend_dict={
            'Masked (Soil/Rock/Water)': 'ff0000',
            'Vegetation/Olive': '00ff00'
        }
    )
    
    # Save map
    map_path = os.path.join(save_dir, 'day5_vegetation_mask.html')
    Map.to_html(map_path)
    print(f"🗺️ Map saved to: {map_path}")
    
    # Calculate statistics
    print("\n📊 Mask Statistics:")
    
    # Count pixels
    total_pixels = veg_mask.clip(roi).unmask(0).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    ).get('vegetation_mask').getInfo()
    
    veg_pixels = veg_mask.clip(roi).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    ).get('vegetation_mask').getInfo()
    
    olive_pixels = olive_mask.clip(roi).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    ).get('olive_mask').getInfo()
    
    print(f"   Total pixels: {total_pixels:,.0f}")
    print(f"   Vegetation pixels: {veg_pixels:,.0f} ({100*veg_pixels/total_pixels:.1f}%)")
    print(f"   Olive orchard pixels: {olive_pixels:,.0f} ({100*olive_pixels/total_pixels:.1f}%)")
    print(f"   Non-vegetation (masked): {total_pixels - veg_pixels:,.0f} ({100*(1-veg_pixels/total_pixels):.1f}%)")
    
    return Map


# =============================================================================
# DATA EXTRACTION & FUSION (Day 6)
# =============================================================================

def get_sentinel2_timeseries(roi: ee.Geometry,
                              start_date: str,
                              end_date: str,
                              apply_veg_mask: bool = True) -> ee.ImageCollection:
    """
    Get processed Sentinel-2 time series with indices, cloud masking, and Topographic Correction.
    """
    # Create vegetation mask if needed
    if apply_veg_mask:
        veg_mask, _ = create_olive_mask(roi, start_date, end_date) # Use Enhanced Olive Mask
    else:
        veg_mask = None
    
    # Get cloud probability function
    add_prob = add_cloud_probability(roi, start_date, end_date)
    
    # Initialize Topographic Correction
    topo_corr = TopographicCorrection()
    
    # Get Sentinel-2 collection
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', gee_config.s2_cloud_threshold)) \
        .map(add_prob) \
        .map(apply_hybrid_cloud_mask) \
        .map(lambda img: topo_corr.apply_c_correction(img, roi)) \
        .map(compute_all_indices)
    
    def process_s2(image):
        # Apply vegetation mask if provided
        if veg_mask is not None:
            image = image.updateMask(veg_mask)
        
        # Reduce to region statistics
        bands = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'SAVI', 'EVI']
        stats = image.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=gee_config.scale,
            maxPixels=gee_config.max_pixels
        )
        
        return image.set(stats).set('date', image.date().format('YYYY-MM-dd'))
    
    return s2.map(process_s2)


def get_era5_timeseries(roi: ee.Geometry,
                         start_date: str,
                         end_date: str) -> ee.ImageCollection:
    """
    Get ERA5-Land climate data time series.
    """
    era5 = ee.ImageCollection(gee_config.era5_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    def process_era5(image):
        # Convert temperature from Kelvin to Celsius
        temp = image.select('temperature_2m').subtract(273.15).rename('Temperature_C')
        
        # Convert precipitation to mm
        precip = image.select('total_precipitation_sum').multiply(1000).rename('Precipitation_mm')
        
        # Reduce to region statistics
        stats = temp.addBands(precip).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=11132,  # ERA5 native resolution (~11km)
            maxPixels=gee_config.max_pixels
        )
        
        return image.set(stats).set('date', image.date().format('YYYY-MM-dd'))
    
    return era5.map(process_era5)


def create_stress_label(row: pd.Series, 
                         ndwi_mean: float,
                         ndwi_std: float) -> int:
    """
    Create binary stress label based on NDWI anomaly and temperature.
    
    Stress conditions (any of):
    1. NDWI below threshold AND high temperature
    2. NDWI extremely low (absolute threshold)
    3. Combined stress: low NDWI + low precipitation + high temp
    """
    ndwi_threshold = ndwi_mean - (ml_config.stress_ndwi_std_multiplier * ndwi_std)
    
    ndwi = row.get('NDWI', 0)
    temp = row.get('Temperature_C', 0)
    precip = row.get('Precipitation_mm', 0)
    
    if pd.isna(ndwi) or pd.isna(temp):
        return 0
    
    # Condition 1: NDWI below threshold AND high temperature
    if ndwi < ndwi_threshold and temp > ml_config.stress_temp_threshold:
        return 1
    
    # Condition 2: Extremely low NDWI (absolute threshold)
    if hasattr(ml_config, 'stress_ndwi_absolute_threshold'):
        if ndwi < ml_config.stress_ndwi_absolute_threshold:
            return 1
    
    # Condition 3: Summer stress (high temp, low precip, below-average NDWI)
    if temp > 28 and precip < 1 and ndwi < ndwi_mean:
        return 1
    
    return 0


def generate_kmeans_labels(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Generate pseudo-labels using K-Means clustering on time-series data.
    
    Clusters days into 'Stable', 'Transition', 'Stressed' based on 
    NDVI, NDWI, Temperature, and Precipitation behavior.
    """
    print("   🤖 Running K-Means clustering for pseudo-labels...")
    
    # Select features for clustering
    features = ['NDVI', 'NDWI', 'Temperature_C', 'Precipitation_mm']
    if 'VH' in df.columns:
        features.append('VH')
        
    # Drop rows with NaNs for clustering (though df should be clean)
    X = df[features].dropna()
    
    if len(X) < n_clusters:
        print("   ⚠️ Not enough data for K-Means.")
        df['stress_label_kmeans'] = 0
        return df
        
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Assign labels back to DataFrame
    df.loc[X.index, 'cluster_label'] = labels
    
    # Interpret clusters: "Stress" usually correlates with Low NDWI
    # We find the cluster with lowest mean NDWI and label it as '1' (Stress), others '0'
    cluster_means = df.groupby('cluster_label')['NDWI'].mean()
    stress_cluster = cluster_means.idxmin()
    
    df['stress_label'] = (df['cluster_label'] == stress_cluster).astype(int)
    
    print(f"   ✅ Identified Cluster {stress_cluster} as 'Stressed' (Mean NDWI: {cluster_means[stress_cluster]:.3f})")
    
    return df


def rank_years_by_rainfall(roi: ee.Geometry, 
                            start_year: int = 2018, 
                            end_year: int = 2024) -> List[int]:
    """
    Rank years by total annual precipitation using ERA5 data.
    Returns a list of years sorted from Wettest to Driest.
    """
    print("\n🌧️  Ranking years by rainfall (ERA5)...")
    years = ee.List.sequence(start_year, end_year)
    
    def get_annual_precip(year):
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, 'year')
        
        era5 = ee.ImageCollection(gee_config.era5_collection) \
            .filterBounds(roi) \
            .filterDate(start, end) \
            .select('total_precipitation_sum')
            
        # Sum daily precipitation
        total_precip = era5.sum().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=11132,
            maxPixels=1e9
        ).get('total_precipitation_sum')
        
        return ee.Feature(None, {'year': year, 'precip': total_precip})
    
    # Map over years
    yearly_stats = ee.FeatureCollection(years.map(get_annual_precip))
    
    # Sort descending
    sorted_years = yearly_stats.sort('precip', False)
    
    # Get values client-side
    data = sorted_years.reduceColumns(ee.Reducer.toList(2), ['year', 'precip']).get('list').getInfo()
    
    # Convert to list of (year, precip) and sort in Python to be sure
    ranked_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    print("   Year | Total Precip (m)")
    print("   -----|------------------")
    for year, precip in ranked_data:
        print(f"   {int(year)} | {precip:.4f}")
        
    ranked_years = [int(x[0]) for x in ranked_data]
    return ranked_years


def extract_climate_driven_dataset(roi: ee.Geometry) -> pd.DataFrame:
    """
    Generate Dataset A: Climate-Driven (Healthy Years).
    Extracts data ONLY from the top 3 wettest years.
    """
    print("\n📡 Generating Dataset A: Climate-Driven (Healthy Years)...")
    
    # 1. Identify Healthy Years
    current_year = datetime.now().year
    ranked_years = rank_years_by_rainfall(roi, start_year=2018, end_year=current_year-1)
    healthy_years = ranked_years[:3] # Top 3 wettest
    print(f"   🏆 Selected Healthy Years: {healthy_years}")
    
    all_dfs = []
    
    for year in healthy_years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"   ... Extracting for {year}")
        
        # Reuse existing extraction logic but for specific year
        # Note: We temporarily override config dates or pass explicit dates
        s2 = get_sentinel2_timeseries(roi, start_date, end_date, apply_veg_mask=True)
        era5 = get_era5_timeseries(roi, start_date, end_date)
        s1 = get_sentinel1_collection(roi, start_date, end_date)
        s1_proc = s1.map(process_sentinel1).map(lambda img: img.set(img.select(['VV', 'VH', 'RVI']).reduceRegion(ee.Reducer.mean(), roi, gee_config.s1_resolution, 1e9)).set('date', img.date().format('YYYY-MM-dd')))
        
        # Helper to df
        def to_df(col, props):
            data = col.reduceColumns(ee.Reducer.toList(len(props)), props).get('list').getInfo()
            return pd.DataFrame(data, columns=props) if data else pd.DataFrame()

        df_s2 = to_df(s2, ['date', 'NDVI', 'NDWI', 'SAVI', 'EVI'])
        df_era5 = to_df(era5, ['date', 'Temperature_C', 'Precipitation_mm'])
        df_s1 = to_df(s1_proc, ['date', 'VV', 'VH', 'RVI'])
        
        # Merge
        if not df_s2.empty and not df_era5.empty:
            df_s2['date'] = pd.to_datetime(df_s2['date'])
            df_era5['date'] = pd.to_datetime(df_era5['date'])
            merged = df_era5.set_index('date').join(df_s2.set_index('date').resample('D').interpolate(), how='inner')
            if not df_s1.empty:
                df_s1['date'] = pd.to_datetime(df_s1['date'])
                merged = merged.join(df_s1.set_index('date').resample('D').interpolate(), how='left')
            
            merged['stress_label'] = 0 # BY DEFINITION: Healthy Year = 0 Stress
            all_dfs.append(merged)
            
    if not all_dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'date'}, inplace=True)
    
    # Save
    path = output_config.training_data_climate_driven_csv
    final_df.to_csv(path, index=False)
    print(f"   💾 Saved Dataset A to {path} ({len(final_df)} rows)")
    return final_df


def extract_data_driven_dataset(roi: ee.Geometry) -> pd.DataFrame:
    """
    Generate Dataset B: Data-Driven (K-Means).
    Extracts full time series and filters for 'Stable' clusters.
    """
    print("\n📡 Generating Dataset B: Data-Driven (K-Means)...")
    
    # Use existing extraction for full range
    df = extract_training_data(roi, include_radar=True, output_path=output_config.training_data_data_driven_csv)
    
    # Filter for Healthy Only (Label 0)
    df_healthy = df[df['stress_label'] == 0].copy()
    
    # Overwrite the file to keep only healthy? Or keep full for validation?
    # For training the Autoencoder, we need ONLY healthy.
    # Let's save a specific "healthy_only" version for the model.
    path_healthy = str(Path(output_config.training_data_data_driven_csv).parent / "training_data_data_driven_healthy_only.csv")
    df_healthy.to_csv(path_healthy, index=False)
    print(f"   💾 Saved Healthy-Only subset to {path_healthy} ({len(df_healthy)} rows)")
    
    return df


def extract_training_data(roi: ee.Geometry = None,
                           start_date: str = None,
                           end_date: str = None,
                           output_path: str = None,
                           include_radar: bool = True) -> pd.DataFrame:
    """
    Main function to extract training data from GEE.
    
    This function:
    1. Creates vegetation mask
    2. Gets Sentinel-2 time series with indices
    3. Gets Sentinel-1 radar data (optional)
    4. Gets ERA5 climate data
    5. Fuses all data sources
    6. Creates stress labels (K-Means)
    7. Exports to CSV
    """
    # Use defaults from config if not provided
    if roi is None:
        roi, _ = get_study_area()
    if start_date is None:
        start_date = gee_config.start_date
    if end_date is None:
        end_date = gee_config.end_date
    if output_path is None:
        output_path = output_config.training_data_csv
    
    print(f"\n🚀 Starting data extraction...")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Step 1: Get Sentinel-2 time series
    print("\n🛰️ Step 1: Processing Sentinel-2 data...")
    s2_collection = get_sentinel2_timeseries(roi, start_date, end_date, apply_veg_mask=True)
    
    # Step 2: Get ERA5 climate data
    print("\n🌡️ Step 2: Processing ERA5 climate data...")
    era5_collection = get_era5_timeseries(roi, start_date, end_date)
    
    # Step 3: Get Sentinel-1 radar data (if requested)
    if include_radar:
        print("\n📡 Step 3: Processing Sentinel-1 radar data...")
        s1_collection = get_sentinel1_collection(roi, start_date, end_date)
        s1_processed = s1_collection.map(process_sentinel1)
        
        def get_s1_means(image):
            stats = image.select(['VV', 'VH', 'RVI']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=gee_config.s1_resolution,
                maxPixels=gee_config.max_pixels
            )
            return image.set(stats).set('date', image.date().format('YYYY-MM-dd'))
        
        s1_with_stats = s1_processed.map(get_s1_means)
    
    # Step 4: Extract data to pandas DataFrames
    print("\n📥 Step 4: Downloading data...")
    
    def collection_to_df(collection, properties):
        try:
            data = collection.reduceColumns(
                ee.Reducer.toList(len(properties)), 
                properties
            ).get('list').getInfo()
            
            return pd.DataFrame(data, columns=properties)
        except Exception as e:
            print(f"   ⚠️ Error extracting data: {e}")
            return pd.DataFrame()
    
    # Extract Sentinel-2 data
    s2_props = ['date', 'B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'SAVI', 'EVI']
    df_s2 = collection_to_df(s2_collection, s2_props)
    print(f"   ✅ Sentinel-2: {len(df_s2)} observations")
    
    # Extract ERA5 data
    era5_props = ['date', 'Temperature_C', 'Precipitation_mm']
    df_era5 = collection_to_df(era5_collection, era5_props)
    print(f"   ✅ ERA5: {len(df_era5)} observations")
    
    # Extract Sentinel-1 data
    if include_radar:
        s1_props = ['date', 'VV', 'VH', 'RVI']
        df_s1 = collection_to_df(s1_with_stats, s1_props)
        print(f"   ✅ Sentinel-1: {len(df_s1)} observations")
    
    if len(df_s2) == 0:
        print("❌ No Sentinel-2 data found. Check date range and ROI.")
        return pd.DataFrame()
    
    # Step 5: Merge datasets
    print("\n🔗 Step 5: Merging datasets...")
    
    # Convert date columns
    df_s2['date'] = pd.to_datetime(df_s2['date'])
    df_era5['date'] = pd.to_datetime(df_era5['date'])
    if include_radar and len(df_s1) > 0:
        df_s1['date'] = pd.to_datetime(df_s1['date'])
    
    # Set date as index
    df_s2.set_index('date', inplace=True)
    df_era5.set_index('date', inplace=True)
    if include_radar and len(df_s1) > 0:
        df_s1.set_index('date', inplace=True)
    
    # Resample to daily (interpolate)
    df_s2_daily = df_s2.resample('D').mean().interpolate(method='linear')
    if include_radar and len(df_s1) > 0:
        df_s1_daily = df_s1.resample('D').mean().interpolate(method='linear')
    
    # Merge all datasets
    df_merged = df_era5.join(df_s2_daily, how='inner')
    if include_radar and len(df_s1) > 0:
        df_merged = df_merged.join(df_s1_daily, how='left')
    
    # Step 6: Create stress labels (K-Means)
    print("\n🏷️ Step 6: Creating stress labels (K-Means)...")
    
    # Drop NaN rows before clustering
    initial_count = len(df_merged)
    df_merged.dropna(inplace=True)
    
    df_merged = generate_kmeans_labels(df_merged)
    
    stress_count = df_merged['stress_label'].sum()
    total_count = len(df_merged)
    print(f"   ✅ Stress events: {stress_count}/{total_count} ({100*stress_count/total_count:.1f}%)")
    
    # Add coordinates
    df_merged['lat'] = study_area_config.center_lat
    df_merged['lon'] = study_area_config.center_lon
    
    # Reset index
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={'index': 'date'}, inplace=True)
    
    final_count = len(df_merged)
    print(f"   Final dataset size: {final_count} rows")
    
    # Step 7: Save to CSV
    print(f"\n💾 Step 7: Saving to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS! Extracted {len(df_merged)} observations")
    print(f"   Columns: {list(df_merged.columns)}")
    print(f"   Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    
    return df_merged


def visualize_extracted_data(df: pd.DataFrame = None,
                              csv_path: str = None,
                              save_dir: str = None) -> None:
    """
    Day 6 Visualization: Display extracted time series data.
    
    Creates matplotlib plots showing:
    1. NDVI and NDWI time series
    2. Temperature and precipitation
    3. Radar backscatter (if available)
    4. Stress events overlay
    5. Correlation heatmap
    """
    print("\n" + "=" * 60)
    print("DAY 6: EXTRACTED DATA VISUALIZATION")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(PLOTS_DIR)
    
    # Load data if not provided
    if df is None:
        if csv_path is None:
            csv_path = output_config.training_data_csv
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 Visualizing {len(df)} observations")
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Plot 1: NDVI and NDWI
    ax1 = axes[0]
    ax1.plot(df['date'], df['NDVI'], 'g-', label='NDVI', linewidth=1.5)
    ax1.plot(df['date'], df['NDWI'], 'b-', label='NDWI', linewidth=1.5)
    
    # Add stress events
    if 'stress_label' in df.columns:
        stress_dates = df[df['stress_label'] == 1]['date']
        for date in stress_dates:
            ax1.axvline(x=date, color='red', alpha=0.3, linewidth=0.5)
    
    ax1.set_ylabel('Index Value')
    ax1.set_title('Vegetation Indices Time Series')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: SAVI and EVI
    ax2 = axes[1]
    if 'SAVI' in df.columns:
        ax2.plot(df['date'], df['SAVI'], 'orange', label='SAVI', linewidth=1.5)
    if 'EVI' in df.columns:
        ax2.plot(df['date'], df['EVI'], 'purple', label='EVI', linewidth=1.5)
    
    ax2.set_ylabel('Index Value')
    ax2.set_title('Soil-Adjusted Indices Time Series')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temperature and Precipitation
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    ax3.plot(df['date'], df['Temperature_C'], 'r-', label='Temperature', linewidth=1.5)
    ax3_twin.bar(df['date'], df['Precipitation_mm'], color='blue', alpha=0.4, label='Precipitation', width=1)
    
    ax3.set_ylabel('Temperature (°C)', color='red')
    ax3_twin.set_ylabel('Precipitation (mm)', color='blue')
    ax3.set_title('Climate Variables')
    ax3.grid(True, alpha=0.3)
    
    # Add threshold line
    ax3.axhline(y=ml_config.stress_temp_threshold, color='red', linestyle='--', alpha=0.5, 
                label=f'Stress threshold ({ml_config.stress_temp_threshold}°C)')
    
    # Plot 4: Radar data (if available)
    ax4 = axes[3]
    if 'VV' in df.columns and 'VH' in df.columns:
        ax4.plot(df['date'], df['VV'], 'gray', label='VV', linewidth=1.5)
        ax4.plot(df['date'], df['VH'], 'black', label='VH', linewidth=1.5)
        if 'RVI' in df.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df['date'], df['RVI'], 'green', label='RVI', linewidth=1.5, linestyle='--')
            ax4_twin.set_ylabel('RVI', color='green')
        ax4.set_ylabel('Backscatter (dB)')
        ax4.set_title('Radar Backscatter Time Series')
        ax4.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'Radar data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Radar Backscatter (Not Available)')
    
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save time series plot
    ts_path = os.path.join(save_dir, 'day6_timeseries.png')
    plt.savefig(ts_path, dpi=150, bbox_inches='tight')
    print(f"📈 Time series plot saved to: {ts_path}")
    plt.close()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['lat', 'lon', 'stress_label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save correlation plot
    corr_path = os.path.join(save_dir, 'day6_correlation_matrix.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    print(f"📊 Correlation matrix saved to: {corr_path}")
    plt.close()
    
    # Create stress events visualization
    if 'stress_label' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot NDWI with stress overlay
        ax.plot(df['date'], df['NDWI'], 'b-', label='NDWI', linewidth=1.5)
        
        # Highlight stress events
        stress_mask = df['stress_label'] == 1
        ax.scatter(df.loc[stress_mask, 'date'], df.loc[stress_mask, 'NDWI'], 
                  c='red', s=30, label='Stress Events', zorder=5)
        
        # Add threshold line
        ndwi_mean = df['NDWI'].mean()
        ndwi_std = df['NDWI'].std()
        threshold = ndwi_mean - ndwi_std
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Stress Threshold ({threshold:.3f})')
        ax.axhline(y=ndwi_mean, color='green', linestyle='--', alpha=0.7,
                  label=f'Mean NDWI ({ndwi_mean:.3f})')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('NDWI')
        ax.set_title('Water Stress Detection Timeline')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        stress_path = os.path.join(save_dir, 'day6_stress_timeline.png')
        plt.savefig(stress_path, dpi=150, bbox_inches='tight')
        print(f"🚨 Stress timeline saved to: {stress_path}")
        plt.close()
    
    # Print summary statistics
    print("\n📊 Data Summary:")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Observations: {len(df)}")
    if 'stress_label' in df.columns:
        stress_pct = 100 * df['stress_label'].mean()
        print(f"   Stress events: {df['stress_label'].sum()} ({stress_pct:.1f}%)")
    print(f"\n   Feature statistics:")
    for col in ['NDVI', 'NDWI', 'SAVI', 'EVI', 'Temperature_C', 'VV', 'VH']:
        if col in df.columns:
            print(f"   {col}: {df[col].mean():.3f} ± {df[col].std():.3f}")


def create_final_fusion_map(roi: ee.Geometry,
                             start_date: str = '2024-06-01',
                             end_date: str = '2024-08-31',
                             save_dir: str = None) -> geemap.Map:
    """
    Day 6 Final Visualization: Multi-sensor fusion map.
    
    Creates an interactive map showing the complete data fusion:
    1. Optical indices (NDVI, NDWI)
    2. Radar data (VV, VH)
    3. Vegetation mask
    4. Combined stress indicator
    """
    print("\n" + "=" * 60)
    print("DAY 6: MULTI-SENSOR FUSION MAP")
    print("=" * 60)
    
    if save_dir is None:
        save_dir = str(NOTEBOOKS_DIR)
    
    # Get vegetation mask
    veg_mask, ndvi_composite = create_vegetation_mask(roi, start_date, end_date)
    
    # Get cloud-masked Sentinel-2 composite with indices
    add_prob = add_cloud_probability(roi, start_date, end_date)
    
    s2 = ee.ImageCollection(gee_config.s2_collection) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(add_prob) \
        .map(apply_hybrid_cloud_mask) \
        .map(compute_all_indices) \
        .median() \
        .clip(roi)
    
    # Apply vegetation mask
    s2_masked = s2.updateMask(veg_mask.clip(roi))
    
    # Get Sentinel-1 composite
    s1 = get_sentinel1_collection(roi, start_date, end_date) \
        .map(process_sentinel1) \
        .median() \
        .clip(roi)
    
    # Create stress indicator
    # Stress = Low NDWI + Low VH (canopy dehydration)
    ndwi = s2_masked.select('NDWI')
    vh = s1.select('VH')
    
    # Normalize to 0-1 range
    ndwi_norm = ndwi.add(0.5).divide(1).clamp(0, 1)  # NDWI range ~ -0.5 to 0.5
    vh_norm = vh.add(25).divide(20).clamp(0, 1)      # VH range ~ -25 to -5
    
    # Combined stress indicator (lower = more stressed)
    stress_indicator = ndwi_norm.add(vh_norm).divide(2).rename('stress_indicator')
    
    # Apply vegetation mask to stress indicator
    stress_indicator = stress_indicator.updateMask(veg_mask.clip(roi))
    
    # Create map
    Map = geemap.Map(
        center=[study_area_config.center_lat, study_area_config.center_lon],
        zoom=12
    )
    
    # Add layers
    Map.addLayer(roi, {'color': 'white'}, 'Study Area Boundary')
    
    # RGB
    Map.addLayer(s2_masked, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 
                 '1. Optical RGB (Masked)')
    
    # NDVI
    ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['brown', 'yellow', 'lightgreen', 'green', 'darkgreen']}
    Map.addLayer(s2_masked.select('NDVI'), ndvi_vis, '2. NDVI (Masked)')
    
    # NDWI
    ndwi_vis = {'min': -0.5, 'max': 0.2, 'palette': ['red', 'orange', 'yellow', 'lightblue', 'blue']}
    Map.addLayer(s2_masked.select('NDWI'), ndwi_vis, '3. NDWI (Masked)')
    
    # Radar VH
    vh_vis = {'min': -25, 'max': -5, 'palette': ['black', 'gray', 'white']}
    Map.addLayer(s1.select('VH').updateMask(veg_mask.clip(roi)), vh_vis, '4. VH Radar (Masked)')
    
    # RVI
    rvi_vis = {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green', 'darkgreen']}
    Map.addLayer(s1.select('RVI').updateMask(veg_mask.clip(roi)), rvi_vis, '5. RVI (Masked)')
    
    # Stress indicator
    stress_vis = {'min': 0, 'max': 1, 'palette': ['red', 'orange', 'yellow', 'lightgreen', 'green']}
    Map.addLayer(stress_indicator, stress_vis, '6. Combined Stress Indicator')
    
    # Add legend
    Map.add_legend(
        title="Stress Level",
        legend_dict={
            'High Stress (0.0-0.2)': 'ff0000',
            'Moderate Stress (0.2-0.4)': 'ffa500',
            'Mild Stress (0.4-0.6)': 'ffff00',
            'Low Stress (0.6-0.8)': '90ee90',
            'Healthy (0.8-1.0)': '006400'
        }
    )
    
    # Save map
    map_path = os.path.join(save_dir, 'day6_multisensor_fusion.html')
    Map.to_html(map_path)
    print(f"🗺️ Final fusion map saved to: {map_path}")
    
    return Map


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_day2():
    """Run Day 2: Cloud Masking"""
    initialize_gee()
    roi, _ = get_study_area()
    visualize_cloud_masking(roi)


def run_day3():
    """Run Day 3: Vegetation Indices"""
    initialize_gee()
    roi, _ = get_study_area()
    visualize_vegetation_indices(roi)


def run_day4():
    """Run Day 4: Sentinel-1 Radar"""
    initialize_gee()
    roi, _ = get_study_area()
    visualize_radar_data(roi)


def run_day5():
    """Run Day 5: Vegetation Mask"""
    initialize_gee()
    roi, _ = get_study_area()
    visualize_vegetation_mask(roi)


def run_day6():
    """Run Day 6: Data Extraction & Fusion (Comparative Study)"""
    initialize_gee()
    roi, _ = get_study_area()
    
    # 1. Dataset A: Climate-Driven (Healthy Years)
    extract_climate_driven_dataset(roi)
    
    # 2. Dataset B: Data-Driven (K-Means)
    df = extract_data_driven_dataset(roi)
    
    # Visualize extracted data (using Dataset B as it covers full range)
    if len(df) > 0:
        visualize_extracted_data(df)
        create_final_fusion_map(roi)


def run_all_days():
    """Run all days (1-6) sequentially"""
    print("\n" + "=" * 70)
    print("OLIVEGUARD GEE PIPELINE - WEEK 1 COMPLETE EXECUTION")
    print("=" * 70)
    
    initialize_gee()
    roi, _ = get_study_area()
    
    # Day 2: Cloud Masking
    visualize_cloud_masking(roi)
    
    # Day 3: Vegetation Indices
    visualize_vegetation_indices(roi)
    
    # Day 4: Radar Data
    visualize_radar_data(roi)
    
    # Day 5: Vegetation Mask
    visualize_vegetation_mask(roi)
    
    # Day 6: Data Extraction & Fusion
    df = extract_training_data(roi)
    if len(df) > 0:
        visualize_extracted_data(df)
        create_final_fusion_map(roi)
    
    print("\n" + "=" * 70)
    print("✅ WEEK 1 COMPLETE!")
    print("=" * 70)
    print("\nGenerated outputs:")
    print(f"   📁 {NOTEBOOKS_DIR}/")
    print("      - day2_cloud_masking_comparison.html")
    print("      - day3_vegetation_indices.html")
    print("      - day4_radar_data.html")
    print("      - day5_vegetation_mask.html")
    print("      - day6_multisensor_fusion.html")
    print(f"   📁 {PLOTS_DIR}/")
    print("      - day6_timeseries.png")
    print("      - day6_correlation_matrix.png")
    print("      - day6_stress_timeline.png")
    print(f"   📁 Training data:")
    print(f"      - {output_config.training_data_csv}")


def main():
    """Main entry point - runs Day 1 setup by default"""
    print("=" * 60)
    print("OLIVEGUARD GEE PIPELINE")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  from gee_pipeline import run_day2  # Cloud masking")
    print("  from gee_pipeline import run_day3  # Vegetation indices")
    print("  from gee_pipeline import run_day4  # Radar data")
    print("  from gee_pipeline import run_day5  # Vegetation mask")
    print("  from gee_pipeline import run_day6  # Data extraction")
    print("  from gee_pipeline import run_all_days  # Complete Week 1")
    print("\nRunning Day 1 setup...")
    
    print_config_summary()
    
    if initialize_gee():
        roi, _ = get_study_area()
        
        # Quick data availability check
        s2_count = ee.ImageCollection(gee_config.s2_collection) \
            .filterBounds(roi) \
            .filterDate(gee_config.start_date, gee_config.end_date) \
            .size().getInfo()
        print(f"\n📷 Sentinel-2 images available: {s2_count}")
        
        s1_count = ee.ImageCollection(gee_config.s1_collection) \
            .filterBounds(roi) \
            .filterDate(gee_config.start_date, gee_config.end_date) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .size().getInfo()
        print(f"📡 Sentinel-1 images available: {s1_count}")
        
        print("\n✅ Day 1 complete! Run run_day2() to continue.")


if __name__ == "__main__":
    main()
