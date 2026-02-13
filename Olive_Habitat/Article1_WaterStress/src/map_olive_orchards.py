import ee
import geemap
import os
import certifi

# Fix for macOS SSL Certificate error
os.environ['SSL_CERT_FILE'] = certifi.where()

def initialize_gee():
    """Initializes Google Earth Engine."""
    try:
        ee.Initialize(project='phd1-481917')
    except:
        ee.Authenticate()
        ee.Initialize(project='phd1-481917')

def get_sentinel2_collection(roi, start_date, end_date):
    """Filters Sentinel-2 collection for the ROI and date range."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(lambda img: img.divide(10000).copyProperties(img, ['system:time_start']))
    return s2

def calculate_ndvi(image):
    """Calculates NDVI for a single image."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def create_olive_mask(roi):
    """
    Creates a mask for olive orchards based on NDVI phenology.
    Olive Logic:
    1. Evergreen: Low NDVI variance throughout the year (unlike crops).
    2. Moderate NDVI: Values typically between 0.3 and 0.7 (not dense forest, not bare soil).
    """
    
    # 1. Get a full year of data to analyze seasonality
    s2_year = get_sentinel2_collection(roi, '2023-01-01', '2023-12-31')
    s2_with_ndvi = s2_year.map(calculate_ndvi)
    
    # 2. Calculate Statistics
    ndvi_stats = s2_with_ndvi.select('NDVI').reduce(ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True
    ))
    
    # 3. Define Thresholds (These need tuning for Ghafsai!)
    # Olives are evergreen, so StdDev should be low (stable greenness).
    # Wheat/Barley change drastically (high StdDev).
    # Bare soil has very low mean NDVI.
    
    olive_mask = ndvi_stats.select('NDVI_mean').gt(0.25) \
        .And(ndvi_stats.select('NDVI_mean').lt(0.70)) \
        .And(ndvi_stats.select('NDVI_stdDev').lt(0.15))
    
    return olive_mask.clip(roi)

def main():
    initialize_gee()
    
    # Redefine ROI from previous step
    coords = [
        [-4.947147546412382, 34.620655908492395],
        [-4.689655480982695, 34.620655908492395],
        [-4.689655480982695, 34.72964124033107],
        [-4.947147546412382, 34.72964124033107],
        [-4.947147546412382, 34.620655908492395]
    ]
    roi = ee.Geometry.Polygon([coords])
    
    # Generate Mask
    print("Generating Olive Orchard Mask...")
    olive_mask = create_olive_mask(roi)
    
    # Visualization
    Map = geemap.Map(center=[34.675, -4.818], zoom=12)
    Map.addLayer(roi, {'color': 'red'}, 'Study Area')
    
    # Load a background image
    image = get_sentinel2_collection(roi, '2023-05-01', '2023-06-30').median().clip(roi)
    Map.addLayer(image, {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}, 'True Color Image')
    
    # Show the mask (Green pixels = Potential Olives)
    Map.addLayer(olive_mask.selfMask(), {'palette': ['green']}, 'Predicted Olive Orchards')
    
    output_file = "Olive_Habitat/Article1_WaterStress/notebooks/olive_mask_map.html"
    Map.to_html(output_file)
    print(f"Map saved to {output_file}. Open this file in your browser to inspect the results.")

if __name__ == "__main__":
    main()
