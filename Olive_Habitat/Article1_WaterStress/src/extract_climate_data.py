import ee
import geemap
import pandas as pd
import os
import certifi

# Fix for macOS SSL Certificate error
os.environ['SSL_CERT_FILE'] = certifi.where()

def initialize_gee():
    try:
        ee.Initialize(project='phd1-481917')
    except:
        ee.Authenticate()
        ee.Initialize(project='phd1-481917')

def get_olive_mask(roi):
    """Re-creates the olive mask (simplified for extraction)."""
    # Use a specific year for masking to stay consistent
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
    
    stats = s2.select('NDVI').reduce(ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True
    ))
    
    # Same thresholds as Step 2
    mask = stats.select('NDVI_mean').gt(0.25) \
        .And(stats.select('NDVI_mean').lt(0.70)) \
        .And(stats.select('NDVI_stdDev').lt(0.15))
    return mask

def extract_data(roi, start_date, end_date, output_csv):
    """Extracts climate and satellite data for olive pixels."""
    print(f"Starting data extraction from {start_date} to {end_date}...")
    
    olive_mask = get_olive_mask(roi)
    
    # 1. Sentinel-2 (Vegetation Indices)
    # ----------------------------------
    def process_s2(img):
        # Calculate indices
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI') # Gao's NDWI (Water content)
        
        # Apply Olive Mask
        img_masked = img.select([]).addBands([ndvi, ndwi]).updateMask(olive_mask)
        
        # Reduce to region mean (average condition of all olive trees)
        stats = img_masked.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=20, # S2 resolution
            maxPixels=1e9
        )
        return img.set(stats)

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(process_s2)
    
    # 2. ERA5-Land (Climate Data)
    # ---------------------------
    # ERA5 is coarser (11km) but daily/hourly
    def process_climate(img):
        # Convert Kelvin to Celsius for Temperature
        temp = img.select('temperature_2m').subtract(273.15).rename('Temperature_C')
        precip = img.select('total_precipitation_sum').multiply(1000).rename('Precipitation_mm')
        
        # Apply Olive Mask (Climate over the olive areas)
        # Note: We reproject mask to match ERA5 or just reduce region directly 
        # Since ERA5 is coarse, reducing over the detailed olive mask works as a weighted average
        img_masked = temp.addBands(precip).updateMask(olive_mask)
        
        stats = img_masked.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=11132, # ERA5 resolution (~11km)
            maxPixels=1e9
        )
        return img.set(stats)

    climate_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .map(process_climate)

    # 3. Fetch Data (Client-side)
    # ---------------------------
    print("Fetching Sentinel-2 Data...")
    s2_data = s2_col.aggregate_array('NDVI').getInfo() # Check if any data exists first
    if not s2_data:
        print("No Sentinel-2 data found for this period.")
        return

    # Helper to convert collection to DataFrame
    def get_df(collection, columns):
        # We fetch the list of properties (dictionaries)
        data_list = collection.reduceColumns(ee.Reducer.toList(len(columns)), columns).get('list').getInfo()
        df = pd.DataFrame(data_list, columns=columns)
        # Convert system:time_start to datetime
        if 'system:time_start' in columns:
            df['Date'] = pd.to_datetime(df['system:time_start'], unit='ms')
            df.drop(columns=['system:time_start'], inplace=True)
        return df

    print("Downloading Satellite Time Series...")
    df_s2 = get_df(s2_col, ['system:time_start', 'NDVI', 'NDWI'])
    
    print("Downloading Climate Time Series...")
    df_climate = get_df(climate_col, ['system:time_start', 'Temperature_C', 'Precipitation_mm'])

    # 4. Merge Datasets
    # -----------------
    # Merge on Date (nearest or daily match). S2 is every ~5 days, Climate is daily.
    # We keep daily climate and fill S2 values (interpolation) or keep S2 days.
    # For "Early Detection", we usually want daily data with interpolated satellite indices.
    
    # Set Date as index
    df_s2.set_index('Date', inplace=True)
    df_climate.set_index('Date', inplace=True)
    
    # Resample S2 to daily (interpolate) to match climate
    df_s2_daily = df_s2.resample('D').mean().interpolate(method='linear')
    
    # Join
    df_final = df_climate.join(df_s2_daily, how='inner') # Only keep days where we have valid interpolated S2 coverage
    
    # Save
    df_final.to_csv(output_csv)
    print(f"Success! Data saved to {output_csv}")
    print(df_final.head())

def main():
    initialize_gee()
    
    # Define ROI (Ghafsai Polygon)
    coords = [
        [-4.947147546412382, 34.620655908492395],
        [-4.689655480982695, 34.620655908492395],
        [-4.689655480982695, 34.72964124033107],
        [-4.947147546412382, 34.72964124033107],
        [-4.947147546412382, 34.620655908492395]
    ]
    roi = ee.Geometry.Polygon([coords])
    
    # Output path
    output_dir = "Olive_Habitat/Article1_WaterStress/data"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "ghafsai_olive_data_2023_2024.csv")
    
    # Run Extraction (2 years of data)
    extract_data(roi, '2023-01-01', '2024-12-31', output_csv)

if __name__ == "__main__":
    main()
