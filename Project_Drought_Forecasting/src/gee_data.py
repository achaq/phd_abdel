import ee
import pandas as pd
import datetime

def initialize_gee():
    """Initializes the Earth Engine API."""
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print("GEE initialization failed. Attempting to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize()
            print("Google Earth Engine authenticated and initialized.")
        except Exception as e2:
            print(f"Failed to authenticate/initialize GEE: {e2}")
            raise

def get_morocco_boundary():
    """Returns the geometry of Morocco."""
    # Simplified geometry or feature collection for Morocco
    # Using LSIB (Large Scale International Boundary) dataset
    dataset = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    morocco = dataset.filter(ee.Filter.eq("country_na", "Morocco"))
    return morocco.geometry()

def get_precipitation_chirps(start_date, end_date, geometry):
    """
    Fetches daily precipitation data from CHIRPS.
    """
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)
    
    def reduce_region(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=5000,
            maxPixels=1e9
        )
        return image.set('precip', mean.get('precipitation'))

    reduced = chirps.map(reduce_region)
    
    # Extract data
    dates = reduced.aggregate_array('system:time_start').getInfo()
    precip = reduced.aggregate_array('precip').getInfo()
    
    df = pd.DataFrame({
        'date': [datetime.datetime.fromtimestamp(d/1000).date() for d in dates],
        'precipitation': precip
    })
    return df

def get_temperature_era5(start_date, end_date, geometry):
    """
    Fetches temperature data from ERA5 Land.
    """
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)
    
    def reduce_region(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=11132,
            maxPixels=1e9
        )
        return image.set('temp_2m', mean.get('temperature_2m'))

    reduced = era5.map(reduce_region)
    
    dates = reduced.aggregate_array('system:time_start').getInfo()
    # Temp is in Kelvin, converting to Celsius could be done here or later
    temp = reduced.aggregate_array('temp_2m').getInfo()
    
    df = pd.DataFrame({
        'date': [datetime.datetime.fromtimestamp(d/1000).date() for d in dates],
        'temperature_k': temp
    })
    return df

def get_ndvi_modis(start_date, end_date, geometry):
    """
    Fetches NDVI data from MODIS (MOD13Q1).
    """
    modis = ee.ImageCollection("MODIS/006/MOD13Q1") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)
    
    def reduce_region(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=250,
            maxPixels=1e9
        )
        return image.set('NDVI', mean.get('NDVI'))

    reduced = modis.map(reduce_region)
    
    dates = reduced.aggregate_array('system:time_start').getInfo()
    ndvi = reduced.aggregate_array('NDVI').getInfo()
    
    df = pd.DataFrame({
        'date': [datetime.datetime.fromtimestamp(d/1000).date() for d in dates],
        'NDVI': ndvi
    })
    # MODIS NDVI is scaled by 0.0001
    df['NDVI'] = df['NDVI'] * 0.0001
    return df

if __name__ == "__main__":
    # Example usage
    try:
        initialize_gee()
        
        start_date = '2020-01-01'
        end_date = '2020-02-01'
        roi = get_morocco_boundary()
        
        print("Fetching CHIRPS data...")
        df_precip = get_precipitation_chirps(start_date, end_date, roi)
        print(df_precip.head())
        
        print("\nFetching ERA5 data...")
        df_temp = get_temperature_era5(start_date, end_date, roi)
        print(df_temp.head())
        
        print("\nFetching MODIS NDVI data...")
        df_ndvi = get_ndvi_modis(start_date, end_date, roi)
        print(df_ndvi.head())
        
    except Exception as e:
        print(f"An error occurred: {e}")
