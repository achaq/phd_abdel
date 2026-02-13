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
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print("Google Earth Engine not initialized. Attempting authentication...")
        try:
            ee.Authenticate()
            ee.Initialize(project='phd1-481917')
            print("Google Earth Engine initialized successfully after authentication.")
        except Exception as e:
            print(f"Failed to initialize Google Earth Engine: {e}")
            return None
    return True

def define_study_area():
    """Defines the Ghafsai study area using the user-provided polygon."""
    # Coordinates from the user's GEE selection
    # Format: [[lon, lat], [lon, lat], ...] - ensuring the polygon is closed
    coords = [
        [-4.947147546412382, 34.620655908492395],
        [-4.689655480982695, 34.620655908492395],
        [-4.689655480982695, 34.72964124033107],
        [-4.947147546412382, 34.72964124033107],
        [-4.947147546412382, 34.620655908492395]  # Closing the loop
    ]
    
    # Create the Polygon geometry
    roi = ee.Geometry.Polygon([coords])
    
    # Calculate the centroid for the point/map center
    point = roi.centroid()
    
    return roi, point

def main():
    if not initialize_gee():
        return

    roi, point = define_study_area()
    print("Study Area Defined (10km buffer around Ghafsai):")
    print(roi.getInfo())

    # Create a map to visualize
    Map = geemap.Map(center=[34.6333, -4.6000], zoom=11)
    Map.addLayer(roi, {'color': 'red'}, 'Ghafsai Study Area')
    Map.addLayer(point, {'color': 'blue'}, 'Ghafsai Center')
    
    # Save the map as HTML
    output_map = "Olive_Habitat/Article1_WaterStress/notebooks/ghafsai_map.html"
    Map.to_html(output_map)
    print(f"Map saved to {output_map}")

if __name__ == "__main__":
    main()
