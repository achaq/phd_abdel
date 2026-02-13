// =============================================================================
// Article 1: Ghafsai Olive Orchard Study Area Setup
// =============================================================================

// 1. Define the Region of Interest (ROI)
// Coordinates extracted from your selection
var roi = ee.Geometry.Polygon([
  [
    [-4.947147546412382, 34.620655908492395],
    [-4.689655480982695, 34.620655908492395],
    [-4.689655480982695, 34.72964124033107],
    [-4.947147546412382, 34.72964124033107],
    // Closing the polygon is good practice
    [-4.947147546412382, 34.620655908492395]
  ]
]);

// 2. Visualization Setup
Map.centerObject(roi, 12);
Map.addLayer(roi, {color: 'red'}, 'Study Area (Ghafsai)');

// 3. Quick Check: Load a recent Sentinel-2 Image to verify the location
// This helps ensure we are looking at the right olive orchards
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2024-01-01', '2024-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median()
    .clip(roi);

var visParams = {
  min: 0,
  max: 3000,
  bands: ['B4', 'B3', 'B2'], // True Color (Red, Green, Blue)
};

Map.addLayer(s2, visParams, 'Sentinel-2 RGB (2024 Median)');

// =============================================================================
// Next Step Preparation: Olive Orchard Masking
// =============================================================================
// For the next step (Week 2), we will calculate indices like NDVI here.
print('Area of Interest defined:', roi.area().divide(10000), 'hectares');
