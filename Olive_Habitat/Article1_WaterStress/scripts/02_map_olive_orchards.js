// =============================================================================
// Article 1: Step 2 - Olive Orchard Mapping (Phenology Based)
// =============================================================================

// 1. Define ROI (Same as Step 1)
var roi = ee.Geometry.Polygon([
  [
    [-4.947147546412382, 34.620655908492395],
    [-4.689655480982695, 34.620655908492395],
    [-4.689655480982695, 34.72964124033107],
    [-4.947147546412382, 34.72964124033107],
    [-4.947147546412382, 34.620655908492395]
  ]
]);

Map.centerObject(roi, 12);

// 2. Helper Function: Calculate NDVI
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

// 3. Load Sentinel-2 Data for a Full Year (2023)
// We need a full year to see the seasonal changes (phenology)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(addNDVI);

// 4. Calculate Phenology Metrics
// Mean: How green is it on average? (Olives are moderately green, ~0.3 - 0.6)
// StdDev: How much does it change? (Olives change little = low StdDev. Crops change a lot = high StdDev)
var ndviStats = s2.select('NDVI').reduce(ee.Reducer.mean().combine({
  reducer2: ee.Reducer.stdDev(),
  sharedInputs: true
}));

// 5. Define Thresholds for Olive Classification
// NOTE: These values are initial estimates. Inspect the map and adjust them!
// Rule 1: Vegetation must be present (Mean NDVI > 0.25)
// Rule 2: Not dense forest (Mean NDVI < 0.70)
// Rule 3: Stable throughout the year (StdDev < 0.15) - separates trees from annual crops
var oliveMask = ndviStats.select('NDVI_mean').gt(0.25)
    .and(ndviStats.select('NDVI_mean').lt(0.70))
    .and(ndviStats.select('NDVI_stdDev').lt(0.15));

// 6. Visualization
var rgbVis = {min: 0, max: 3000, bands: ['B4', 'B3', 'B2']};
var maskVis = {min: 0, max: 1, palette: ['white', 'green']};

// Background Image (Median of Summer to see the trees clearly)
var summerImg = s2.filterDate('2023-06-01', '2023-08-31').median().clip(roi);

Map.addLayer(summerImg, rgbVis, 'True Color (Summer)');
Map.addLayer(oliveMask.updateMask(oliveMask).clip(roi), {palette: ['green']}, 'Predicted Olive Orchards');

// 7. Validation Helper
// Click on the map to see values to help tune thresholds
Map.onClick(function(coords) {
  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var stats = ndviStats.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 10
  });
  print('Clicked Pixel Stats:', stats);
});
