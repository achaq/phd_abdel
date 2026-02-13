# Data Acquisition Strategy

## 1. Groundwater PINN (Project 1)

### Primary Data: Piezometric Heads
To train the PINN, we need historical groundwater head ($h$) data.
*   **Source:** Agence du Bassin Hydraulique (ABH) de Souss-Massa.
*   **Action:**
    1.  Locate the "Plan Directeur d'Aménagement Intégré des Ressources en Eau (PDAIRE)" reports for Souss-Massa (available online or via ABH library).
    2.  Extract Piezometric Maps (contour lines) for years 2010, 2015, 2020.
    3.  **Digitization:** Use QGIS (Georeferencer plugin) to digitize these maps into Point Shapefiles (x, y, h).
    4.  **Target Volume:** ~50-100 data points distributed across the aquifer.

### Secondary Data: Boundary Conditions
*   **Aquifer Geometry:** Use the ABH maps to define the polygon boundary of the Souss-Massa unconfined aquifer.
*   **Recharge ($R$):**
    *   *Option A (Simple):* Use average annual recharge values from literature (e.g., 20-50 mm/year).
    *   *Option B (Advanced):* Download CHIRPS precipitation data from GEE and apply a recharge coefficient (e.g., 5% of rainfall).

## 2. Orchard Segmentation (Project 2)

### Primary Data: High-Resolution Imagery
To segment individual trees, we need sub-meter resolution imagery.
*   **Source:** Google Earth Satellite Tiles (via `leafmap` / `geemap`).
*   **Resolution:** ~30-50 cm/pixel (Zoom level 19).
*   **Locations (Test Sites):**
    *   **Almonds:** Region of Tafraout or outlying Souss plains.
    *   **Olives:** Haouz plain (Marrakech) or Taroudant.
*   **Action:** Use the provided `data_download.ipynb` to download GeoTIFFs for 2-3 specific orchards (approx. 10 hectares each).

### Validation Data: Vegetation Indices
To correlate canopy size with tree health.
*   **Source:** Sentinel-2 (Level-2A) via Google Earth Engine.
*   **Indices:** NDVI, NDWI.
*   **Action:** Extract time-series data for the years 2023-2025 over the segmented masks.

