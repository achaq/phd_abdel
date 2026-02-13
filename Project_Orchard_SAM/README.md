# Project 2: Zero-Shot Orchard Monitoring with SAM (Segment Anything Model)

## Overview
This project leverages the "Segment Anything Model" (SAM) and its geospatial adapters (`segment-geospatial`, `LangSAM`) to perform automated delineation of high-value tree crops (Almond, Olive) in Morocco. The goal is to calculate tree canopy areas to estimate water requirements without the need for massive, manually labeled datasets.

## Objectives
1.  **Data Acquisition**: Download high-resolution RGB imagery of orchard test sites (Souss-Massa, Haouz) using Google Earth Engine or Google Maps tiles.
2.  **Segmentation**:
    -   Use **LangSAM** (Language-based SAM) to segment trees using text prompts ("tree", "orchard").
    -   Use **SAMGeo** for grid-based or point-prompted segmentation.
3.  **Analysis**: Calculate the total canopy area ($m^2$) per hectare and correlate with remote sensing indices (NDVI/ETa).

## Workflow
1.  **Define ROI**: Select Regions of Interest (ROIs) in the Souss-Massa basin containing almond/olive orchards.
2.  **Download Imagery**: Run `data_download.ipynb` to fetch high-res satellite/aerial basemaps.
3.  **Run SAM**:
    -   Load the image.
    -   Initialize LangSAM.
    -   Prompt: "tree".
    -   Filter results by area (to remove bushes or noise) and shape (circularity).
4.  **Export Results**: Save the segmentation masks as GeoTIFFs or Shapefiles for GIS analysis.

## Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Open `data_download.ipynb` to acquire imagery.
3.  Run the segmentation scripts (to be developed in Phase 2).

