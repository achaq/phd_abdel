# Deep Learning for Drought and Yield Forecasting in Morocco

## Overview
This project applies Deep Learning (CNN-LSTM) to forecast drought indices and crop yields in Moroccan agro-climatic zones. It integrates remote sensing data (precipication, temperature, vegetation indices) from Google Earth Engine with local ground truth data.

## Structure
- `data/`: Raw and processed data.
- `src/`: Source code for GEE extraction (`gee_data.py`), preprocessing, and modeling.
- `notebooks/`: Analysis and visualization.
- `models/`: Trained model checkpoints.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate Google Earth Engine:
   ```bash
   earthengine authenticate
   ```
