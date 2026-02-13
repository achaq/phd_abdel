---
name: Deep_Learning_Drought_Forecast_Weekly_Plan
overview: A 10-12 week research plan for the 'Deep Learning for Drought and Yield Forecasting in Morocco' article, tailored for a PhD thesis on AI and Climate Change. Includes data acquisition, model development, and writing phases.
todos:
  - id: refine_gee_provinces
    content: Refine `src/gee_data.py` to extract data for specific provinces (Settat, Al Haouz, Meknes)
    status: pending
  - id: download_historical_data
    content: Download historical data (2000-2024) for Precip, Temp, NDVI, Soil Moisture
    status: pending
  - id: clean_yield_data
    content: Digitize and clean historical yield statistics for Wheat/Barley
    status: pending
  - id: calculate_drought_indices
    content: Calculate SPI/SPEI drought indices
    status: pending
  - id: implement_preprocessing
    content: Implement preprocessing pipeline (resampling, normalization, windowing)
    status: pending
  - id: train_baselines
    content: Train baseline models (ARIMA, RF, XGBoost)
    status: pending
  - id: train_cnn_lstm
    content: Train and tune CNN-LSTM model
    status: pending
  - id: generalization_test
    content: Perform spatial generalization tests (Fes-Meknes -> Souss-Massa)
    status: pending
  - id: shap_analysis
    content: Generate SHAP feature importance plots
    status: pending
  - id: draft_paper_core
    content: Draft Methodology and Results sections in LaTeX
    status: pending
---

# Weekly Research Plan: Deep Learning for Drought & Yield Forecasting in Morocco

## Phase 1: Data Acquisition & Preprocessing (Weeks 1-3)

### Week 1: Satellite Data Pipeline (GEE)

- **Objective**: Finalize the automated extraction of climate variables for all target Moroccan regions.
- **Tasks**:
- Refine `src/gee_data.py` to extract data for specific provinces (e.g., Settat, Al Haouz, Meknes).
- Download historical data (2000-2024):
- **Precipitation**: CHIRPS (Daily).
- **Temperature**: ERA5-Land (Daily Aggregated).
- **Vegetation**: MODIS NDVI (16-day) and Sentinel-2 (since 2015).
- **Soil Moisture**: NASA-USDA SMAP (if resolution allows) or ERA5 Soil Water.
- **Deliverable**: Raw CSVs for each region stored in `data/raw/`.

### Week 2: Ground Truth & Socio-Economic Data

- **Objective**: Build the "Target" variables dataset.
- **Tasks**:
- Digitize/Clean historical yield statistics (Ministry of Agriculture MAPMDREF reports) for Cereal (Wheat/Barley).
- Calculate meteorological drought indices (SPI-3, SPI-6, SPEI) using the CHIRPS/ERA5 data.
- **Thesis alignment**: Search for socio-economic indicators (market prices, rural employment) to discuss impacts.
- **Deliverable**: Cleaned `ground_truth.csv`.

### Week 3: Preprocessing & Alignment

- **Objective**: Create the "Analysis-Ready" dataset (ARD).
- **Tasks**:
- Implement `src/preprocessing.py` to:
- Resample all satellite data to a common temporal resolution (e.g., Dekadal/10-day or Monthly).
- Handle missing values (linear interpolation).
- Normalize features (MinMax or Z-Score).
- Create sliding windows (Inputs: t-12 to t, Output: t+1 to t+3).
- Split data: Training (2000-2018), Validation (2019-2021), Testing (2022-2024).
- **Deliverable**: `data/processed/train_tensor.pt`, `test_tensor.pt`.

## Phase 2: Model Development & Experiments (Weeks 4-7)

### Week 4: Baseline Models

- **Objective**: Establish performance benchmarks.
- **Tasks**:
- Train statistical baselines: ARIMA, Linear Regression.
- Train ML baselines: Random Forest, XGBoost (tabular approach).
- Track metrics: RMSE, MAE, R².
- **Deliverable**: Baseline results table.

### Week 5: Deep Learning Implementation (CNN-LSTM)

- **Objective**: Train the core spatio-temporal model.
- **Tasks**:
- Refine `src/model.py`:
- **1D-CNN**: Extract short-term temporal patterns.
- **LSTM/GRU**: Capture long-term seasonal dependencies.
- Hyperparameter tuning (Learning rate, hidden dims, dropout).
- **Deliverable**: Best model checkpoint `models/cnn_lstm_best.pth`.

### Week 6: Spatial Generalization (Thesis Contribution)

- **Objective**: Test model transferability between zones.
- **Tasks**:
- Train on North/Central (Fes-Meknes) -> Test on South (Souss-Massa).
- Analyze where the model fails (e.g., different crop varieties, irrigation practices).
- **Deliverable**: "Generalization Gap" analysis report.

### Week 7: Hybrid Interpretability

- **Objective**: Open the "Black Box" (Critical for PhD).
- **Tasks**:
- Apply SHAP (SHapley Additive exPlanations).
- Identify primary drought drivers in Morocco.
- **Deliverable**: Feature importance plots.

## Phase 3: Writing & Thesis Integration (Weeks 8-10)

### Week 8: Results Visualization

- **Objective**: Generate publication-quality figures.
- **Tasks**:
- Plot Study Area Map.
- Plot Time-series comparison (Predicted vs Observed).
- Plot Scatter plots of performance.
- Plot SHAP summary.

### Week 9: Drafting the Article (Methodology & Results)

- **Objective**: Write the technical core.
- **Tasks**:
- Write "Methodology": Mathematical formulation of CNN-LSTM.
- Write "Results": Comparative tables, Generalization study.
- Update LaTeX draft.

### Week 10: Introduction, Discussion & Polishing

- **Objective**: Frame the narrative.
- **Tasks**:
- Write "Introduction": Context of Climate Change in MENA.
- Write "Discussion": Implications for Moroccan water policy.
- **Thesis Connection**: Link findings to "AI for Adaptation".
- **Deliverable**: Full Draft v1.0.