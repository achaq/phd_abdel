# Weekly Research Plan: Deep Learning for Drought & Yield Forecasting in Morocco
**PhD Thesis Context:** AI Applications in Climate Change Impacts in Morocco

This plan is designed to produce a high-impact journal article (e.g., *Journal of Hydrology*, *Computers and Electronics in Agriculture*) within 10-12 weeks. It emphasizes rigorous methodology, reproducibility, and specific relevance to Moroccan agro-climatic zones.

---

## Phase 1: Data Acquisition & Preprocessing (Weeks 1-3)

### Week 1: Satellite Data Pipeline (GEE)
- **Objective**: Finalize the automated extraction of climate variables for all target Moroccan regions.
- **Tasks**:
  - [ ] Refine `src/gee_data.py` to extract data for specific provinces (e.g., Settat, Al Haouz, Meknes).
  - [ ] Download historical data (2000-2024):
    - **Precipitation**: CHIRPS (Daily).
    - **Temperature**: ERA5-Land (Daily Aggregated).
    - **Vegetation**: MODIS NDVI (16-day) and Sentinel-2 (since 2015).
    - **Soil Moisture**: NASA-USDA SMAP (if resolution allows) or ERA5 Soil Water.
  - [ ] **Deliverable**: Raw CSVs for each region stored in `data/raw/`.

### Week 2: Ground Truth & Socio-Economic Data
- **Objective**: Build the "Target" variables dataset.
- **Tasks**:
  - [ ] Digitize/Clean historical yield statistics (Ministry of Agriculture MAPMDREF reports) for Cereal (Wheat/Barley).
  - [ ] Calculate meteorological drought indices (SPI-3, SPI-6, SPEI) using the CHIRPS/ERA5 data to serve as secondary ground truth.
  - [ ] **Thesis alignment**: Search for socio-economic indicators (market prices, rural employment) to discuss impacts in the introduction.
  - [ ] **Deliverable**: Cleaned `ground_truth.csv` with columns `[Region, Date, Yield, SPI, SPEI]`.

### Week 3: Preprocessing & alignment
- **Objective**: Create the "Analysis-Ready" dataset (ARD).
- **Tasks**:
  - [ ] Implement `src/preprocessing.py` to:
    - Resample all satellite data to a common temporal resolution (e.g., Dekadal/10-day or Monthly).
    - Handle missing values (linear interpolation for short gaps).
    - Normalize features (MinMax or Z-Score).
    - Create sliding windows (Inputs: $t-12$ to $t$, Output: $t+1$ to $t+3$).
  - [ ] Split data: Training (2000-2018), Validation (2019-2021), Testing (2022-2024).
  - [ ] **Deliverable**: `data/processed/train_tensor.pt`, `test_tensor.pt`.

---

## Phase 2: Model Development & Experiments (Weeks 4-7)

### Week 4: Baseline Models
- **Objective**: Establish performance benchmarks.
- **Tasks**:
  - [ ] Train statistical baselines: ARIMA, Linear Regression.
  - [ ] Train ML baselines: Random Forest, XGBoost (tabular approach).
  - [ ] Metric tracking: RMSE, MAE, R².
  - [ ] **Deliverable**: Baseline results table.

### Week 5: Deep Learning Implementation (CNN-LSTM)
- **Objective**: Train the core spatio-temporal model.
- **Tasks**:
  - [ ] Refine `src/model.py`:
    - **1D-CNN**: To extract short-term temporal patterns from weather windows.
    - **LSTM/GRU**: To capture long-term seasonal dependencies.
    - **Attention (Optional)**: Implement simple attention mechanism to weigh important time steps.
  - [ ] Hyperparameter tuning (Learning rate, hidden dims, dropout).
  - [ ] **Deliverable**: Best model checkpoint `models/cnn_lstm_best.pth`.

### Week 6: Spatial Generalization (Thesis Contribution)
- **Objective**: Test model transferability between zones.
- **Tasks**:
  - [ ] Train on North/Central (Fes-Meknes) -> Test on South (Souss-Massa).
  - [ ] Analyze where the model fails (e.g., different crop varieties, irrigation practices).
  - [ ] **Deliverable**: "Generalization Gap" analysis report.

### Week 7: Hybrid Interpretability
- **Objective**: Open the "Black Box" (Critical for PhD).
- **Tasks**:
  - [ ] Apply SHAP (SHapley Additive exPlanations) or Integrated Gradients.
  - [ ] Identify which variables drive drought predictions in Morocco (e.g., "Is Precipitation lag-3 more important than Temperature?").
  - [ ] **Deliverable**: Feature importance plots.

---

## Phase 3: Writing & Thesis Integration (Weeks 8-10)

### Week 8: Results Visualization
- **Objective**: Generate publication-quality figures.
- **Tasks**:
  - [ ] Plot 1: Study Area Map (Moroccan Agro-Climatic Zones).
  - [ ] Plot 2: Time-series comparison (Predicted vs Observed Yields/SPI).
  - [ ] Plot 3: Scatter plots of performance ($Predicted \times Observed$).
  - [ ] Plot 4: SHAP summary plot.

### Week 9: Drafting the Article (Methodology & Results)
- **Objective**: Write the technical core.
- **Tasks**:
  - [ ] Write "Methodology": Mathematical formulation of CNN-LSTM.
  - [ ] Write "Results": Comparative tables (DL vs Baselines), Generalization study.
  - [ ] Update LaTeX draft.

### Week 10: Introduction, Discussion & Polishing
- **Objective**: Frame the narrative.
- **Tasks**:
  - [ ] Write "Introduction": Context of Climate Change in MENA, Data Scarcity Paradox.
  - [ ] Write "Discussion": Implications for Moroccan water policy (Plan Maroc Vert / Generation Green).
  - [ ] **Thesis Connection**: Explicitly link findings to the broader thesis theme of "AI for Adaptation".
  - [ ] **Deliverable**: Full Draft v1.0 ready for review.

---

## Key PhD Milestones
- **Methodological Contribution**: Proving that hybrid DL (CNN-LSTM) outperforms standard hydrological models in data-scarce MENA regions.
- **Practical Contribution**: A working prototype for seasonal drought early warning.
