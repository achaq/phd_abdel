# OliveGuard: AI-Powered Digital Twin for Water Stress Detection

**Project**: AI-Based Olive Stress Detection Dashboard & Scientific Research  
**Target Region**: Ghafsai, Morocco (Pre-Rif)  
**Objective**: Build a full-stack Decision Support System (DSS) for water stress detection in olive orchards using Google Earth Engine (GEE), Machine Learning, and publish a high-impact scientific article.

---

## 🎯 Research Novelty & Impact

Based on the state-of-the-art analysis, this project addresses **three critical gaps**:

| Gap | Description | Our Solution |
|-----|-------------|--------------|
| **Geographic Gap** | All existing Moroccan studies focus on Tadla, Haouz, or Saïss plains | First AI-based water stress study targeting **Ghafsai/Pre-Rif** |
| **Methodological Gap** | Standard approaches fail in mountainous/mixed terrain | **Topographic Correction** + **Unsupervised Deep Learning (LSTM Autoencoder)** |
| **Operational Gap** | Research stays in academia | **MVP Digital Twin**: Instant, session-based analysis tool for farmers |

**Key Innovations**:
- **Unsupervised Anomaly Detection**: Using **LSTM Autoencoders** to learn "Normal" growth patterns and flagging deviations (High Reconstruction Error) as stress, solving the lack of ground truth.
- **Phenological Matching**: Using **Dynamic Time Warping (DTW)** to compare current seasonal curves against historical baselines, robust to seasonal shifts (e.g., late rain).
- **Topographic Correction**: Illumination correction for optical and radiometric terrain flattening for radar.

---

## 🛠 Tech Stack (MVP Edition)

| Component | Technology |
|-----------|------------|
| **Core Engine** | Google Earth Engine (Python API) |
| **Backend** | Python FastAPI (Stateless, Synchronous) |
| **ML Framework** | PyTorch (LSTM Autoencoder) + `fastdtw` (Phenology) |
| **Frontend** | Next.js 14 + TypeScript + Tailwind CSS |
| **Mapping** | Leaflet + React-Leaflet |
| **Data Viz** | Recharts (Responsive & Interactive) |
| **Deployment** | Docker (Single container or simple Compose) |

---

## 📊 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-SENSOR DATA FUSION                              │
├─────────────────┬─────────────────┬─────────────────┬──────────────────────┤
│   Sentinel-2    │   Sentinel-1    │  Landsat 8/9    │      ERA5-Land       │
│   (Optical)     │   (Radar)       │   (Thermal)     │     (Climate)        │
├─────────────────┼─────────────────┼─────────────────┼──────────────────────┤
│ B2, B3, B4, B8  │ VV, VH          │ LST (Band 10)   │ Temperature          │
│ NDVI, NDWI      │ Backscatter     │ CWSI            │ Precipitation        │
│ SAVI, EVI       │ Diurnal Diff    │                 │ VPD                  │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬───────────┘
         │                 │                 │                   │
         └─────────────────┴─────────────────┴───────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   PREPROCESSING PIPELINE      │
                    ├───────────────────────────────┤
                    │ • Topographic Correction      │
                    │ • Pure Pixel Filtering        │
                    │ • Resolution Harmonization    │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   UNSUPERVISED LEARNING       │
                    │ Train on "Healthy Years"      │
                    │ (Programmatic ERA5 selection) │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     STRESS DETECTION          │
                    │ 1. LSTM Reconstruction Error  │
                    │ 2. DTW Dissimilarity Score    │
                    └───────────────────────────────┘
```

---

## 🗓 Timeline Overview

| Phase | Weeks | Focus | Deliverable |
|-------|-------|-------|-------------|
| **Phase 1** | 1-2 | Data Backend | Robust GEE Pipeline + Topographic Correction |
| **Phase 2** | 3-4 | MVP App | Functional "Draw & Analyze" Dashboard |
| **Phase 3** | 5-6 | Science & AI | LSTM Autoencoder + DTW Implementation |
| **Phase 4** | 7-8 | Article Writing | Draft Paper Ready for Submission |
| **Phase 5** | 9-10 | Advanced Features | Thermal/CWSI + Attention Map Visualization |
| **Phase 6** | 11-12 | Validation | Scientific Validation & Polish |

---

## 🗓 Phase 1: The Data Backend (Weeks 1-2)

*Focus: Building a scientifically rigorous multi-sensor data extraction pipeline.*

### Week 1: Enhanced GEE Data Pipeline

- [ ] **Day 1: Environment & Architecture Setup**
    - Install `earthengine-api`, `geemap`, `geopandas`
    - Authenticate GEE
    - Define Ghafsai ROI coordinates
    - **New**: Implement `TopographicCorrection` class (Sun-Canopy-Sensor + C-Correction)

- [ ] **Day 2: Sentinel-2 Collection & Corrections**
    - Fetch Sentinel-2 SR Harmonized
    - Apply **Topographic Illumination Correction** (using DEM)
    - Implement **QA60 + s2cloudless** masking (Aggressive threshold for mountains)
    - *Scientific Note*: Compare corrected vs. uncorrected spectral signatures

- [ ] **Day 3: Vegetation Index & Pure Pixel Filtering**
    - Calculate NDVI, NDWI, SAVI, EVI
    - **New**: Implement **Pure Pixel Filtering**. Only keep pixels where `NDVI > 0.4` and variance within 3x3 window is low (homogeneity).
    - This avoids training on "soil noise" in mixed orchards.

- [ ] **Day 4: Sentinel-1 Radar & Terrain Flattening**
    - Fetch Sentinel-1 GRD
    - Apply **Radiometric Terrain Flattening** (Volumetric model)
    - Apply Speckle filtering (Refined Lee)
    - *Scientific Note*: Essential for Ghafsai's steep slopes where radar shadows occur.

- [ ] **Day 5: Climate Data Downscaling**
    - Fetch ERA5-Land (9km)
    - **New**: Implement **Bilinear Interpolation** or Lapse-Rate correction to estimate temperature at plot altitude.
    - Merge with satellite data.

- [ ] **Day 6: Reference Dataset Creation (Comparative Study)**
    - **Objective**: Create two distinct datasets to validate the "Healthy" baseline.
    - **Dataset A (Climate-Driven)**: `training_data_climate_driven.csv`
        - Programmatically identify the top 3 "Wettest Years" using ERA5 precipitation data.
        - Extract vectors ONLY from these years.
    - **Dataset B (Data-Driven)**: `training_data_data_driven.csv`
        - Use K-Means clustering on the full time-series (2022-2025).
        - Filter to keep only the "Stable/Healthy" cluster.
    - *Scientific Note*: This duality allows us to compare "Climate-defined health" vs. "Statistically-defined health".

### Week 2: FastAPI Backend (MVP)

- [ ] **Day 1: Stateless Backend Setup**
    - Initialize FastAPI project (`api/`)
    - Define Pydantic models (Input: GeoJSON, Output: AnalysisResult)
    - **No Database**: Data flows strictly Input -> GEE -> Output.

- [ ] **Day 2: Core Analysis Endpoint**
    - `POST /analyze`:
        - Receive Polygon
        - Trigger `gee_pipeline.extract_time_series(polygon)`
        - Return JSON: `{ "ndvi_series": [...], "anomaly_score": 0.15, "dtw_distance": 42.5 }`
    - Implement robust timeout handling (GEE can be slow).

- [ ] **Day 3: GEE Service Optimization**
    - Optimize GEE scripts (`.reduceRegion` instead of `.sample`) for speed.
    - Ensure execution completes < 30s for small parcels.
    - Add basic caching via `functools.lru_cache` (in-memory only) for repeated calls.

- [ ] **Day 4: MVP "Mock" Inference**
    - Before the deep model is ready, implement a placeholder DTW using `scipy` or `fastdtw` between the fetched series and a static "Ideal Curve".
    - This proves the pipeline works end-to-end.

- [ ] **Day 5-6: Testing & Validation**
    - Unit tests for API endpoints.
    - Validate that Topographic Correction doesn't produce artifacts (negative values).
    - Test with "Known" locations (e.g., a reservoir, a dense forest).

---

## 🖥 Phase 2: The Interactive MVP (Weeks 3-4)

*Focus: A beautiful, functional, single-session tool. No logins, no database.*

### Week 3: Frontend Foundation & UX

- [ ] **Day 1: Next.js Setup & UI Library**
    - Initialize Next.js 14 + TypeScript.
    - Install **Tailwind CSS** + **shadcn/ui** (or similar) for polished look.
    - Structure: `components/Map`, `components/Charts`, `components/Layout`.

- [ ] **Day 2: Modern Map Interface**
    - `LeafletMap.tsx`: Full-screen or large container.
    - Add `Geosearch` to jump to Ghafsai/Locations.
    - Custom Map Controls (Zoom, Layer Toggle) for a "Pro" feel.

- [ ] **Day 3: Drawing Experience**
    - `leaflet-draw`: Customize the drawing tools (only Polygon/Rectangle).
    - **UX Polish**: "Click to start drawing", "Double click to finish".
    - Instant feedback: "Calculating area..." (Warn if > 10 Hectares to save GEE quotas).

- [ ] **Day 4: API Integration**
    - Hook up `POST /analyze`.
    - **Loading State**: Create a beautiful Skeleton Loader or Progress Bar ("Fetching Satellite Data...", "Correcting Terrain...", "Detecting Anomalies...").
    - Error Handling: "Cloud cover too high", "No vegetation found".

- [ ] **Day 5-6: Dashboard Layout**
    - Single Page View: Map on Left (or Top), Analysis on Right (or Bottom).
    - **Stat Cards**: "Anomaly Score (AI)", "Season Similarity (DTW)", "Water Content".
    - Use color coding (Green/Yellow/Red) based on the Anomaly Score.

### Week 4: Visualizations & Polish

- [ ] **Day 1: Interactive Time Series**
    - `Recharts` implementation.
    - **Innovation**: Plot the "Actual" curve (Solid Line) vs. "Reconstructed/Ideal" curve (Dashed Line).
    - The *gap* between them visually explains the stress.

- [ ] **Day 2: Comparative Analysis UI**
    - Toggle: "Show last 3 years".
    - Allow user to click points on the chart to update the "Status Card" to that date.

- [ ] **Day 3: Mobile Responsiveness**
    - Ensure map works on touch screens.
    - Stack charts vertically on mobile.
    - "Field Mode": High contrast option for viewing outside.

- [ ] **Day 4: PDF Report Generation (Client Side)**
    - Use `jspdf` or `html2canvas`.
    - "Download Report": Captures current view/charts and saves as PDF.
    - Value add for farmers/consultants.

- [ ] **Day 5-6: MVP Final Polish**
    - Fix z-index issues.
    - Smooth transitions/animations.
    - **Deployment**: Dockerize frontend + backend. Deploy to a single VPS or local runner for demo.

---

## 🔬 Phase 3: Science & AI (Weeks 5-6)

*Focus: Implementing the Unsupervised Anomaly Detection Engines.*

### Week 5: LSTM Autoencoder (The "Normal" Learner)

- [ ] **Day 1: Data Preparation**
    - Load `reference_healthy_vectors.csv`.
    - Normalize data (MinMax Scaler is critical for LSTMs).
    - Create sequences (e.g., 12-month sliding windows).

- [ ] **Day 2: Architecture Design**
    - Build PyTorch Model:
        - **Encoder**: LSTM (Input: 12xFeatures -> Hidden: 32 -> Latent: 8)
        - **Decoder**: LSTM (Latent: 8 -> Hidden: 32 -> Output: 12xFeatures)
    - Loss Function: Mean Squared Error (MSE) between Input and Output.

- [ ] **Day 3: Training**
    - Train ONLY on healthy/normal years.
    - The model learns to compress and reproduce "healthy olive signatures".
    - Save model as `lstm_autoencoder.pt`.

- [ ] **Day 4: Inference Logic (Anomaly Detection)**
    - Feed *current* (potentially stressed) data.
    - Calculate **Reconstruction Error** (MSE).
    - Define Threshold: If Error > $\mu + 2\sigma$, flag as **Anomaly/Stress**.

### Week 6: Dynamic Time Warping (The "Season" Matcher)

- [ ] **Day 1: Baseline Construction**
    - Create a "Standard Phenology Curve" for Ghafsai (averaged over 10 years).
    - Represents the ideal growth cycle.

- [ ] **Day 2: DTW Implementation**
    - Use `fastdtw`.
    - Compute distance between "Current Year Parcel" and "Standard Curve".
    - Normalized Distance = Stress Indicator.

- [ ] **Day 3: Integration & Validation**
    - Compare LSTM Error vs. DTW Distance.
    - **Ensemble Score**: Combine both metrics for robust detection.
    - Validate against 2024 (known drought year) - should show High Error & High Distance.

---

## 📝 Phase 4: Article Writing (Weeks 7-8)

*Content pivots to "Unsupervised Anomaly Detection" for data-scarce regions.*

---

## 📁 MVP File Structure

```
Article1_WaterStress/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── schemas.py               # Pydantic models
│   ├── gee_service.py           # GEE wrapper
│   ├── anomaly_detection.py     # LSTM + DTW Logic
│   └── dtw_utils.py             # Phenology matching
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── Map/
│   │   │   ├── LeafletMap.tsx
│   │   │   └── DrawControl.tsx
│   │   ├── Dashboard/
│   │   │   ├── StatCard.tsx
│   │   │   └── TimeSeriesChart.tsx (Recharts)
│   │   └── UI/
│   │       └── ...
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   ├── package.json
│   └── next.config.js
├── src/
│   ├── gee_pipeline.py          # Data Extraction + Topo Correction
│   ├── train_autoencoder.py     # Unsupervised Training
│   ├── lstm_model.py            # PyTorch Architecture
│   └── spatial_utils.py         # Coordinate utilities
├── models/
│   ├── scaler.pkl
│   └── lstm_autoencoder.pt
├── data/
│   └── reference_healthy_vectors.csv
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
