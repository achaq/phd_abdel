---
name: OliveGuard Master Specification
overview: "Complete end-to-end architecture for OliveGuard: AI-powered digital twin system for water stress detection in Ghafsai olive orchards. 12-week timeline covering multi-sensor GEE pipeline (Sentinel-1/2 + Landsat + ERA5), CNN-Attention-LSTM with SHAP explainability, FastAPI backend with Redis/Celery, and Next.js dashboard."
todos:
  - id: gee-setup
    content: "Week 1 Day 1: Environment setup, config module, GEE initialization, study area definition"
    status: completed
  - id: gee-cloud-mask
    content: "Week 1 Day 2: Implement QA60 + s2cloudless hybrid cloud masking"
    status: completed
  - id: gee-indices
    content: "Week 1 Day 3: Calculate NDVI, NDWI, SAVI, EVI indices"
    status: completed
  - id: gee-sentinel1
    content: "Week 1 Day 4: Add Sentinel-1 VV/VH extraction with speckle filtering"
    status: completed
  - id: gee-veg-mask
    content: "Week 1 Day 5: Apply vegetation fragmentation mask (NDVI > 0.2)"
    status: completed
  - id: gee-export
    content: "Week 1 Day 6: Create extract_training_data() with 1000 sample points"
    status: completed
  - id: api-setup
    content: "Week 2 Day 1: Initialize FastAPI project with schemas"
    status: pending
  - id: api-endpoints
    content: "Week 2 Day 2-3: Implement /analyze-parcel and /history endpoints"
    status: pending
  - id: api-redis
    content: "Week 2 Day 5: Add Redis caching layer for GEE queries"
    status: pending
  - id: fe-setup
    content: "Week 3 Day 1: Initialize Next.js 14 with TypeScript"
    status: pending
  - id: fe-map
    content: "Week 3 Day 2: Create Leaflet map centered on Ghafsai"
    status: pending
  - id: fe-draw
    content: "Week 3 Day 3: Implement polygon drawing tool"
    status: pending
  - id: fe-stress-card
    content: "Week 4 Day 1: Create StressCard component"
    status: pending
  - id: fe-history
    content: "Week 4 Day 2: Create HistoryGraph with dual-axis chart"
    status: pending
  - id: fe-integration
    content: "Week 4 Day 5-6: Full end-to-end integration"
    status: pending
  - id: ml-data-prep
    content: "Week 5 Day 1: Create 30-day sliding window sequences for LSTM"
    status: pending
  - id: ml-cnn-lstm
    content: "Week 5 Day 2: Build CNN-LSTM hybrid architecture in PyTorch"
    status: pending
  - id: ml-train
    content: "Week 5 Day 3: Train with time-series CV and early stopping"
    status: pending
  - id: ml-rf-enhance
    content: "Week 5 Day 4: Enhance Random Forest with new features"
    status: pending
  - id: shap-rf
    content: "Week 6 Day 1: Implement SHAP for Random Forest"
    status: pending
  - id: shap-deep
    content: "Week 6 Day 2: Implement SHAP for CNN-LSTM"
    status: pending
  - id: shap-api
    content: "Week 6 Day 4: Integrate SHAP into API response"
    status: pending
  - id: fe-shap-chart
    content: "Week 6 Day 5: Create SHAPChart component"
    status: pending
  - id: paper-intro
    content: "Week 7 Day 1: Write Introduction (Ghafsai Gap)"
    status: pending
  - id: paper-methods
    content: "Week 7 Day 3: Write Methodology section"
    status: pending
  - id: paper-results
    content: "Week 7 Day 4-6: Write Results with figures"
    status: pending
  - id: paper-discussion
    content: "Week 8 Day 1-2: Write Discussion section"
    status: pending
  - id: paper-final
    content: "Week 8 Day 6: Final formatting and review"
    status: pending
  - id: thermal-lst
    content: "Week 9 Day 1: Add Landsat thermal (LST) to pipeline"
    status: pending
  - id: cwsi
    content: "Week 9 Day 2: Implement CWSI calculation"
    status: pending
  - id: radar-diurnal
    content: "Week 9 Day 3: Add diurnal radar hysteresis feature"
    status: pending
  - id: multiclass
    content: "Week 9 Day 4: Upgrade to 4-class stress levels"
    status: pending
  - id: uncertainty
    content: "Week 9 Day 5: Add Monte Carlo Dropout uncertainty"
    status: pending
  - id: attention
    content: "Week 10 Day 1-2: Add self-attention layer to model"
    status: pending
  - id: phenology
    content: "Week 10 Day 3: Implement DTW phenological anomaly detection"
    status: pending
  - id: ensemble
    content: "Week 10 Day 5: Create RF+XGB+CNN-LSTM ensemble"
    status: pending
  - id: docker
    content: "Week 11 Day 1: Create Docker configuration"
    status: pending
  - id: celery
    content: "Week 11 Day 2: Set up Celery background tasks"
    status: pending
  - id: batch-api
    content: "Week 11 Day 3: Add batch analysis endpoint"
    status: pending
  - id: ablation
    content: "Week 12 Day 1-2: Conduct ablation study"
    status: pending
  - id: validation
    content: "Week 12 Day 3: Cross-validate with Tadla/Haouz studies"
    status: pending
  - id: final-test
    content: "Week 12 Day 5: Final integration testing"
    status: pending
isProject: true
---

# OliveGuard: Master Specification Document

## Executive Summary

**Project**: OliveGuard - AI-Powered Digital Twin for Water Stress Detection

**Target Region**: Ghafsai, Morocco (Pre-Rif)

**Timeline**: 12 Weeks (8 core + 4 extended)

**Architecture**: End-to-end system from satellite data extraction to web dashboard

**Tech Stack**: Python (GEE, FastAPI), PyTorch (Deep Learning), Next.js (Frontend), Leaflet (Mapping)

---

## Research Novelty & Impact

| Gap | Description | Our Solution |

|-----|-------------|--------------|

| **Geographic Gap** | All existing Moroccan studies focus on Tadla, Haouz, or Saïss plains | First AI-based water stress study targeting **Ghafsai/Pre-Rif** |

| **Methodological Gap** | Most studies use single-sensor approaches | **Multi-sensor fusion**: Sentinel-1 + Sentinel-2 + Landsat + ERA5 |

| **Operational Gap** | Research stays in academia | First **operational Digital Twin dashboard** for Moroccan olive monitoring |

---

## Part 1: Enhanced Geospatial Data Pipeline

### 1.1 Core Module: `gee_pipeline.py`

**Location**: `Olive_Habitat/Article1_WaterStress/src/gee_pipeline.py`

**Key Functions**:

- `initialize_gee()`: GEE authentication and initialization
- `mask_clouds_qa60(img)`: Aggressive cloud masking using QA60 band (bits 10, 11)
- `mask_clouds_s2cloudless(img)`: Probability-based masking (threshold 40%)
- `apply_vegetation_mask(img)`: Fragmentation mask excluding pixels where NDVI < 0.2
- `compute_indices(img)`: Calculate NDVI, NDWI, SAVI (L=0.5), EVI
- `get_sentinel1_backscatter(roi, date_range)`: Extract VV and VH with speckle filtering
- `get_landsat_thermal(roi, date_range)`: Extract LST from Landsat 8/9 TIRS (Week 9)
- `compute_cwsi(lst, era5)`: Calculate Crop Water Stress Index (Week 9)
- `extract_training_data(roi, start_date, end_date, n_samples=1000)`: Main extraction function

**Data Sources**:

- Sentinel-2 Level-2A (COPERNICUS/S2_SR_HARMONIZED) - Last 3 years
- Sentinel-1 GRD (COPERNICUS/S1_GRD) - VV/VH backscatter
- Landsat 8/9 (LANDSAT/LC08/C02/T1_L2) - Thermal bands (Extended)
- ERA5-Land (ECMWF/ERA5_LAND/DAILY_AGGR) - Temperature, Precipitation, VPD
- Cloud masking: QA60 band + s2cloudless collection

**Output Format**:

CSV with columns: `[date, lat, lon, B2, B3, B4, B8, NDVI, NDWI, SAVI, EVI, VV, VH, Temp, Precip, stress_label]`

Extended columns (Week 9): `[..., LST, CWSI, VH_diurnal_diff]`

---

## Part 2: AI/ML Training Pipeline

### 2.1 Baseline Model: Random Forest

**Location**: `Olive_Habitat/Article1_WaterStress/src/train_model.py`

**Enhancements over existing code**:

- Add new features: VV, VH, SAVI, EVI
- Hyperparameter tuning with GridSearchCV
- Feature importance visualization
- Save as `rf_stress_model.pkl`

### 2.2 Advanced Model: CNN-LSTM Hybrid

**Location**: `Olive_Habitat/Article1_WaterStress/src/train_deep_model.py` (NEW)

**Architecture**:

```
Input: [batch, timesteps=30, features=12]
    ↓
Conv1D(64, kernel=3) → BatchNorm → ReLU
    ↓
Conv1D(128, kernel=3) → BatchNorm → ReLU
    ↓
[Optional: MultiHeadAttention(4 heads)] ← Week 10 addition
    ↓
LSTM(64, return_sequences=True)
    ↓
LSTM(32)
    ↓
Dense(64) → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Stress Probability
```

**Training Configuration**:

- Time-series cross-validation (5 folds)
- Class imbalance: weighted loss + SMOTE
- Early stopping (patience=10)
- Save as `cnn_lstm_stress.h5`

### 2.3 Extended Models (Weeks 9-10)

**Multi-Class Classification**:

- Level 0: Healthy (NDWI > μ)
- Level 1: Mild Stress (μ-σ < NDWI < μ)
- Level 2: Moderate Stress (μ-2σ < NDWI < μ-σ)
- Level 3: Severe Stress (NDWI < μ-2σ)

**Uncertainty Quantification**:

- Monte Carlo Dropout: N=50 forward passes
- Report mean ± confidence interval

**Ensemble Model**:

- Average predictions from RF + XGBoost + CNN-LSTM
- Voting with calibrated probabilities

### 2.4 Explainability Module: SHAP

**Location**: `Olive_Habitat/Article1_WaterStress/src/explainability.py` (NEW)

**Functions**:

- `generate_shap_values(model, X_sample)`: Compute SHAP values
- `plot_shap_summary(shap_values, feature_names)`: Generate summary plot
- `get_top_contributors(shap_values, k=3)`: Extract main causes
- `map_to_recommendation(causes)`: Generate human-readable advice

**Output**:

- SHAP summary plots (saved as PNG)
- Feature importance rankings
- Main cause explanations:
  - "Low NDWI" → "Possible irrigation deficit"
  - "High Temperature" → "Heat stress detected"
  - "Low VH Backscatter" → "Canopy structure change"

---

## Part 3: FastAPI Backend

### 3.1 Main API Server

**Location**: `Olive_Habitat/Article1_WaterStress/api/main.py` (NEW)

**Endpoints**:

#### POST `/analyze-parcel`

**Request Body**:

```json
{
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon, lat], ...]]
  }
}
```

**Response**:

```json
{
  "stress_level": "High",
  "stress_class": 3,
  "confidence": 0.85,
  "uncertainty": 0.08,
  "main_causes": [
    {
      "feature": "NDWI",
      "contribution": -0.45,
      "explanation": "Low water content"
    },
    {
      "feature": "Temperature_30d",
      "contribution": 0.32,
      "explanation": "Heat wave detected"
    },
    {
      "feature": "VH",
      "contribution": -0.12,
      "explanation": "Canopy structure change"
    }
  ],
  "recommendation": "Irrigate within 48 hours - water deficit detected",
  "timestamp": "2026-02-01T12:00:00Z"
}
```

#### GET `/history/{lat}/{lon}`

**Response**: 2-year NDVI/NDWI/Temperature time-series

#### POST `/analyze-batch` (Week 11)

**Request**: Array of GeoJSON polygons

**Response**: Aggregated farm-level report

#### GET `/compare-years` (Week 10)

**Query**: `?year1=2024&year2=2025&geometry={GeoJSON}`

**Response**: Year-over-year stress comparison

### 3.2 Supporting Services

- `api/gee_service.py`: GEE wrapper with caching
- `api/ml_service.py`: Model loading and inference (singleton pattern)
- `api/shap_service.py`: SHAP explanation generation
- `api/tasks.py`: Celery background tasks for long-running GEE queries

### 3.3 Infrastructure

- **Redis**: Cache GEE responses (TTL: 24h)
- **Celery**: Background task processing for async GEE calls
- **Docker**: Containerized deployment

---

## Part 4: Next.js Frontend Dashboard

### 4.1 Project Structure

**Location**: `Olive_Habitat/Article1_WaterStress/frontend/` (NEW)

```
frontend/
├── app/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── Map/
│   │   ├── LeafletMap.tsx      # Centered on Ghafsai [34.675, -4.818]
│   │   └── DrawControl.tsx     # Polygon drawing
│   ├── Dashboard/
│   │   ├── StressCard.tsx      # Red/Yellow/Green indicator
│   │   ├── SHAPChart.tsx       # Feature contribution bars
│   │   ├── HistoryGraph.tsx    # NDVI/NDWI timeline
│   │   └── CompareYears.tsx    # Year-over-year (Week 10)
│   └── Layout/
│       └── Sidebar.tsx
├── lib/
│   ├── api.ts                  # Axios client
│   ├── types.ts                # TypeScript interfaces
│   └── store.ts                # Zustand state management
└── hooks/
    └── useParcelAnalysis.ts
```

### 4.2 Core Features

**Phase 2 (Weeks 3-4)**:

- Full-screen Leaflet map with satellite basemap
- Polygon drawing tool
- StressCard with confidence score
- HistoryGraph with dual-axis (NDVI/NDWI + Temperature)
- Loading states and error handling

**Phase 5-6 (Weeks 9-12)**:

- Multi-class stress gradient (4 levels)
- Uncertainty display (confidence intervals)
- SHAPChart with color-coded bars
- Year-over-year comparison view
- Parcel management (save/name favorites)
- PDF report export
- Alert thresholds (notify when stress > X)

---

## Part 5: Extended Features (Weeks 9-12)

### 5.1 Thermal Data Integration (Week 9)

**Landsat LST Extraction**:

- Landsat 8/9 TIRS (Band 10)
- Thermal sharpening using Sentinel-2 optical data
- Resolution: 100m → 30m (sharpened)

**CWSI Calculation**:

```
CWSI = (Tc - Twet) / (Tdry - Twet)
```

Where:

- Tc = Canopy temperature (from LST)
- Twet/Tdry = Theoretical limits from VPD

### 5.2 Diurnal Radar Hysteresis (Week 9)

Following Ouaadi et al. (Reference [20]):

- Morning pass (descending orbit): VH_morning
- Evening pass (ascending orbit): VH_evening
- Diurnal difference: VH_diff = VH_morning - VH_evening
- High difference = active transpiration = healthy
- Low difference = stomatal closure = stressed

### 5.3 Self-Attention Mechanism (Week 10)

Upgrade CNN-LSTM to CNN-Attention-LSTM:

```
Conv1D → Conv1D → MultiHeadAttention(heads=4) → LSTM → Dense
```

Benefits:

- Highlights critical time steps (when stress began)
- Attention weights are interpretable
- Following CNN-ViT approach (Reference [21])

### 5.4 Phenological Anomaly Detection (Week 10)

**Dynamic Time Warping (DTW)**:

- Compute 5-year historical "normal" NDVI curve
- Compare current year using DTW distance
- Anomaly score = DTW distance / baseline std
- Following Spanish approach (Reference [22])

### 5.5 Climate Attribution (Week 9)

Rule-based cause identification:

```python
if ndwi_drop AND temp_30d > 35:
    cause = "Heat Wave Stress"
elif ndwi_drop AND precip_30d < 10:
    cause = "Drought Stress"
elif vh_drop AND ndvi_stable:
    cause = "Hydraulic Failure (Root Zone)"
elif cwsi > 0.7:
    cause = "Severe Evaporative Demand"
```

---

## Part 6: Validation Strategy (Week 12)

### 6.1 Ablation Study

Test model performance with incremental features:

| Model | Features | Expected Accuracy |

|-------|----------|-------------------|

| A | Sentinel-2 only | Baseline |

| B | + Sentinel-1 radar | +3-5% |

| C | + ERA5 climate | +2-4% |

| D | + Temporal (LSTM) | +5-8% |

| E | + Thermal (CWSI) | +2-3% |

| F | + Attention | +1-2% |

### 6.2 Cross-Validation

Compare stress patterns with published Tadla/Haouz studies:

- Temporal trends should match regional drought events
- Seasonal patterns should align with Mediterranean climate

### 6.3 Performance Benchmarks

Target metrics:

- API response time: < 10s (cached), < 60s (fresh GEE query)
- Model inference: < 100ms
- Frontend load time: < 3s
- F1-score: > 0.85

---

## Timeline Summary

| Week | Phase | Key Deliverables |

|------|-------|------------------|

| 1 | Data Backend | Enhanced GEE pipeline with multi-sensor fusion |

| 2 | Data Backend | FastAPI foundation with Redis caching |

| 3 | Prototype | Next.js + Leaflet map + polygon drawing |

| 4 | Prototype | StressCard + HistoryGraph + full integration |

| 5 | Science & AI | CNN-LSTM training + model comparison |

| 6 | Science & AI | SHAP integration + API + frontend |

| 7 | Article | Introduction + Methodology + Results draft |

| 8 | Article | Discussion + Final formatting |

| 9 | Extended | Thermal/CWSI + Diurnal radar + Multi-class |

| 10 | Extended | Attention + Phenology DTW + Ensemble |

| 11 | Production | Docker + Celery + Batch API |

| 12 | Validation | Ablation study + Testing + Documentation |

---

## Key Milestones

1. **GEE Pipeline Complete**: Week 1 (Multi-sensor extraction working)
2. **API Foundation**: Week 2 (FastAPI with core endpoints)
3. **POC Live**: Week 4 (Functional dashboard prototype)
4. **AI Models Trained**: Week 5 (RF + CNN-LSTM models saved)
5. **SHAP Integrated**: Week 6 (Explainability in API + Frontend)
6. **Paper Drafted**: Week 8 (Ready for internal review)
7. **Advanced Features**: Week 10 (Thermal + Attention + Phenology)
8. **Production Ready**: Week 12 (Docker + Validation complete)

---

## Scientific Contribution Summary

1. **Novel Study Area**: First ML-based olive stress detection in Pre-Rif/Ghafsai
2. **Multi-Sensor Methodology**: Sentinel-1 + Sentinel-2 + Landsat + ERA5 fusion
3. **Advanced AI Architecture**: CNN-Attention-LSTM with uncertainty quantification
4. **Explainable AI**: SHAP integration for transparency and farmer trust
5. **Thermal Integration**: CWSI for direct physiological stress measurement
6. **Operational System**: Digital Twin dashboard for Morocco's Green Generation Strategy
7. **Phenological Analysis**: DTW-based anomaly detection for early warning
8. **Comprehensive Validation**: Ablation study + cross-validation with published studies
