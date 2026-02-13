# OliveGuard: Week 1 Progress Report
**Phase 1: The Data Backend & Scientific Foundation**

**Date:** February 6, 2026
**To:** Project Supervision Team
**From:** Lead Developer / Researcher

---

## 1. Executive Summary

This week focused on building the **"Data Engine"** that powers the OliveGuard Decision Support System (DSS). The primary objective was to transition from "Raw Satellite Data" to "Science-Ready Signals".

We successfully implemented a robust **Google Earth Engine (GEE) Pipeline** capable of handling the complex, mountainous terrain of Ghafsai. Crucially, we established a **Comparative Scientific Strategy** to validate our AI models, ensuring the final tool provides trusted, verifiable insights to farmers.

---

## 2. Key Achievements & Value Proposition

### A. Topographic Correction (Optical Data)
*   **What we did:** Implemented the **Sun-Canopy-Sensor (SCS) + C-Correction** algorithm.
*   **Technical POV:** In mountainous regions, slopes facing away from the sun appear darker/stressed even if they are healthy. We used a Digital Elevation Model (DEM) to mathematically normalize illumination angles across all pixels.
*   **Business/End-User POV:** **False Alarm Prevention.** Without this, a farmer on a north-facing slope would constantly receive "Water Stress" alerts simply because their orchard is in a shadow. This ensures our analysis is fair and accurate regardless of terrain.

### B. Radiometric Terrain Flattening (Radar Data)
*   **What we did:** Integrated **Volumetric Model Flattening** for Sentinel-1 Radar.
*   **Technical POV:** Radar is essential because it sees through clouds (common in Rif winters). However, mountains cause severe geometric distortions (layover/shadow). We implemented correction logic to normalize backscatter based on local incidence angles.
*   **Business/End-User POV:** **All-Weather Reliability.** Farmers need data even when it's cloudy. This step ensures that our "Cloud-Proof" radar backup is actually usable and accurate, allowing continuous monitoring during the critical wet season.

### C. Pure Pixel Filtering (The "Clean Data" Initiative)
*   **What we did:** Enhanced the masking logic to include **Texture Analysis (GLCM Entropy)** and dynamic NDVI thresholds (> 0.4).
*   **Technical POV:** Olive orchards in Ghafsai are fragmented and spaced out. A standard 10m pixel often contains 50% tree and 50% soil ("Mixed Pixel"). By using texture entropy, we filter out pixels that are too heterogeneous (noisy).
*   **Business/End-User POV:** **Precision Diagnostics.** We ensure the AI is analyzing the *tree health*, not the *weeds or soil* between rows. This dramatically increases the signal-to-noise ratio of our stress alerts.

### D. The "Three-Way" Comparative Strategy (Scientific Rigor)
*   **What we did:** Redesigned the data extraction pipeline to generate **Two Distinct Datasets** for a future comparative study.
    1.  **Dataset A (Climate-Driven):** Data extracted *only* from years historically proven to be wet/healthy (via ERA5 rainfall ranking).
    2.  **Dataset B (Data-Driven):** Data extracted from all years, but filtered using unsupervised clustering (K-Means) to find "stable" behavior.
*   **Technical POV:** We moved away from arbitrary "Mock Labels". We are now set up to train two competing LSTM models and validate them against a third "Naive Baseline".
*   **Business/End-User POV:** **Trust & Validation.** "How do we know the AI is right?" This strategy allows us to prove scientifically which method detects drought (e.g., the 2024 crisis) more accurately. It provides the "Confidence Score" needed for a commercial-grade application.

---

## 3. Technical Implementation Details (Pipeline Architecture)

The system is now fully automated via `src/gee_pipeline.py`.

| Component | Status | Implementation Detail | Impact on Final App |
| :--- | :--- | :--- | :--- |
| **Sentinel-2** | ✅ Ready | Cloud Probability Masking + Topo Correction | The visual "Map" and "NDVI Charts" the user sees. |
| **Sentinel-1** | ✅ Ready | Speckle Filtering + Terrain Flattening | The "Backup Signal" when optical data is missing. |
| **Climate (ERA5)** | ✅ Ready | Bilinear Downscaling (9km → Plot Level) | Contextual data ("Is it hot/dry?") displayed on the dashboard. |
| **Labeling** | ✅ Ready | **K-Means Clustering** (Unsupervised) | The "Brain" that decides if a tree is Stressed vs. Healthy. |

---

## 4. Next Steps (Transition to Phase 2)

With the backend data engine secured, we are proceeding to **Phase 2: The Interactive MVP**.

*   **Immediate Goal:** Connect this Python pipeline to the React/Next.js frontend.
*   **User Action:** Allow the supervisor/user to draw a polygon on the map.
*   **System Action:** Trigger this exact pipeline in real-time (or near real-time) to produce the analysis report.

---

*This report confirms that Week 1 objectives have been met with a focus on scientific novelty (Terrain Correction) and operational robustness (Radar Integration).*
