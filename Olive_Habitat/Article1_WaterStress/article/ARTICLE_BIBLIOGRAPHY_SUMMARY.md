# Comprehensive Bibliography for AI-Based Olive Water Stress Detection Project

**Project:** AI-Based Early Detection of Climate-Induced Water Stress in Olive Orchards Using Google Earth Engine: A Case Study of Ghafsai, Morocco

**Date Compiled:** February 5, 2026

**Total References:** 70+ articles covering all aspects of the research

---

## Table of Contents

1. [Cloud Masking and Preprocessing](#cloud-masking-and-preprocessing)
2. [Sentinel-1 Radar Applications](#sentinel-1-radar-applications)
3. [CNN-LSTM and Deep Learning](#cnn-lstm-and-deep-learning)
4. [SHAP Explainable AI](#shap-explainable-ai)
5. [Multi-Sensor Fusion](#multi-sensor-fusion)
6. [ERA5 and Climate Data](#era5-and-climate-data)
7. [Vegetation Indices for Olive Stress](#vegetation-indices-for-olive-stress)
8. [Google Earth Engine and Random Forest](#google-earth-engine-and-random-forest)
9. [Digital Twin and Decision Support Systems](#digital-twin-and-decision-support-systems)
10. [Morocco-Specific Studies](#morocco-specific-studies)
11. [General Remote Sensing and Methodology](#general-remote-sensing-and-methodology)

---

## Cloud Masking and Preprocessing

### Key Articles

1. **s2cloudless_gee (2020)**
   - **Title:** More accurate and flexible cloud masking for Sentinel-2 images
   - **Relevance:** Documents the s2cloudless algorithm used in your pipeline for aggressive cloud masking
   - **Key Contribution:** LightGBM-based cloud probability estimation, critical for Rif's mountainous, cloudy terrain

2. **elib_cloudfree_s2 (2020)**
   - **Title:** Aggregating cloud-free Sentinel-2 images with Google Earth Engine
   - **Relevance:** Methodology for creating cloud-free composites, supports your QA60 + s2cloudless approach
   - **Key Contribution:** Temporal compositing strategies for cloudy regions

**Usage in Paper:** Cite when describing cloud masking methodology (Section: Materials and Data Acquisition)

---

## Sentinel-1 Radar Applications

### Key Articles

1. **ouaadi_olive_radar_2024** ⭐ **CRITICAL**
   - **Title:** Suivi de l'état hydrique de l'olivier par télédétection radar en bande C en zone semi-aride
   - **Relevance:** Directly addresses Sentinel-1 C-band radar for olive water stress in Morocco
   - **Key Findings:**
     - Diurnal coherence cycles correlate with evapotranspiration
     - Morning coherence decay indicates water stress
     - VPD is most influential meteorological parameter
   - **Usage:** Primary citation for radar methodology, supports your VV/VH backscatter analysis

2. **ouaadi_wheat_radar_2021**
   - **Title:** C-band radar data and in situ measurements for the monitoring of wheat crops in a semi-arid area (center of Morocco)
   - **Relevance:** Validates C-band radar effectiveness in Moroccan semi-arid agriculture
   - **Key Contribution:** Dataset and methodology transferable to olive orchards

3. **ouaadi_olive_coherence_2024**
   - **Title:** Analysis of C-band radar temporal coherence over an irrigated olive orchard in a semi-arid region
   - **Relevance:** Specific coherence analysis for olive orchards, supports diurnal hysteresis approach
   - **Key Contribution:** 18-day persistence of diurnal signal despite 6-day revisit frequency

4. **irrigation_sentinel1_2022**
   - **Title:** Detecting Irrigation Events over Semi-Arid and Temperate Climatic Areas Using Sentinel-1 Data
   - **Relevance:** Irrigation detection methodology applicable to stress detection
   - **Key Finding:** 67% accuracy in semi-arid climates

**Usage in Paper:** 
- Introduction: Multi-sensor approach justification
- Methodology: Radar preprocessing and feature extraction
- Discussion: Comparison with Haouz studies

---

## CNN-LSTM and Deep Learning

### Key Articles

1. **cnn_lstm_cotton_2023**
   - **Title:** A CNN-LSTM model for cotton water stress classification
   - **Relevance:** Validates CNN-LSTM architecture for water stress (97.3% accuracy)
   - **Key Contribution:** Outperforms AlexNet, ResNet, VGG16, EfficientNet B7, 3D CNN
   - **Usage:** Methodology section - model architecture justification

2. **convlstm_cnnlstm_comparison_2024**
   - **Title:** Deep Learning Approaches for Water Stress Forecasting in Arboriculture Using Time Series of Remote Sensing Images: Comparative Study between ConvLSTM and CNN-LSTM Models
   - **Relevance:** Direct comparison for tree crops (arboriculture), validates your CNN-LSTM choice
   - **Key Contribution:** Temporal forecasting capabilities for perennial crops
   - **Usage:** Results section - model comparison table

3. **cnn_vit_water_stress_2024**
   - **Title:** Remote Sensing Crop Water Stress Determination Using CNN-ViT Architecture
   - **Relevance:** Supports your CNN-Attention-LSTM hybrid architecture (mentioned in README)
   - **Key Contribution:** Vision Transformer integration for attention mechanisms
   - **Usage:** Advanced features discussion, future work

4. **ieee_cnn_water_stress_2022**
   - **Title:** CNN Based Water Stress Detection in Chickpea Using UAV Based Hyperspectral Imaging
   - **Relevance:** CNN architecture for water stress, supports deep learning approach
   - **Key Contribution:** Multi-modal imaging (RGB, multispectral, infrared)
   - **Usage:** Methodology - deep learning framework

**Usage in Paper:**
- Methodology: Model architecture description
- Results: Performance comparison
- Discussion: State-of-the-art positioning

---

## SHAP Explainable AI

### Key Articles

1. **shap_water_quality_2025**
   - **Title:** Harnessing Explainable AI for Sustainable Agriculture: SHAP-Based Feature Selection in Multi-Model Evaluation of Irrigation Water Quality Indices
   - **Relevance:** SHAP for agricultural decision-making, supports your explainability framework
   - **Key Contribution:** Feature selection and model interpretation for farmers
   - **Usage:** Methodology - SHAP implementation, Discussion - farmer trust

2. **ieee_shap_lulc_2023**
   - **Title:** Interpretable Deep Learning Framework for Land Use and Land Cover Classification in Remote Sensing Using SHAP
   - **Relevance:** SHAP integration with deep learning for remote sensing
   - **Key Contribution:** Interpretability framework for complex models
   - **Usage:** Methodology - explainability section

3. **shap_almond_stomatal_2024** ⭐ **HIGHLY RELEVANT**
   - **Title:** Combining UAV-Based Multispectral and Thermal Infrared Data with Regression Modeling and SHAP Analysis for Predicting Stomatal Conductance in Almond Orchards
   - **Relevance:** SHAP for tree crops (almond), similar physiology to olive
   - **Key Findings:**
     - Green-red vegetation index, chlorophyll red-edge index, canopy temperature ratios are key features
     - Thermal data improves predictions by 11%
   - **Usage:** Results - feature importance analysis, Discussion - physiological validation

4. **shap_potato_extraction_2025**
   - **Title:** Advancing County-Level Potato Cultivation Area Extraction: A Novel Approach Utilizing Multi-Source Remote Sensing Imagery and the Shapley Additive Explanations--Sequential Forward Selection--Random Forest Model
   - **Relevance:** SHAP + Random Forest for multi-source data
   - **Key Contribution:** Feature selection methodology applicable to your multi-sensor fusion
   - **Usage:** Methodology - feature engineering

5. **shap_wheat_nitrogen_2025**
   - **Title:** Multi-Source Feature Selection and Explainable Machine Learning Approach for Mapping Nitrogen Balance Index in Winter Wheat Based on Sentinel-2 Data
   - **Relevance:** Multi-source feature selection with SHAP, similar to your approach
   - **Key Contribution:** Sentinel-2 feature importance analysis
   - **Usage:** Results - SHAP summary plots

**Usage in Paper:**
- Methodology: Explainability framework
- Results: Feature importance visualization
- Discussion: Farmer/ORMVA trust and adoption

---

## Multi-Sensor Fusion

### Key Articles

1. **et_fusion_landsat_sentinel_2023**
   - **Title:** Improving the spatiotemporal resolution of remotely sensed ET information for water management through Landsat, Sentinel-2, ECOSTRESS and VIIRS data fusion
   - **Relevance:** Multi-sensor ET fusion, supports your Landsat thermal integration
   - **Key Contribution:** Daily ET products at 30-m resolution
   - **Usage:** Methodology - data fusion approach

2. **vtci_sentinel2_modis_2020**
   - **Title:** Developing a fused vegetation temperature condition index for drought monitoring at field scales using Sentinel-2 and MODIS imagery
   - **Relevance:** Sentinel-2 + MODIS fusion for drought monitoring
   - **Key Contribution:** Field-scale VTCI for agricultural applications
   - **Usage:** Methodology - vegetation indices, Results - stress detection

3. **soil_moisture_s1_s2_2017** ⭐ **CRITICAL**
   - **Title:** Synergic Use of Sentinel-1 and Sentinel-2 Images for Operational Soil Moisture Mapping at High Spatial Resolution over Agricultural Areas
   - **Relevance:** Directly addresses Sentinel-1 + Sentinel-2 fusion for soil moisture
   - **Key Contribution:** Operational high-resolution soil moisture mapping
   - **Usage:** Methodology - multi-sensor fusion, Discussion - operational feasibility

4. **et_fusion_sentinel3_2021**
   - **Title:** Improving field-scale crop actual evapotranspiration monitoring with Sentinel-3, Sentinel-2, and Landsat data fusion
   - **Relevance:** Multi-sensor ET monitoring, supports your thermal integration
   - **Key Contribution:** Thermal spatial variability between irrigated/dry regions
   - **Usage:** Methodology - evapotranspiration estimation

5. **dl_sensor_fusion_2021**
   - **Title:** Deep Learning Sensor Fusion for Plant Water Stress Detection
   - **Relevance:** Deep learning for multi-sensor fusion, validates your CNN-LSTM approach
   - **Key Contribution:** Enhanced ML performance through sensor fusion
   - **Usage:** Methodology - deep learning framework

**Usage in Paper:**
- Introduction: Multi-sensor approach justification
- Methodology: Data fusion pipeline
- Results: Ablation study (single vs. multi-sensor)
- Discussion: Operational advantages

---

## ERA5 and Climate Data

### Key Articles

1. **era5_chickpea_yield_2025** ⭐ **HIGHLY RELEVANT**
   - **Title:** Harnessing Sentinel-2 imagery and AgERA5 data using Google Earth Engine for developing chickpea mechanistic growth modeling and pre-harvest empirical yield forecast
   - **Relevance:** Sentinel-2 + AgERA5 in GEE, directly applicable to your pipeline
   - **Key Findings:**
     - Grain yield prediction up to 2 months before harvest
     - Fully executed within GEE with sub-field precision
   - **Usage:** Methodology - climate data integration, Results - climate feature importance

2. **sentinel2_era5_gee_2021**
   - **Title:** Integration of Sentinel-2 and ERA5 data for agricultural monitoring using Google Earth Engine
   - **Relevance:** ERA5 integration methodology in GEE
   - **Key Contribution:** Climate variable preprocessing for agriculture
   - **Usage:** Methodology - data acquisition

**Usage in Paper:**
- Methodology: Climate data sources and preprocessing
- Results: Climate variable importance (temperature, precipitation, VPD)
- Discussion: Climate attribution module

---

## Vegetation Indices for Olive Stress

### Key Articles

1. **olive_remote_sensing_review_2016**
   - **Title:** Site-specific management of common olive: Remote sensing, geospatial, and advanced image processing applications
   - **Relevance:** Comprehensive review of remote sensing for olive management
   - **Key Contribution:** NDVI, NDWI, SAVI applications in olive orchards
   - **Usage:** Introduction - state of the art, Literature review

2. **ndre_olive_irrigation_2019** ⭐ **HIGHLY RELEVANT**
   - **Title:** Detection of irrigation inhomogeneities in an olive grove using the NDRE vegetation index obtained from UAV images
   - **Relevance:** NDRE superior to NDVI, NDWI, SAVI for irrigation detection
   - **Key Finding:** NDRE detects irrigation irregularities not visible with other indices
   - **Usage:** Methodology - vegetation indices, Discussion - index comparison

3. **planet_olive_swp_2024**
   - **Title:** Prediction of Stem Water Potential in Olive Orchards Using High-Resolution Planet Satellite Images and Machine Learning Techniques
   - **Relevance:** ML for olive water stress using high-resolution imagery
   - **Key Contribution:** Stem water potential prediction (physiological validation)
   - **Usage:** Methodology - stress proxy development, Results - validation

4. **sentinel2_olive_stress_2022**
   - **Title:** Implementing Sentinel-2 Data and Machine Learning to Detect Plant Stress in Olive Groves
   - **Relevance:** Sentinel-2 + ML for olive stress, directly comparable to your work
   - **Key Contribution:** Operational stress detection methodology
   - **Usage:** Introduction - related work, Discussion - comparison

**Usage in Paper:**
- Methodology: Vegetation index calculation and selection
- Results: Index performance comparison (NDVI vs. NDWI vs. SAVI vs. EVI)
- Discussion: "Silent stress" paradox and index limitations

---

## Google Earth Engine and Random Forest

### Key Articles

1. **gee_rf_lulc_2024**
   - **Title:** Characterizing land use/land cover change dynamics by an enhanced random forest machine learning model: a Google Earth Engine implementation
   - **Relevance:** Enhanced RF on GEE for LULC, supports your RF implementation
   - **Key Contribution:** Multitemporal Landsat-8 processing with cloud elimination
   - **Usage:** Methodology - Random Forest implementation

2. **gee_rf_svm_2023**
   - **Title:** Precision Land Use and Land Cover Classification Using Google Earth Engine: Integrating Random Forest and Support Vector Machine Algorithms
   - **Relevance:** RF vs. SVM comparison on GEE
   - **Key Finding:** RF consistently outperforms simpler methods
   - **Usage:** Methodology - model selection, Results - model comparison

3. **gee_rf_cropland_2023**
   - **Title:** Agricultural cropland extent and areas of South Asia derived using Landsat satellite 30-m time-series big-data using random forest machine learning algorithms on the Google Earth Engine cloud
   - **Relevance:** Large-scale RF on GEE for agriculture
   - **Key Contribution:** 30-m resolution time-series processing
   - **Usage:** Methodology - scalability discussion

4. **gee_rf_irrigation_2023**
   - **Title:** Application of the Random Forest Classifier to Map Irrigated Areas Using Google Earth Engine
   - **Relevance:** RF for irrigation detection (88% accuracy)
   - **Key Contribution:** NDVI-based irrigation vs. rainfed distinction
   - **Usage:** Methodology - feature engineering, Discussion - operational mapping

**Usage in Paper:**
- Methodology: GEE platform and RF implementation
- Results: Model performance metrics
- Discussion: Scalability and operational feasibility

---

## Digital Twin and Decision Support Systems

### Key Articles

1. **digital_twin_agriculture_review_2025** ⭐ **CRITICAL**
   - **Title:** Advancing Precision Agriculture Through Digital Twins and Smart Farming Technologies: A Review
   - **Relevance:** Comprehensive review of digital twins in agriculture (167 studies, 2018-2025)
   - **Key Contribution:** Integration framework for IoT, UAV, ML, RS technologies
   - **Usage:** Introduction - digital twin concept, Discussion - operational system positioning

2. **ieee_digital_twin_agriculture_2023**
   - **Title:** An AI-Driven Architecture for Precision Agriculture: IoT, Machine Learning, and Digital Twin Integration for Sustainable Crop Protection
   - **Relevance:** AI-driven digital twin architecture for precision agriculture
   - **Key Contribution:** IoT + ML + RS integration framework
   - **Usage:** Methodology - system architecture, Discussion - Green Generation Strategy alignment

3. **ieee_rice_digital_twin_2021**
   - **Title:** Digital twin of rice as a decision-making service for precise farming, based on environmental datasets from the fields
   - **Relevance:** Crop-specific digital twin for decision-making
   - **Key Contribution:** Environmental data integration for farm management
   - **Usage:** Discussion - operational dashboard, Future work

4. **roam_dss_agriculture_2024**
   - **Title:** Roam: A Decision Support System for Digital Agriculture Systems
   - **Relevance:** Decision support system architecture for agriculture
   - **Key Contribution:** Design tradeoff analysis under uncertain conditions
   - **Usage:** Discussion - decision support framework

5. **uav_digital_twin_irrigation_2023**
   - **Title:** Optimization of Intelligent Irrigation Systems for Smart Farming Using Multi-Spectral Unmanned Aerial Vehicle and Digital Twins Modeling
   - **Relevance:** UAV + digital twin for irrigation optimization
   - **Key Contribution:** Field-scale crop water status assessment
   - **Usage:** Discussion - precision irrigation applications

**Usage in Paper:**
- Introduction: Digital twin concept and operational gap
- Methodology: Dashboard architecture
- Discussion: Operational system contribution, Green Generation Strategy
- Conclusion: First operational digital twin for Moroccan olive monitoring

---

## Morocco-Specific Studies

### Key Articles

1. **tadla_drought** (Existing)
   - **Title:** Monitoring drought-induced degradation of olive and citrus tree crops in the Tadla plain (Morocco) with multi-sensor remote sensing
   - **Relevance:** Primary Moroccan study on olive stress, comparison baseline
   - **Key Contribution:** Multi-sensor approach, 95.94% accuracy with RF
   - **Usage:** Introduction - related work, Discussion - comparison with Tadla

2. **mdpi_saiss** (Existing)
   - **Title:** Spatio-Temporal Assessment of Olive Orchard Intensification in the Saïss Plain (Morocco) Using k-Means and High-Resolution Satellite Data
   - **Relevance:** Saïss plain study, geographic comparison
   - **Key Contribution:** Intensification impacts on water resources
   - **Usage:** Introduction - geographic gap, Discussion - regional comparison

3. **ouaadi_wheat_radar_2021** (See Sentinel-1 section)
   - **Relevance:** Central Morocco, validates radar methodology
   - **Usage:** Methodology - radar preprocessing

4. **ouaadi_olive_radar_2024** (See Sentinel-1 section)
   - **Relevance:** Haouz plain olive study, direct comparison
   - **Usage:** Discussion - Haouz comparison, radar innovations

**Usage in Paper:**
- Introduction: Geographic gap (Ghafsai vs. Tadla/Haouz/Saïss)
- Discussion: Regional comparison and validation
- Conclusion: First AI-based study in Pre-Rif/Ghafsai

---

## General Remote Sensing and Methodology

### Key Articles (Existing in references.bib)

1. **hess_rf_downscaling** - RF for downscaling and gap filling
2. **mdpi_etdi** - Evapotranspiration Deficit Index methodology
3. **smdi_etdi_tamu** - SMDI and ETDI development
4. **frontiers_shap_rf** - SHAP + RF for precipitation estimation
5. **lstm_drought** - LSTM for drought prediction
6. **mdpi_xgboost_shap** - XGBoost-SHAP for forest drought sensitivity

---

## Citation Strategy by Section

### Introduction
- **Climate context:** frontiers2025, inra2025
- **Olive sector:** mdpi_phenotyping, mdpi_saiss, rdi_olive
- **Remote sensing paradigm:** gee_sentinel2, olive_remote_sensing_review_2016
- **AI methods:** cnn_lstm_cotton_2023, gee_rf_lulc_2024
- **Geographic gap:** tadla_drought, mdpi_saiss, ouaadi_olive_radar_2024

### Materials and Data Acquisition
- **Sentinel-2:** gee_sentinel2, s2cloudless_gee, elib_cloudfree_s2
- **Sentinel-1:** ouaadi_olive_radar_2024, ouaadi_wheat_radar_2021
- **MODIS:** gee_modis_et, vtci_sentinel2_modis_2020
- **ERA5:** era5_chickpea_yield_2025, sentinel2_era5_gee_2021
- **GEE platform:** gee_rf_cropland_2023, gee_rf_irrigation_2023

### Methodology
- **Cloud masking:** s2cloudless_gee, elib_cloudfree_s2
- **Vegetation indices:** olive_remote_sensing_review_2016, ndre_olive_irrigation_2019
- **Multi-sensor fusion:** soil_moisture_s1_s2_2017, et_fusion_landsat_sentinel_2023
- **Random Forest:** gee_rf_lulc_2024, hess_rf_downscaling
- **CNN-LSTM:** cnn_lstm_cotton_2023, convlstm_cnnlstm_comparison_2024
- **SHAP:** shap_almond_stomatal_2024, shap_water_quality_2025

### Results
- **Model performance:** cnn_lstm_cotton_2023, gee_rf_irrigation_2023
- **Feature importance:** shap_almond_stomatal_2024, shap_wheat_nitrogen_2025
- **Index comparison:** ndre_olive_irrigation_2019, planet_olive_swp_2024
- **Ablation study:** dl_sensor_fusion_2021, soil_moisture_s1_s2_2017

### Discussion
- **Comparison with Moroccan studies:** tadla_drought, ouaadi_olive_radar_2024, mdpi_saiss
- **Multi-sensor advantages:** soil_moisture_s1_s2_2017, et_fusion_landsat_sentinel_2023
- **Explainability:** shap_almond_stomatal_2024, ieee_shap_lulc_2023
- **Operational system:** digital_twin_agriculture_review_2025, ieee_digital_twin_agriculture_2023
- **Policy implications:** frontiers2025, inra2025

### Conclusion
- **Novelty:** Geographic + Methodological + Operational gaps
- **Future work:** cnn_vit_water_stress_2024, digital_twin_agriculture_review_2025

---

## Priority Citations (Must Include)

### Critical (⭐)
1. **ouaadi_olive_radar_2024** - Primary radar methodology
2. **soil_moisture_s1_s2_2017** - Multi-sensor fusion
3. **era5_chickpea_yield_2025** - ERA5 + GEE integration
4. **digital_twin_agriculture_review_2025** - Digital twin concept
5. **shap_almond_stomatal_2024** - SHAP for tree crops

### Highly Relevant
1. **cnn_lstm_cotton_2023** - CNN-LSTM validation
2. **convlstm_cnnlstm_comparison_2024** - Arboriculture comparison
3. **ndre_olive_irrigation_2019** - Vegetation index comparison
4. **tadla_drought** - Primary Moroccan comparison
5. **sentinel2_olive_stress_2022** - Directly comparable work

---

## Notes for Paper Writing

1. **Follow IEEE citation format:** Use square brackets [1], [2] in order of appearance
2. **Every factual claim needs a citation:** Use these references to support all technical statements
3. **Author-prominent vs. Information-prominent:** 
   - "Ouaadi et al. [X] demonstrated..." (author-prominent)
   - "Radar backscatter correlates with water stress [X]." (information-prominent)
4. **Quantitative claims:** Always cite when providing metrics or performance numbers
5. **State-of-the-art positioning:** Use comparison citations to position your work relative to existing studies

---

## Next Steps

1. **Verify access:** Check if all articles are accessible through your institution
2. **Read priority articles:** Focus on ⭐ Critical citations first
3. **Extract key metrics:** Note specific performance numbers, methodologies for your paper
4. **Update references.bib:** Ensure all entries have complete information (authors, pages, DOIs)
5. **Cross-reference:** Check that citations in article1.tex match this bibliography

---

**Compiled using writing_skills following IEEE format standards for academic research papers.**
