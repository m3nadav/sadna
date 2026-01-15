# Exploratory Data Analysis Documentation
## CMI: Detect Behavior with Sensor Data

**Project**: Kaggle Competition - Gesture Recognition with Multimodal Sensor Data
**Date**: 2026-01-14
**Analyst**: Claude + Nadav
**Status**: EDA Complete ✅

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Sensor Analysis](#sensor-analysis)
5. [Key Findings](#key-findings)
6. [Recommendations](#recommendations)
7. [Code Repository](#code-repository)

---

## Overview

This document summarizes the comprehensive exploratory data analysis (EDA) conducted on all sensor modalities in the CMI gesture recognition dataset. The analysis covers:

- **Time-of-Flight (ToF)** depth sensors (320 features)
- **Accelerometer** (3-axis IMU) (3 features)
- **Rotation/Quaternion** sensors (4 features)
- **Thermal** sensors (5 features)

**Total Features**: 341 columns (332 sensor features + 9 metadata columns)
**Total Samples**: 574,945 frames across 8,151 sequences
**Target Classes**: 18 distinct gesture types

---

## Dataset Description

### File Structure

```
/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data/
├── train.csv                    (574,945 rows × 341 columns)
├── test.csv                     (107 rows × 336 columns)
├── train_demographics.csv       (81 subjects)
└── test_demographics.csv        (2 subjects)
```

### Target Variable: Gesture Classes

| Gesture | Sequences | Category |
|---------|-----------|----------|
| Forehead - scratch | 640 | Contact (upper body) |
| Text on phone | 640 | Manipulation |
| Forehead - pull hairline | 640 | Contact (upper body) |
| Neck - scratch | 640 | Contact (upper body) |
| Neck - pinch skin | 640 | Contact (upper body) |
| Eyelash - pull hair | 640 | Contact (upper body) |
| Above ear - pull hair | 638 | Contact (upper body) |
| Eyebrow - pull hair | 638 | Contact (upper body) |
| Cheek - pinch skin | 637 | Contact (upper body) |
| Wave hello | 478 | Air gesture |
| Write name in air | 477 | Air gesture |
| Pull air toward your face | 477 | Air gesture |
| Feel around in tray | 161 | Complex manipulation |
| Write name on leg | 161 | Contact (lower body) |
| Pinch knee/leg skin | 161 | Contact (lower body) |
| Scratch knee/leg skin | 161 | Contact (lower body) |
| Drink from bottle/cup | 161 | Manipulation |
| Glasses on/off | 161 | Manipulation |

**⚠️ Class Imbalance**: Upper body gestures (640 sequences) vs. lower body/misc (161 sequences)
**Implication**: Critical for Macro F1-Score evaluation metric!

### Sequence Characteristics

- **Sequence Length**: 29-700 frames (mean: 70.5, median: 59)
- **Subjects**: 81 unique individuals in training
- **Orientations**: 4 body positions (seated, lying on side, lying on back, leaning)
- **Behaviors**: 4 action phases (performs gesture, moves hand, hand at target, relaxes)

---

## Methodology

### Analysis Workflow

1. **Load Data**: Read train.csv using pandas
2. **Per-Modality Analysis**:
   - Statistical summaries (mean, std, min, max, missing data)
   - Distribution visualizations (histograms, boxplots)
   - Per-gesture comparisons (group by gesture class)
   - Temporal pattern analysis (time series plots for sample sequences)
   - Derived feature engineering (magnitude, Euler angles, aggregations)
3. **Cross-Modality Insights**: Identify gesture categories and sensor complementarity
4. **Documentation**: Create reproducible notebook cells with code blocks and interpretations

### Tools Used

- **Python 3.x**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualizations
- **Jupyter Notebook**: Interactive analysis

### Code Location

All analysis code is in: `Nadav_Sadna_Project.ipynb`

Sections:
- Cells 1-36: ToF sensor analysis (pre-existing)
- Cells 37+: Accelerometer, Rotation, Thermal analysis (newly added)

---

## Sensor Analysis

### 1. Time-of-Flight (ToF) Sensors

**Status**: ✅ Previously completed (existing in notebook)

#### Description
- **5 sensors** arranged in cross pattern on bracelet
- Each sensor: **8×8 pixel grid** = 64 features
- **Total**: 320 ToF features (tof_1_v0 through tof_5_v63)
- **Physical arrangement**:
  ```
          ToF 2
  ToF 5   ToF 1   ToF 3
          ToF 4
  ```

#### Key Statistics
- **Valid Range**: 0-249 mm (millimeters)
- **Invalid Value**: -1.0 (out of range / no object detected)
- **Global Sparsity**: 59.4% invalid readings
- **Mean Valid Depth**: 108.21 mm

#### Per-Sensor Performance
| Sensor | Sparsity | Mean Depth (mm) | Notes |
|--------|----------|----------------|-------|
| ToF 1 | 58.77% | 111.75 | Center sensor |
| ToF 2 | 62.93% | 101.65 | Top sensor |
| ToF 3 | 60.14% | 105.34 | Left sensor (rotated 90°) |
| ToF 4 | 52.34% | 112.38 | **Best** - bottom sensor |
| ToF 5 | 62.84% | 109.95 | Right sensor (rotated 90°) |

#### Visualizations Created
- Heatmaps of separated vs. overlapping sensor frames
- Animated GIF sequences showing depth changes over time
- Per-pixel sparsity analysis

#### Modeling Implications
1. **Challenge**: High sparsity (59%) requires robust handling
   - Tree-based models can handle -1 natively
   - Deep learning needs imputation or masking
2. **Opportunity**: Dimensionality reduction via aggregation
   - Per-sensor mean depth (5 features vs. 320)
   - Valid pixel count per sensor
   - Spatial center of mass
3. **Best Sensor**: ToF 4 (bottom) most reliable for features

---

### 2. Accelerometer Sensors

**Status**: ✅ Newly completed (analysis added to notebook)

#### Description
- **3-axis IMU** capturing linear acceleration
- **Features**: acc_x, acc_y, acc_z
- **Units**: Likely m/s² or g (gravity units)

#### Key Statistics
| Feature | Min | Max | Mean | Std | Missing |
|---------|-----|-----|------|-----|---------|
| acc_x | -12.79 | 14.75 | 0.26 | 2.04 | 0% |
| acc_y | -12.45 | 14.43 | -0.18 | 1.89 | 0% |
| acc_z | -11.54 | 14.38 | 0.03 | 1.96 | 0% |

#### Derived Features
- **3D Magnitude**: `sqrt(acc_x² + acc_y² + acc_z²)`
  - Range: 0.00 - 20.37
  - Mean: 3.33
  - **Highly informative** for gesture intensity

#### Per-Gesture Patterns
**High Acceleration Gestures** (dynamic):
- Write name in air: 4.2 magnitude
- Wave hello: 4.0 magnitude
- Pull air toward face: 3.8 magnitude

**Low Acceleration Gestures** (static):
- Text on phone: 2.7 magnitude
- Forehead scratch: 2.9 magnitude
- Glasses on/off: 3.0 magnitude

#### Temporal Patterns
- **Clear phases**: Start → Peak → End visible in time series
- **Periodic motions**: Wave hello shows sinusoidal pattern
- **Sharp spikes**: Pinching gestures show brief acceleration bursts

#### Visualizations Created
- 3-panel histogram (acc_x, acc_y, acc_z distributions)
- 3D magnitude distribution + boxplot comparison
- Per-gesture mean acceleration ranking (horizontal bar chart)
- Time series plots for 4 sample gestures

#### Modeling Implications
1. **Complete Data**: No missing values - highest quality sensor
2. **Strong Discriminator**: Magnitude clearly separates dynamic vs. static gestures
3. **Temporal Information**: Moving averages and jerk will enhance features
4. **Feature Priority**: Magnitude is **must-have** baseline feature

---

### 3. Rotation/Quaternion Sensors

**Status**: ✅ Newly completed (analysis added to notebook)

#### Description
- **Quaternion representation** of 3D wrist orientation
- **Features**: rot_w, rot_x, rot_y, rot_z (4D rotation vector)
- **Normalization**: Magnitude ≈ 1.0 (confirmed ✓)

#### Key Statistics
| Feature | Min | Max | Mean | Std | Missing |
|---------|-----|-----|------|-----|---------|
| rot_w | -0.999 | 0.999 | -0.078 | 0.548 | 0% |
| rot_x | -0.997 | 0.998 | 0.010 | 0.388 | 0% |
| rot_y | -0.994 | 0.999 | -0.002 | 0.429 | 0% |
| rot_z | -0.998 | 0.999 | 0.025 | 0.473 | 0% |

#### Derived Features: Euler Angles
Quaternions converted to intuitive Euler angles (degrees):

| Angle | Physical Meaning | Range |
|-------|------------------|-------|
| **Roll** | Pronation/Supination (x-axis) | -180° to +180° |
| **Pitch** | Flexion/Extension (y-axis) | -90° to +90° |
| **Yaw** | Radial/Ulnar Deviation (z-axis) | -180° to +180° |

#### Per-Gesture Rotation Ranges
**High Rotation Gestures** (multi-axis movement):
- Write name in air: Roll 350°, Pitch 170°, Yaw 340°
- Wave hello: Roll 320°, Pitch 150°, Yaw 350°
- Pull air toward face: Roll 300°, Pitch 160°, Yaw 310°

**Low Rotation Gestures** (stable orientation):
- Text on phone: Roll 180°, Pitch 90°, Yaw 200°
- Forehead scratch: Roll 200°, Pitch 100°, Yaw 220°

#### Temporal Patterns
- **Wave hello**: Large yaw oscillations (side-to-side wrist movement)
- **Write name in air**: Complex 3-axis rotation patterns (cursive motion)
- **Forehead scratch**: Stable orientation with small noise
- **Text on phone**: Minimal rotation (static typing posture)

#### Visualizations Created
- 2×2 grid of quaternion component distributions
- 3-panel Euler angle histograms (roll, pitch, yaw)
- Per-gesture rotation range comparison (3 subplots for roll/pitch/yaw)
- Time series plots showing orientation changes over time

#### Modeling Implications
1. **Feature Conversion**: Euler angles more interpretable than raw quaternions
2. **Angular Velocity**: Δorientation/Δt will capture rotation dynamics
3. **Gesture Complexity**: Rotation range is strong discriminator
4. **Complete Data**: No missing values - reliable sensor

---

### 4. Thermal Sensors

**Status**: ✅ Newly completed (analysis added to notebook)

#### Description
- **5 thermal sensors** measuring temperature
- **Features**: thm_1, thm_2, thm_3, thm_4, thm_5
- **Units**: Degrees Celsius (°C)

#### Key Statistics
| Feature | Min (°C) | Max (°C) | Mean (°C) | Std (°C) | Missing |
|---------|----------|----------|-----------|----------|---------|
| thm_1 | 23.0 | 36.5 | 30.8 | 2.1 | 0% |
| thm_2 | 23.2 | 36.8 | 31.0 | 2.0 | 0% |
| thm_3 | 23.1 | 36.7 | 30.9 | 2.1 | 0% |
| thm_4 | 23.0 | 37.0 | 31.1 | 2.0 | 0% |
| thm_5 | 23.3 | 36.9 | 31.0 | 2.0 | 0% |

**Overall Mean**: 30.98°C (typical skin temperature)

#### Inter-Sensor Correlation
| | thm_1 | thm_2 | thm_3 | thm_4 | thm_5 |
|---|-------|-------|-------|-------|-------|
| thm_1 | 1.000 | 0.967 | 0.970 | 0.963 | 0.965 |
| thm_2 | 0.967 | 1.000 | 0.972 | 0.968 | 0.969 |
| thm_3 | 0.970 | 0.972 | 1.000 | 0.968 | 0.970 |
| thm_4 | 0.963 | 0.968 | 0.968 | 1.000 | 0.967 |
| thm_5 | 0.965 | 0.969 | 0.970 | 0.967 | 1.000 |

**Very high correlation** (>0.96) → sensors measure similar temperatures

#### Derived Features
- **thm_mean**: Average temperature across all sensors
  - Range: 24.4 - 36.2°C
  - Mean: 30.98°C
- **thm_range**: Temperature difference (max - min)
  - Range: 0.0 - 4.8°C
  - Mean: 0.85°C
  - **Interpretation**: >2°C suggests uneven/partial contact
- **thm_std**: Standard deviation across sensors
  - Range: 0.0 - 1.9°C
  - Mean: 0.35°C

#### Per-Gesture Temperature Patterns
**Warmest Gestures** (skin contact):
- Neck pinch skin: 31.3°C
- Forehead scratch: 31.2°C
- Eyelash pull hair: 31.2°C

**Coolest Gestures** (air exposure):
- Wave hello: 30.5°C
- Write name in air: 30.6°C
- Pull air toward face: 30.7°C

**Temperature Difference**: ~0.6-0.8°C between contact and air gestures

#### Temporal Patterns
- **Forehead scratch**: Gradual temperature increase during sustained contact
- **Wave hello**: Stable temperature (no skin contact, air cooling)
- **Text on phone**: Stable moderate temperature (static position)
- **Neck pinch**: Sharp temperature spike during brief pinching

#### Visualizations Created
- 2×3 grid of thermal sensor distributions + boxplot comparison
- Derived features (mean, range, std) histograms
- Per-gesture mean temperature ranking (horizontal bar chart)
- Time series plots showing temperature changes over time
- Correlation heatmap (5×5 matrix)

#### Modeling Implications
1. **Contact Detection**: Thermal features distinguish contact vs. air gestures
2. **Temperature Change Rate**: ΔT/Δt captures contact initiation/release
3. **Redundancy**: High correlation means mean temperature is sufficient
4. **Complete Data**: No missing values - reliable sensor

---

## Key Findings

### 1. Multi-Modal Sensor Complementarity

Each sensor modality provides unique information:

| Sensor | Captures | Best For |
|--------|----------|----------|
| **ToF** | Spatial position, object proximity | Hand-object distance, spatial configuration |
| **Accelerometer** | Linear motion intensity | Dynamic vs. static gestures, movement speed |
| **Rotation** | Wrist orientation | Gesture complexity, rotation-based actions |
| **Thermal** | Contact state | Contact vs. air gestures, sustained contact |

**Insight**: Multi-modal fusion is essential for comprehensive gesture understanding.

### 2. Gesture Categories (Data-Driven Taxonomy)

Based on sensor patterns, gestures cluster into 4 categories:

#### Category 1: **High-Motion Air Gestures**
- Examples: Wave hello, Write name in air, Pull air toward face
- **Accelerometer**: High magnitude (>3.8)
- **Rotation**: High range (>300° in multiple axes)
- **Thermal**: Low/stable temperature (~30.6°C)
- **ToF**: Variable depth changes

#### Category 2: **Contact-Based Upper Body Gestures**
- Examples: Forehead scratch, Neck pinch, Eyelash pull hair
- **Accelerometer**: Low to moderate magnitude (2.7-3.2)
- **Rotation**: Low range (<250°)
- **Thermal**: Higher temperature (~31.2°C), increases during contact
- **ToF**: Closer depths when hand approaches body

#### Category 3: **Precision Manipulation**
- Examples: Text on phone, Glasses on/off
- **Accelerometer**: Low magnitude (~2.7-3.0)
- **Rotation**: Low range, stable orientation
- **Thermal**: Moderate temperature (~30.8°C)
- **ToF**: Static patterns

#### Category 4: **Complex Lower Body / Object Manipulation**
- Examples: Feel around in tray, Write name on leg, Drink from bottle
- **Accelerometer**: Moderate magnitude (3.2-3.5)
- **Rotation**: Moderate range changes
- **Thermal**: Variable
- **ToF**: Complex object interaction patterns

### 3. Class Imbalance Impact

**Critical Issue**: 640 sequences vs. 161 sequences (4:1 ratio)

| Class Group | Sequences | % of Total |
|-------------|-----------|------------|
| Upper body contact (9 classes) | 5,758 | 70.6% |
| Air gestures (3 classes) | 1,432 | 17.6% |
| Lower body / manipulation (6 classes) | 966 | 11.8% |

**Implication for Macro F1-Score**:
- Majority class accuracy ≠ good Macro F1
- **Must ensure minority classes (13-18) perform well**
- Mitigation: Class-balanced loss, SMOTE, threshold tuning

### 4. Data Quality Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Completeness** | ✅ Excellent | Only ToF has missingness (59% by design, not error) |
| **Data Range** | ✅ Excellent | All sensors within expected physical ranges |
| **Consistency** | ✅ Excellent | No anomalies or outliers detected |
| **Temporal Coherence** | ✅ Excellent | Smooth time series, no discontinuities |
| **Class Balance** | ⚠️ Poor | 4:1 imbalance between majority/minority classes |

**Overall Assessment**: High-quality dataset with manageable challenges (ToF sparsity, class imbalance).

### 5. Feature Engineering Priorities

Based on discriminative power observed in EDA:

**Tier 1 - Critical Features** (implement immediately):
1. Accelerometer 3D magnitude
2. Euler angles (roll, pitch, yaw) from quaternions
3. Thermal mean temperature
4. ToF valid pixel count per sensor
5. ToF mean depth per sensor

**Tier 2 - High-Value Features**:
6. Accelerometer jerk (Δacceleration/Δt)
7. Angular velocity (Δorientation/Δt)
8. Thermal temperature range (max-min)
9. Moving averages (window=5,10 frames) for acc and rotation
10. ToF center of mass (spatial features)

**Tier 3 - Refinement Features** (if time permits):
11. Temperature change rate (ΔT/Δt)
12. Frequency domain features (FFT) for periodic gestures
13. Cross-sensor correlations
14. Subject-specific normalization using demographics

---

## ⚠️ CRITICAL: Test Set Challenge

### Hidden Test Set Composition

**IMPORTANT**: The hidden test set (~3,500 sequences) has a unique challenge:

- **50% of sequences**: **IMU data ONLY**
  - `acc_x`, `acc_y`, `acc_z`, `rot_w`, `rot_x`, `rot_y`, `rot_z` present
  - **`thm_*` and `tof_*` columns contain NULL values**
- **50% of sequences**: **All sensors** (IMU + thermal + ToF)

### Modeling Implications

**This fundamentally changes our strategy!**

1. **Cannot Rely Solely on ToF/Thermal Features**
   - Models must perform reasonably well with IMU-only data
   - ToF and thermal are "bonus" features when available

2. **Required Model Architectures**:

   **Option A: Ensemble Approach** (Recommended)
   - **Model 1**: IMU-only model (acc + rot features)
     - Trains on all data using only IMU features
     - Serves as fallback for IMU-only test sequences
   - **Model 2**: Full multi-modal model (IMU + thermal + ToF)
     - Trains on all features
     - Used when all sensors available
   - **Inference**: Check for null values → route to appropriate model

   **Option B: Missing-Tolerant Model**
   - Train single model that handles missing modalities
   - Use masking or imputation for missing ToF/thermal
   - Risk: May underperform compared to ensemble

3. **Feature Engineering Priorities UPDATE**:

   **TIER 0 - CRITICAL (IMU-based features)**:
   - These must be HIGHLY discriminative since they're all we have for 50% of test
   1. Accelerometer 3D magnitude + jerk
   2. Euler angles (roll, pitch, yaw) from quaternions
   3. Angular velocity and acceleration
   4. Temporal IMU features (moving averages, velocity, peaks)
   5. IMU-based sequence statistics (mean, std, range)

   **TIER 1 - High Value (when available)**:
   6. Thermal mean temperature
   7. ToF valid pixel count per sensor
   8. ToF mean depth per sensor
   9. Temperature range and change rate

4. **Evaluation Strategy UPDATE**:
   - **Simulate test conditions**: Create validation split with 50% IMU-only
   - Measure Macro F1 separately for:
     - IMU-only sequences
     - Full sensor sequences
     - Overall (weighted average)
   - Target: **IMU-only Macro F1 > 0.60**, **Full Macro F1 > 0.75**

5. **Known Sensor Failures**:
   - Beyond designed IMU-only sequences, there are "communication failures"
   - Some sequences may have partial sensor data
   - Models must be robust to missing data patterns

### Revised Success Criteria

| Scenario | Target Macro F1 | Strategy |
|----------|----------------|----------|
| **IMU-only** | > 0.60 | Strong IMU feature engineering + temporal modeling |
| **Full sensors** | > 0.75 | Multi-modal fusion leveraging all sensor information |
| **Overall** | > 0.68 | Ensemble approach with fallback logic |

**Key Insight**: This competition tests whether ToF + thermal sensors **add value beyond IMU alone**. We must demonstrate improvement when additional sensors are available while maintaining reasonable performance with IMU-only data.

---

## Recommendations

### Immediate Next Steps (Week 1) - UPDATED FOR TEST SET CHALLENGE

1. **Feature Engineering Pipeline** (Priority 1)
   - Implement Tier 1 + Tier 2 features
   - Create `feature_engineering.py` module
   - Generate engineered feature matrix for train/test
   - Target: Reduce from 341 → ~150-200 meaningful features

2. **Preprocessing Pipeline** (Priority 2)
   - Handle ToF sparsity (option: tree-based models handle -1 naturally)
   - Normalize accelerometer (standardize per axis)
   - No scaling needed for ToF (already in mm) or thermal (already in °C)
   - Create `preprocessing.py` module

3. **Baseline Model - XGBoost** (Priority 3)
   - Use sequence-level aggregated features (mean, std, min, max per sequence)
   - Apply `class_weight='balanced'` for imbalance
   - 5-fold stratified group cross-validation (group by subject)
   - Target: **Macro F1 > 0.65**

4. **Model Evaluation Framework**
   - Implement Macro F1-Score calculation
   - Per-class F1-score reporting (identify weak classes)
   - Confusion matrix visualization
   - Create `evaluation.py` module

### Medium-Term Goals (Week 2-3)

5. **Deep Learning Experiments**
   - 1D CNN for temporal patterns (local GPU)
   - LSTM/GRU for long-term dependencies (cloud GPU)
   - Use focal loss to handle class imbalance
   - Target: **Macro F1 > 0.75**

6. **Multi-Modal Fusion**
   - Train modality-specific models (ToF-only, motion-only, thermal-only)
   - Ensemble via weighted voting or meta-classifier
   - Target: **Macro F1 > 0.80**

7. **Threshold Tuning**
   - Optimize classification thresholds per class
   - Maximize Macro F1 on validation set
   - Expected gain: +2-5% Macro F1

### Advanced Optimizations (If Time Permits)

8. **Handedness Normalization**
   - Mirror features for left-handed subjects
   - Flip ToF sensors (tof_3 ↔ tof_5)
   - Negate appropriate accelerometer/rotation axes

9. **Demographics Integration**
   - Merge with `train_demographics.csv`
   - Normalize features by arm length, height
   - Subject-specific scaling

10. **Data Augmentation**
    - Time warping (stretch/compress sequences)
    - Noise injection to sensor readings
    - SMOTE on sequence-level features for minority classes

---

## Code Repository

### File Structure

```
/Users/nadav/code/openu/sadna/project/
├── Nadav_Sadna_Project.ipynb          # Main EDA notebook (updated)
├── plans/
│   └── feature_engineering_and_model_selection_plan.md   # Detailed plan
├── EDA_DOCUMENTATION.md               # This file
├── feature_engineering.py             # To be created
├── preprocessing.py                   # To be created
├── baseline_models.ipynb              # To be created
├── train_deep_models.py               # To be created
└── evaluation.py                      # To be created
```

### Notebook Sections

**Nadav_Sadna_Project.ipynb**:
- Cells 1-3: Data loading
- Cells 4-9: Dataset overview, gesture distribution
- Cells 10-30: ToF sensor analysis (pre-existing)
- Cells 31-36: Test set exploration
- **Cells 37+: Accelerometer analysis** (newly added)
  - Basic statistics, distributions
  - 3D magnitude derivation
  - Per-gesture patterns
  - Time series visualizations
- **Cells 37+: Rotation/Quaternion analysis** (newly added)
  - Quaternion validation
  - Euler angle conversion
  - Per-gesture rotation ranges
  - Time series visualizations
- **Cells 37+: Thermal sensor analysis** (newly added)
  - Temperature distributions
  - Derived features (mean, range, std)
  - Per-gesture temperature patterns
  - Correlation analysis
- **Cells 37+: EDA Summary** (newly added)
  - Comprehensive markdown summary
  - Cross-modality insights
  - Recommendations

### Reproducibility

All analysis is reproducible by:
1. Downloading dataset from Kaggle: `kagglehub.competition_download('cmi-detect-behavior-with-sensor-data')`
2. Running cells sequentially in `Nadav_Sadna_Project.ipynb`
3. All code is self-contained with clear comments

**Dependencies**:
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
import kagglehub
```

---

## Appendix: Sensor Specifications

### Time-of-Flight Sensor: VL53L7CX
- **Manufacturer**: STMicroelectronics
- **Model**: VL53L7CX
- **Resolution**: 8×8 pixels = 64 pixels per sensor
- **Units**: **Uncalibrated sensor values 0-254** (NOT millimeters!)
- **Invalid Reading**: -1 (no object detected / no reflection)
- **Reading Convention**: Row-wise from top-left to bottom-right
- **Field of View**: 45° diagonal
- **Datasheet**: [VL53L7CX PDF](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Competitions/Helios2025/Time_of_Flight_Sensor.pdf)

### Inertial Measurement Unit (IMU): BNO080/BNO085
- **Type**: Integrated 9-axis IMU (accelerometer + gyroscope + magnetometer)
- **Manufacturer**: Hillcrest Labs / CEVA
- **Accelerometer Output**: acc_x, acc_y, acc_z
  - **Units**: meters per second squared (m/s²)
  - **Range**: ±16 g (estimated from data: ±12.79 to 14.75 m/s²)
- **Rotation Output**: rot_w, rot_x, rot_y, rot_z
  - **Type**: Quaternion representation (fused sensor data)
  - **Processing**: Onboard sensor fusion (accelerometer + gyroscope + magnetometer)
  - **Normalization**: Unit quaternions (magnitude = 1)
- **Sampling Rate**: Aligned with sequence_counter (frame-based)

### Thermopile Sensors: MLX90632
- **Type**: Non-contact infrared temperature sensors
- **Manufacturer**: Melexis
- **Count**: 5 sensors (indexed 1-5, positions shown in competition overview image)
- **Measurement**: Infrared radiation from skin/objects
- **Units**: Degrees Celsius (°C)
- **Observed Range**: 23-37°C (human body temperature)
- **Advantage**: Non-contact measurement allows detection without touching surface

---

## Contact & Attribution

**Primary Analyst**: Claude (Anthropic AI Assistant)
**Project Lead**: Nadav
**Institution**: Open University - Sadna Project
**Competition**: [CMI: Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/)
**Evaluation Metric**: Macro F1-Score

---

**Last Updated**: 2026-01-14
**Version**: 1.0
**Status**: EDA Complete ✅ | Feature Engineering In Progress 🔄
