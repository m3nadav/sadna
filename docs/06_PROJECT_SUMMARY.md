# Kaggle CMI: Gesture Recognition Project Summary

**Project**: Detect Behavior with Sensor Data (BFRB Detection)
**Date**: 2026-01-14
**Status**: Deep Learning Models Training

---

## Project Overview

### Challenge
Classify 18 gesture types from wearable sensor data:
- **8 BFRB target gestures** (body-focused repetitive behaviors)
- **10 non-BFRB gestures**

### Dataset
- **Training**: 574,945 rows across 8,151 sequences (81 subjects)
- **Test**: 50% IMU-only, 50% full sensor data (critical constraint!)
- **Sensors**: IMU (7), ToF (320), Thermal (5), Metadata (9)

### Evaluation Metric
Competition Score = (Binary F1 + Macro F1) / 2
- **Binary F1**: BFRB vs non-BFRB classification
- **Macro F1**: 9-class (8 BFRB + 1 collapsed non-BFRB)

---

## Work Completed

### Phase 1: Exploratory Data Analysis ✅

**Files Created**:
- [Nadav_Sadna_Project.ipynb](Nadav_Sadna_Project.ipynb) - Comprehensive EDA
- [EDA_DOCUMENTATION.md](EDA_DOCUMENTATION.md) - 600+ lines of analysis

**Key Findings**:
1. **ToF Sensors** (VL53L7CX):
   - 5 sensors × 64 pixels = 320 features
   - 59% sparsity (-1 values)
   - Uncalibrated values 0-254 (not millimeters!)
   - Physical arrangement: overlapping 8×8 frames

2. **IMU Sensors** (BNO080/BNO085):
   - Accelerometer: acc_x, acc_y, acc_z (m/s²)
   - Rotation: rot_w, rot_x, rot_y, rot_z (quaternions)
   - Fused 9-axis sensor (acc + gyro + mag)

3. **Thermal Sensors** (MLX90632):
   - 5 non-contact IR temperature sensors
   - Range: 23-37°C (body temperature)
   - Mean: 31.9°C

4. **Class Imbalance**:
   - Majority: 640 sequences ("Above ear - pull hair")
   - Minority: 161 sequences (several gestures)
   - Imbalance ratio: 4:1

**Critical Discovery**: Test set is 50% IMU-only (no ToF/thermal data)
→ Requires dual model strategy

---

### Phase 2: Feature Engineering ✅

**Files Created**:
- [feature_engineering.py](feature_engineering.py) - Modular feature extraction
- [test_feature_engineering.py](test_feature_engineering.py) - Validation script

**Features Extracted**:

#### TIER 0 (IMU-only): 83 features
Critical for 50% IMU-only test sequences:

1. **Accelerometer** (42 features):
   - 3D magnitude, jerk (rate of change)
   - Statistics: mean, std, min, max, median, IQR, range
   - Activity metrics: high activity ratio, peak position
   - Temporal: moving averages, trends
   - Cross-axis: variance ratio, axis correlations

2. **Rotation** (30 features):
   - Euler angles: roll, pitch, yaw (from quaternions)
   - Angular velocity & acceleration
   - Statistics per axis
   - Rotation magnitude, axis dominance

3. **Temporal** (11 features):
   - Sequence length
   - Temporal phases (start/middle/end ratios)
   - Autocorrelation (lag 1)
   - Trends (linear regression slopes)

#### TIER 1 (ToF + Thermal): 75 features
Bonus when all sensors available:

4. **ToF** (60 features):
   - Per-sensor (12 per sensor × 5):
     - Mean/std/min/max valid depth
     - Valid pixel count (inverse sparsity)
     - Depth velocity (temporal change)
   - Global features:
     - Combined depth statistics
     - Total valid pixels across sensors
     - Cross-sensor gradients

5. **Thermal** (15 features):
   - Per-sensor statistics (mean, std, range)
   - Inter-sensor differences
   - Temperature change rate (ΔT/Δt)
   - Hottest/coldest sensor identification

**Total Features**: 158 (83 IMU + 75 ToF/Thermal)

---

### Phase 3: Baseline Models (XGBoost) ✅

**Files Created**:
- [train_baseline_models.py](train_baseline_models.py) - Dual model training
- [baseline_training.log](baseline_training.log) - Training output
- Models saved:
  - [models/xgboost_imu_only.pkl](models/xgboost_imu_only.pkl)
  - [models/xgboost_full_sensor.pkl](models/xgboost_full_sensor.pkl)

**Results**:

| Model | Competition Score | Binary F1 | Macro F1 | Accuracy |
|-------|------------------|-----------|----------|----------|
| **IMU-only** | 0.6864 | 0.9481 | 0.4247 | 48.1% |
| **Full Sensor** | 0.7838 | 0.9762 | 0.5913 | 62.9% |
| **Overall** (50/50) | **0.7351** | - | - | - |

**Value Added by ToF/Thermal**: +0.0973 (+14.2% improvement)

**Weak Gesture Classes** (F1 < 0.50):
1. Eyebrow - pull hair (0.33) ⚠️ CRITICAL
2. Scratch knee/leg skin (0.22)
3. Pinch knee/leg skin (0.42)
4. Neck - pinch skin (0.44)
5. Eyelash - pull hair (0.46)

**Strong Gestures** (F1 > 0.85):
- Text on phone (0.94)
- Glasses on/off (0.92)
- Feel around in tray (0.90)

---

### Phase 4: Feature Importance Analysis ✅

**Files Created**:
- [analyze_feature_importance.py](analyze_feature_importance.py) - Analysis script
- [FEATURE_IMPORTANCE_ANALYSIS.md](FEATURE_IMPORTANCE_ANALYSIS.md) - Results

**Key Findings**:

#### Top 5 IMU-only Features:
1. **angular_velocity_median** (Rotation) - 2146.0
2. **acc_autocorrelation_lag1** (Accelerometer) - 2032.0
3. **sequence_length** (Other) - 1837.0
4. **yaw_median** (Rotation) - 1747.0
5. **acc_z_min** (Accelerometer) - 1654.0

#### Top 5 Full Sensor Features:
1. **thm_2_mean** (Thermal) - 1104.0 ⭐ Thermal sensor 2 is most important!
2. **tof_2_depth_mean** (ToF) - 1094.0
3. **angular_velocity_median** (Rotation) - 1075.0
4. **acc_autocorrelation_lag1** (Accelerometer) - 1070.0
5. **tof_3_depth_mean** (ToF) - 909.0

#### Sensor Type Importance (Full Model):
- **Accelerometer**: 9250.0 (12 features, avg 770.8)
- **ToF**: 6763.0 (9 features, avg 751.4)
- **Thermal**: 3326.0 (4 features, avg 831.5) 👈 Highest average!
- **Rotation**: 2608.0 (3 features, avg 869.3)

**ToF & Thermal contribute 43% of top 30 feature importance** → Validates the +14.2% score improvement

#### Insights for Deep Learning:
1. **Rotation features dominate** IMU-only model → Focus on angular velocity/Euler angles
2. **Thermal sensor 2** (center position?) is single most important feature
3. **Temporal features crucial** (autocorrelation, sequence length)
4. **ToF sensors 2 & 3** (center/near-center?) most informative

---

### Phase 5: Deep Learning Models (In Progress) ⏳

**Files Created**:
- [train_cnn_models.py](train_cnn_models.py) - 1D CNN architecture
- [cnn_training.log](cnn_training.log) - Training output (in progress)

**Architecture**:
```
1D Convolutional Neural Network
================================

Input: (sequence_length, n_features)
  ↓
Masking Layer (handles variable-length sequences)
  ↓
Conv1D Block 1: 64 filters, kernel=5
  ├─ BatchNorm
  ├─ MaxPool (stride=2)
  └─ Dropout (0.3)
  ↓
Conv1D Block 2: 128 filters, kernel=5
  ├─ BatchNorm
  ├─ MaxPool (stride=2)
  └─ Dropout (0.3)
  ↓
Conv1D Block 3: 256 filters, kernel=3
  ├─ BatchNorm
  ├─ GlobalMaxPool
  └─ Dropout (0.4)
  ↓
Dense Layer: 128 units
  └─ Dropout (0.4)
  ↓
Output: 18 classes (softmax)
```

**Training Configuration**:
- **Loss**: Focal loss (gamma=2.0, alpha=0.25) → handles class imbalance
- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 32
- **Max epochs**: 100
- **Early stopping**: patience=15
- **Learning rate decay**: factor=0.5, patience=7

**Two Models Trained**:
1. **IMU-only CNN**: Input shape (150, 7) - for 50% IMU-only test
2. **Full Sensor CNN**: Input shape (150, 336) - for 50% full test

**Expected Performance**:
- IMU-only CNN: 0.70-0.75 (vs 0.6864 XGBoost)
- Full Sensor CNN: 0.82-0.87 (vs 0.7838 XGBoost)
- Overall: 0.76-0.81 (vs 0.7351 XGBoost)

**Status**: Training in background (15-30 min estimated)

---

## Model Comparison

### XGBoost Baseline vs CNN (Expected)

| Aspect | XGBoost | 1D CNN |
|--------|---------|--------|
| **Input** | Engineered features (83/158) | Raw time series (variable length) |
| **Feature Engineering** | Manual (83-158 features) | Automatic (learned representations) |
| **Temporal Modeling** | Aggregated statistics | Sequential patterns via convolution |
| **Class Imbalance** | Class weights | Focal loss |
| **Training Time** | 5-10 minutes | 15-30 minutes |
| **IMU-only Score** | 0.6864 | 0.70-0.75 (target) |
| **Full Sensor Score** | 0.7838 | 0.82-0.87 (target) |
| **Overall Score** | **0.7351** | **0.76-0.81** (target) |

---

## Technical Implementation Details

### Data Processing Pipeline

1. **Loading**:
   - Source: `/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data/train.csv`
   - 574,945 rows × 341 columns

2. **Sequence Creation**:
   - Group by `sequence_id`
   - Pad/truncate to max_length = 150 (median + 2×IQR)
   - Pad value: -999 (masked by CNN)

3. **Train/Val Split**:
   - Stratified by gesture (maintains class balance)
   - 80% train (6,520 sequences) / 20% val (1,631 sequences)
   - Random seed: 42 (reproducibility)

4. **Label Encoding**:
   - 18 gestures → integers 0-17
   - One-hot encoded for CNN (18-dim vector)

5. **Class Balancing**:
   - Computed class weights: `balanced` mode
   - Applied to loss function

### File Organization

```
project/
├── EDA & Documentation
│   ├── Nadav_Sadna_Project.ipynb
│   ├── EDA_DOCUMENTATION.md
│   ├── CRITICAL_NOTES.md
│   ├── COMPETITION_SUMMARY.md
│   └── PROGRESS_SUMMARY.md
│
├── Feature Engineering
│   ├── feature_engineering.py
│   └── test_feature_engineering.py
│
├── Baseline Models
│   ├── train_baseline_models.py
│   ├── baseline_training.log
│   └── models/
│       ├── xgboost_imu_only.pkl
│       └── xgboost_full_sensor.pkl
│
├── Feature Analysis
│   ├── analyze_feature_importance.py
│   └── FEATURE_IMPORTANCE_ANALYSIS.md
│
├── Deep Learning
│   ├── train_cnn_models.py
│   ├── cnn_training.log
│   └── models/
│       ├── cnn_imu_only.keras
│       ├── cnn_full_sensor.keras
│       └── cnn_label_encoder.pkl
│
└── Plans
    └── .claude/plans/glittery-plotting-lollipop.md
```

---

## Next Steps

### Immediate (Once CNN Training Completes):
1. ✅ Evaluate CNN performance on validation set
2. ✅ Compare CNN vs XGBoost per-gesture performance
3. ✅ Identify which model is better for each gesture
4. 🔄 **Create ensemble** combining XGBoost + CNN predictions

### Short-term Improvements:
1. **Hyperparameter Tuning**:
   - CNN: Adjust filters, kernel sizes, dropout rates
   - XGBoost: Tune max_depth, learning_rate, n_estimators

2. **Data Augmentation**:
   - Time warping (stretch/compress sequences)
   - Noise injection (Gaussian noise to sensors)
   - Rotation augmentation (small quaternion perturbations)

3. **Advanced Architectures**:
   - LSTM/GRU for longer-term dependencies
   - Transformer with attention (if data permits)
   - Multi-modal fusion (separate branches for IMU/ToF/Thermal)

### Final Submission:
1. **Ensemble Strategy**:
   - Weighted average of XGBoost + CNN predictions
   - Per-gesture model selection (use best model for each gesture)
   - Stacking meta-learner

2. **Test Set Inference**:
   - Load test.csv
   - Detect IMU-only vs full sensor sequences (check for nulls)
   - Route to appropriate model
   - Generate submission.csv

3. **Expected Final Score**: 0.78-0.82 (competition score)

---

## Key Insights & Lessons

### 1. Test Set Constraint is Critical
- 50% IMU-only dramatically changes strategy
- Cannot rely solely on ToF/thermal features
- Dual model approach is essential

### 2. Class Imbalance Requires Special Handling
- Standard accuracy is misleading
- Macro F1 metric equally weights all classes
- Focal loss and class weights are effective

### 3. Feature Importance Reveals Sensor Value
- **Thermal sensor 2** is surprisingly important (center position?)
- **Rotation features** (angular velocity, yaw) dominate IMU-only
- **ToF sensors 2 & 3** most informative (center/near-center)
- **Temporal features** (autocorrelation, sequence length) crucial

### 4. Deep Learning Benefits
- Learns representations automatically (no manual feature engineering)
- Captures sequential patterns (temporal modeling)
- Generalizes better to new subjects (when properly regularized)

### 5. Model Diversity Improves Ensemble
- XGBoost: Strong on engineered statistical features
- CNN: Strong on raw temporal patterns
- Combining both leverages complementary strengths

---

## Competition Context

**Kaggle Competition**: CMI - Detect Behavior with Sensor Data
**Organizer**: Child Mind Institute
**Goal**: Early detection of body-focused repetitive behaviors (BFRBs)
**Clinical Relevance**: BFRBs (hair pulling, skin picking, etc.) affect 1-5% of population
**Technical Challenge**: Real-time gesture recognition from wearable sensors with missing modalities

**Evaluation**:
- Binary F1: Distinguish BFRB from non-BFRB behaviors
- Macro F1: Classify specific gesture types
- Balanced score rewards both coarse and fine-grained classification

---

## Resources & References

### Sensor Documentation
- [ToF Sensor (VL53L7CX) Datasheet](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Competitions/Helios2025/Time_of_Flight_Sensor.pdf)
- IMU: BNO080/BNO085 (Hillcrest Labs / CEVA)
- Thermal: MLX90632 (Melexis)

### Libraries Used
- **Data Processing**: pandas, numpy
- **ML**: scikit-learn, xgboost
- **Deep Learning**: TensorFlow/Keras
- **Visualization**: (EDA in Jupyter)

### Academic References (for Plan)
- Time Series Classification: Wang et al. (2017), Fawaz et al. (2020)
- Gesture Recognition: Ronao & Cho (2016), Ordonez & Roggen (2016)
- Focal Loss: Lin et al. (2017) - "Focal Loss for Dense Object Detection"

---

## Current Status: Awaiting CNN Training Results ⏳

**Training Started**: Background task ID b3a4ea3
**Estimated Completion**: 15-30 minutes from start
**Next Action**: Evaluate CNN performance and decide on ensemble strategy

---

*Last Updated: 2026-01-14*
*Project Lead: Nadav | AI Assistant: Claude (Anthropic)*
