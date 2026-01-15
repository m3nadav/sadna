# Project Documentation

This folder contains all documentation for the Kaggle CMI Gesture Recognition project, organized in chronological order.

---

## Documents (in order of creation)

### [01_EDA_DOCUMENTATION.md](docs/01_EDA_DOCUMENTATION.md)
**Exploratory Data Analysis**

Comprehensive analysis of all sensor types:
- Time-of-Flight (ToF) sensors: 5 sensors × 64 pixels
- Accelerometer: 3-axis motion data
- Rotation: Quaternion-based orientation
- Thermal: 5 non-contact temperature sensors

**Key Findings**:
- ToF: 59% sparsity, uncalibrated values 0-254
- IMU: BNO080/BNO085 fused 9-axis sensor
- Critical constraint: 50% test set is IMU-only!

---

### [02_CRITICAL_NOTES.md](docs/02_CRITICAL_NOTES.md)
**Critical Insights & Corrections**

Important discoveries made during the project:
- Test set composition (50% IMU-only)
- ToF units correction (not millimeters!)
- IMU sensor type identification
- Modeling strategy implications

---

### [03_COMPETITION_SUMMARY.md](docs/03_COMPETITION_SUMMARY.md)
**Competition Overview**

Details about the Kaggle competition:
- Evaluation metric: (Binary F1 + Macro F1) / 2
- BFRB detection goal
- 18 gesture classes
- Dataset structure

---

### [04_PROGRESS_SUMMARY.md](docs/04_PROGRESS_SUMMARY.md)
**Mid-Project Progress Report**

Status update after completing EDA and feature engineering:
- Achievements summary
- Files created
- Next steps planned
- Technical decisions made

---

### [05_FEATURE_IMPORTANCE_ANALYSIS.md](docs/05_FEATURE_IMPORTANCE_ANALYSIS.md)
**XGBoost Feature Importance**

Analysis of which features drive predictions:
- **Top IMU feature**: angular_velocity_median (2146.0)
- **Top Full sensor feature**: thm_2_mean (1104.0)
- ToF/Thermal contribute 43% of top 30 importance
- Insights for deep learning architecture

**Key Insight**: Thermal sensor 2 (center position) is the single most important feature when all sensors are available!

---

### [06_PROJECT_SUMMARY.md](docs/06_PROJECT_SUMMARY.md)
**Complete Project Summary**

Comprehensive overview of the entire project:
- All phases from EDA to deep learning
- Results comparison (XGBoost vs CNN)
- File organization
- Technical implementation details
- Next steps and final submission strategy

---

## Data Split Strategy

### Train/Validation Split (80/20)

**Approach**:
```python
train_test_split(
    sequences,                      # 8,151 unique sequences
    test_size=0.2,                 # 20% validation
    stratify=sequences['gesture'], # Maintain gesture balance
    random_state=42                # Reproducible
)
```

**Results**:
- **Training**: 6,520 sequences (80%)
- **Validation**: 1,631 sequences (20%)

**Properties**:
1. ✅ **Sequence-level split**: No data leakage (entire sequences kept together)
2. ✅ **Stratified by gesture**: All 18 classes proportionally represented
3. ✅ **Reproducible**: Random seed = 42
4. ⚠️ **Subject overlap**: Same subject can appear in both train/val with different sequences
   - More rigorous: Subject-grouped split (would require StratifiedGroupKFold)
   - Current approach may slightly overestimate generalization

**Validation Set Simulation**:
To match test conditions, validation set is further split:
- 50% IMU-only (815 sequences): ToF/thermal masked as null
- 50% Full sensor (816 sequences): All sensors available

This ensures evaluation mimics the actual test set composition.

---

## Model Results Summary

### XGBoost Baseline (Trained on 6,520 sequences, validated on 1,631)

| Model | Competition Score | Binary F1 | Macro F1 |
|-------|------------------|-----------|----------|
| **IMU-only** | 0.6864 | 0.9481 | 0.4247 |
| **Full Sensor** | 0.7838 | 0.9762 | 0.5913 |
| **Overall** (50/50 mix) | **0.7351** | - | - |

**Value Added by ToF/Thermal**: +0.0973 (+14.2% improvement)

### 1D CNN Models (In Training)

**Expected Performance**:
- IMU-only CNN: 0.70-0.75 (target)
- Full Sensor CNN: 0.82-0.87 (target)
- Overall: 0.76-0.81 (target)

**Architecture**:
- 3 convolutional blocks (64 → 128 → 256 filters)
- Focal loss for class imbalance
- Masking for variable-length sequences
- Trained on raw time series (no manual feature engineering)

---

## File Organization

```
project/
├── docs/                          # 👈 You are here
│   ├── 00_README.md              # This file
│   ├── 01_EDA_DOCUMENTATION.md
│   ├── 02_CRITICAL_NOTES.md
│   ├── 03_COMPETITION_SUMMARY.md
│   ├── 04_PROGRESS_SUMMARY.md
│   ├── 05_FEATURE_IMPORTANCE_ANALYSIS.md
│   └── 06_PROJECT_SUMMARY.md
│
├── Nadav_Sadna_Project.ipynb     # Main EDA notebook
├── feature_engineering.py         # Feature extraction module
├── train_baseline_models.py       # XGBoost training
├── analyze_feature_importance.py  # Feature analysis
├── train_cnn_models.py            # Deep learning training
│
├── models/                        # Trained models
│   ├── xgboost_imu_only.pkl
│   ├── xgboost_full_sensor.pkl
│   ├── cnn_imu_only.keras        # (in training)
│   └── cnn_full_sensor.keras     # (in training)
│
└── baseline_training.log          # Training logs
```

---

## Quick Navigation

**Start Here**: [06_PROJECT_SUMMARY.md](docs/06_PROJECT_SUMMARY.md) - Full project overview

**Technical Deep Dives**:
- Sensor Analysis → [01_EDA_DOCUMENTATION.md](docs/01_EDA_DOCUMENTATION.md)
- Feature Engineering → [04_PROGRESS_SUMMARY.md](docs/04_PROGRESS_SUMMARY.md)
- Model Performance → [05_FEATURE_IMPORTANCE_ANALYSIS.md](docs/05_FEATURE_IMPORTANCE_ANALYSIS.md)

**Critical Constraints**: [02_CRITICAL_NOTES.md](docs/02_CRITICAL_NOTES.md) - Must-read for understanding test set challenge

---

*Documentation last updated: 2026-01-14*
