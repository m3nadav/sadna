# Feature Engineering & Model Selection Plan
## Kaggle CMI: Detect Behavior with Sensor Data

---

## Current Status

### Completed Work
- ✅ Comprehensive EDA on Time-of-Flight (ToF) sensors (5 sensors × 64 pixels = 320 features)
- ✅ Understanding of ToF physical arrangement and sensor specifications
- ✅ Visualization framework for ToF data (overlapping vs. separated sensor frames)
- ✅ Statistical analysis: 59% sparsity, valid range 0-249mm, mean distance ~108mm
- ✅ Identified ToF sensor orientations need normalization (sensors 3 & 5 require 90° rotation)

### Dataset Overview
- **Files**: train.csv (574,945 rows × 341 cols), test.csv (107 rows × 336 cols), demographics files
- **Sequences**: 8,151 training sequences (29-700 frames each, mean 70.5)
- **Target**: 18 gesture classes with class imbalance (640 vs 161 sequences per class)
- **Subjects**: 81 unique subjects with demographic data (age, sex, handedness, anthropometry)

### Available Sensors (341 total features)
1. **ToF Sensors (320 features)** - ✅ Explored
2. **Accelerometer (3 features)** - ❌ Not explored: acc_x, acc_y, acc_z
3. **Rotation/Quaternion (4 features)** - ❌ Not explored: rot_w, rot_x, rot_y, rot_z
4. **Thermal Sensors (5 features)** - ❌ Not explored: thm_1 through thm_5
5. **Metadata (9 features)** - Subject, orientation, behavior, phase, sequence_type

---

## Research Capabilities Assessment

### Available Tools for This Project
✅ **Web Search** - Can search scientific papers and documentation
✅ **Context7 MCP** - Access up-to-date library documentation (scikit-learn, PyTorch, TensorFlow, etc.)
✅ **Scientific Skills** - 142+ specialized skills available including:
- Time series ML (aeon)
- Deep learning frameworks
- Data processing (pandas, polars, dask)
- Visualization tools

### Limitations Identified
⚠️ **No direct scientific paper database access** - Cannot query ArXiv, IEEE, PubMed directly via MCP
- **Workaround**: Use WebSearch to find and fetch papers, then extract content

⚠️ **Cannot execute code during planning** - Can only read and plan
- **Workaround**: Detailed step-by-step implementation plan for execution phase

---

## Phase 1: Understanding Unexplored Sensors

### 1.1 Accelerometer Data Analysis
**Goal**: Understand motion patterns for gesture classification

**Tasks**:
- Load and visualize acc_x, acc_y, acc_z distributions
- Calculate derived features:
  - Magnitude: √(x² + y² + z²)
  - Jerk: d(acceleration)/dt
  - Moving averages (window sizes: 5, 10, 20 frames)
- Analyze per-gesture patterns
- Check for correlation with gesture classes
- Investigate if normalization by subject handedness is needed

### 1.2 Rotation/Quaternion Data Analysis
**Goal**: Capture wrist orientation changes during gestures

**Tasks**:
- Convert quaternions to Euler angles (roll, pitch, yaw)
- Calculate angular velocity: Δorientation/Δt
- Compute rotation magnitude and direction changes
- Analyze gesture-specific rotation patterns (e.g., "Wave hello" vs "Scratch forehead")
- Check stability during different orientations (seated, lying)

### 1.3 Thermal Sensor Analysis
**Goal**: Detect skin contact and temperature changes

**Tasks**:
- Visualize thermal sensor distributions (thm_1 to thm_5)
- Calculate inter-sensor temperature differences (proximity patterns)
- Compute temporal temperature changes (ΔT/Δt)
- Identify correlation with contact-based gestures (scratch, pinch vs. air gestures)
- Check for subject-specific baseline temperatures

---

## Phase 2: Feature Engineering Strategy

### 2.1 ToF Feature Engineering (Existing Data)
**Current**: 320 raw pixel values per frame

**Proposed Features**:
1. **Per-Sensor Aggregations** (5 sensors × 6 features = 30 features)
   - Mean valid depth
   - Std valid depth
   - Valid pixel count (sparsity inverse)
   - Min/max depth
   - Center of mass (x, y coordinates)

2. **Cross-Sensor Features** (10 features)
   - Depth gradient between adjacent sensors
   - Correlation between overlapping regions
   - Total valid pixel count across all sensors

3. **Spatial Patterns** (20 features)
   - Global frame center of mass (using overlapping frame technique)
   - Spatial moments (compactness, eccentricity)
   - Distance to frame center
   - Depth histogram features (percentiles)

4. **Temporal ToF Features** (per sensor, 5×4 = 20 features)
   - Velocity (Δdepth/Δt)
   - Acceleration (Δ²depth/Δt²)
   - Moving average (windows: 5, 10 frames)
   - Rate of change in valid pixel count

**Rationale**: Reduce 320 dimensions while preserving spatial and temporal information

### 2.2 Motion Feature Engineering (Accelerometer + Rotation)
**Raw**: 7 features (3 acc + 4 rot)

**Proposed Features**:
1. **Accelerometer Derived** (12 features)
   - 3D magnitude
   - Jerk (3 axes + magnitude)
   - Moving average (3 axes, window=5)
   - Peak acceleration in sequence
   - Acceleration variance

2. **Rotation Derived** (15 features)
   - Euler angles (roll, pitch, yaw)
   - Angular velocity (3 axes)
   - Angular acceleration (3 axes)
   - Total rotation magnitude
   - Rotation rate variance
   - Peak angular velocity

3. **Fusion Features** (8 features)
   - Correlation(acc_magnitude, angular_velocity)
   - Movement type indicator (linear vs rotational dominant)
   - Rest vs. motion periods (threshold-based segmentation)
   - Coordination score (alignment of acceleration and rotation changes)

### 2.3 Thermal Feature Engineering
**Raw**: 5 sensors

**Proposed Features**:
1. **Spatial Thermal** (10 features)
   - Mean temperature
   - Temperature range (max - min)
   - Inter-sensor differences (adjacency-based)
   - Hottest/coldest sensor ID

2. **Temporal Thermal** (10 features)
   - Temperature change rate (ΔT/Δt per sensor)
   - Moving average temperature
   - Peak temperature change
   - Contact detection (rapid temperature increase)

### 2.4 Demographic-Normalized Features
**Approach**: Scale features by subject anthropometry

**Normalization Factors**:
- Accelerometer magnitude → normalize by arm length (leverage pendulum effect)
- ToF depth readings → normalize by shoulder_to_wrist distance
- Rotation ranges → normalize by handedness (left vs. right mirroring)

### 2.5 Context-Aware Features
**Metadata to Incorporate**:
- One-hot encode: orientation (4 classes), behavior (4 classes), phase (2 classes)
- Subject ID → Target encoding or leave-one-out encoding
- Sequence position: normalized sequence_counter (0-1 scale within sequence)

### 2.6 Sequence-Level Aggregations
**For each sequence, compute** (across all frames):
- Statistical moments: mean, std, min, max, median, IQR
- Temporal features: trend (linear regression slope), autocorrelation
- Gesture duration (number of frames)
- Activity level: % of frames with high acceleration/rotation

**Total Engineered Features Estimate**: ~200-300 features (from 341 raw)

---

## Phase 3: Preprocessing Strategy

### 3.1 Missing Value Handling
**ToF -1.0 values (59% sparsity)**:
- **Option A**: Impute with sensor-specific mean/median
- **Option B**: Forward-fill within sequence (assume object persistence)
- **Option C**: Leave as-is and use tree-based models (handle missing naturally)
- **Recommendation**: Option C for initial modeling, Option B if using deep learning

### 3.2 Normalization
**Sensor-specific normalization**:
- ToF: Already in mm, no scaling needed initially
- Accelerometer: Standardize (zero mean, unit variance) per axis
- Rotation: Quaternions already normalized, Euler angles need scaling
- Thermal: MinMax scaling to [0, 1] range

**Subject-specific normalization**:
- Z-score normalization within subject (removes individual baseline differences)
- Apply after demographic normalization

### 3.3 Sequence Padding/Truncation
**Challenge**: Variable sequence lengths (29-700 frames)

**Options**:
1. **Pad short sequences** to fixed length (e.g., 700 or median 59) with zeros
2. **Truncate long sequences** to fixed length
3. **Sliding window approach**: Fixed windows (e.g., 50 frames) with overlap
4. **Sequence aggregation**: Convert to fixed-size feature vector (Phase 2.6)

**Recommendation**: Use sliding windows for RNN/CNN, aggregation for traditional ML

### 3.4 Handedness Normalization
**Goal**: Make left and right hand gestures comparable

**Approach**:
- Flip horizontal ToF sensors for left-handed subjects (mirror tof_3 ↔ tof_5)
- Negate appropriate accelerometer/rotation axes (x or y depending on coordinate system)
- Swap thermal sensor positions accordingly

### 3.5 Data Augmentation (Optional)
**For addressing class imbalance**:
- Time warping (stretch/compress sequences slightly)
- Noise injection (Gaussian noise to sensor readings)
- Rotation augmentation (small quaternion perturbations)
- SMOTE on sequence-level features

---

## Phase 4: Model Selection Strategy

### 4.1 Baseline Models (Traditional ML)

#### Model 1: Random Forest
**Input**: Sequence-level aggregated features (~300 features)
**Advantages**:
- Handles missing values naturally (-1 in ToF)
- No need for sequence padding
- Feature importance analysis
- Fast training

**Configuration**:
- n_estimators: 100-500
- max_depth: 10-30
- class_weight: 'balanced' (handle imbalance)

#### Model 2: XGBoost
**Input**: Sequence-level aggregated features
**Advantages**:
- Superior performance to Random Forest typically
- Built-in handling of missing values
- Regularization prevents overfitting

**Configuration**:
- scale_pos_weight: for imbalanced classes
- learning_rate: 0.01-0.1
- max_depth: 5-10

#### Model 3: LightGBM
**Input**: Sequence-level aggregated features
**Advantages**:
- Faster than XGBoost
- Efficient with high-dimensional data
- Good with categorical features (encode subject, orientation)

**Expected Performance**: 70-85% accuracy (baseline)

---

### 4.2 Deep Learning Models (Sequential Data)

#### Model 4: 1D CNN
**Architecture**:
```
Input: (sequence_length, n_features_per_frame)
Conv1D(filters=64, kernel=5) → ReLU → MaxPool
Conv1D(filters=128, kernel=3) → ReLU → MaxPool
Conv1D(filters=256, kernel=3) → ReLU → GlobalMaxPool
Dense(128) → Dropout(0.5)
Dense(18, softmax)
```

**Advantages**:
- Captures local temporal patterns
- Translation invariant (gesture can occur anywhere in sequence)
- Faster training than RNN

**Preprocessing**: Pad/truncate to fixed length, impute missing ToF values

#### Model 5: LSTM/GRU
**Architecture**:
```
Input: (sequence_length, n_features_per_frame)
LSTM(128, return_sequences=True)
LSTM(64)
Dense(64) → Dropout(0.5)
Dense(18, softmax)
```

**Advantages**:
- Captures long-term dependencies
- Handles variable sequence lengths (with masking)
- Good for gesture temporal structure

**Disadvantages**:
- Slower training
- Risk of vanishing gradients

**Variant**: Bidirectional LSTM to capture future context

#### Model 6: Transformer
**Architecture**:
```
Input: (sequence_length, n_features_per_frame)
Positional Encoding
Multi-Head Attention (4-8 heads, 128-256 dims)
Feed-Forward Network
GlobalAveragePooling
Dense(18, softmax)
```

**Advantages**:
- Captures long-range dependencies better than LSTM
- Parallelizable (faster training)
- Attention weights provide interpretability

**Disadvantages**:
- Requires more data
- More hyperparameters to tune

---

### 4.3 Multi-Modal Fusion Models

#### Model 7: Late Fusion Ensemble
**Approach**:
1. Train separate models for each sensor modality:
   - ToF model (CNN on spatial frames)
   - Motion model (LSTM on acc + rotation)
   - Thermal model (simple MLP)
2. Combine predictions via weighted voting or meta-classifier

**Advantages**:
- Leverages modality-specific architectures
- Robust to sensor failures
- Interpretable (see which modality contributes most)

#### Model 8: Early Fusion CNN-LSTM Hybrid
**Architecture**:
```
ToF branch: Conv2D on 16×16 frames → Flatten
Motion branch: Dense layers on acc/rot
Thermal branch: Dense layers
→ Concatenate
→ LSTM(128) for temporal modeling
→ Dense(18, softmax)
```

**Advantages**:
- End-to-end learning
- Learns cross-modal interactions

---

### 4.4 Specialized Architectures (Research-Backed)

#### Model 9: Temporal Convolutional Network (TCN)
**Reference**: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)

**Advantages**:
- Matches or exceeds LSTM performance
- Faster training and inference
- Handles very long sequences
- Parallelizable

**Architecture**: Dilated causal convolutions with residual connections

#### Model 10: Vision Transformer for ToF Sequences
**Approach**: Treat ToF sequence as video (frames of 16×16 spatial images)

**Architecture**:
- Patch embedding (split each frame into patches)
- Spatial attention (within frame)
- Temporal attention (across frames)

**Advantages**:
- State-of-art for video classification
- Captures spatio-temporal patterns

**Disadvantages**:
- Data-hungry (may need augmentation)

---

### 4.5 Model Selection Criteria

| Model | Data Efficiency | Training Speed | Interpretability | Expected Performance |
|-------|----------------|----------------|------------------|---------------------|
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 70-80% |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 75-85% |
| 1D CNN | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 80-88% |
| LSTM | ⭐⭐ | ⭐⭐ | ⭐⭐ | 82-90% |
| Transformer | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 85-92% |
| TCN | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 83-91% |
| Multi-Modal Fusion | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 88-94% |

**Recommendation**:
1. **Start with XGBoost** (quick baseline, feature importance insights)
2. **Build 1D CNN** (strong sequential performance, fast iteration)
3. **Experiment with TCN or Transformer** (if CNN performs well, push further)
4. **Final ensemble**: Combine top 3 models for best Kaggle submission

---

## Phase 5: Evaluation Strategy

### 5.1 Validation Approach
**Challenge**: Ensure generalization across subjects and orientations

**Strategy**: Stratified Group K-Fold Cross-Validation
- **Grouping**: By subject (prevent data leakage - same subject not in train/val)
- **Stratification**: Maintain gesture class proportions
- **K**: 5 folds

### 5.2 Metrics
**Primary Metric** (Kaggle Competition):
- **Macro F1-Score** ⚠️ CRITICAL: Average F1 across all 18 gesture classes
- **Why this matters**: Class imbalance (640 vs 161 sequences) means accuracy is misleading
- **Implication**: Must ensure minority classes (gestures 13-18) perform well, not just majority

**Secondary Metrics**:
- Per-class F1-scores (identify weak gestures - especially classes 13-18!)
- Confusion matrix (understand misclassifications between similar gestures)
- Macro Precision & Macro Recall (diagnose if model is too conservative/aggressive)

### 5.3 Ablation Studies
**Test importance of**:
- Each sensor modality (ToF only vs. + motion vs. + thermal)
- Demographic normalization (with vs. without)
- Handedness normalization
- Context variables (orientation, phase)

---

## Phase 6: Implementation Roadmap

### Step 1: Exploratory Data Analysis (EDA) on Unexplored Sensors
**Files**: [Nadav_Sadna_Project.ipynb](Nadav_Sadna_Project.ipynb)
- Add cells for accelerometer analysis
- Add cells for rotation/quaternion analysis
- Add cells for thermal sensor analysis
- Visualize per-gesture patterns for each modality

### Step 2: Feature Engineering Pipeline
**Create new notebook or script**: `feature_engineering.py`
- Implement all features from Phase 2
- Create feature extraction functions (modular, reusable)
- Generate feature matrix for train/test sets
- Save engineered features to CSV or pickle

### Step 3: Preprocessing Pipeline
**Create script**: `preprocessing.py`
- Implement normalization functions
- Handedness normalization logic
- Sequence padding/truncation utilities
- Train/validation split with stratified group K-fold

### Step 4: Baseline Model Training
**Create notebook**: `baseline_models.ipynb`
- Train Random Forest, XGBoost, LightGBM
- Hyperparameter tuning (GridSearchCV or Optuna)
- Feature importance analysis
- Cross-validation results

### Step 5: Deep Learning Model Training
**Create script**: `train_deep_models.py`
- Implement CNN, LSTM, Transformer architectures
- Training loop with early stopping
- Model checkpointing
- Experiment tracking (MLflow or Weights & Biases)

### Step 6: Ensemble & Final Submission
**Create notebook**: `ensemble_predictions.ipynb`
- Load best models
- Generate predictions on test set
- Weighted ensemble or stacking
- Create Kaggle submission file

---

## Phase 7: Potential Challenges & Mitigations

### Challenge 1: Class Imbalance (640 vs. 161 sequences) ⚠️ CRITICAL FOR MACRO F1
**Why Critical**: Macro F1 equally weights all classes - poor performance on minority classes (13-18) will tank the score!

**Mitigations (PRIORITY)**:
1. **Class-balanced loss**: Use `class_weight='balanced'` or compute inverse frequency weights
2. **Oversampling minority classes**: SMOTE on sequence-level features, or duplicate minority sequences
3. **Focal loss** for deep learning: `FL(pt) = -(1-pt)^γ * log(pt)` focuses on hard examples
4. **Stratified sampling**: Ensure all classes in every train/val fold
5. **Threshold tuning**: Adjust classification thresholds per class based on validation F1
6. **Ensemble diversity**: Train some models with balanced sampling, others with original distribution

### Challenge 2: ToF Data Sparsity (59% missing)
**Mitigations**:
- Tree-based models handle missing naturally
- Forward-fill imputation for sequences
- Mask-aware deep learning layers
- Feature engineering reduces reliance on raw pixels

### Challenge 3: Subject Variability (81 subjects)
**Mitigations**:
- Subject-grouped cross-validation (prevent leakage)
- Demographic normalization
- Subject-level target encoding
- Domain adaptation techniques (if test subjects differ)

### Challenge 4: Variable Sequence Lengths (29-700 frames)
**Mitigations**:
- Sequence aggregation (for traditional ML)
- Padding with masking (for deep learning)
- Sliding window approach
- Attention mechanisms (automatically weight informative frames)

### Challenge 5: Optimizing for Fast Iterations
**Available Resources**: Local GPU + Cloud GPU (Colab/Kaggle)

**Fast Iteration Strategy**:
1. **Quick prototyping on CPU**: XGBoost with aggregated features (minutes)
2. **Local GPU for CNNs**: 1D CNN training (10-30 min per experiment)
3. **Cloud GPU for heavy models**: Transformers, large ensembles (hours)
4. **Parallel experimentation**: Train baseline on local while testing deep models on cloud

---

## Key Files to Create/Modify

1. **[Nadav_Sadna_Project.ipynb](Nadav_Sadna_Project.ipynb)** - Continue EDA with unexplored sensors
2. **feature_engineering.py** - Feature extraction pipeline (new file)
3. **preprocessing.py** - Data preprocessing utilities (new file)
4. **baseline_models.ipynb** - Traditional ML experiments (new file)
5. **train_deep_models.py** - Deep learning training script (new file)
6. **ensemble_predictions.ipynb** - Final predictions & submission (new file)
7. **config.yaml** - Configuration file for hyperparameters (new file)

---

## Recommended Next Steps (Immediate) - Optimized for Fast Iterations

### Week 1: Foundation (Fast CPU-based work)
1. ✅ **Confirmed: Macro F1-Score is evaluation metric**
2. **Complete EDA on accelerometer, rotation, thermal sensors** (2-3 hours)
   - Add cells to [Nadav_Sadna_Project.ipynb](Nadav_Sadna_Project.ipynb)
   - Visualize per-gesture patterns
3. **Implement feature engineering pipeline** (4-6 hours)
   - Create `feature_engineering.py`
   - Start with sequence-level aggregations (no deep learning needed)
4. **Baseline XGBoost with class balancing** (2 hours)
   - Use `class_weight='balanced'`
   - Target: >0.65 Macro F1
   - Analyze per-class F1 scores (identify weak classes)

### Week 2: Deep Learning Experiments (GPU-accelerated)
5. **1D CNN on local GPU** (parallel with baseline)
   - Quick architecture iterations (10-20 min per experiment)
   - Use focal loss to handle imbalance
   - Target: >0.75 Macro F1
6. **LSTM/GRU experiments on cloud GPU** (if CNN shows promise)
   - Longer training times (1-2 hours)
   - Target: >0.80 Macro F1

### Week 3: Advanced & Ensemble (Cloud GPU)
7. **Transformer or TCN architecture** (if time permits)
8. **Multi-modal fusion ensemble**
9. **Threshold tuning per class** to maximize Macro F1
10. **Final Kaggle submission**

### Parallelization Strategy:
- **Local**: Quick experiments (XGBoost, feature engineering, EDA)
- **Cloud**: Long-running deep learning training
- **Simultaneous**: Train baseline while prototyping CNN architecture

---

## Research Papers to Review (Optional Deep Dive)

**Gesture Recognition with Wearables**:
- "Deep Convolutional Neural Networks for Human Activity Recognition with Smartphone Sensors" (Ronao & Cho, 2016)
- "Ensembles of Deep LSTM Learners for Activity Recognition using Wearables" (Ordonez & Roggen, 2016)

**Time Series Classification**:
- "Time Series Classification from Scratch with Deep Neural Networks" (Wang et al., 2017)
- "InceptionTime: Finding AlexNet for Time Series Classification" (Fawaz et al., 2020)

**Multi-Modal Fusion**:
- "Multimodal Deep Learning" (Ngiam et al., 2011)
- "ActionVLAD: Learning spatio-temporal aggregation for action classification" (Girdhar et al., 2017)

**Temporal Convolutional Networks**:
- "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)

---

## Limitations & What You Can Provide

### My Capabilities ✅
- Web search for papers and documentation
- Access to up-to-date library docs (Context7 MCP)
- 142+ scientific skills for implementation
- Code reading and planning

### Current Limitations ⚠️
- Cannot directly query ArXiv/IEEE/PubMed databases
- Cannot execute code during planning phase

### How You Can Help
1. **Share Kaggle evaluation metric** (if known)
2. **Provide any baseline notebooks** from Kaggle community (if useful)
3. **Share domain knowledge** about gestures (clinical context, if any)
4. **Computational resources available** (local GPU, cloud credits)
5. **Time constraints** for project completion

---

## Summary

This plan provides a comprehensive roadmap from unexplored sensor EDA through feature engineering, preprocessing, and model selection **optimized for Macro F1-Score and fast iteration**.

### Key Success Factors:
1. ⚠️ **Class imbalance handling is CRITICAL** - Macro F1 punishes poor minority class performance
2. 🚀 **Fast iteration strategy** - CPU baselines → Local GPU CNNs → Cloud GPU Transformers
3. 🔬 **Multi-modal fusion** - Leverage ToF + motion + thermal sensors
4. 📊 **Per-class analysis** - Monitor F1 for gestures 13-18 (minority classes)
5. 🎯 **Threshold tuning** - Optimize decision boundaries per class

### Expected Performance Trajectory:
- **Baseline XGBoost**: 0.60-0.70 Macro F1 (Week 1)
- **1D CNN with focal loss**: 0.70-0.80 Macro F1 (Week 2)
- **Ensemble + tuning**: 0.80-0.88 Macro F1 (Week 3)

### Resource Allocation:
- **Local CPU**: EDA, feature engineering, XGBoost (fast prototyping)
- **Local GPU**: 1D CNN experiments (10-30 min iterations)
- **Cloud GPU**: Transformers, LSTM, heavy ensembles (1-3 hour jobs)

**Confidence Level**: High - This plan is grounded in established practices for multimodal time series classification, addresses dataset-specific challenges (sparsity, imbalance, variable lengths), and is optimized for your computational resources and Macro F1 metric.

**Ready to proceed with implementation!** The plan prioritizes quick wins while building toward state-of-the-art performance.
