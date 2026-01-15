# Train/Validation Split Strategy

Detailed explanation of how the training data was split for model evaluation.

---

## Overview

**Total Training Data**: 574,945 frames across 8,151 sequences (81 subjects)

**Split Approach**: Sequence-level stratified split
- **Training Set**: 6,520 sequences (80%) = 457,938 frames
- **Validation Set**: 1,631 sequences (20%) = 117,007 frames

---

## Visual Representation

```
Original Dataset (train.csv)
┌─────────────────────────────────────────────────────────────┐
│  574,945 frames                                             │
│  8,151 sequences                                            │
│  81 subjects                                                │
│  18 gesture classes                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─── Unique sequences extracted
                            │    (one row per sequence)
                            │
                            ▼
            ┌───────────────────────────────┐
            │   8,151 Unique Sequences      │
            │   with metadata:              │
            │   - sequence_id               │
            │   - gesture (18 classes)      │
            │   - subject (81 subjects)     │
            └───────────────────────────────┘
                            │
                            ├─── train_test_split()
                            │    - test_size = 0.2
                            │    - stratify = gesture
                            │    - random_state = 42
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │   Training Set       │  │   Validation Set     │
    │   6,520 sequences    │  │   1,631 sequences    │
    │   (80%)              │  │   (20%)              │
    │                      │  │                      │
    │   457,938 frames     │  │   117,007 frames     │
    └──────────────────────┘  └──────────────────────┘
                │                       │
                │                       ├─── Simulate test conditions
                │                       │    (50% IMU-only, 50% full)
                │                       │
                ▼                       ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Feature Extraction  │  │   815 IMU-only       │
    │  - IMU: 83 features  │  │   816 Full sensor    │
    │  - Full: 158 features│  └──────────────────────┘
    └──────────────────────┘
                │
                ▼
    ┌──────────────────────────────────────┐
    │  Train Models                        │
    │  - XGBoost (IMU-only + Full)         │
    │  - 1D CNN (IMU-only + Full)          │
    │  - Class-balanced weights            │
    └──────────────────────────────────────┘
```

---

## Implementation Details

### Step 1: Extract Unique Sequences

```python
sequences = train_df[['sequence_id', 'gesture', 'subject']].drop_duplicates()
# Result: 8,151 rows (one per sequence)
```

**Why sequence-level?**
- Prevents data leakage (same sequence's frames not in both sets)
- Mimics real-world scenario (entire gesture performed at once)
- More realistic performance estimate

### Step 2: Stratified Split

```python
from sklearn.model_selection import train_test_split

train_sequences, val_sequences = train_test_split(
    sequences,
    test_size=0.2,                 # 20% validation
    stratify=sequences['gesture'], # Maintain class balance
    random_state=42                # Reproducible
)
```

**Stratification by gesture ensures**:
- Each of 18 gestures has ~80% in train, ~20% in val
- Class imbalance preserved (640 vs 161 sequences)
- Representative validation set

### Step 3: Filter Frames by Sequence IDs

```python
train_data = train_df[train_df['sequence_id'].isin(train_sequences['sequence_id'])]
val_data = train_df[train_df['sequence_id'].isin(val_sequences['sequence_id'])]
```

**Result**:
- Training: 457,938 frames
- Validation: 117,007 frames
- Total: 574,945 frames (all accounted for)

---

## Gesture Distribution

Stratification ensures proportional representation:

| Gesture | Total Seqs | Train | Val | Train % | Val % |
|---------|-----------|-------|-----|---------|-------|
| Above ear - pull hair | 638 | 510 | 128 | 79.9% | 20.1% |
| Cheek - pinch skin | 637 | 509 | 128 | 79.9% | 20.1% |
| Drink from bottle/cup | 161 | 129 | 32 | 80.1% | 19.9% |
| Eyebrow - pull hair | 638 | 510 | 128 | 79.9% | 20.1% |
| Eyelash - pull hair | 640 | 512 | 128 | 80.0% | 20.0% |
| Feel around in tray | 161 | 129 | 32 | 80.1% | 19.9% |
| Forehead - pull hairline | 640 | 512 | 128 | 80.0% | 20.0% |
| Forehead - scratch | 640 | 512 | 128 | 80.0% | 20.0% |
| Glasses on/off | 161 | 129 | 32 | 80.1% | 19.9% |
| Neck - pinch skin | 640 | 512 | 128 | 80.0% | 20.0% |
| Neck - scratch | 640 | 512 | 128 | 80.0% | 20.0% |
| Pinch knee/leg skin | 161 | 129 | 32 | 80.1% | 19.9% |
| Pull air toward face | 477 | 381 | 96 | 79.9% | 20.1% |
| Scratch knee/leg skin | 161 | 129 | 32 | 80.1% | 19.9% |
| Text on phone | 640 | 512 | 128 | 80.0% | 20.0% |
| Wave hello | 478 | 382 | 96 | 79.9% | 20.1% |
| Write name in air | 477 | 382 | 95 | 80.1% | 19.9% |
| Write name on leg | 161 | 129 | 32 | 80.1% | 19.9% |

**Key Observations**:
- All gestures have ~80/20 split (within 0.2%)
- Class imbalance maintained: 640 vs 161 sequences
- No gesture missing from validation set

---

## Validation Set Simulation

To match test set conditions (50% IMU-only), validation set is randomly split:

```python
np.random.seed(42)
val_size = len(val_sequences)
imu_only_mask = np.random.rand(val_size) < 0.5

imu_only_sequences = val_sequences[imu_only_mask]     # 815 sequences
full_sensor_sequences = val_sequences[~imu_only_mask] # 816 sequences
```

**Two Evaluation Scenarios**:

1. **IMU-only evaluation** (815 sequences):
   - Uses only IMU features (83 features)
   - Tests model with acc + rotation data only
   - Mimics 50% of test set

2. **Full sensor evaluation** (816 sequences):
   - Uses all features (158 features)
   - Tests model with IMU + ToF + thermal
   - Mimics other 50% of test set

3. **Overall score**: Average of both scenarios
   - `(IMU_score + Full_score) / 2`
   - Directly estimates test set performance

---

## Advantages of This Approach

### ✅ Pros:

1. **No data leakage**: Sequences are atomic units
2. **Class balance**: Stratification maintains gesture proportions
3. **Realistic**: Validation mimics test conditions (50% IMU-only)
4. **Reproducible**: Random seed ensures same split every run
5. **Efficient**: Single 80/20 split (not computationally expensive)

### ⚠️ Limitations:

1. **Subject overlap**: Same subject can appear in train and val
   - **Impact**: May slightly overestimate generalization
   - **Why**: Subject-specific patterns could "leak"
   - **Example**: Subject 42 has 10 sequences → 8 in train, 2 in val

2. **Not cross-validation**: Single split may not represent all data
   - **Alternative**: 5-fold StratifiedGroupKFold (more robust but slower)
   - **Justification**: Single split sufficient for baseline + prototyping

3. **Sequence length variation**: Some sequences much longer than others
   - **29-700 frames per sequence** (median: 59)
   - **Impact**: Short sequences may be underrepresented in training
   - **Mitigation**: CNN uses masking to handle variable lengths

---

## Alternative Approaches Considered

### Option 1: Subject-Grouped Split (More Rigorous)

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in sgkf.split(sequences, y=sequences['gesture'], groups=sequences['subject']):
    train_sequences = sequences.iloc[train_idx]
    val_sequences = sequences.iloc[val_idx]
    # Use first fold as train/val split
    break
```

**Advantages**:
- No subject overlap between train/val
- Better estimates generalization to new subjects

**Disadvantages**:
- Harder to maintain exact 80/20 split
- May have uneven gesture distribution
- More complex implementation

**Decision**: Deferred to future work (current approach sufficient for baseline)

### Option 2: Time-Based Split

Split by sequence timestamp (if available):
- Train: Earlier sequences
- Val: Later sequences

**Advantages**:
- Tests temporal generalization
- Realistic for deployment (future data)

**Disadvantages**:
- Requires timestamp metadata (not available)
- May have distribution shift (subjects change over time)

---

## Validation Results

### XGBoost Baseline (on 1,631 validation sequences)

| Evaluation Set | Sequences | Score | Binary F1 | Macro F1 |
|---------------|-----------|-------|-----------|----------|
| **All validation** | 1,631 | 0.7351 | 0.9622 | 0.5080 |
| **IMU-only subset** | 815 | 0.6864 | 0.9481 | 0.4247 |
| **Full sensor subset** | 816 | 0.7838 | 0.9762 | 0.5913 |

**Key Insight**: Full sensor model improves score by +0.0973 (+14.2%) compared to IMU-only, validating the value of ToF/thermal sensors.

---

## Code Reference

Full implementation in [train_baseline_models.py](../train_baseline_models.py):
- Lines 52-93: `create_train_val_split()` function
- Lines 96-130: `extract_features()` with sequence filtering
- Lines 195-218: Validation set simulation (50% IMU-only)

---

## Conclusion

The train/validation split strategy successfully:
1. ✅ Prevents data leakage at sequence level
2. ✅ Maintains class balance through stratification
3. ✅ Simulates test conditions (50% IMU-only)
4. ✅ Provides reproducible results (random seed)

**Validation score of 0.7351** provides a reliable estimate of expected test set performance, accounting for the dual-modality challenge.

---

*Created: 2026-01-14*
