# Competition Summary: CMI Detect Behavior with Sensor Data

## Objective

Develop a model to:
1. **Distinguish** BFRB-like gestures from non-BFRB-like gestures
2. **Identify** the specific type of BFRB-like gesture

**Real-World Impact**: Determine if thermopile + ToF sensors add value beyond IMU for BFRB detection.

---

## Gestures

### BFRB-Like Gestures (Target = 8 classes)
1. Above ear - Pull hair
2. Forehead - Pull hairline
3. Forehead - Scratch
4. Eyebrow - Pull hair
5. Eyelash - Pull hair
6. Neck - Pinch skin
7. Neck - Scratch
8. Cheek - Pinch skin

### Non-BFRB-Like Gestures (Non-Target = 10 classes)
9. Drink from bottle/cup
10. Glasses on/off
11. Pull air toward your face
12. Pinch knee/leg skin
13. Scratch knee/leg skin
14. Write name on leg
15. Text on phone
16. Feel around in tray and pull out an object
17. Write name in air
18. Wave hello

---

## Evaluation Metric

**Formula**:
```python
Score = (Binary_F1 + Macro_F1) / 2
```

### Component 1: Binary F1
- **Task**: Classify as BFRB (target) vs. non-BFRB (non-target)
- **Classes**: 2 (binary)
- **Metric**: F1-score

### Component 2: Macro F1
- **Task**: Classify specific gesture type
- **Classes**: 9 (8 BFRB gestures + 1 "non_target" class for all non-BFRB)
- **Metric**: Macro F1 (average F1 across all 9 classes)

### Example Calculation:
```
Prediction: "Forehead - Scratch"
Ground Truth: "Forehead - Scratch"

Binary Classification:
  Predicted: BFRB (target)
  Actual: BFRB (target)
  → Contributes to Binary F1

Multiclass Classification:
  Predicted: "Forehead - Scratch"
  Actual: "Forehead - Scratch"
  → Contributes to Macro F1 for this class

Final Score = (Binary_F1 + Macro_F1) / 2
```

---

## Sequence Structure

Each sequence contains 3 phases:
1. **Transition**: Moving hand from rest to target location
2. **Pause**: Brief pause (~1-2 seconds)
3. **Gesture**: Performing the actual gesture

**Important**: We classify the **Gesture** phase, but data includes all 3 phases.

---

## Test Set Constraints

### 50% IMU-Only Sequences
- **Available**: acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z
- **Missing**: All thm_* and tof_* columns (NULL)

### 50% Full Sensor Sequences
- **Available**: All sensors (IMU + thermal + ToF)

**Critical**: Model must handle both scenarios gracefully.

---

## Submission Format

- Use Kaggle evaluation API (sequence-by-sequence inference)
- Predict `gesture` for each `sequence_id`
- Must predict one of the 18 training gestures (no new labels allowed)

---

## Modeling Strategy

### Two-Model Approach (Recommended)

**Model A: IMU-Only**
- Train on: acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z features
- Use for: Sequences with NULL thm/tof columns
- Target: Binary F1 > 0.70, Macro F1 > 0.55

**Model B: Full Sensors**
- Train on: All features (IMU + thermal + ToF)
- Use for: Sequences with complete sensor data
- Target: Binary F1 > 0.85, Macro F1 > 0.75

**Inference Logic**:
```python
if all_thermal_and_tof_are_null:
    prediction = model_A.predict(imu_features)
else:
    prediction = model_B.predict(all_features)
```

### Alternative: Hierarchical Approach

**Step 1**: Binary classification (BFRB vs. non-BFRB)
- Train binary classifier
- High accuracy expected (distinct motion patterns)

**Step 2**: Multi-class refinement
- If BFRB: Classify among 8 BFRB gestures
- If non-BFRB: Assign "non_target" label

**Advantage**: Specialized models for each task
**Risk**: Error propagation from Step 1 to Step 2

---

## Success Criteria

| Metric | IMU-Only Target | Full Sensors Target | Overall Target |
|--------|-----------------|---------------------|----------------|
| **Binary F1** | > 0.70 | > 0.85 | > 0.77 |
| **Macro F1** | > 0.55 | > 0.75 | > 0.65 |
| **Final Score** | > 0.62 | > 0.80 | > 0.71 |

**Leaderboard Goal**: Top 25% (estimated score: 0.75+)

---

## Key Insights from EDA

### BFRB vs. Non-BFRB Patterns

**BFRB Gestures (Contact-based, upper body)**:
- Lower acceleration magnitude (~2.9-3.2)
- Stable orientation (low rotation range)
- Higher temperature (~31.2°C) due to skin contact
- Closer ToF readings (hand near face/neck)

**Non-BFRB Gestures (Mixed)**:
- Higher variability in patterns
- Air gestures: High acceleration, high rotation, lower temperature
- Manipulation gestures: Low acceleration, stable temperature

### Discriminative Features (Priority Order)

**For Binary Classification (BFRB vs. non-BFRB)**:
1. Accelerometer magnitude (air vs. contact gestures)
2. Thermal mean (contact detection)
3. Rotation range (dynamic vs. static)
4. ToF proximity patterns

**For Multi-Class Classification (Specific gesture)**:
1. Temporal IMU patterns (gesture signature)
2. ToF spatial configuration (hand position)
3. Body position context (orientation column)
4. Sequence duration and intensity

---

## Timeline

- **Week 1**: Feature engineering + IMU-only baseline (Target: 0.62)
- **Week 2**: Full sensor model + ensemble (Target: 0.71)
- **Week 3**: Optimization + final submission (Target: 0.75+)

---

**Competition Dates**:
- Start: May 30, 2025
- Close: Sep 3, 2025
- Merger & Entry: TBD

**Status**: Planning Complete ✅ | Feature Engineering In Progress 🔄
