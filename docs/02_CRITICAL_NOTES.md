# CRITICAL PROJECT NOTES

## ⚠️ Test Set Constraints (MUST READ!)

### The Challenge
The hidden test set (~3,500 sequences) is split:
- **50% IMU-only sequences** (acc + rot data, thm & tof are NULL)
- **50% Full sensor sequences** (all sensors present)

###  Why This Matters
**Competition Goal**: Determine if ToF + thermal sensors improve BFRB detection beyond IMU alone.

**Implication**: We cannot build a model that relies heavily on ToF/thermal, as it will fail on 50% of test cases!

### Required Strategy

1. **Build TWO models**:
   - **Model A (IMU-only)**: For sequences with missing ToF/thermal
   - **Model B (Full)**: For sequences with all sensors

2. **Inference Logic**:
   ```python
   if (thm_cols.isnull().all() and tof_cols.isnull().all()):
       prediction = model_A.predict(imu_features)
   else:
       prediction = model_B.predict(all_features)
   ```

3. **Evaluation**: Simulate test conditions
   - Randomly mask ToF/thermal for 50% of validation sequences
   - Measure Macro F1 separately for IMU-only vs. Full
   - Target: IMU-only F1 > 0.60, Full F1 > 0.75, Overall > 0.68

### Feature Engineering Priority

**TIER 0 (CRITICAL - IMU features)**:
- Must be highly discriminative since they're all we have for 50% of test
1. Accelerometer magnitude + jerk
2. Euler angles + angular velocity
3. Temporal IMU statistics
4. Movement phases (start, peak, end detection)

**TIER 1 (Bonus - ToF/Thermal)**:
- Only used when available
5. Thermal mean + range
6. ToF aggregations
7. Multi-modal fusion features

### Corrected Understanding

| Item | Previous (WRONG) | Corrected (RIGHT) |
|------|------------------|-------------------|
| ToF units | "millimeters (mm)" | "Uncalibrated sensor values 0-254" |
| IMU type | "Accelerometer only" | "9-axis IMU (acc + gyro + mag) with sensor fusion" |
| Thermal type | "Contact thermometer" | "Non-contact infrared (thermopile)" |
| Test set | "All sensors" | "50% IMU-only, 50% full sensors" |

### Action Items (UPDATED)

1. ✅ EDA complete - all sensors analyzed
2. 🔄 Feature engineering - **Focus on IMU features first!**
3. ⏳ Baseline XGBoost:
   - Train Model A on IMU features only
   - Train Model B on all features
   - Implement fallback logic
4. ⏳ Validation strategy:
   - Create IMU-only validation split (50% masked)
   - Measure performance on both splits separately

**Remember**: The goal is to show ToF/thermal ADD VALUE, not that they're essential!
