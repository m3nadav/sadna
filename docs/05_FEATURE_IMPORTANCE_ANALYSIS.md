# Feature Importance Analysis
## XGBoost Baseline Models

*Generated from trained XGBoost models*

---

## IMU-Only Model: Top 30 Features

**Model Score**: 0.6864 (Competition Score)

| Rank | Feature | Sensor Type | Importance |
|------|---------|-------------|------------|
| 1 | angular_velocity_median | Rotation | 2146.0 |
| 2 | acc_autocorrelation_lag1 | Accelerometer | 2032.0 |
| 3 | sequence_length | Other | 1837.0 |
| 4 | yaw_median | Rotation | 1747.0 |
| 5 | acc_z_min | Accelerometer | 1654.0 |
| 6 | acc_high_activity_ratio | Accelerometer | 1641.0 |
| 7 | acc_y_max | Accelerometer | 1601.0 |
| 8 | acc_peak_position | Accelerometer | 1590.0 |
| 9 | angular_acceleration_mean | Rotation | 1561.0 |
| 10 | acc_axis_variance_ratio | Accelerometer | 1544.0 |
| 11 | quat_magnitude_std | Other | 1490.0 |
| 12 | acc_y_iqr | Accelerometer | 1460.0 |
| 13 | acc_magnitude_min | Accelerometer | 1392.0 |
| 14 | yaw_mean | Rotation | 1390.0 |
| 15 | quat_magnitude_mean | Other | 1352.0 |
| 16 | acc_y_min | Accelerometer | 1349.0 |
| 17 | acc_x_iqr | Accelerometer | 1318.0 |
| 18 | acc_magnitude_q25 | Accelerometer | 1298.0 |
| 19 | acc_y_range | Accelerometer | 1274.0 |
| 20 | jerk_mean | Accelerometer | 1271.0 |
| 21 | rotation_axis_dominance_ratio | Other | 1236.0 |
| 22 | pitch_min | Rotation | 1234.0 |
| 23 | acc_start_to_middle_ratio | Accelerometer | 1219.0 |
| 24 | acc_middle_to_end_ratio | Accelerometer | 1194.0 |
| 25 | jerk_max | Accelerometer | 1190.0 |
| 26 | acc_x_min | Accelerometer | 1188.0 |
| 27 | acc_z_q25 | Accelerometer | 1166.0 |
| 28 | acc_y_q75 | Accelerometer | 1163.0 |
| 29 | yaw_max | Rotation | 1159.0 |
| 30 | roll_median | Rotation | 1158.0 |

### IMU Feature Importance by Sensor Type

| Sensor Type | Total Score | Count | Avg Score |
|-------------|-------------|-------|----------|
| Accelerometer | 26544.0 | 19 | 1397.1 |
| Rotation | 10395.0 | 7 | 1485.0 |
| Other | 5915.0 | 4 | 1478.8 |

---

## Full Sensor Model: Top 30 Features

**Model Score**: 0.7838 (Competition Score)

| Rank | Feature | Sensor Type | Importance |
|------|---------|-------------|------------|
| 1 | thm_2_mean | Thermal | 1104.0 |
| 2 | tof_2_depth_mean | ToF | 1094.0 |
| 3 | angular_velocity_median | Rotation | 1075.0 |
| 4 | acc_autocorrelation_lag1 | Accelerometer | 1070.0 |
| 5 | tof_3_depth_mean | ToF | 909.0 |
| 6 | acc_z_min | Accelerometer | 897.0 |
| 7 | thm_2_std | Thermal | 866.0 |
| 8 | sequence_length | Other | 848.0 |
| 9 | acc_y_max | Accelerometer | 847.0 |
| 10 | acc_high_activity_ratio | Accelerometer | 842.0 |
| 11 | acc_y_iqr | Accelerometer | 800.0 |
| 12 | acc_peak_position | Accelerometer | 788.0 |
| 13 | angular_acceleration_mean | Rotation | 788.0 |
| 14 | yaw_median | Rotation | 745.0 |
| 15 | acc_axis_variance_ratio | Accelerometer | 723.0 |
| 16 | tof_4_depth_velocity_mean | ToF | 718.0 |
| 17 | tof_global_depth_std | ToF | 703.0 |
| 18 | tof_2_depth_velocity_mean | ToF | 699.0 |
| 19 | acc_y_range | Accelerometer | 696.0 |
| 20 | tof_2_depth_min | ToF | 694.0 |
| 21 | tof_3_depth_velocity_mean | ToF | 685.0 |
| 22 | thm_5_mean | Thermal | 679.0 |
| 23 | acc_magnitude_q25 | Accelerometer | 678.0 |
| 24 | thm_5_std | Thermal | 677.0 |
| 25 | rotation_axis_dominance_ratio | Other | 666.0 |
| 26 | acc_x_iqr | Accelerometer | 645.0 |
| 27 | acc_magnitude_min | Accelerometer | 642.0 |
| 28 | tof_5_depth_mean | ToF | 641.0 |
| 29 | acc_z_q25 | Accelerometer | 622.0 |
| 30 | tof_global_depth_mean | ToF | 620.0 |

### Full Sensor Feature Importance by Sensor Type

| Sensor Type | Total Score | Count | Avg Score |
|-------------|-------------|-------|----------|
| Accelerometer | 9250.0 | 12 | 770.8 |
| ToF | 6763.0 | 9 | 751.4 |
| Thermal | 3326.0 | 4 | 831.5 |
| Rotation | 2608.0 | 3 | 869.3 |
| Other | 1514.0 | 2 | 757.0 |

---

## Value Added by ToF & Thermal Sensors

**ToF & Thermal features in Top 30**: 13 features

**Combined importance score**: 10089.0

**Percentage of total importance**: 43.0%

### Top ToF/Thermal Features

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | thm_2_mean | Thermal | 1104.0 |
| 2 | tof_2_depth_mean | ToF | 1094.0 |
| 5 | tof_3_depth_mean | ToF | 909.0 |
| 7 | thm_2_std | Thermal | 866.0 |
| 16 | tof_4_depth_velocity_mean | ToF | 718.0 |
| 17 | tof_global_depth_std | ToF | 703.0 |
| 18 | tof_2_depth_velocity_mean | ToF | 699.0 |
| 20 | tof_2_depth_min | ToF | 694.0 |
| 21 | tof_3_depth_velocity_mean | ToF | 685.0 |
| 22 | thm_5_mean | Thermal | 679.0 |
| 24 | thm_5_std | Thermal | 677.0 |
| 28 | tof_5_depth_mean | ToF | 641.0 |
| 30 | tof_global_depth_mean | ToF | 620.0 |

---

## Key Insights

### 1. IMU-Only Model
- Most important sensor type: Accelerometer
- Top feature: angular_velocity_median
- Competition score: 0.6864

### 2. Full Sensor Model
- Most important sensor type: Accelerometer
- Top feature: thm_2_mean
- Competition score: 0.7838 (+14.2% improvement)

### 3. Sensor Value Added
- ToF/Thermal contribute 43.0% of top 30 feature importance
- This justifies the +9.7 point score improvement (0.6864 → 0.7838)

### 4. Recommendations for Deep Learning
Based on feature importance analysis:

- **Focus on Accelerometer and Rotation features** in CNN/LSTM models
- **Use attention mechanisms** to learn which features matter per gesture
- **Multi-modal architecture**: Separate branches for IMU vs ToF/Thermal
- **Temporal modeling**: Many important features are temporal (trends, autocorrelation)

---

*Analysis complete. Proceed to deep learning model development.*
