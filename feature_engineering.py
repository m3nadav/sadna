"""
Feature Engineering Pipeline for CMI Gesture Recognition
=========================================================

This module implements feature extraction from multimodal sensor data:
- TIER 0 (CRITICAL): IMU-based features (accelerometer + rotation)
- TIER 1: ToF and Thermal features (when available)

Strategy: 50% of test set has IMU-only, so IMU features must be highly discriminative.

Author: Claude + Nadav
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IMUFeatureExtractor:
    """
    Extract features from IMU sensors (accelerometer + rotation/quaternion).

    These features are CRITICAL as they're the only data available for 50% of test sequences.
    """

    def __init__(self):
        self.acc_cols = ['acc_x', 'acc_y', 'acc_z']
        self.rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']

    def quaternion_to_euler(self, w: np.ndarray, x: np.ndarray,
                           y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert quaternions to Euler angles (roll, pitch, yaw) in radians.

        Args:
            w, x, y, z: Quaternion components (numpy arrays)

        Returns:
            roll, pitch, yaw: Euler angles in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1,
                        np.sign(sinp) * np.pi / 2,
                        np.arcsin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def extract_accelerometer_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract accelerometer-based features from a sequence.

        Features:
        - 3D magnitude (overall movement intensity)
        - Jerk (rate of change of acceleration)
        - Statistical moments (mean, std, min, max, range)
        - Percentiles (25th, 50th, 75th)
        - Peak values and timing

        Args:
            df: DataFrame with acc_x, acc_y, acc_z columns (single sequence)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # 3D Acceleration Magnitude
        acc_magnitude = np.sqrt(
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        )

        # Per-axis statistics
        for axis in self.acc_cols:
            features[f'{axis}_mean'] = df[axis].mean()
            features[f'{axis}_std'] = df[axis].std()
            features[f'{axis}_min'] = df[axis].min()
            features[f'{axis}_max'] = df[axis].max()
            features[f'{axis}_range'] = df[axis].max() - df[axis].min()
            features[f'{axis}_median'] = df[axis].median()
            features[f'{axis}_q25'] = df[axis].quantile(0.25)
            features[f'{axis}_q75'] = df[axis].quantile(0.75)
            features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']

        # Magnitude statistics
        features['acc_magnitude_mean'] = acc_magnitude.mean()
        features['acc_magnitude_std'] = acc_magnitude.std()
        features['acc_magnitude_min'] = acc_magnitude.min()
        features['acc_magnitude_max'] = acc_magnitude.max()
        features['acc_magnitude_range'] = acc_magnitude.max() - acc_magnitude.min()
        features['acc_magnitude_median'] = acc_magnitude.median()
        features['acc_magnitude_q25'] = acc_magnitude.quantile(0.25)
        features['acc_magnitude_q75'] = acc_magnitude.quantile(0.75)

        # Jerk (rate of change of acceleration) - critical for gesture dynamics
        jerk_x = np.diff(df['acc_x'].values, prepend=df['acc_x'].values[0])
        jerk_y = np.diff(df['acc_y'].values, prepend=df['acc_y'].values[0])
        jerk_z = np.diff(df['acc_z'].values, prepend=df['acc_z'].values[0])
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

        features['jerk_mean'] = jerk_magnitude.mean()
        features['jerk_std'] = jerk_magnitude.std()
        features['jerk_max'] = jerk_magnitude.max()

        # Peak detection
        features['acc_peak_value'] = acc_magnitude.max()
        features['acc_peak_position'] = acc_magnitude.argmax() / len(acc_magnitude)  # Normalized position

        # Activity level (% of frames above threshold)
        threshold = acc_magnitude.mean() + acc_magnitude.std()
        features['acc_high_activity_ratio'] = (acc_magnitude > threshold).sum() / len(acc_magnitude)

        # Dominant axis (which axis has highest variance)
        variances = [df[axis].var() for axis in self.acc_cols]
        features['acc_dominant_axis'] = np.argmax(variances)  # 0=x, 1=y, 2=z
        features['acc_axis_variance_ratio'] = max(variances) / sum(variances) if sum(variances) > 0 else 0

        return features

    def extract_rotation_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract rotation/quaternion-based features from a sequence.

        Features:
        - Euler angles (roll, pitch, yaw) statistics
        - Angular velocity (rate of rotation change)
        - Angular acceleration
        - Rotation magnitude and stability

        Args:
            df: DataFrame with rot_w, rot_x, rot_y, rot_z columns (single sequence)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Convert quaternions to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            df['rot_w'].values,
            df['rot_x'].values,
            df['rot_y'].values,
            df['rot_z'].values
        )

        # Convert to degrees for interpretability
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        # Euler angle statistics
        for angle_name, angle_values in [('roll', roll_deg), ('pitch', pitch_deg), ('yaw', yaw_deg)]:
            features[f'{angle_name}_mean'] = angle_values.mean()
            features[f'{angle_name}_std'] = angle_values.std()
            features[f'{angle_name}_min'] = angle_values.min()
            features[f'{angle_name}_max'] = angle_values.max()
            features[f'{angle_name}_range'] = angle_values.max() - angle_values.min()
            features[f'{angle_name}_median'] = np.median(angle_values)

        # Angular velocity (rate of rotation change)
        roll_velocity = np.diff(roll_deg, prepend=roll_deg[0])
        pitch_velocity = np.diff(pitch_deg, prepend=pitch_deg[0])
        yaw_velocity = np.diff(yaw_deg, prepend=yaw_deg[0])

        angular_velocity_magnitude = np.sqrt(
            roll_velocity**2 + pitch_velocity**2 + yaw_velocity**2
        )

        features['angular_velocity_mean'] = angular_velocity_magnitude.mean()
        features['angular_velocity_std'] = angular_velocity_magnitude.std()
        features['angular_velocity_max'] = angular_velocity_magnitude.max()
        features['angular_velocity_median'] = np.median(angular_velocity_magnitude)

        # Angular acceleration (rate of angular velocity change)
        angular_acceleration = np.diff(angular_velocity_magnitude, prepend=angular_velocity_magnitude[0])
        features['angular_acceleration_mean'] = angular_acceleration.mean()
        features['angular_acceleration_std'] = angular_acceleration.std()
        features['angular_acceleration_max'] = np.abs(angular_acceleration).max()

        # Rotation stability (low std = stable orientation)
        features['rotation_stability'] = 1 / (1 + angular_velocity_magnitude.std())

        # Dominant rotation axis
        axis_ranges = [features['roll_range'], features['pitch_range'], features['yaw_range']]
        features['dominant_rotation_axis'] = np.argmax(axis_ranges)  # 0=roll, 1=pitch, 2=yaw
        features['rotation_axis_dominance_ratio'] = max(axis_ranges) / sum(axis_ranges) if sum(axis_ranges) > 0 else 0

        # Quaternion magnitude check (should be ~1)
        quat_magnitude = np.sqrt(
            df['rot_w']**2 + df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2
        )
        features['quat_magnitude_mean'] = quat_magnitude.mean()
        features['quat_magnitude_std'] = quat_magnitude.std()

        return features

    def extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract temporal features (moving averages, trends, phases).

        Args:
            df: DataFrame with IMU columns (single sequence)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Sequence length
        features['sequence_length'] = len(df)

        # Acceleration magnitude for temporal analysis
        acc_magnitude = np.sqrt(
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        )

        # Moving averages (window=5 frames)
        window = min(5, len(df))
        if window > 1:
            acc_ma = pd.Series(acc_magnitude).rolling(window=window, min_periods=1).mean()
            features['acc_ma_mean'] = acc_ma.mean()
            features['acc_ma_std'] = acc_ma.std()
        else:
            features['acc_ma_mean'] = acc_magnitude.mean()
            features['acc_ma_std'] = 0.0

        # Temporal phases (divide sequence into start, middle, end)
        n = len(df)
        if n >= 3:
            start_idx = n // 3
            end_idx = 2 * n // 3

            acc_start = acc_magnitude[:start_idx].mean()
            acc_middle = acc_magnitude[start_idx:end_idx].mean()
            acc_end = acc_magnitude[end_idx:].mean()

            features['acc_start_mean'] = acc_start
            features['acc_middle_mean'] = acc_middle
            features['acc_end_mean'] = acc_end
            features['acc_start_to_middle_ratio'] = acc_start / acc_middle if acc_middle > 0 else 1.0
            features['acc_middle_to_end_ratio'] = acc_middle / acc_end if acc_end > 0 else 1.0
        else:
            features['acc_start_mean'] = acc_magnitude.mean()
            features['acc_middle_mean'] = acc_magnitude.mean()
            features['acc_end_mean'] = acc_magnitude.mean()
            features['acc_start_to_middle_ratio'] = 1.0
            features['acc_middle_to_end_ratio'] = 1.0

        # Trend (linear regression slope)
        if n > 1:
            x = np.arange(n)
            coeffs = np.polyfit(x, acc_magnitude, 1)
            features['acc_trend_slope'] = coeffs[0]
        else:
            features['acc_trend_slope'] = 0.0

        # Autocorrelation (at lag=1) - measures periodicity
        if n > 1:
            acc_normalized = (acc_magnitude - acc_magnitude.mean()) / (acc_magnitude.std() + 1e-8)
            autocorr = np.correlate(acc_normalized[:-1], acc_normalized[1:], mode='valid')[0] / (n - 1)
            features['acc_autocorrelation_lag1'] = autocorr
        else:
            features['acc_autocorrelation_lag1'] = 0.0

        return features

    def extract_imu_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all IMU features from a sequence.

        Args:
            df: DataFrame with acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z columns

        Returns:
            Dictionary of all IMU features
        """
        features = {}

        # Accelerometer features
        acc_features = self.extract_accelerometer_features(df)
        features.update(acc_features)

        # Rotation features
        rot_features = self.extract_rotation_features(df)
        features.update(rot_features)

        # Temporal features
        temporal_features = self.extract_temporal_features(df)
        features.update(temporal_features)

        return features


class ToFFeatureExtractor:
    """
    Extract features from Time-of-Flight sensors.

    These are TIER 1 features - only available for 50% of test sequences.
    """

    def __init__(self):
        self.tof_sensors = ['tof_1', 'tof_2', 'tof_3', 'tof_4', 'tof_5']

    def get_tof_columns(self, sensor_id: int) -> List[str]:
        """Get column names for a specific ToF sensor."""
        return [f'tof_{sensor_id}_v{i}' for i in range(64)]

    def extract_tof_sensor_features(self, df: pd.DataFrame, sensor_id: int) -> Dict[str, float]:
        """
        Extract features from a single ToF sensor.

        Args:
            df: DataFrame with ToF columns for one sensor
            sensor_id: Sensor number (1-5)

        Returns:
            Dictionary of features for this sensor
        """
        features = {}
        sensor_cols = self.get_tof_columns(sensor_id)

        # Check if sensor data exists
        if not all(col in df.columns for col in sensor_cols):
            return {}

        sensor_data = df[sensor_cols].values  # Shape: (n_frames, 64 pixels)

        # Replace -1 (invalid) with NaN
        sensor_data_clean = np.where(sensor_data == -1, np.nan, sensor_data)

        # Valid pixel count (sparsity inverse)
        valid_count = np.sum(~np.isnan(sensor_data_clean), axis=1)
        features[f'tof_{sensor_id}_valid_count_mean'] = valid_count.mean()
        features[f'tof_{sensor_id}_valid_count_std'] = valid_count.std()
        features[f'tof_{sensor_id}_sparsity'] = np.sum(np.isnan(sensor_data_clean)) / sensor_data_clean.size

        # Mean depth (ignoring invalid pixels)
        mean_depth_per_frame = np.nanmean(sensor_data_clean, axis=1)
        features[f'tof_{sensor_id}_depth_mean'] = np.nanmean(mean_depth_per_frame)
        features[f'tof_{sensor_id}_depth_std'] = np.nanstd(mean_depth_per_frame)
        features[f'tof_{sensor_id}_depth_min'] = np.nanmin(mean_depth_per_frame)
        features[f'tof_{sensor_id}_depth_max'] = np.nanmax(mean_depth_per_frame)
        features[f'tof_{sensor_id}_depth_range'] = features[f'tof_{sensor_id}_depth_max'] - features[f'tof_{sensor_id}_depth_min']

        # Depth change rate (temporal)
        depth_velocity = np.diff(mean_depth_per_frame)
        features[f'tof_{sensor_id}_depth_velocity_mean'] = np.nanmean(np.abs(depth_velocity))
        features[f'tof_{sensor_id}_depth_velocity_max'] = np.nanmax(np.abs(depth_velocity))

        return features

    def extract_tof_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all ToF features from a sequence.

        Args:
            df: DataFrame with ToF columns

        Returns:
            Dictionary of all ToF features
        """
        features = {}

        # Per-sensor features
        for sensor_id in range(1, 6):
            sensor_features = self.extract_tof_sensor_features(df, sensor_id)
            features.update(sensor_features)

        # Cross-sensor features
        all_tof_cols = []
        for sensor_id in range(1, 6):
            all_tof_cols.extend(self.get_tof_columns(sensor_id))

        if all(col in df.columns for col in all_tof_cols):
            all_tof_data = df[all_tof_cols].values
            all_tof_data_clean = np.where(all_tof_data == -1, np.nan, all_tof_data)

            # Global statistics
            features['tof_global_sparsity'] = np.sum(np.isnan(all_tof_data_clean)) / all_tof_data_clean.size
            features['tof_global_depth_mean'] = np.nanmean(all_tof_data_clean)
            features['tof_global_depth_std'] = np.nanstd(all_tof_data_clean)

        return features


class ThermalFeatureExtractor:
    """
    Extract features from thermal (thermopile) sensors.

    These are TIER 1 features - only available for 50% of test sequences.
    """

    def __init__(self):
        self.thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']

    def extract_thermal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract thermal features from a sequence.

        Args:
            df: DataFrame with thm_1 through thm_5 columns

        Returns:
            Dictionary of thermal features
        """
        features = {}

        # Check if thermal data exists
        if not all(col in df.columns for col in self.thm_cols):
            return {}

        # Mean temperature across sensors (per frame)
        temp_mean = df[self.thm_cols].mean(axis=1)
        features['thm_mean_mean'] = temp_mean.mean()
        features['thm_mean_std'] = temp_mean.std()
        features['thm_mean_min'] = temp_mean.min()
        features['thm_mean_max'] = temp_mean.max()
        features['thm_mean_range'] = temp_mean.max() - temp_mean.min()

        # Temperature range (max - min across sensors per frame)
        temp_range = df[self.thm_cols].max(axis=1) - df[self.thm_cols].min(axis=1)
        features['thm_range_mean'] = temp_range.mean()
        features['thm_range_std'] = temp_range.std()
        features['thm_range_max'] = temp_range.max()

        # Temperature std across sensors (per frame)
        temp_std = df[self.thm_cols].std(axis=1)
        features['thm_std_mean'] = temp_std.mean()
        features['thm_std_max'] = temp_std.max()

        # Temperature change rate (temporal)
        temp_velocity = np.diff(temp_mean.values, prepend=temp_mean.values[0])
        features['thm_velocity_mean'] = np.abs(temp_velocity).mean()
        features['thm_velocity_max'] = np.abs(temp_velocity).max()

        # Per-sensor statistics
        for col in self.thm_cols:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()

        return features


class FeatureEngineering:
    """
    Main feature engineering pipeline.

    Combines all feature extractors and handles both IMU-only and full-sensor scenarios.
    """

    def __init__(self):
        self.imu_extractor = IMUFeatureExtractor()
        self.tof_extractor = ToFFeatureExtractor()
        self.thermal_extractor = ThermalFeatureExtractor()

    def extract_sequence_features(self, df: pd.DataFrame, include_tof_thermal: bool = True) -> Dict[str, float]:
        """
        Extract all features from a single sequence.

        Args:
            df: DataFrame for one sequence (all frames)
            include_tof_thermal: Whether to extract ToF and thermal features

        Returns:
            Dictionary of all features
        """
        features = {}

        # TIER 0: IMU features (ALWAYS present)
        imu_features = self.imu_extractor.extract_imu_features(df)
        features.update(imu_features)

        if include_tof_thermal:
            # TIER 1: ToF features (when available)
            tof_features = self.tof_extractor.extract_tof_features(df)
            features.update(tof_features)

            # TIER 1: Thermal features (when available)
            thermal_features = self.thermal_extractor.extract_thermal_features(df)
            features.update(thermal_features)

        return features

    def process_dataset(self, df: pd.DataFrame, include_tof_thermal: bool = True) -> pd.DataFrame:
        """
        Process entire dataset and extract features for all sequences.

        Args:
            df: Full dataset (train or test)
            include_tof_thermal: Whether to include ToF and thermal features

        Returns:
            DataFrame with features (one row per sequence)
        """
        print(f"Processing {df['sequence_id'].nunique()} sequences...")

        features_list = []
        sequence_ids = []

        for sequence_id in df['sequence_id'].unique():
            seq_df = df[df['sequence_id'] == sequence_id].copy()

            # Extract features
            features = self.extract_sequence_features(seq_df, include_tof_thermal)

            # Add sequence metadata
            features['sequence_id'] = sequence_id
            if 'gesture' in seq_df.columns:
                features['gesture'] = seq_df['gesture'].iloc[0]
            if 'subject' in seq_df.columns:
                features['subject'] = seq_df['subject'].iloc[0]

            features_list.append(features)
            sequence_ids.append(sequence_id)

            if len(features_list) % 100 == 0:
                print(f"  Processed {len(features_list)} sequences...")

        features_df = pd.DataFrame(features_list)
        print(f"✓ Feature extraction complete: {features_df.shape}")
        print(f"  Total features: {features_df.shape[1] - 1}")  # Exclude sequence_id

        return features_df


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nThis module provides feature extraction for:")
    print("  - TIER 0 (IMU): Accelerometer + Rotation features")
    print("  - TIER 1 (ToF/Thermal): When available")
    print("\nUsage:")
    print("  from feature_engineering import FeatureEngineering")
    print("  fe = FeatureEngineering()")
    print("  features_df = fe.process_dataset(train_df)")
