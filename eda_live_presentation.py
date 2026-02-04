"""
EDA Live Presentation Script
============================

Comprehensive exploratory data analysis for the Kaggle CMI gesture recognition project.
This script reproduces all findings documented in docs/01_EDA_DOCUMENTATION.md.

Usage:
    python eda_live_presentation.py                    # Full analysis
    python eda_live_presentation.py --quick            # Quick overview only
    python eda_live_presentation.py --sensor tof       # ToF sensors only
    python eda_live_presentation.py --save-viz         # Save all visualizations

Author: Nadav Sadna Project
Date: 2026-01-16
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
OUTPUT_DIR = Path('eda_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Sensor offsets for frame reconstruction
SENSOR_OFFSETS_OVERLAPPING = {
    'tof_1': (8, 8),   # Center
    'tof_2': (4, 8),   # Up
    'tof_3': (8, 4),   # Left
    'tof_4': (12, 8),  # Bottom
    'tof_5': (8, 12),  # Right
}

SENSOR_OFFSETS_SEPARATED = {
    'tof_1': (8, 8),
    'tof_2': (0, 8),
    'tof_3': (8, 0),
    'tof_4': (16, 8),
    'tof_5': (8, 16),
}

# BFRB gesture classes
BFRB_GESTURES = [
    'Above ear - pull hair', 'Eyebrow - pull hair', 'Eyelash - pull hair',
    'Forehead - pull hairline', 'Forehead - scratch', 'Cheek - pinch skin',
    'Neck - pinch skin', 'Neck - scratch'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text, level=1):
    """Print formatted section header."""
    if level == 1:
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)
    elif level == 2:
        print("\n" + "-"*80)
        print(f"  {text}")
        print("-"*80)
    else:
        print(f"\n>>> {text}")


def load_data():
    """Load training dataset."""
    print_header("LOADING DATA", level=1)

    train_path = os.path.join(DATA_PATH, 'train.csv')
    print(f"Loading from: {train_path}")

    train_df = pd.read_csv(train_path)

    print(f"✓ Loaded: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"✓ Sequences: {train_df['sequence_id'].nunique():,}")
    print(f"✓ Subjects: {train_df['subject'].nunique()}")
    print(f"✓ Gestures: {train_df['gesture'].nunique()}")

    return train_df


# ============================================================================
# TOF SENSOR FUNCTIONS
# ============================================================================

def tof_columns(df, sensor_id=""):
    """Extract ToF column names."""
    return [col for col in df.columns if col.startswith("tof_" + str(sensor_id))]


def build_samples_df(df):
    """Build samples dataframe for frame reconstruction."""
    tof_df = df[tof_columns(df)]
    sensors_pixels = tof_df.T.reset_index(names=["sensor_pixel"])
    sensors_pixels["sensor_id"] = sensors_pixels["sensor_pixel"].apply(
        lambda x: x.rsplit("_", maxsplit=1)[0]
    )
    sensors_pixels["pixel"] = sensors_pixels["sensor_pixel"].apply(
        lambda x: int(x.rsplit("_", maxsplit=1)[-1].replace('v', ''))
    )

    # Replace -1 (missing) with NaN
    normalized_sensors_pixels = sensors_pixels.replace(-1, np.nan).fillna(np.nan)
    return normalized_sensors_pixels


def build_sensor_frame(samples_df, sensor_id):
    """Build 8×8 frame for a single sensor."""
    sample_cols = sorted([col for col in samples_df.columns if isinstance(col, int)])
    n_samples = len(sample_cols)

    sensor_df = samples_df[samples_df['sensor_id'] == sensor_id].copy()
    if sensor_df.empty:
        return None

    sensor_df = sensor_df.sort_values('pixel')
    values = sensor_df[sample_cols].T.values
    sensor_frame = values.reshape(n_samples, 8, 8)

    # Flip axes for proper orientation
    sensor_frame = np.flip(sensor_frame, axis=(1, 2))

    # Rotate sensors 3 & 5 (90° clockwise)
    if sensor_id in ['tof_3', 'tof_5']:
        sensor_frame = np.rot90(sensor_frame, k=-1, axes=(1, 2))

    return sensor_frame


def build_global_frames(samples_df, offsets, grid_size=24):
    """Combine all sensors into a global frame with averaging of overlapping pixels."""
    sample_cols = sorted([col for col in samples_df.columns if isinstance(col, int)])
    n_samples = len(sample_cols)

    # Numerator and denominator for averaging overlapping pixels
    num = np.zeros((n_samples, grid_size, grid_size), dtype=float)
    den = np.zeros((n_samples, grid_size, grid_size), dtype=np.int32)

    for sensor_id, (r_off, c_off) in offsets.items():
        sensor_frame = build_sensor_frame(samples_df, sensor_id)
        if sensor_frame is None:
            continue

        num_roi = num[:, r_off:r_off+8, c_off:c_off+8]
        den_roi = den[:, r_off:r_off+8, c_off:c_off+8]

        valid = ~np.isnan(sensor_frame)
        num_roi[valid] += sensor_frame[valid]
        den_roi[valid] += 1

    # Compute average where we have data
    frames = np.full((n_samples, grid_size, grid_size), np.nan, dtype=float)
    mask = den > 0
    frames[mask] = num[mask] / den[mask]

    return frames


def analyze_tof_sensors(df, save_viz=False):
    """Comprehensive ToF sensor analysis."""
    print_header("TIME-OF-FLIGHT (ToF) SENSORS", level=1)

    # Basic statistics
    tof_cols = tof_columns(df)
    print(f"\n📊 Basic Statistics:")
    print(f"  - Total ToF features: {len(tof_cols)}")
    print(f"  - Number of sensors: 5 (each with 8×8 = 64 pixels)")
    print(f"  - Total frames: {df.shape[0]:,}")

    # Sparsity analysis per sensor
    print_header("Sparsity Analysis", level=2)

    mean_depth_per_sensor = {}
    sparsity_per_sensor = {}

    for sensor_id in range(1, 6):
        sensor_cols = tof_columns(df, sensor_id)
        sensor_data = df[sensor_cols]

        # Calculate sparsity (percentage of -1 values)
        sparsity = (sensor_data == -1).sum().sum() / sensor_data.size * 100
        sparsity_per_sensor[sensor_id] = sparsity

        # Calculate mean depth (excluding -1)
        valid_data = sensor_data.values.ravel()
        valid_data = valid_data[valid_data != -1]
        mean_depth_per_sensor[sensor_id] = valid_data.mean()

    # Overall statistics
    overall_sparsity = np.mean(list(sparsity_per_sensor.values()))
    overall_mean_depth = np.mean(list(mean_depth_per_sensor.values()))

    print(f"\n  Overall Sparsity: {overall_sparsity:.1f}% (missing values)")
    print(f"  Overall Mean Depth: {overall_mean_depth:.2f} mm")
    print(f"\n  Per-Sensor Breakdown:")

    for sensor_id in range(1, 6):
        print(f"    Sensor {sensor_id}: "
              f"{sparsity_per_sensor[sensor_id]:.1f}% missing, "
              f"mean depth = {mean_depth_per_sensor[sensor_id]:.2f} mm")

    # Visualize sparsity
    if save_viz:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sensors = list(sparsity_per_sensor.keys())
        sparsity_vals = list(sparsity_per_sensor.values())
        depth_vals = list(mean_depth_per_sensor.values())

        # Sparsity plot
        ax1.bar(sensors, sparsity_vals, color='steelblue', alpha=0.8)
        ax1.axhline(overall_sparsity, color='red', linestyle='--',
                   label=f'Overall: {overall_sparsity:.1f}%')
        ax1.set_xlabel('Sensor ID')
        ax1.set_ylabel('Missing Data (%)')
        ax1.set_title('ToF Sensor Sparsity Analysis')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Mean depth plot
        ax2.bar(sensors, depth_vals, color='forestgreen', alpha=0.8)
        ax2.axhline(overall_mean_depth, color='red', linestyle='--',
                   label=f'Overall: {overall_mean_depth:.1f} mm')
        ax2.set_xlabel('Sensor ID')
        ax2.set_ylabel('Mean Depth (mm)')
        ax2.set_title('ToF Sensor Mean Depth')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / 'tof_sparsity_depth.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {output_path}")

    # Frame reconstruction example
    print_header("Frame Reconstruction Example", level=2)

    # Select a sample sequence with good data
    sample_seq = df[df['gesture'] == 'Text on phone']['sequence_id'].iloc[0]
    seq_df = df[df['sequence_id'] == sample_seq].head(10)

    print(f"  Using sequence: {sample_seq} (gesture: Text on phone)")
    print(f"  Frames: {len(seq_df)}")

    samples_df = build_samples_df(seq_df)

    # Build both overlapping and separated frames
    frames_overlap = build_global_frames(samples_df, SENSOR_OFFSETS_OVERLAPPING, grid_size=24)
    frames_sep = build_global_frames(samples_df, SENSOR_OFFSETS_SEPARATED, grid_size=24)

    print(f"  ✓ Built overlapping frame: {frames_overlap.shape}")
    print(f"  ✓ Built separated frame: {frames_sep.shape}")

    if save_viz:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(frames_overlap[0], cmap='viridis', vmin=0, vmax=250)
        axes[0].set_title('Overlapping Sensors (24×24)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Depth (mm)')

        im2 = axes[1].imshow(frames_sep[0], cmap='viridis', vmin=0, vmax=250)
        axes[1].set_title('Separated Sensors (24×24)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Depth (mm)')

        plt.suptitle(f'ToF Frame Reconstruction (Sequence {sample_seq}, Frame 0)')

        output_path = OUTPUT_DIR / 'tof_frame_reconstruction.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")

    # Key insights
    print_header("Key Insights", level=2)
    print("  1. 59.4% of ToF data is missing (-1 values) → Need robust imputation")
    print("  2. Valid depth range: 0-249 mm (sensor specification)")
    print("  3. Mean captured distance: ~108 mm (typical hand-to-sensor distance)")
    print("  4. Sensors 3 & 5 require 90° rotation for proper orientation")
    print("  5. Overlapping sensor fields allow for higher-resolution reconstruction")


# ============================================================================
# ACCELEROMETER FUNCTIONS
# ============================================================================

def analyze_accelerometer(df, save_viz=False):
    """Comprehensive accelerometer analysis."""
    print_header("ACCELEROMETER (IMU)", level=1)

    acc_cols = ['acc_x', 'acc_y', 'acc_z']

    print(f"\n📊 Basic Statistics:")
    print(f"  - Features: {len(acc_cols)} (3-axis)")
    print(f"  - Total frames: {df.shape[0]:,}")

    # Calculate 3D magnitude
    df_copy = df.copy()
    df_copy['acc_magnitude'] = np.sqrt(
        df_copy['acc_x']**2 + df_copy['acc_y']**2 + df_copy['acc_z']**2
    )

    # Overall statistics
    print_header("Overall Statistics", level=2)
    for col in acc_cols + ['acc_magnitude']:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        min_val = df_copy[col].min()
        max_val = df_copy[col].max()

        print(f"  {col:15s}: mean={mean_val:7.3f}, std={std_val:7.3f}, "
              f"range=[{min_val:7.3f}, {max_val:7.3f}]")

    # Per-gesture analysis
    print_header("Per-Gesture Analysis", level=2)

    gesture_stats = df_copy.groupby('gesture')['acc_magnitude'].agg(['mean', 'std', 'max'])
    gesture_stats = gesture_stats.sort_values('mean', ascending=False)

    print("\n  Top 5 gestures by mean acceleration magnitude:")
    for i, (gesture, row) in enumerate(gesture_stats.head(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.3f} ± {row['std']:.3f}")

    print("\n  Bottom 5 gestures by mean acceleration magnitude:")
    for i, (gesture, row) in enumerate(gesture_stats.tail(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.3f} ± {row['std']:.3f}")

    # BFRB vs non-BFRB comparison
    print_header("BFRB vs Non-BFRB Comparison", level=2)

    df_copy['is_bfrb'] = df_copy['gesture'].isin(BFRB_GESTURES)

    bfrb_mean = df_copy[df_copy['is_bfrb']]['acc_magnitude'].mean()
    non_bfrb_mean = df_copy[~df_copy['is_bfrb']]['acc_magnitude'].mean()
    ratio = non_bfrb_mean / bfrb_mean

    print(f"  BFRB gestures:     {bfrb_mean:.3f}")
    print(f"  Non-BFRB gestures: {non_bfrb_mean:.3f}")
    print(f"  Ratio:             {ratio:.2f}x")
    print(f"\n  → Non-BFRB gestures have {((ratio - 1) * 100):.1f}% higher acceleration")

    # Visualization
    if save_viz:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution of acc_magnitude
        axes[0, 0].hist(df_copy['acc_magnitude'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Acceleration Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Acceleration Magnitude')
        axes[0, 0].grid(alpha=0.3)

        # Per-gesture boxplot (top 10)
        top_gestures = gesture_stats.head(10).index
        data_to_plot = [df_copy[df_copy['gesture'] == g]['acc_magnitude'].values
                        for g in top_gestures]
        bp = axes[0, 1].boxplot(data_to_plot, labels=range(1, 11), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 1].set_xlabel('Gesture Rank (by mean acceleration)')
        axes[0, 1].set_ylabel('Acceleration Magnitude')
        axes[0, 1].set_title('Top 10 Gestures by Acceleration')
        axes[0, 1].grid(alpha=0.3)

        # BFRB vs non-BFRB
        bp = axes[1, 0].boxplot([df_copy[df_copy['is_bfrb']]['acc_magnitude'].values,
                                  df_copy[~df_copy['is_bfrb']]['acc_magnitude'].values],
                                 labels=['BFRB', 'Non-BFRB'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        axes[1, 0].set_ylabel('Acceleration Magnitude')
        axes[1, 0].set_title('BFRB vs Non-BFRB Gestures')
        axes[1, 0].grid(alpha=0.3)

        # Time series example
        sample_seq = df_copy[df_copy['gesture'] == 'Wave hello']['sequence_id'].iloc[0]
        seq_data = df_copy[df_copy['sequence_id'] == sample_seq]

        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['acc_x'],
                       label='acc_x', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['acc_y'],
                       label='acc_y', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['acc_z'],
                       label='acc_z', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['acc_magnitude'],
                       label='magnitude', linewidth=2, color='black')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Acceleration')
        axes[1, 1].set_title(f'Time Series (Sequence {sample_seq}: Wave hello)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / 'accelerometer_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {output_path}")

    # Key insights
    print_header("Key Insights", level=2)
    print("  1. Non-BFRB gestures (wave, text, drink) have 2-3× higher acceleration")
    print("  2. BFRB gestures (scratch, pinch) involve subtle, low-motion actions")
    print("  3. Magnitude is highly discriminative for binary classification")
    print("  4. Per-axis features needed for distinguishing similar gestures")


# ============================================================================
# ROTATION FUNCTIONS
# ============================================================================

def quaternion_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
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

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def analyze_rotation(df, save_viz=False):
    """Comprehensive rotation/quaternion analysis."""
    print_header("ROTATION (QUATERNION)", level=1)

    rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']

    print(f"\n📊 Basic Statistics:")
    print(f"  - Features: {len(rot_cols)} (quaternion: w, x, y, z)")
    print(f"  - Total frames: {df.shape[0]:,}")

    # Verify quaternion normalization
    df_copy = df.copy()
    df_copy['quat_norm'] = np.sqrt(
        df_copy['rot_w']**2 + df_copy['rot_x']**2 +
        df_copy['rot_y']**2 + df_copy['rot_z']**2
    )

    print(f"\n  Quaternion Normalization Check:")
    print(f"    Mean norm: {df_copy['quat_norm'].mean():.6f} (should be ~1.0)")
    print(f"    Std norm:  {df_copy['quat_norm'].std():.6f}")

    # Convert to Euler angles
    print_header("Euler Angle Conversion", level=2)

    roll, pitch, yaw = quaternion_to_euler(
        df_copy['rot_w'].values,
        df_copy['rot_x'].values,
        df_copy['rot_y'].values,
        df_copy['rot_z'].values
    )

    df_copy['roll'] = roll
    df_copy['pitch'] = pitch
    df_copy['yaw'] = yaw

    print(f"\n  Euler Angle Ranges (degrees):")
    for col in ['roll', 'pitch', 'yaw']:
        min_val = df_copy[col].min()
        max_val = df_copy[col].max()
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()

        print(f"    {col:6s}: [{min_val:7.1f}, {max_val:7.1f}], "
              f"mean={mean_val:7.1f}, std={std_val:6.1f}")

    # Calculate rotation magnitude
    df_copy['rotation_magnitude'] = np.sqrt(
        df_copy['roll']**2 + df_copy['pitch']**2 + df_copy['yaw']**2
    )

    # Per-gesture rotation analysis
    print_header("Per-Gesture Rotation Analysis", level=2)

    gesture_stats = df_copy.groupby('gesture')['rotation_magnitude'].agg(['mean', 'std', 'max'])
    gesture_stats = gesture_stats.sort_values('mean', ascending=False)

    print("\n  Top 5 gestures by mean rotation magnitude:")
    for i, (gesture, row) in enumerate(gesture_stats.head(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.1f}° ± {row['std']:.1f}°")

    print("\n  Bottom 5 gestures by mean rotation magnitude:")
    for i, (gesture, row) in enumerate(gesture_stats.tail(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.1f}° ± {row['std']:.1f}°")

    # Visualization
    if save_viz:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Euler angle distributions
        axes[0, 0].hist(df_copy['roll'], bins=50, alpha=0.6, label='Roll', color='red')
        axes[0, 0].hist(df_copy['pitch'], bins=50, alpha=0.6, label='Pitch', color='green')
        axes[0, 0].hist(df_copy['yaw'], bins=50, alpha=0.6, label='Yaw', color='blue')
        axes[0, 0].set_xlabel('Angle (degrees)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Euler Angle Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Rotation magnitude distribution
        axes[0, 1].hist(df_copy['rotation_magnitude'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Rotation Magnitude (degrees)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Overall Rotation Magnitude')
        axes[0, 1].grid(alpha=0.3)

        # Per-gesture comparison (top 10)
        top_gestures = gesture_stats.head(10).index
        data_to_plot = [df_copy[df_copy['gesture'] == g]['rotation_magnitude'].values
                        for g in top_gestures]
        bp = axes[1, 0].boxplot(data_to_plot, labels=range(1, 11), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightyellow')
        axes[1, 0].set_xlabel('Gesture Rank (by mean rotation)')
        axes[1, 0].set_ylabel('Rotation Magnitude (degrees)')
        axes[1, 0].set_title('Top 10 Gestures by Rotation')
        axes[1, 0].grid(alpha=0.3)

        # Time series example
        sample_seq = df_copy[df_copy['gesture'] == 'Wave hello']['sequence_id'].iloc[0]
        seq_data = df_copy[df_copy['sequence_id'] == sample_seq]

        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['roll'],
                       label='Roll', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['pitch'],
                       label='Pitch', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(seq_data['sequence_counter'], seq_data['yaw'],
                       label='Yaw', alpha=0.7, linewidth=1.5)
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].set_title(f'Time Series (Sequence {sample_seq}: Wave hello)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / 'rotation_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {output_path}")

    # Key insights
    print_header("Key Insights", level=2)
    print("  1. Quaternions are properly normalized (mean norm ≈ 1.0)")
    print("  2. Roll ranges from -180° to +180° (full wrist rotation)")
    print("  3. Pitch and yaw have narrower ranges (limited by wrist anatomy)")
    print("  4. 'Wave hello' and 'Pull air' have highest rotation magnitudes")
    print("  5. BFRB gestures have low rotation (head/face contact gestures)")


# ============================================================================
# THERMAL SENSOR FUNCTIONS
# ============================================================================

def analyze_thermal(df, save_viz=False):
    """Comprehensive thermal sensor analysis."""
    print_header("THERMAL SENSORS", level=1)

    thm_cols = [f'thm_{i}' for i in range(1, 6)]

    print(f"\n📊 Basic Statistics:")
    print(f"  - Features: {len(thm_cols)} (5 temperature sensors)")
    print(f"  - Total frames: {df.shape[0]:,}")

    # Overall statistics
    print_header("Overall Statistics", level=2)

    df_copy = df.copy()

    for col in thm_cols:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        min_val = df_copy[col].min()
        max_val = df_copy[col].max()

        print(f"  {col:6s}: mean={mean_val:6.2f}, std={std_val:5.2f}, "
              f"range=[{min_val:6.2f}, {max_val:6.2f}]")

    # Calculate derived features
    df_copy['thm_range'] = df_copy[thm_cols].max(axis=1) - df_copy[thm_cols].min(axis=1)
    df_copy['thm_mean'] = df_copy[thm_cols].mean(axis=1)

    print(f"\n  Temperature Range (max - min):")
    print(f"    Mean: {df_copy['thm_range'].mean():.2f}")
    print(f"    Std:  {df_copy['thm_range'].std():.2f}")

    # Per-gesture analysis
    print_header("Per-Gesture Analysis", level=2)

    gesture_stats = df_copy.groupby('gesture')['thm_mean'].agg(['mean', 'std'])
    gesture_stats = gesture_stats.sort_values('mean', ascending=False)

    print("\n  Top 5 gestures by mean temperature:")
    for i, (gesture, row) in enumerate(gesture_stats.head(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.2f} ± {row['std']:.2f}")

    print("\n  Bottom 5 gestures by mean temperature:")
    for i, (gesture, row) in enumerate(gesture_stats.tail(5).iterrows(), 1):
        print(f"    {i}. {gesture:30s}: {row['mean']:.2f} ± {row['std']:.2f}")

    # BFRB vs non-BFRB comparison
    print_header("BFRB vs Non-BFRB Comparison", level=2)

    df_copy['is_bfrb'] = df_copy['gesture'].isin(BFRB_GESTURES)

    bfrb_mean = df_copy[df_copy['is_bfrb']]['thm_mean'].mean()
    non_bfrb_mean = df_copy[~df_copy['is_bfrb']]['thm_mean'].mean()

    print(f"  BFRB gestures:     {bfrb_mean:.2f}")
    print(f"  Non-BFRB gestures: {non_bfrb_mean:.2f}")
    print(f"  Difference:        {abs(bfrb_mean - non_bfrb_mean):.2f}")

    if bfrb_mean > non_bfrb_mean:
        print(f"\n  → BFRB gestures are {((bfrb_mean / non_bfrb_mean - 1) * 100):.1f}% "
              f"warmer (skin contact)")
    else:
        print(f"\n  → Non-BFRB gestures are {((non_bfrb_mean / bfrb_mean - 1) * 100):.1f}% warmer")

    # Visualization
    if save_viz:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Per-sensor distributions
        for i, col in enumerate(thm_cols):
            axes[0, 0].hist(df_copy[col], bins=30, alpha=0.5, label=f'Sensor {i+1}')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Temperature Distributions per Sensor')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Temperature range distribution
        axes[0, 1].hist(df_copy['thm_range'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Temperature Range (max - min)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Inter-Sensor Temperature Range')
        axes[0, 1].grid(alpha=0.3)

        # BFRB vs non-BFRB
        bp = axes[1, 0].boxplot([df_copy[df_copy['is_bfrb']]['thm_mean'].values,
                                  df_copy[~df_copy['is_bfrb']]['thm_mean'].values],
                                 labels=['BFRB', 'Non-BFRB'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        axes[1, 0].set_ylabel('Mean Temperature')
        axes[1, 0].set_title('BFRB vs Non-BFRB Gestures')
        axes[1, 0].grid(alpha=0.3)

        # Time series example
        sample_seq = df_copy[df_copy['gesture'].isin(BFRB_GESTURES)]['sequence_id'].iloc[0]
        seq_data = df_copy[df_copy['sequence_id'] == sample_seq]

        for i, col in enumerate(thm_cols):
            axes[1, 1].plot(seq_data['sequence_counter'], seq_data[col],
                           label=f'Sensor {i+1}', alpha=0.7, linewidth=1.5)
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Temperature')
        gesture_name = seq_data['gesture'].iloc[0]
        axes[1, 1].set_title(f'Time Series (Sequence {sample_seq}: {gesture_name})')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / 'thermal_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved: {output_path}")

    # Key insights
    print_header("Key Insights", level=2)
    print("  1. Thermal sensors detect skin contact (BFRB gestures warmer)")
    print("  2. Temperature range indicates spatial proximity patterns")
    print("  3. Sensor 2 (center) typically warmest during contact gestures")
    print("  4. Temporal temperature changes indicate gesture transitions")
    print("  5. Critical for distinguishing contact vs. air gestures")


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(df):
    """Print comprehensive summary of all findings."""
    print_header("COMPREHENSIVE SUMMARY", level=1)

    print_header("Dataset Overview", level=2)
    print(f"  Total frames: {df.shape[0]:,}")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Unique sequences: {df['sequence_id'].nunique():,}")
    print(f"  Unique subjects: {df['subject'].nunique()}")
    print(f"  Gesture classes: {df['gesture'].nunique()}")

    print_header("Feature Categories", level=2)
    print("  1. ToF Sensors (320 features)")
    print("     - 5 sensors × 8×8 pixels")
    print("     - 59.4% sparsity (missing values)")
    print("     - Mean depth: 108.21 mm")
    print("     - Critical for spatial understanding")

    print("  2. Accelerometer (3 features)")
    print("     - 3-axis IMU: acc_x, acc_y, acc_z")
    print("     - Non-BFRB gestures: 2-3× higher magnitude")
    print("     - Critical for motion intensity")

    print("  3. Rotation (4 features)")
    print("     - Quaternion: rot_w, rot_x, rot_y, rot_z")
    print("     - Converted to Euler angles (roll, pitch, yaw)")
    print("     - Wave/pull gestures: highest rotation")
    print("     - Critical for orientation changes")

    print("  4. Thermal (5 features)")
    print("     - 5 temperature sensors: thm_1 to thm_5")
    print("     - BFRB gestures: warmer (skin contact)")
    print("     - Critical for contact detection")

    print_header("Key Findings for Machine Learning", level=2)
    print("  1. Feature Engineering Priorities:")
    print("     ✓ ToF: Spatial aggregations (mean depth, center of mass, sparsity)")
    print("     ✓ Accelerometer: Magnitude, jerk, moving averages")
    print("     ✓ Rotation: Euler angles, angular velocity, rotation magnitude")
    print("     ✓ Thermal: Mean temperature, range, contact indicators")

    print("\n  2. Preprocessing Requirements:")
    print("     ✓ ToF: Handle 59% sparsity (forward-fill or tree models)")
    print("     ✓ Accelerometer: Z-score normalization per axis")
    print("     ✓ Rotation: Euler conversion, angle normalization")
    print("     ✓ Thermal: MinMax scaling to [0, 1]")

    print("\n  3. Model Recommendations:")
    print("     ✓ Baseline: XGBoost with sequence-level aggregations (DONE: 0.7351)")
    print("     ✓ Advanced: 1D CNN or LSTM for temporal patterns")
    print("     ✓ Multi-modal: Fusion for leveraging all sensors")

    print("\n  4. Critical Challenges:")
    print("     ✓ Class imbalance: 640 vs 161 sequences")
    print("     ✓ ToF sparsity: 59% missing values")
    print("     ✓ Variable sequence length: 29-700 frames")
    print("     ✓ Subject variability: 81 different subjects")

    print_header("Cross-Modality Insights", level=2)
    print("  ✓ Motion + Rotation → Distinguishes wave/pull (high) vs. scratch/pinch (low)")
    print("  ✓ Thermal + ToF → Detects contact gestures (BFRB) vs. air gestures")
    print("  ✓ All modalities → Text/phone gestures have unique signatures")
    print("  ✓ IMU-only → Reasonable for non-contact gestures")
    print("  ✓ Full sensor → Best performance, especially for BFRB gestures")

    print("\n" + "="*80)
    print("  For detailed documentation, see: docs/01_EDA_DOCUMENTATION.md")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='EDA Live Presentation for Kaggle CMI Gesture Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eda_live_presentation.py                    # Full analysis
  python eda_live_presentation.py --quick            # Quick overview only
  python eda_live_presentation.py --sensor tof       # ToF sensors only
  python eda_live_presentation.py --save-viz         # Save all visualizations
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick overview only (skip detailed analysis)')
    parser.add_argument('--sensor', choices=['tof', 'acc', 'rot', 'thm', 'all'],
                       default='all', help='Analyze specific sensor only')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save all visualizations to eda_outputs/')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip final summary')

    args = parser.parse_args()

    # Load data
    df = load_data()

    # Run analyses based on arguments
    if args.sensor == 'all' or args.sensor == 'tof':
        analyze_tof_sensors(df, save_viz=args.save_viz)

    if args.sensor == 'all' or args.sensor == 'acc':
        analyze_accelerometer(df, save_viz=args.save_viz)

    if args.sensor == 'all' or args.sensor == 'rot':
        analyze_rotation(df, save_viz=args.save_viz)

    if args.sensor == 'all' or args.sensor == 'thm':
        analyze_thermal(df, save_viz=args.save_viz)

    # Print summary
    if not args.no_summary:
        print_summary(df)

    # Final message
    print_header("ANALYSIS COMPLETE", level=1)
    if args.save_viz:
        print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
        print(f"  - tof_sparsity_depth.png")
        print(f"  - tof_frame_reconstruction.png")
        print(f"  - accelerometer_analysis.png")
        print(f"  - rotation_analysis.png")
        print(f"  - thermal_analysis.png")

    print("\n📊 Current Model Performance (XGBoost Baseline):")
    print("  - Overall Score: 0.7351")
    print("  - IMU-only:      0.6864")
    print("  - Full Sensor:   0.7838")

    print("\n📖 Documentation:")
    print("  - Full EDA:      docs/01_EDA_DOCUMENTATION.md")
    print("  - Final Results: docs/08_FINAL_RESULTS_SUMMARY.md")

    print("\n🎉 Thank you for attending the presentation!")


if __name__ == "__main__":
    main()
