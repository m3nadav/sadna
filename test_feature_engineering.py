"""
Test Feature Engineering Pipeline
==================================

This script tests the feature engineering module on a sample of training data.

Usage:
    python test_feature_engineering.py
"""

import os
import sys
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineering

def main():
    print("=" * 70)
    print("TESTING FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading training data...")
    try:
        path = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        print(f"✓ Loaded: {train.shape[0]:,} rows × {train.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Sample a few sequences for testing
    print("\n[2/5] Sampling sequences for testing...")
    sample_sequences = train['sequence_id'].unique()[:10]  # First 10 sequences
    train_sample = train[train['sequence_id'].isin(sample_sequences)].copy()
    print(f"✓ Sampled {len(sample_sequences)} sequences")
    print(f"  Gestures: {train_sample['gesture'].unique().tolist()}")

    # Test IMU-only features
    print("\n[3/5] Extracting IMU-only features...")
    fe = FeatureEngineering()
    try:
        imu_features = fe.process_dataset(train_sample, include_tof_thermal=False)
        print(f"✓ IMU features extracted: {imu_features.shape}")
        print(f"  Feature count: {imu_features.shape[1] - 3} (excluding metadata)")  # -3 for sequence_id, gesture, subject
        print(f"  Sample features:")
        feature_cols = [col for col in imu_features.columns if col not in ['sequence_id', 'gesture', 'subject']]
        print(f"    - {', '.join(feature_cols[:5])} ...")
    except Exception as e:
        print(f"✗ Error extracting IMU features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test full features (IMU + ToF + Thermal)
    print("\n[4/5] Extracting full sensor features...")
    try:
        full_features = fe.process_dataset(train_sample, include_tof_thermal=True)
        print(f"✓ Full features extracted: {full_features.shape}")
        print(f"  Feature count: {full_features.shape[1] - 3} (excluding metadata)")
        print(f"  Additional ToF/Thermal features: {full_features.shape[1] - imu_features.shape[1]}")
    except Exception as e:
        print(f"✗ Error extracting full features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validation checks
    print("\n[5/5] Running validation checks...")

    # Check for NaN/Inf values
    nan_count = imu_features.drop(['sequence_id', 'gesture', 'subject'], axis=1).isna().sum().sum()
    inf_count = np.isinf(imu_features.select_dtypes(include=[np.number]).values).sum()

    if nan_count > 0:
        print(f"⚠ Warning: {nan_count} NaN values found in IMU features")
    else:
        print(f"✓ No NaN values in IMU features")

    if inf_count > 0:
        print(f"⚠ Warning: {inf_count} Inf values found in IMU features")
    else:
        print(f"✓ No Inf values in IMU features")

    # Check feature ranges
    print("\n  Sample feature statistics:")
    feature_cols = [col for col in imu_features.columns if col not in ['sequence_id', 'gesture', 'subject']]
    for feat in feature_cols[:5]:
        values = imu_features[feat].dropna()
        print(f"    {feat}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Feature engineering pipeline is working!")
    print(f"\n  IMU-only features: {imu_features.shape[1] - 3}")
    print(f"  Full sensor features: {full_features.shape[1] - 3}")
    print(f"  Sequences processed: {len(sample_sequences)}")
    print(f"\n  Next steps:")
    print(f"    1. Process full training dataset")
    print(f"    2. Train IMU-only XGBoost baseline")
    print(f"    3. Train full sensor XGBoost model")
    print(f"    4. Compare performance")
    print("=" * 70)

    return imu_features, full_features

if __name__ == "__main__":
    imu_features, full_features = main()
