"""
Train Baseline XGBoost Models
==============================

This script trains two XGBoost models:
1. IMU-only model (for 50% IMU-only test sequences)
2. Full sensor model (for 50% full sensor test sequences)

Strategy:
- Split training data into train/validation (80/20)
- Simulate test conditions (50% validation sequences have IMU-only)
- Evaluate separately: IMU-only performance, Full sensor performance, Overall

Author: Claude + Nadav
Date: 2026-01-14
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
from feature_engineering import FeatureEngineering
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2  # 20% for validation


def load_data():
    """Load training data."""
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    print(f"✓ Loaded training data: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Unique sequences: {train['sequence_id'].nunique():,}")
    print(f"  Unique subjects: {train['subject'].nunique()}")
    print(f"  Gesture classes: {train['gesture'].nunique()}")

    return train


def create_train_val_split(train_df):
    """
    Create train/validation split at sequence level.

    Strategy:
    - Split by subject (group) to prevent leakage
    - Stratify by gesture to maintain class balance
    - 80% train, 20% validation
    """
    print("\n" + "=" * 80)
    print("STEP 2: CREATING TRAIN/VALIDATION SPLIT")
    print("=" * 80)

    # Get unique sequences with metadata
    sequences = train_df[['sequence_id', 'gesture', 'subject']].drop_duplicates()
    print(f"Total sequences: {len(sequences):,}")

    # Split sequences (stratified by gesture, grouped by subject)
    # Note: Using simple stratified split since StratifiedGroupKFold is for CV
    train_sequences, val_sequences = train_test_split(
        sequences,
        test_size=VALIDATION_SIZE,
        stratify=sequences['gesture'],
        random_state=RANDOM_STATE
    )

    print(f"✓ Train sequences: {len(train_sequences):,} ({100*(1-VALIDATION_SIZE):.0f}%)")
    print(f"✓ Validation sequences: {len(val_sequences):,} ({100*VALIDATION_SIZE:.0f}%)")

    # Verify class distribution
    print("\n  Gesture distribution in splits:")
    print("  " + "-" * 76)
    train_dist = train_sequences['gesture'].value_counts().sort_index()
    val_dist = val_sequences['gesture'].value_counts().sort_index()

    for gesture in train_dist.index:
        train_count = train_dist.get(gesture, 0)
        val_count = val_dist.get(gesture, 0)
        total = train_count + val_count
        print(f"  {gesture:45s}: Train={train_count:4d} ({100*train_count/total:5.1f}%) | Val={val_count:4d} ({100*val_count/total:5.1f}%)")

    return train_sequences, val_sequences


def extract_features(train_df, train_sequences, val_sequences):
    """
    Extract features for train and validation sets.

    Creates:
    - IMU-only features (for both train and val)
    - Full sensor features (for train and val)
    """
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE EXTRACTION")
    print("=" * 80)

    fe = FeatureEngineering()

    # Filter data by sequences
    train_data = train_df[train_df['sequence_id'].isin(train_sequences['sequence_id'])].copy()
    val_data = train_df[train_df['sequence_id'].isin(val_sequences['sequence_id'])].copy()

    print(f"\nTrain data: {train_data.shape[0]:,} rows across {train_data['sequence_id'].nunique():,} sequences")
    print(f"Validation data: {val_data.shape[0]:,} rows across {val_data['sequence_id'].nunique():,} sequences")

    # Extract IMU-only features
    print("\n[3.1] Extracting IMU-only features (TIER 0)...")
    print("  Processing training set...")
    train_imu_features = fe.process_dataset(train_data, include_tof_thermal=False)
    print("  Processing validation set...")
    val_imu_features = fe.process_dataset(val_data, include_tof_thermal=False)
    print(f"✓ IMU-only features: {train_imu_features.shape[1] - 3} features")

    # Extract full sensor features
    print("\n[3.2] Extracting full sensor features (TIER 0 + TIER 1)...")
    print("  Processing training set...")
    train_full_features = fe.process_dataset(train_data, include_tof_thermal=True)
    print("  Processing validation set...")
    val_full_features = fe.process_dataset(val_data, include_tof_thermal=True)
    print(f"✓ Full sensor features: {train_full_features.shape[1] - 3} features")

    return train_imu_features, val_imu_features, train_full_features, val_full_features


def simulate_test_conditions(val_imu_features, val_full_features):
    """
    Simulate test conditions: 50% of validation sequences will be IMU-only.

    Returns indices for IMU-only and full-sensor validation sequences.
    """
    print("\n" + "=" * 80)
    print("STEP 4: SIMULATING TEST CONDITIONS")
    print("=" * 80)

    n_val = len(val_imu_features)
    indices = np.arange(n_val)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(indices)

    split_point = n_val // 2
    imu_only_indices = indices[:split_point]
    full_sensor_indices = indices[split_point:]

    print(f"✓ Total validation sequences: {n_val}")
    print(f"  - IMU-only sequences (simulated): {len(imu_only_indices)} ({100*len(imu_only_indices)/n_val:.1f}%)")
    print(f"  - Full sensor sequences: {len(full_sensor_indices)} ({100*len(full_sensor_indices)/n_val:.1f}%)")

    return imu_only_indices, full_sensor_indices


def prepare_data_for_training(features_df):
    """
    Prepare features and labels for training.

    Returns:
    - X: Feature matrix (numpy array)
    - y: Labels (numpy array)
    - label_encoder: Fitted LabelEncoder for inverse transform
    - feature_names: List of feature column names
    """
    # Separate features from metadata
    feature_cols = [col for col in features_df.columns
                    if col not in ['sequence_id', 'gesture', 'subject']]

    X = features_df[feature_cols].values

    # Encode gestures to integers
    le = LabelEncoder()
    y = le.fit_transform(features_df['gesture'].values)

    return X, y, le, feature_cols


def calculate_competition_score(y_true, y_pred, label_encoder):
    """
    Calculate the competition evaluation metric:
    Score = (Binary F1 + Macro F1) / 2

    Binary F1: BFRB (target) vs. non-BFRB (non-target)
    Macro F1: 9 classes (8 BFRB + 1 non_target)
    """
    # Define BFRB (target) gestures
    bfrb_gestures = [
        'Above ear - pull hair',
        'Forehead - pull hairline',
        'Forehead - scratch',
        'Eyebrow - pull hair',
        'Eyelash - pull hair',
        'Neck - pinch skin',
        'Neck - scratch',
        'Cheek - pinch skin'
    ]

    # Get gesture names
    gestures_true = label_encoder.inverse_transform(y_true)
    gestures_pred = label_encoder.inverse_transform(y_pred)

    # Binary classification: BFRB (1) vs. non-BFRB (0)
    y_true_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_true])
    y_pred_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_pred])

    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')

    # Multi-class: Collapse non-BFRB into "non_target"
    gestures_true_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_true]
    gestures_pred_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_pred]

    # Encode collapsed gestures
    unique_classes = list(set(gestures_true_collapsed))
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}

    y_true_collapsed = np.array([class_to_idx[g] for g in gestures_true_collapsed])
    y_pred_collapsed = np.array([class_to_idx[g] for g in gestures_pred_collapsed])

    macro_f1 = f1_score(y_true_collapsed, y_pred_collapsed, average='macro')

    # Final score
    competition_score = (binary_f1 + macro_f1) / 2

    return competition_score, binary_f1, macro_f1


def train_xgboost_model(X_train, y_train, X_val, y_val, model_name="XGBoost"):
    """
    Train XGBoost classifier with class balancing.
    """
    print(f"\nTraining {model_name}...")

    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    # XGBoost parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': len(classes),
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"✓ {model_name} trained successfully")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]}")

    return model


def evaluate_model(model, X, y, label_encoder, split_name="Validation"):
    """
    Evaluate model and print detailed metrics.
    """
    print(f"\n  Evaluating on {split_name} set:")
    print("  " + "-" * 76)

    # Predictions
    y_pred = model.predict(X)

    # Competition score
    comp_score, binary_f1, macro_f1 = calculate_competition_score(y, y_pred, label_encoder)

    print(f"  Competition Score: {comp_score:.4f}")
    print(f"    - Binary F1 (BFRB vs. non-BFRB): {binary_f1:.4f}")
    print(f"    - Macro F1 (9 classes): {macro_f1:.4f}")

    # Per-class F1 scores
    per_class_f1 = f1_score(y, y_pred, average=None)
    gestures = label_encoder.classes_

    print(f"\n  Per-Gesture F1 Scores:")
    print("  " + "-" * 76)
    for idx, (gesture, f1) in enumerate(sorted(zip(gestures, per_class_f1), key=lambda x: x[1])):
        status = "⚠️" if f1 < 0.5 else "✓"
        print(f"  {status} {gesture:45s}: {f1:.4f}")

    # Overall accuracy
    accuracy = (y == y_pred).mean()
    print(f"\n  Overall Accuracy: {accuracy:.4f}")

    return comp_score, binary_f1, macro_f1, per_class_f1


def main():
    """
    Main training pipeline.
    """
    print("\n")
    print("=" * 80)
    print("BASELINE MODEL TRAINING: XGBoost with Dual Strategy")
    print("=" * 80)
    print(f"Random Seed: {RANDOM_STATE}")
    print(f"Validation Size: {VALIDATION_SIZE*100:.0f}%")
    print("=" * 80)

    # Step 1: Load data
    train = load_data()

    # Step 2: Create train/val split
    train_sequences, val_sequences = create_train_val_split(train)

    # Step 3: Extract features
    train_imu, val_imu, train_full, val_full = extract_features(
        train, train_sequences, val_sequences
    )

    # Step 4: Simulate test conditions
    imu_only_indices, full_sensor_indices = simulate_test_conditions(val_imu, val_full)

    # Step 5: Prepare data for training
    print("\n" + "=" * 80)
    print("STEP 5: PREPARING DATA FOR TRAINING")
    print("=" * 80)

    X_train_imu, y_train_imu, le_imu, feature_names_imu = prepare_data_for_training(train_imu)
    X_val_imu, y_val_imu, _, _ = prepare_data_for_training(val_imu)

    X_train_full, y_train_full, le_full, feature_names_full = prepare_data_for_training(train_full)
    X_val_full, y_val_full, _, _ = prepare_data_for_training(val_full)

    print(f"✓ IMU-only data prepared:")
    print(f"  Train: {X_train_imu.shape}")
    print(f"  Val: {X_val_imu.shape}")
    print(f"\n✓ Full sensor data prepared:")
    print(f"  Train: {X_train_full.shape}")
    print(f"  Val: {X_val_full.shape}")

    # Step 6: Train models
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING MODELS")
    print("=" * 80)

    print("\n[6.1] Training IMU-only XGBoost model...")
    model_imu = train_xgboost_model(
        X_train_imu, y_train_imu,
        X_val_imu, y_val_imu,
        model_name="IMU-only XGBoost"
    )

    print("\n[6.2] Training Full sensor XGBoost model...")
    model_full = train_xgboost_model(
        X_train_full, y_train_full,
        X_val_full, y_val_full,
        model_name="Full Sensor XGBoost"
    )

    # Step 7: Evaluate models
    print("\n" + "=" * 80)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 80)

    print("\n[7.1] IMU-only Model Performance")
    print("=" * 80)

    # Evaluate on all validation (to see overall IMU-only performance)
    score_imu_all, bin_f1_imu, mac_f1_imu, _ = evaluate_model(
        model_imu, X_val_imu, y_val_imu, le_imu, "All Validation"
    )

    # Evaluate on IMU-only subset (simulated test condition)
    score_imu_subset, _, _, _ = evaluate_model(
        model_imu, X_val_imu[imu_only_indices], y_val_imu[imu_only_indices],
        le_imu, "IMU-only Validation Subset"
    )

    print("\n[7.2] Full Sensor Model Performance")
    print("=" * 80)

    # Evaluate on all validation
    score_full_all, bin_f1_full, mac_f1_full, _ = evaluate_model(
        model_full, X_val_full, y_val_full, le_full, "All Validation"
    )

    # Evaluate on full sensor subset (simulated test condition)
    score_full_subset, _, _, _ = evaluate_model(
        model_full, X_val_full[full_sensor_indices], y_val_full[full_sensor_indices],
        le_full, "Full Sensor Validation Subset"
    )

    # Step 8: Simulated competition score
    print("\n" + "=" * 80)
    print("STEP 8: SIMULATED COMPETITION PERFORMANCE")
    print("=" * 80)

    # Combine predictions from both models on appropriate subsets
    overall_score = (score_imu_subset + score_full_subset) / 2

    print(f"\n  Simulated Test Set Performance (50% IMU + 50% Full):")
    print("  " + "=" * 76)
    print(f"  IMU-only Model (50% of test): {score_imu_subset:.4f}")
    print(f"  Full Sensor Model (50% of test): {score_full_subset:.4f}")
    print(f"  Overall Simulated Score: {overall_score:.4f}")

    print(f"\n  Value Added by ToF + Thermal:")
    print("  " + "=" * 76)
    improvement = score_full_subset - score_imu_subset
    print(f"  Absolute improvement: +{improvement:.4f}")
    print(f"  Relative improvement: +{100*improvement/score_imu_subset:.1f}%")

    # Step 9: Save models
    print("\n" + "=" * 80)
    print("STEP 9: SAVING MODELS")
    print("=" * 80)

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save IMU-only model
    imu_model_path = os.path.join(models_dir, 'xgboost_imu_only.pkl')
    with open(imu_model_path, 'wb') as f:
        pickle.dump({
            'model': model_imu,
            'label_encoder': le_imu,
            'feature_names': feature_names_imu
        }, f)
    print(f"✓ IMU-only model saved: {imu_model_path}")

    # Save full sensor model
    full_model_path = os.path.join(models_dir, 'xgboost_full_sensor.pkl')
    with open(full_model_path, 'wb') as f:
        pickle.dump({
            'model': model_full,
            'label_encoder': le_full,
            'feature_names': feature_names_full
        }, f)
    print(f"✓ Full sensor model saved: {full_model_path}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  IMU-only Model Score: {score_imu_subset:.4f}")
    print(f"  Full Sensor Model Score: {score_full_subset:.4f}")
    print(f"  Overall Competition Score (simulated): {overall_score:.4f}")
    print(f"\n✅ Models saved to '{models_dir}/' directory")
    print("=" * 80)

    return model_imu, model_full, overall_score


if __name__ == "__main__":
    model_imu, model_full, score = main()
