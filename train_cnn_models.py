"""
Train 1D CNN Models for Gesture Classification
==============================================

This script trains 1D CNN models on raw time series data for gesture classification.
Uses dual model strategy (IMU-only + Full sensor) to match test set conditions.

Architecture based on feature importance findings:
- Temporal modeling for sequences
- Separate branches for different sensor modalities
- Focal loss to handle class imbalance

Usage:
    python train_cnn_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pickle
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configuration
VAL_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

print("="*80)
print("1D CNN TRAINING: Dual Strategy (IMU-only + Full Sensor)")
print("="*80)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Validation Size: {int(VAL_SIZE * 100)}%")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Epochs: {EPOCHS}")
print("="*80)


def load_and_prepare_data():
    """Load training data and prepare for CNN."""
    print("\n" + "="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)

    # Load data
    path = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
    train = pd.read_csv(os.path.join(path, 'train.csv'))

    print(f"✓ Loaded training data: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Unique sequences: {train['sequence_id'].nunique():,}")
    print(f"  Gesture classes: {train['gesture'].nunique()}")

    return train


def create_sequences(df, max_length=None, pad_value=-999):
    """
    Convert dataframe to 3D array (samples, timesteps, features).

    Args:
        df: DataFrame with sequence_id, gesture, and sensor columns
        max_length: Maximum sequence length (for padding/truncation)
        pad_value: Value to use for padding

    Returns:
        X: 3D numpy array (n_sequences, max_length, n_features)
        y: 1D numpy array (n_sequences,)
        sequence_ids: List of sequence IDs
        feature_cols: List of feature column names
    """
    print("\n[Creating sequences from raw time series data...]")

    # Define feature columns
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    thm_cols = [col for col in df.columns if col.startswith('thm_')]

    feature_cols = imu_cols + tof_cols + thm_cols

    # Get unique sequences
    sequence_ids = df['sequence_id'].unique()

    # Determine max length
    if max_length is None:
        max_length = df.groupby('sequence_id').size().max()
        print(f"  Auto-detected max sequence length: {max_length}")
    else:
        print(f"  Using fixed max length: {max_length}")

    # Initialize arrays
    n_sequences = len(sequence_ids)
    n_features = len(feature_cols)

    X = np.full((n_sequences, max_length, n_features), pad_value, dtype=np.float32)
    y = np.zeros(n_sequences, dtype=object)

    print(f"  Processing {n_sequences:,} sequences...")

    # Fill sequences
    for idx, seq_id in enumerate(sequence_ids):
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1:,} sequences...")

        seq_data = df[df['sequence_id'] == seq_id]
        seq_features = seq_data[feature_cols].values

        # Handle variable length sequences
        seq_len = min(len(seq_features), max_length)
        X[idx, :seq_len, :] = seq_features[:seq_len]

        # Store label
        y[idx] = seq_data['gesture'].iloc[0]

    print(f"✓ Created sequence array: {X.shape}")
    print(f"  Shape: (sequences={n_sequences}, timesteps={max_length}, features={n_features})")

    return X, y, sequence_ids, feature_cols


def create_imu_only_sequences(X_full, feature_cols):
    """Extract only IMU features from full sequence data."""
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    imu_indices = [i for i, col in enumerate(feature_cols) if col in imu_cols]

    X_imu = X_full[:, :, imu_indices]
    print(f"✓ Extracted IMU-only sequences: {X_imu.shape}")

    return X_imu, [feature_cols[i] for i in imu_indices]


def create_masking_layer(pad_value=-999):
    """Create masking layer to handle variable-length sequences."""
    return layers.Masking(mask_value=pad_value)


def build_1d_cnn_model(input_shape, num_classes, model_name="1D_CNN"):
    """
    Build 1D CNN model for time series classification.

    Architecture:
    - Conv1D layers to extract temporal patterns
    - MaxPooling for dimensionality reduction
    - Batch normalization for stable training
    - Dropout for regularization
    - Dense layers for classification
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # Masking layer for variable-length sequences
    x = create_masking_layer()(inputs)

    # First convolutional block
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Second convolutional block
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Third convolutional block
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.4)(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance.
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    """
    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy
        ce = -y_true * keras.backend.log(y_pred)

        # Focal term
        focal_term = keras.backend.pow(1.0 - y_pred, gamma)

        # Combine
        loss = alpha * focal_term * ce

        return keras.backend.sum(loss, axis=-1)

    return loss_fn


def calculate_competition_score(y_true, y_pred, label_encoder):
    """
    Calculate competition metric: (Binary F1 + Macro F1) / 2

    Binary: BFRB vs non-BFRB
    Macro: 9 classes (8 BFRB + 1 collapsed non-BFRB)
    """
    # Define BFRB gestures
    bfrb_gestures = [
        'Above ear - pull hair', 'Eyebrow - pull hair', 'Eyelash - pull hair',
        'Forehead - pull hairline', 'Forehead - scratch', 'Cheek - pinch skin',
        'Neck - pinch skin', 'Neck - scratch'
    ]

    # Get gesture names
    gestures_true = label_encoder.inverse_transform(y_true)
    gestures_pred = label_encoder.inverse_transform(y_pred)

    # Binary F1: BFRB vs non-BFRB
    y_true_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_true])
    y_pred_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_pred])
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')

    # Macro F1: Collapse non-BFRB into single class
    gestures_true_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_true]
    gestures_pred_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_pred]
    macro_f1 = f1_score(gestures_true_collapsed, gestures_pred_collapsed, average='macro')

    # Competition score
    competition_score = (binary_f1 + macro_f1) / 2

    return competition_score, binary_f1, macro_f1


def train_model(model, X_train, y_train, X_val, y_val, class_weights, model_path):
    """Train CNN model with focal loss and early stopping."""

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X, y, label_encoder, split_name="Validation"):
    """Evaluate model and print detailed metrics."""
    print(f"\n  Evaluating on {split_name} set:")
    print("  " + "-" * 76)

    # Predictions
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate competition score
    comp_score, binary_f1, macro_f1 = calculate_competition_score(y, y_pred, label_encoder)

    print(f"  Competition Score: {comp_score:.4f}")
    print(f"    - Binary F1 (BFRB vs. non-BFRB): {binary_f1:.4f}")
    print(f"    - Macro F1 (9 classes): {macro_f1:.4f}")

    # Per-gesture F1 scores
    gestures = label_encoder.inverse_transform(y)
    gestures_pred = label_encoder.inverse_transform(y_pred)

    print(f"\n  Per-Gesture F1 Scores:")
    print("  " + "-" * 76)

    unique_gestures = sorted(set(gestures))
    gesture_f1_scores = []

    for gesture in unique_gestures:
        mask = np.array([g == gesture for g in gestures])
        if mask.sum() > 0:
            y_true_binary = mask.astype(int)
            y_pred_binary = np.array([g == gesture for g in gestures_pred]).astype(int)
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            gesture_f1_scores.append((gesture, f1))

    # Sort by F1 score
    gesture_f1_scores.sort(key=lambda x: x[1])

    for gesture, f1 in gesture_f1_scores:
        status = "✓" if f1 >= 0.5 else "⚠️"
        print(f"  {status} {gesture:<50} : {f1:.4f}")

    # Overall accuracy
    accuracy = (y == y_pred).mean()
    print(f"\n  Overall Accuracy: {accuracy:.4f}")

    return comp_score


def main():
    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Load data
    train = load_and_prepare_data()

    # Create sequences (with reasonable max length)
    print("\n" + "="*80)
    print("STEP 2: CREATING SEQUENCES")
    print("="*80)

    # Use median + 2*IQR as max length (covers ~95% of sequences)
    seq_lengths = train.groupby('sequence_id').size()
    median_len = seq_lengths.median()
    iqr = seq_lengths.quantile(0.75) - seq_lengths.quantile(0.25)
    max_length = int(median_len + 2 * iqr)

    print(f"  Sequence length statistics:")
    print(f"    Min: {seq_lengths.min()}, Max: {seq_lengths.max()}")
    print(f"    Median: {median_len:.0f}, IQR: {iqr:.0f}")
    print(f"  Using max_length: {max_length} (covers ~95% of sequences)")

    X_full, y, sequence_ids, feature_cols = create_sequences(train, max_length=max_length)

    # Create IMU-only version
    X_imu, imu_feature_cols = create_imu_only_sequences(X_full, feature_cols)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    print(f"✓ Encoded {num_classes} gesture classes")

    # Convert to categorical
    y_categorical = keras.utils.to_categorical(y_encoded, num_classes)

    # Train/val split (stratified by gesture)
    print("\n" + "="*80)
    print("STEP 3: CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)

    # Split IMU-only
    X_imu_train, X_imu_val, y_train, y_val, seq_train, seq_val = train_test_split(
        X_imu, y_categorical, sequence_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    # Split full sensor (using same indices)
    train_indices = [i for i, s in enumerate(sequence_ids) if s in seq_train]
    val_indices = [i for i, s in enumerate(sequence_ids) if s in seq_val]

    X_full_train = X_full[train_indices]
    X_full_val = X_full[val_indices]

    print(f"✓ Train sequences: {len(seq_train):,} ({(1-VAL_SIZE)*100:.0f}%)")
    print(f"✓ Validation sequences: {len(seq_val):,} ({VAL_SIZE*100:.0f}%)")

    # Calculate class weights
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.arange(num_classes)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train_labels)
    class_weights = {i: w for i, w in enumerate(class_weights_array)}

    print(f"\n✓ Computed class weights for {num_classes} classes")

    # Build models
    print("\n" + "="*80)
    print("STEP 4: BUILDING CNN MODELS")
    print("="*80)

    print("\n[4.1] Building IMU-only 1D CNN...")
    model_imu = build_1d_cnn_model(
        input_shape=(max_length, X_imu.shape[2]),
        num_classes=num_classes,
        model_name="CNN_IMU_Only"
    )
    model_imu.summary()

    print("\n[4.2] Building Full Sensor 1D CNN...")
    model_full = build_1d_cnn_model(
        input_shape=(max_length, X_full.shape[2]),
        num_classes=num_classes,
        model_name="CNN_Full_Sensor"
    )

    # Train models
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODELS")
    print("="*80)

    print("\n[5.1] Training IMU-only CNN...")
    model_imu, history_imu = train_model(
        model_imu, X_imu_train, y_train, X_imu_val, y_val,
        class_weights, 'models/cnn_imu_only.keras'
    )

    print("\n[5.2] Training Full Sensor CNN...")
    model_full, history_full = train_model(
        model_full, X_full_train, y_train, X_full_val, y_val,
        class_weights, 'models/cnn_full_sensor.keras'
    )

    # Evaluate models
    print("\n" + "="*80)
    print("STEP 6: MODEL EVALUATION")
    print("="*80)

    # Decode validation labels for evaluation
    y_val_labels = np.argmax(y_val, axis=1)

    print("\n[6.1] IMU-only Model Performance")
    print("="*80)
    score_imu = evaluate_model(model_imu, X_imu_val, y_val_labels, label_encoder, "Validation")

    print("\n[6.2] Full Sensor Model Performance")
    print("="*80)
    score_full = evaluate_model(model_full, X_full_val, y_val_labels, label_encoder, "Validation")

    # Simulated competition performance
    print("\n" + "="*80)
    print("STEP 7: SIMULATED COMPETITION PERFORMANCE")
    print("="*80)

    overall_score = (score_imu + score_full) / 2

    print(f"\n  Simulated Test Set Performance (50% IMU + 50% Full):")
    print("  " + "="*76)
    print(f"  IMU-only Model (50% of test): {score_imu:.4f}")
    print(f"  Full Sensor Model (50% of test): {score_full:.4f}")
    print(f"  Overall Simulated Score: {overall_score:.4f}")

    # Compare with XGBoost baseline
    print(f"\n  Comparison with XGBoost Baseline:")
    print("  " + "="*76)
    print(f"  XGBoost overall: 0.7351")
    print(f"  CNN overall: {overall_score:.4f}")
    print(f"  Improvement: {(overall_score - 0.7351):.4f} ({((overall_score - 0.7351) / 0.7351 * 100):+.1f}%)")

    # Save metadata
    print("\n" + "="*80)
    print("STEP 8: SAVING MODELS AND METADATA")
    print("="*80)

    # Save label encoder and feature info
    with open('models/cnn_label_encoder.pkl', 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'imu_features': imu_feature_cols,
            'full_features': feature_cols,
            'max_length': max_length
        }, f)

    print("✓ Saved label encoder and metadata")
    print("✓ Models saved as Keras format (.keras files)")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"  IMU-only CNN Score: {score_imu:.4f}")
    print(f"  Full Sensor CNN Score: {score_full:.4f}")
    print(f"  Overall Competition Score: {overall_score:.4f}")
    print(f"\n✅ Models saved to 'models/' directory")
    print("="*80)


if __name__ == "__main__":
    main()
