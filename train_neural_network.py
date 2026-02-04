"""
Train Fully-Connected Neural Network (MLP)
==========================================

Uses engineered features from feature_engineering.py.
Simpler and faster than CNN on sequence data.

Usage:
    python train_neural_network.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pickle
from pathlib import Path

# Configuration
RANDOM_SEED = 42
VAL_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("="*80)
print("NEURAL NETWORK TRAINING (MLP on Engineered Features)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("="*80)


class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Multi-layer perceptron for gesture classification."""

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def calculate_competition_score(y_true, y_pred, label_encoder):
    """Calculate (Binary F1 + Macro F1) / 2."""
    bfrb_gestures = [
        'Above ear - pull hair', 'Eyebrow - pull hair', 'Eyelash - pull hair',
        'Forehead - pull hairline', 'Forehead - scratch', 'Cheek - pinch skin',
        'Neck - pinch skin', 'Neck - scratch'
    ]

    gestures_true = label_encoder.inverse_transform(y_true)
    gestures_pred = label_encoder.inverse_transform(y_pred)

    # Binary F1
    y_true_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_true])
    y_pred_binary = np.array([1 if g in bfrb_gestures else 0 for g in gestures_pred])
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)

    # Macro F1
    gestures_true_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_true]
    gestures_pred_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_pred]
    macro_f1 = f1_score(gestures_true_collapsed, gestures_pred_collapsed, average='macro', zero_division=0)

    return (binary_f1 + macro_f1) / 2, binary_f1, macro_f1


def train_model(model, train_loader, val_loader, epochs, label_encoder, model_name):
    """Train the MLP model."""

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

    best_score = 0
    patience_counter = 0
    patience = 15

    print(f"\nTraining {model_name}...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = focal_loss(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = model(X_batch)
                loss = focal_loss(outputs, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        # Calculate score
        score, binary_f1, macro_f1 = calculate_competition_score(
            np.array(all_targets), np.array(all_preds), label_encoder
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={train_loss/len(train_loader):.4f}/{val_loss/len(val_loader):.4f} "
                  f"Score={score:.4f} (Bin={binary_f1:.4f}, Mac={macro_f1:.4f})")

        # Early stopping
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), f'models/{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best score: {best_score:.4f})")
                break

    # Load best model
    model.load_state_dict(torch.load(f'models/{model_name}.pth'))

    return model, best_score


def evaluate_model(model, data_loader, label_encoder, split_name):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    score, binary_f1, macro_f1 = calculate_competition_score(
        np.array(all_targets), np.array(all_preds), label_encoder
    )

    print(f"\n{split_name}:")
    print(f"  Competition Score: {score:.4f}")
    print(f"  Binary F1: {binary_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")

    return score


def main():
    Path('models').mkdir(exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    path = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    print(f"✓ Loaded: {train.shape[0]:,} rows")

    # Extract features
    print("\n[2/5] Extracting features...")
    from feature_engineering import FeatureEngineering
    fe = FeatureEngineering()

    sequences = train[['sequence_id', 'gesture', 'subject']].drop_duplicates()

    # Split sequences
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=VAL_SIZE, stratify=sequences['gesture'], random_state=RANDOM_SEED
    )

    train_data = train[train['sequence_id'].isin(train_sequences['sequence_id'])]
    val_data = train[train['sequence_id'].isin(val_sequences['sequence_id'])]

    # Extract features
    print("  Train set...")
    train_imu_feat = fe.process_dataset(train_data, include_tof_thermal=False)
    train_full_feat = fe.process_dataset(train_data, include_tof_thermal=True)

    print("  Validation set...")
    val_imu_feat = fe.process_dataset(val_data, include_tof_thermal=False)
    val_full_feat = fe.process_dataset(val_data, include_tof_thermal=True)

    print(f"✓ Features extracted: IMU={train_imu_feat.shape[1]-3}, Full={train_full_feat.shape[1]-3}")

    # Prepare data
    print("\n[3/5] Preparing datasets...")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(sequences['gesture'])

    # IMU-only
    X_train_imu = train_imu_feat.drop(['sequence_id', 'gesture', 'subject'], axis=1).values
    X_val_imu = val_imu_feat.drop(['sequence_id', 'gesture', 'subject'], axis=1).values
    y_train = label_encoder.transform(train_imu_feat['gesture'])
    y_val = label_encoder.transform(val_imu_feat['gesture'])

    # Full sensor
    X_train_full = train_full_feat.drop(['sequence_id', 'gesture', 'subject'], axis=1).values
    X_val_full = val_full_feat.drop(['sequence_id', 'gesture', 'subject'], axis=1).values

    # Create datasets
    train_imu_ds = GestureDataset(X_train_imu, y_train)
    val_imu_ds = GestureDataset(X_val_imu, y_val)
    train_full_ds = GestureDataset(X_train_full, y_train)
    val_full_ds = GestureDataset(X_val_full, y_val)

    # Create loaders
    train_imu_loader = DataLoader(train_imu_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_imu_loader = DataLoader(val_imu_ds, batch_size=BATCH_SIZE)
    train_full_loader = DataLoader(train_full_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_full_loader = DataLoader(val_full_ds, batch_size=BATCH_SIZE)

    print(f"✓ Train: {len(train_imu_ds):,} sequences, Val: {len(val_imu_ds):,} sequences")

    # Build models
    print("\n[4/5] Training models...")
    num_classes = len(label_encoder.classes_)

    # IMU-only model
    print("\n[4.1] IMU-only Neural Network")
    model_imu = MLP(X_train_imu.shape[1], num_classes).to(DEVICE)
    model_imu, score_imu = train_model(
        model_imu, train_imu_loader, val_imu_loader,
        EPOCHS, label_encoder, 'nn_imu'
    )

    # Full sensor model
    print("\n[4.2] Full Sensor Neural Network")
    model_full = MLP(X_train_full.shape[1], num_classes).to(DEVICE)
    model_full, score_full = train_model(
        model_full, train_full_loader, val_full_loader,
        EPOCHS, label_encoder, 'nn_full'
    )

    # Final evaluation
    print("\n" + "="*80)
    print("[5/5] FINAL EVALUATION")
    print("="*80)

    score_imu_final = evaluate_model(model_imu, val_imu_loader, label_encoder, "IMU-only Model")
    score_full_final = evaluate_model(model_full, val_full_loader, label_encoder, "Full Sensor Model")

    overall_score = (score_imu_final + score_full_final) / 2

    print(f"\n{'='*80}")
    print("COMPARISON: XGBoost vs Neural Network")
    print(f"{'='*80}")
    print(f"{'Model':<20} | {'XGBoost':<10} | {'Neural Net':<10} | {'Difference':<10}")
    print(f"{'-'*80}")
    print(f"{'IMU-only':<20} | {0.6864:<10.4f} | {score_imu_final:<10.4f} | {(score_imu_final-0.6864):+.4f}")
    print(f"{'Full Sensor':<20} | {0.7838:<10.4f} | {score_full_final:<10.4f} | {(score_full_final-0.7838):+.4f}")
    print(f"{'Overall (50/50)':<20} | {0.7351:<10.4f} | {overall_score:<10.4f} | {(overall_score-0.7351):+.4f}")
    print(f"{'='*80}")

    if overall_score > 0.7351:
        print(f"\n✅ Neural Network BEATS XGBoost by {(overall_score-0.7351):.4f}!")
    elif overall_score > 0.73:
        print(f"\n⚖️  Neural Network comparable to XGBoost (within 0.01)")
    else:
        print(f"\n⚠️  XGBoost still better - stick with tree-based models")

    # Save metadata
    with open('models/nn_metadata.pkl', 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'feature_names_imu': train_imu_feat.columns.tolist()[3:],
            'feature_names_full': train_full_feat.columns.tolist()[3:]
        }, f)

    print("\n✓ Models saved:")
    print("  - models/nn_imu.pth")
    print("  - models/nn_full.pth")
    print("  - models/nn_metadata.pkl")


if __name__ == "__main__":
    main()
