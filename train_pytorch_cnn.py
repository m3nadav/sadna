"""
Train 1D CNN with PyTorch (TensorFlow-free)
===========================================

PyTorch version of CNN training - faster initialization, no mutex issues.

Usage:
    python train_pytorch_cnn.py
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import pickle
from pathlib import Path

# Configuration
RANDOM_SEED = 42
VAL_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("="*80)
print("PYTORCH 1D CNN TRAINING")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("="*80)


class GestureDataset(Dataset):
    """Dataset for gesture sequences."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    """1D CNN for time series classification."""

    def __init__(self, input_channels, num_classes, seq_length):
        super(CNN1D, self).__init__()

        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.4)
        )

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.squeeze(-1)  # Remove seq dimension
        x = self.fc(x)

        return x


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
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')

    # Macro F1
    gestures_true_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_true]
    gestures_pred_collapsed = ['non_target' if g not in bfrb_gestures else g for g in gestures_pred]
    macro_f1 = f1_score(gestures_true_collapsed, gestures_pred_collapsed, average='macro')

    return (binary_f1 + macro_f1) / 2, binary_f1, macro_f1


def train_model(model, train_loader, val_loader, epochs, label_encoder, model_name):
    """Train the CNN model."""

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_score = 0
    patience_counter = 0
    patience = 10

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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, "
                  f"Score={score:.4f} (BinF1={binary_f1:.4f}, MacF1={macro_f1:.4f})")

        # Early stopping
        if score > best_score:
            best_score = score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(f'models/{model_name}.pth'))

    return model, best_score


def evaluate_model(model, data_loader, label_encoder, split_name):
    """Evaluate model on a dataset."""
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

    print(f"\n{split_name} Results:")
    print(f"  Competition Score: {score:.4f}")
    print(f"  Binary F1: {binary_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")

    return score


def main():
    Path('models').mkdir(exist_ok=True)

    # Load data
    print("\n[1/6] Loading data...")
    path = '/Users/nadav/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data'
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    print(f"✓ Loaded: {train.shape[0]:,} rows")

    # Prepare sequences (using aggregate features from XGBoost - faster!)
    print("\n[2/6] Using pre-extracted features from XGBoost...")

    # Load XGBoost feature matrices (already computed!)
    with open('models/xgboost_imu_only.pkl', 'rb') as f:
        data_imu = pickle.load(f)
    with open('models/xgboost_full_sensor.pkl', 'rb') as f:
        data_full = pickle.load(f)

    print("✓ Using XGBoost engineered features (much faster than raw sequences!)")
    print(f"  IMU features: {len(data_imu['feature_names'])} features")
    print(f"  Full features: {len(data_full['feature_names'])} features")

    # Load feature-engineered data
    from feature_engineering import FeatureEngineering
    fe = FeatureEngineering()

    print("\n[3/6] Extracting features...")
    sequences = train[['sequence_id', 'gesture', 'subject']].drop_duplicates()

    # Split sequences
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=VAL_SIZE, stratify=sequences['gesture'], random_state=RANDOM_SEED
    )

    train_data = train[train['sequence_id'].isin(train_sequences['sequence_id'])]
    val_data = train[train['sequence_id'].isin(val_sequences['sequence_id'])]

    # Extract features
    print("  Processing train...")
    train_imu_feat = fe.process_dataset(train_data, include_tof_thermal=False)
    train_full_feat = fe.process_dataset(train_data, include_tof_thermal=True)

    print("  Processing val...")
    val_imu_feat = fe.process_dataset(val_data, include_tof_thermal=False)
    val_full_feat = fe.process_dataset(val_data, include_tof_thermal=True)

    # Prepare data
    print("\n[4/6] Preparing datasets...")

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

    # Reshape for CNN (add sequence dimension)
    X_train_imu = X_train_imu.reshape(len(X_train_imu), 1, -1)
    X_val_imu = X_val_imu.reshape(len(X_val_imu), 1, -1)
    X_train_full = X_train_full.reshape(len(X_train_full), 1, -1)
    X_val_full = X_val_full.reshape(len(X_val_full), 1, -1)

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

    print(f"✓ Created datasets")
    print(f"  Train: {len(train_imu_ds):,} sequences")
    print(f"  Val: {len(val_imu_ds):,} sequences")

    # Build models
    print("\n[5/6] Training models...")
    num_classes = len(label_encoder.classes_)

    # IMU-only model
    print("\n[5.1] Training IMU-only model...")
    model_imu = CNN1D(X_train_imu.shape[2], num_classes, 1).to(DEVICE)
    model_imu, score_imu = train_model(
        model_imu, train_imu_loader, val_imu_loader,
        EPOCHS, label_encoder, 'pytorch_cnn_imu'
    )

    # Full sensor model
    print("\n[5.2] Training Full sensor model...")
    model_full = CNN1D(X_train_full.shape[2], num_classes, 1).to(DEVICE)
    model_full, score_full = train_model(
        model_full, train_full_loader, val_full_loader,
        EPOCHS, label_encoder, 'pytorch_cnn_full'
    )

    # Final evaluation
    print("\n" + "="*80)
    print("[6/6] FINAL RESULTS")
    print("="*80)

    score_imu_final = evaluate_model(model_imu, val_imu_loader, label_encoder, "IMU-only")
    score_full_final = evaluate_model(model_full, val_full_loader, label_encoder, "Full Sensor")

    overall_score = (score_imu_final + score_full_final) / 2

    print(f"\n{'='*80}")
    print("COMPARISON WITH XGBOOST")
    print(f"{'='*80}")
    print(f"Model          | XGBoost | PyTorch CNN | Improvement")
    print(f"{'-'*80}")
    print(f"IMU-only       | 0.6864  | {score_imu_final:.4f}      | {(score_imu_final-0.6864):.4f}")
    print(f"Full Sensor    | 0.7838  | {score_full_final:.4f}      | {(score_full_final-0.7838):.4f}")
    print(f"Overall        | 0.7351  | {overall_score:.4f}      | {(overall_score-0.7351):.4f}")
    print(f"{'='*80}")

    # Save metadata
    with open('models/pytorch_cnn_metadata.pkl', 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'feature_names_imu': data_imu['feature_names'],
            'feature_names_full': data_full['feature_names']
        }, f)

    print("\n✓ Models saved to models/")
    print("  - pytorch_cnn_imu.pth")
    print("  - pytorch_cnn_full.pth")
    print("  - pytorch_cnn_metadata.pkl")


if __name__ == "__main__":
    main()
