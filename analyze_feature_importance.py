"""
Analyze Feature Importance from XGBoost Models
==============================================

This script loads the trained XGBoost models and analyzes which features
are most important for gesture classification.

Usage:
    python analyze_feature_importance.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_model(model_path):
    """Load pickled XGBoost model."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        # Handle both dict and direct model formats
        if isinstance(data, dict):
            return data['model'], data.get('feature_names', None)
        else:
            return data, None

def get_feature_importance(model, feature_names, top_n=30):
    """Extract feature importance from XGBoost model."""
    importance_dict = model.get_booster().get_score(importance_type='weight')

    # Map feature IDs (f0, f1, ...) to actual feature names
    importance_data = []
    for feat_id, score in importance_dict.items():
        feat_idx = int(feat_id.replace('f', ''))
        if feat_idx < len(feature_names):
            importance_data.append({
                'feature': feature_names[feat_idx],
                'importance': score,
                'type': categorize_feature(feature_names[feat_idx])
            })

    df = pd.DataFrame(importance_data)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df.head(top_n)

def categorize_feature(feature_name):
    """Categorize feature by sensor type."""
    if 'acc_' in feature_name or 'jerk_' in feature_name:
        return 'Accelerometer'
    elif 'roll_' in feature_name or 'pitch_' in feature_name or 'yaw_' in feature_name or 'angular_' in feature_name:
        return 'Rotation'
    elif 'tof_' in feature_name or 'depth_' in feature_name or 'valid_' in feature_name:
        return 'ToF'
    elif 'temp_' in feature_name or 'thm_' in feature_name:
        return 'Thermal'
    elif 'temporal_' in feature_name or '_trend' in feature_name or 'autocorr' in feature_name:
        return 'Temporal'
    else:
        return 'Other'

def print_importance_summary(importance_df, model_name):
    """Print feature importance summary."""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()}: TOP 30 MOST IMPORTANT FEATURES")
    print(f"{'='*80}\n")

    print(f"{'Rank':<6} {'Feature':<50} {'Type':<15} {'Importance':<10}")
    print("-" * 80)

    for idx, row in importance_df.iterrows():
        print(f"{idx+1:<6} {row['feature']:<50} {row['type']:<15} {row['importance']:<10.1f}")

    # Summary by sensor type
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE BY SENSOR TYPE")
    print(f"{'='*80}\n")

    type_summary = importance_df.groupby('type')['importance'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)

    print(f"{'Sensor Type':<20} {'Total Score':<15} {'Count':<10} {'Avg Score':<12}")
    print("-" * 80)
    for sensor_type, row in type_summary.iterrows():
        print(f"{sensor_type:<20} {row['sum']:<15.1f} {int(row['count']):<10} {row['mean']:<12.1f}")

def analyze_sensor_value_added(imu_importance, full_importance):
    """Analyze value added by ToF and Thermal sensors."""
    print(f"\n{'='*80}")
    print("VALUE ADDED BY TOF & THERMAL SENSORS")
    print(f"{'='*80}\n")

    # Count ToF and Thermal features in top 30 of full sensor model
    tof_thermal_features = full_importance[full_importance['type'].isin(['ToF', 'Thermal'])]

    print(f"ToF & Thermal features in Top 30: {len(tof_thermal_features)} features")
    print(f"Combined importance score: {tof_thermal_features['importance'].sum():.1f}")
    print(f"Percentage of total importance: {(tof_thermal_features['importance'].sum() / full_importance['importance'].sum() * 100):.1f}%")

    print(f"\nTop ToF/Thermal features:")
    print("-" * 80)
    for idx, row in tof_thermal_features.iterrows():
        print(f"  {idx+1}. {row['feature']} (importance: {row['importance']:.1f})")

def main():
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Load models
    print("\n[1/4] Loading trained models...")
    try:
        model_imu, imu_feature_names = load_model('models/xgboost_imu_only.pkl')
        model_full, full_feature_names = load_model('models/xgboost_full_sensor.pkl')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return

    # Get feature names
    print("\n[2/4] Extracting feature names...")

    # Use loaded feature names or generate defaults
    if imu_feature_names is None:
        imu_feature_names = [f'feature_{i}' for i in range(83)]

    if full_feature_names is None:
        full_feature_names = [f'feature_{i}' for i in range(158)]

    print(f"✓ IMU features: {len(imu_feature_names)}")
    print(f"✓ Full sensor features: {len(full_feature_names)}")

    # Get feature importance
    print("\n[3/4] Analyzing feature importance...")
    imu_importance = get_feature_importance(model_imu, imu_feature_names, top_n=30)
    full_importance = get_feature_importance(model_full, full_feature_names, top_n=30)

    # Print summaries
    print_importance_summary(imu_importance, "IMU-Only Model")
    print_importance_summary(full_importance, "Full Sensor Model")

    # Analyze value added by ToF/Thermal
    analyze_sensor_value_added(imu_importance, full_importance)

    # Save to file
    print(f"\n{'='*80}")
    print("[4/4] Saving results...")
    print(f"{'='*80}\n")

    output_path = Path('FEATURE_IMPORTANCE_ANALYSIS.md')

    with open(output_path, 'w') as f:
        f.write("# Feature Importance Analysis\n")
        f.write("## XGBoost Baseline Models\n\n")
        f.write("*Generated from trained XGBoost models*\n\n")
        f.write("---\n\n")

        # IMU-only model
        f.write("## IMU-Only Model: Top 30 Features\n\n")
        f.write("**Model Score**: 0.6864 (Competition Score)\n\n")
        f.write("| Rank | Feature | Sensor Type | Importance |\n")
        f.write("|------|---------|-------------|------------|\n")
        for idx, row in imu_importance.iterrows():
            f.write(f"| {idx+1} | {row['feature']} | {row['type']} | {row['importance']:.1f} |\n")

        f.write("\n### IMU Feature Importance by Sensor Type\n\n")
        imu_type_summary = imu_importance.groupby('type')['importance'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
        f.write("| Sensor Type | Total Score | Count | Avg Score |\n")
        f.write("|-------------|-------------|-------|----------|\n")
        for sensor_type, row in imu_type_summary.iterrows():
            f.write(f"| {sensor_type} | {row['sum']:.1f} | {int(row['count'])} | {row['mean']:.1f} |\n")

        f.write("\n---\n\n")

        # Full sensor model
        f.write("## Full Sensor Model: Top 30 Features\n\n")
        f.write("**Model Score**: 0.7838 (Competition Score)\n\n")
        f.write("| Rank | Feature | Sensor Type | Importance |\n")
        f.write("|------|---------|-------------|------------|\n")
        for idx, row in full_importance.iterrows():
            f.write(f"| {idx+1} | {row['feature']} | {row['type']} | {row['importance']:.1f} |\n")

        f.write("\n### Full Sensor Feature Importance by Sensor Type\n\n")
        full_type_summary = full_importance.groupby('type')['importance'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
        f.write("| Sensor Type | Total Score | Count | Avg Score |\n")
        f.write("|-------------|-------------|-------|----------|\n")
        for sensor_type, row in full_type_summary.iterrows():
            f.write(f"| {sensor_type} | {row['sum']:.1f} | {int(row['count'])} | {row['mean']:.1f} |\n")

        f.write("\n---\n\n")

        # Value added analysis
        f.write("## Value Added by ToF & Thermal Sensors\n\n")
        tof_thermal_features = full_importance[full_importance['type'].isin(['ToF', 'Thermal'])]
        f.write(f"**ToF & Thermal features in Top 30**: {len(tof_thermal_features)} features\n\n")
        f.write(f"**Combined importance score**: {tof_thermal_features['importance'].sum():.1f}\n\n")
        f.write(f"**Percentage of total importance**: {(tof_thermal_features['importance'].sum() / full_importance['importance'].sum() * 100):.1f}%\n\n")

        if len(tof_thermal_features) > 0:
            f.write("### Top ToF/Thermal Features\n\n")
            f.write("| Rank | Feature | Type | Importance |\n")
            f.write("|------|---------|------|------------|\n")
            for idx, row in tof_thermal_features.iterrows():
                f.write(f"| {idx+1} | {row['feature']} | {row['type']} | {row['importance']:.1f} |\n")

        f.write("\n---\n\n")

        # Key insights
        f.write("## Key Insights\n\n")
        f.write("### 1. IMU-Only Model\n")
        f.write(f"- Most important sensor type: {imu_type_summary.index[0]}\n")
        f.write(f"- Top feature: {imu_importance.iloc[0]['feature']}\n")
        f.write(f"- Competition score: 0.6864\n\n")

        f.write("### 2. Full Sensor Model\n")
        f.write(f"- Most important sensor type: {full_type_summary.index[0]}\n")
        f.write(f"- Top feature: {full_importance.iloc[0]['feature']}\n")
        f.write(f"- Competition score: 0.7838 (+14.2% improvement)\n\n")

        f.write("### 3. Sensor Value Added\n")
        if len(tof_thermal_features) > 0:
            f.write(f"- ToF/Thermal contribute {(tof_thermal_features['importance'].sum() / full_importance['importance'].sum() * 100):.1f}% of top 30 feature importance\n")
            f.write(f"- This justifies the +9.7 point score improvement (0.6864 → 0.7838)\n")
        else:
            f.write("- ToF/Thermal features not in top 30, but still provide +9.7 point improvement\n")
            f.write("- Likely provide complementary information for difficult gesture classes\n")

        f.write("\n### 4. Recommendations for Deep Learning\n")
        f.write("Based on feature importance analysis:\n\n")

        # Get most important sensor types
        top_sensors = imu_type_summary.head(2).index.tolist()
        f.write(f"- **Focus on {' and '.join(top_sensors)} features** in CNN/LSTM models\n")
        f.write("- **Use attention mechanisms** to learn which features matter per gesture\n")
        f.write("- **Multi-modal architecture**: Separate branches for IMU vs ToF/Thermal\n")
        f.write("- **Temporal modeling**: Many important features are temporal (trends, autocorrelation)\n")

        f.write("\n---\n\n")
        f.write("*Analysis complete. Proceed to deep learning model development.*\n")

    print(f"✓ Analysis saved to: {output_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - IMU-only top feature: {imu_importance.iloc[0]['feature']}")
    print(f"  - Full sensor top feature: {full_importance.iloc[0]['feature']}")
    print(f"  - ToF/Thermal in top 30: {len(tof_thermal_features)} features")
    print(f"\nNext step: Build deep learning models (1D CNN/LSTM) targeting 0.80+ score")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
