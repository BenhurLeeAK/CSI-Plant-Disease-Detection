"""
ADVANCED STATISTICAL FEATURES EXTRACTION
Fixed: Saves output to csi_plant_data directory
"""

import numpy as np
import pandas as pd
import re
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set paths
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'
output_path = data_path  # Save to same directory

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def load_csi_data(filename, label, max_packets=None):
    """Load CSI data from text files"""
    full_path = f"{data_path}/{filename}"
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    pattern = r'\[CSI #(\d+)\].*?RSSI:(-?\d+)\nAmp:\s*([\d\.\s]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    features = []
    for match in matches:
        rssi = int(match[1])
        amp_str = match[2].strip()
        amp_list = [float(x) for x in amp_str.split()]
        
        if len(amp_list) >= 64:
            amp_64 = np.array(amp_list[:64])
            
            # Basic features
            feat = {
                'label': label,
                'rssi': rssi,
                'mean_amp': np.mean(amp_64),
                'std_amp': np.std(amp_64),
                'max_amp': np.max(amp_64),
                'min_amp': np.min(amp_64),
                'median_amp': np.median(amp_64),
            }
            
            # ========== ADVANCED STATISTICAL FEATURES ==========
            
            # Higher-order moments
            feat['skewness'] = stats.skew(amp_64)
            feat['kurtosis'] = stats.kurtosis(amp_64)
            
            # Variance ratio
            feat['variance_ratio'] = np.var(amp_64) / (np.mean(amp_64) + 1e-6)
            
            # Absolute deviations
            feat['mean_absolute_deviation'] = np.mean(np.abs(amp_64 - np.mean(amp_64)))
            feat['median_absolute_deviation'] = np.median(np.abs(amp_64 - np.median(amp_64)))
            
            # Percentile-based features
            feat['q1'] = np.percentile(amp_64, 25)
            feat['q3'] = np.percentile(amp_64, 75)
            feat['iqr'] = feat['q3'] - feat['q1']
            feat['percentile_95'] = np.percentile(amp_64, 95)
            feat['percentile_5'] = np.percentile(amp_64, 5)
            feat['percentile_range_95'] = feat['percentile_95'] - feat['percentile_5']
            
            # Shape statistics
            feat['harmonic_mean'] = stats.hmean(amp_64 + 1e-6)
            feat['geometric_mean'] = stats.gmean(amp_64 + 1e-6)
            
            # Zero-crossing rate
            feat['zero_crossing_rate'] = np.sum(np.diff(np.signbit(amp_64)))
            
            # Signal smoothness - autocorrelation peak
            autocorr = np.correlate(amp_64, amp_64, mode='same')
            feat['autocorrelation_peak'] = np.max(autocorr) / len(amp_64)
            
            # Energy features
            feat['total_energy'] = np.sum(amp_64 ** 2)
            feat['rms'] = np.sqrt(np.mean(amp_64 ** 2))
            
            # Crest factor (peak-to-RMS ratio)
            feat['crest_factor'] = np.max(amp_64) / (feat['rms'] + 1e-6)
            
            # Shape factor (RMS / mean absolute)
            feat['shape_factor'] = feat['rms'] / (np.mean(np.abs(amp_64)) + 1e-6)
            
            # Impulse factor (peak / mean absolute)
            feat['impulse_factor'] = np.max(amp_64) / (np.mean(np.abs(amp_64)) + 1e-6)
            
            # Margin factor
            feat['margin_factor'] = np.max(amp_64) / (np.mean(np.sqrt(np.abs(amp_64))) + 1e-6)
            
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

print("="*80)
print("ADVANCED STATISTICAL FEATURES EXTRACTION")
print("="*80)

# Load datasets
print("\n📂 Loading data...")
baseline_df = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
healthy_df = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
diseased_df = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

# Combine
all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)

print(f"Total samples: {len(all_data)}")
print(f"Features extracted: {len([c for c in all_data.columns if c != 'label'])}")

# Summary statistics
print("\n📊 Feature Statistics by Condition:")
feature_cols = [c for c in all_data.columns if c != 'label']

for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    print(f"\n{name} (n={len(subset)}):")
    for feat in ['mean_amp', 'std_amp', 'skewness', 'kurtosis', 'iqr', 'crest_factor']:
        if feat in subset.columns:
            print(f"  {feat}: {subset[feat].mean():.3f} ± {subset[feat].std():.3f}")

# Save to CSV in the correct directory
output_file = os.path.join(output_path, 'advanced_statistical_features.csv')
all_data.to_csv(output_file, index=False)
print(f"\n✅ Saved to '{output_file}'")

# Also save a summary CSV
summary_file = os.path.join(output_path, 'advanced_statistical_features_summary.csv')
summary_data = []
for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    row = {'Condition': name, 'Count': len(subset)}
    for feat in ['mean_amp', 'std_amp', 'skewness', 'kurtosis', 'iqr', 'crest_factor', 'rssi']:
        if feat in subset.columns:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_path, 'advanced_statistical_features_summary.csv'), index=False)
print(f"✅ Saved summary to '{os.path.join(output_path, 'advanced_statistical_features_summary.csv')}'")

# Feature correlation with labels
print("\n📈 Feature Correlation with Disease Label:")
correlations = []
for col in feature_cols:
    if col in all_data.columns and not pd.isna(all_data[col]).all():
        corr = all_data[col].corr(all_data['label'])
        if not pd.isna(corr):
            correlations.append((col, abs(corr)))
    
# Top 10 most correlated features
top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:10]
print("Top 10 features correlated with disease label:")
for i, (feat, corr) in enumerate(top_features, 1):
    print(f"  {i}. {feat}: {corr:.4f}")