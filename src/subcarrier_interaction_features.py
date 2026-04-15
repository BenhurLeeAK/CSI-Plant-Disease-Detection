"""
SUBCARRIER INTERACTION FEATURES EXTRACTION
Extracts subcarrier differences, ratios, symmetry, and peak detection features
Saves output to csi_plant_data directory
"""

import numpy as np
import pandas as pd
import re
import os
from scipy import stats
from scipy.signal import find_peaks
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
            
            feat = {
                'label': label,
                'rssi': rssi,
                'mean_amp': np.mean(amp_64),
                'std_amp': np.std(amp_64),
            }
            
            # ========== ADJACENT SUBCARRIER DIFFERENCES ==========
            diff = np.diff(amp_64)
            feat['diff_mean'] = np.mean(diff)
            feat['diff_std'] = np.std(diff)
            feat['diff_max'] = np.max(diff)
            feat['diff_min'] = np.min(diff)
            feat['diff_abs_mean'] = np.mean(np.abs(diff))
            feat['diff_range'] = np.max(diff) - np.min(diff)
            
            # ========== SECOND-ORDER DIFFERENCES ==========
            diff2 = np.diff(diff)
            feat['diff2_mean'] = np.mean(diff2)
            feat['diff2_std'] = np.std(diff2)
            feat['diff2_max'] = np.max(diff2)
            feat['diff2_min'] = np.min(diff2)
            
            # ========== SYMMETRY FEATURES ==========
            # Left vs Right subcarriers (32 vs 32)
            left = amp_64[0:32]
            right = amp_64[32:64][::-1]  # Reverse to align symmetric positions
            
            feat['left_mean'] = np.mean(left)
            feat['right_mean'] = np.mean(right)
            feat['left_right_ratio'] = feat['left_mean'] / (feat['right_mean'] + 1e-6)
            feat['left_right_diff'] = feat['left_mean'] - feat['right_mean']
            
            # Symmetry error (mean absolute difference between symmetric positions)
            feat['symmetry_error'] = np.mean(np.abs(left - right))
            feat['symmetry_max_error'] = np.max(np.abs(left - right))
            
            # ========== QUARTER SYMMETRY ==========
            q1 = amp_64[0:16]
            q2 = amp_64[16:32]
            q3 = amp_64[32:48]
            q4 = amp_64[48:64]
            
            feat['q1_mean'] = np.mean(q1)
            feat['q2_mean'] = np.mean(q2)
            feat['q3_mean'] = np.mean(q3)
            feat['q4_mean'] = np.mean(q4)
            
            feat['q2_q1_ratio'] = feat['q2_mean'] / (feat['q1_mean'] + 1e-6)
            feat['q3_q1_ratio'] = feat['q3_mean'] / (feat['q1_mean'] + 1e-6)
            feat['q4_q1_ratio'] = feat['q4_mean'] / (feat['q1_mean'] + 1e-6)
            
            # ========== PEAK DETECTION FEATURES ==========
            # Find peaks (local maxima)
            peaks, peak_props = find_peaks(amp_64, height=np.mean(amp_64), distance=2)
            
            feat['num_peaks'] = len(peaks)
            if len(peaks) > 0:
                feat['peak_mean_height'] = np.mean(peak_props['peak_heights'])
                feat['peak_max_height'] = np.max(peak_props['peak_heights'])
                feat['peak_std_height'] = np.std(peak_props['peak_heights'])
                # Prominence approximation
                feat['peak_mean_prominence'] = np.mean(peak_props['peak_heights'] - amp_64[peaks-1] if len(peaks) > 1 else peak_props['peak_heights'])
            else:
                feat['peak_mean_height'] = 0
                feat['peak_max_height'] = 0
                feat['peak_std_height'] = 0
                feat['peak_mean_prominence'] = 0
            
            # Find valleys (local minima)
            valleys, valley_props = find_peaks(-amp_64, height=-np.mean(amp_64), distance=2)
            feat['num_valleys'] = len(valleys)
            
            # Peak-to-peak distances
            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                feat['peak_distance_mean'] = np.mean(peak_distances)
                feat['peak_distance_std'] = np.std(peak_distances)
            else:
                feat['peak_distance_mean'] = 0
                feat['peak_distance_std'] = 0
            
            # ========== SUBCARRIER GROUP RATIOS ==========
            # Low subcarriers (0-15), Mid (16-31), High (32-47), Very High (48-63)
            low = amp_64[0:16]
            mid_low = amp_64[16:32]
            mid_high = amp_64[32:48]
            high = amp_64[48:64]
            
            feat['low_mean'] = np.mean(low)
            feat['mid_low_mean'] = np.mean(mid_low)
            feat['mid_high_mean'] = np.mean(mid_high)
            feat['high_mean'] = np.mean(high)
            
            feat['mid_low_low_ratio'] = feat['mid_low_mean'] / (feat['low_mean'] + 1e-6)
            feat['mid_high_low_ratio'] = feat['mid_high_mean'] / (feat['low_mean'] + 1e-6)
            feat['high_low_ratio'] = feat['high_mean'] / (feat['low_mean'] + 1e-6)
            
            # ========== CROSS-CORRELATION FEATURES ==========
            # Correlation between symmetric subcarrier groups
            if len(left) > 1 and len(right) > 1 and np.std(left) > 0 and np.std(right) > 0:
                feat['left_right_correlation'] = np.corrcoef(left, right)[0, 1]
            else:
                feat['left_right_correlation'] = 0
                
            if len(q1) > 1 and len(q4) > 1 and np.std(q1) > 0 and np.std(q4) > 0:
                feat['q1_q4_correlation'] = np.corrcoef(q1, q4)[0, 1]
            else:
                feat['q1_q4_correlation'] = 0
                
            if len(q2) > 1 and len(q3) > 1 and np.std(q2) > 0 and np.std(q3) > 0:
                feat['q2_q3_correlation'] = np.corrcoef(q2, q3)[0, 1]
            else:
                feat['q2_q3_correlation'] = 0
            
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

print("="*80)
print("SUBCARRIER INTERACTION FEATURES EXTRACTION")
print("="*80)

# Load datasets
print("\n📂 Loading data...")
baseline_df = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
healthy_df = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
diseased_df = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

# Combine
all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)

print(f"Total samples: {len(all_data)}")
print(f"Subcarrier interaction features extracted: {len([c for c in all_data.columns if c != 'label'])}")

# Summary statistics
print("\n📊 Subcarrier Interaction Feature Comparison:")
feature_cols = [c for c in all_data.columns if c != 'label']

for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    print(f"\n{name} (n={len(subset)}):")
    for feat in ['left_right_ratio', 'symmetry_error', 'num_peaks', 'peak_mean_height', 
                 'mid_low_low_ratio', 'left_right_correlation']:
        if feat in subset.columns:
            print(f"  {feat}: {subset[feat].mean():.4f} ± {subset[feat].std():.4f}")

# Save to CSV in the correct directory
output_file = os.path.join(output_path, 'subcarrier_interaction_features.csv')
all_data.to_csv(output_file, index=False)
print(f"\n✅ Saved to '{output_file}'")

# Also save a summary CSV
summary_file = os.path.join(output_path, 'subcarrier_interaction_features_summary.csv')
summary_data = []
for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    row = {'Condition': name, 'Count': len(subset)}
    for feat in ['left_right_ratio', 'symmetry_error', 'num_peaks', 'peak_mean_height', 
                 'mid_low_low_ratio', 'left_right_correlation', 'diff_mean', 'diff_std']:
        if feat in subset.columns:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_path, 'subcarrier_interaction_features_summary.csv'), index=False)
print(f"✅ Saved summary to '{os.path.join(output_path, 'subcarrier_interaction_features_summary.csv')}'")

# Feature correlation with labels
print("\n📈 Subcarrier Features Correlation with Disease Label:")
correlations = []
for col in feature_cols:
    if 'ratio' in col or 'symmetry' in col or 'peak' in col or 'correlation' in col or 'diff' in col:
        if col in all_data.columns and not pd.isna(all_data[col]).all():
            corr = all_data[col].corr(all_data['label'])
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))

top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:15]
print("Top 15 subcarrier features correlated with disease label:")
for i, (feat, corr) in enumerate(top_features, 1):
    print(f"  {i}. {feat}: {corr:.4f}")

# Print all generated files
print("\n" + "="*80)
print("📁 FILES GENERATED:")
print("="*80)
print(f"  1. {os.path.join(output_path, 'subcarrier_interaction_features.csv')}")
print(f"  2. {os.path.join(output_path, 'subcarrier_interaction_features_summary.csv')}")