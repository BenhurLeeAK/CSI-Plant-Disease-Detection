"""
TEMPORAL FEATURES EXTRACTION - FULLY FIXED VERSION
Extracts time-series features across sequential packets
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

def load_temporal_data(filename, label, max_packets=None):
    """Load CSI data preserving temporal order"""
    full_path = f"{data_path}/{filename}"
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    pattern = r'\[CSI #(\d+)\].*?RSSI:(-?\d+)\nAmp:\s*([\d\.\s]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    features = []
    
    for match in matches:
        packet_num = int(match[0])
        rssi = int(match[1])
        amp_str = match[2].strip()
        amp_list = [float(x) for x in amp_str.split()]
        
        if len(amp_list) >= 64:
            amp_64 = np.array(amp_list[:64])
            
            feat = {
                'label': label,
                'packet_num': packet_num,
                'rssi': rssi,
                'mean_amp': np.mean(amp_64),
                'std_amp': np.std(amp_64),
                'max_amp': np.max(amp_64),
                'min_amp': np.min(amp_64),
                'median_amp': np.median(amp_64),
                'energy': np.sum(amp_64 ** 2),
            }
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

def extract_temporal_features(df, window_size=50):
    """Extract features from temporal sequences"""
    
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Rolling statistics
    df['rolling_mean_50'] = df['mean_amp'].rolling(window=window_size, center=True).mean()
    df['rolling_std_50'] = df['mean_amp'].rolling(window=window_size, center=True).std()
    df['rolling_max_50'] = df['mean_amp'].rolling(window=window_size, center=True).max()
    df['rolling_min_50'] = df['mean_amp'].rolling(window=window_size, center=True).min()
    df['rolling_energy_50'] = df['energy'].rolling(window=window_size, center=True).mean()
    
    # Rate of change (derivative)
    df['derivative'] = df['mean_amp'].diff()
    df['derivative_abs'] = np.abs(df['derivative'])
    df['derivative_squared'] = df['derivative'] ** 2
    
    # Cumulative statistics
    df['cumulative_mean'] = df['mean_amp'].expanding().mean()
    df['cumulative_std'] = df['mean_amp'].expanding().std()
    df['cumulative_min'] = df['mean_amp'].expanding().min()
    df['cumulative_max'] = df['mean_amp'].expanding().max()
    
    # Lag features (autocorrelation)
    for lag in [1, 5, 10, 20, 50]:
        df[f'autocorr_lag_{lag}'] = df['mean_amp'].shift(lag)
        df[f'autocorr_diff_lag_{lag}'] = df['mean_amp'] - df[f'autocorr_lag_{lag}']
        df[f'autocorr_ratio_lag_{lag}'] = df['mean_amp'] / (df[f'autocorr_lag_{lag}'] + 1e-6)
    
    return df

def aggregate_temporal_stats(df, condition_name):
    """Aggregate temporal features into summary statistics"""
    
    # Remove NaN values from rolling calculations
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        return {'condition': condition_name, 'error': 'No valid data'}
    
    # Calculate autocorrelation at lag 1 for the whole series
    if len(df['mean_amp']) > 1:
        autocorr_lag1 = df['mean_amp'].corr(df['mean_amp'].shift(1))
        autocorr_lag1 = autocorr_lag1 if not pd.isna(autocorr_lag1) else 0
    else:
        autocorr_lag1 = 0
    
    stats_dict = {
        'condition': condition_name,
        'total_packets': len(df),
        'valid_packets': len(df_clean),
        'mean_amp_mean': df['mean_amp'].mean(),
        'mean_amp_std': df['mean_amp'].std(),
        'mean_amp_var': df['mean_amp'].var(),
        'mean_amp_range': df['mean_amp'].max() - df['mean_amp'].min(),
        'rssi_mean': df['rssi'].mean(),
        'rssi_std': df['rssi'].std(),
        'rolling_mean_mean': df['rolling_mean_50'].mean(),
        'rolling_std_mean': df['rolling_std_50'].mean(),
        'derivative_mean': df['derivative'].mean(),
        'derivative_abs_mean': df['derivative_abs'].mean(),
        'derivative_max': df['derivative'].max(),
        'derivative_min': df['derivative'].min(),
        'derivative_std': df['derivative'].std(),
        'autocorr_stability': df['autocorr_diff_lag_10'].std(),
        'autocorr_ratio_mean': df['autocorr_ratio_lag_10'].mean(),
        'autocorrelation_lag1': autocorr_lag1,
    }
    
    # Trend slope (linear fit)
    try:
        x = np.arange(len(df))
        y = df['mean_amp'].fillna(method='bfill').fillna(method='ffill').values
        slope, intercept = np.polyfit(x, y, 1)
        stats_dict['trend_slope'] = slope
        stats_dict['trend_intercept'] = intercept
    except:
        stats_dict['trend_slope'] = 0
        stats_dict['trend_intercept'] = 0
    
    # Detect change points (where derivative exceeds 2 standard deviations)
    threshold = 2 * df['derivative'].std()
    changes = np.where(np.abs(df['derivative']) > threshold)[0]
    stats_dict['num_significant_changes'] = len(changes)
    stats_dict['change_rate'] = len(changes) / len(df) if len(df) > 0 else 0
    
    # Coefficient of variation (CV = std/mean) - lower is more stable
    stats_dict['coefficient_variation'] = df['mean_amp'].std() / (df['mean_amp'].mean() + 1e-6)
    
    # Mean squared successive difference (MSSD) - lower is more stable
    diff_values = df['mean_amp'].diff().dropna()
    stats_dict['mssd'] = np.mean(diff_values ** 2) if len(diff_values) > 0 else 0
    
    # Percentage of time within 1 std of mean
    mean_val = df['mean_amp'].mean()
    std_val = df['mean_amp'].std()
    within_1std = np.mean((df['mean_amp'] >= mean_val - std_val) & (df['mean_amp'] <= mean_val + std_val)) * 100
    stats_dict['pct_within_1std'] = within_1std
    
    # Hurst exponent estimate (simplified)
    try:
        n = len(df['mean_amp'])
        if n > 10:
            mean_vals = df['mean_amp'].values
            R = np.max(np.cumsum(mean_vals - np.mean(mean_vals))) - np.min(np.cumsum(mean_vals - np.mean(mean_vals)))
            S = np.std(mean_vals)
            stats_dict['hurst_estimate'] = np.log(R / (S + 1e-6)) / np.log(n) if S > 0 else 0
        else:
            stats_dict['hurst_estimate'] = 0
    except:
        stats_dict['hurst_estimate'] = 0
    
    return stats_dict

print("="*80)
print("TEMPORAL FEATURES EXTRACTION")
print("="*80)

# Load full temporal data (all packets, not sampled)
print("\n📂 Loading temporal data...")
baseline_df = load_temporal_data('baseline_no_plant.txt', 0)
healthy_df = load_temporal_data('with_plant_30cm.txt', 1)
diseased_df = load_temporal_data('with_disease_plant_30cm.txt', 2)

print(f"Baseline: {len(baseline_df)} packets")
print(f"Healthy: {len(healthy_df)} packets")
print(f"Diseased: {len(diseased_df)} packets")

# Apply temporal feature extraction
print("\n⏱️ Extracting temporal features...")
baseline_df = extract_temporal_features(baseline_df)
healthy_df = extract_temporal_features(healthy_df)
diseased_df = extract_temporal_features(diseased_df)

# Save individual temporal dataframes
baseline_df.to_csv(os.path.join(output_path, 'temporal_features_baseline.csv'), index=False)
healthy_df.to_csv(os.path.join(output_path, 'temporal_features_healthy.csv'), index=False)
diseased_df.to_csv(os.path.join(output_path, 'temporal_features_diseased.csv'), index=False)
print("\n✅ Saved individual temporal feature files")

# Aggregate to get per-file statistics (one row per condition)
print("\n📊 Aggregating temporal statistics...")
baseline_stats = aggregate_temporal_stats(baseline_df, 'Baseline')
healthy_stats = aggregate_temporal_stats(healthy_df, 'Healthy')
diseased_stats = aggregate_temporal_stats(diseased_df, 'Diseased')

# Create summary DataFrame
summary_df = pd.DataFrame([baseline_stats, healthy_stats, diseased_stats])
print("\n📈 Temporal Feature Summary:")
print(summary_df.to_string(index=False))

# Calculate temporal stability metrics (without using non-existent columns)
print("\n🔍 Temporal Stability Analysis:")
stability_metrics = []

for name, df in [('Baseline', baseline_df), ('Healthy', healthy_df), ('Diseased', diseased_df)]:
    # Coefficient of variation (CV = std/mean) - lower is more stable
    cv = df['mean_amp'].std() / (df['mean_amp'].mean() + 1e-6)
    
    # Mean squared successive difference (MSSD) - lower is more stable
    diff_values = df['mean_amp'].diff().dropna()
    mssd = np.mean(diff_values ** 2) if len(diff_values) > 0 else 0
    
    # Percentage of time within 1 std of mean
    mean_val = df['mean_amp'].mean()
    std_val = df['mean_amp'].std()
    within_1std = np.mean((df['mean_amp'] >= mean_val - std_val) & (df['mean_amp'] <= mean_val + std_val)) * 100
    
    # Trend direction
    try:
        y = df['mean_amp'].fillna(method='bfill').fillna(method='ffill').values
        slope = np.polyfit(range(len(y)), y, 1)[0]
    except:
        slope = 0
    
    # Autocorrelation at lag 1
    if len(df['mean_amp']) > 1:
        autocorr = df['mean_amp'].corr(df['mean_amp'].shift(1))
        autocorr = autocorr if not pd.isna(autocorr) else 0
    else:
        autocorr = 0
    
    # Rolling stability (std of rolling mean)
    rolling_stability = df['rolling_std_50'].std() if 'rolling_std_50' in df.columns else 0
    
    stability_metrics.append({
        'Condition': name,
        'CV (lower=more stable)': f"{cv:.3f}",
        'MSSD (lower=more stable)': f"{mssd:.3f}",
        'Time within ±1σ (%)': f"{within_1std:.1f}%",
        'Trend Slope': f"{slope:.4f}",
        'Autocorrelation (lag1)': f"{autocorr:.3f}",
        'Rolling Std Stability': f"{rolling_stability:.3f}"
    })

stability_df = pd.DataFrame(stability_metrics)
print("\n📊 Stability Metrics:")
print(stability_df.to_string(index=False))

# Save summary to CSV
summary_file = os.path.join(output_path, 'temporal_features_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"\n✅ Saved summary to '{summary_file}'")

# Save stability metrics
stability_file = os.path.join(output_path, 'temporal_stability_metrics.csv')
stability_df.to_csv(stability_file, index=False)
print(f"✅ Saved stability metrics to '{stability_file}'")

# Identify most stable vs most variable condition
print("\n🎯 Key Findings:")
cv_values = []
for row in stability_df['CV (lower=more stable)']:
    try:
        cv_values.append(float(row.split()[0]) if ' ' in row else float(row))
    except:
        cv_values.append(0)
        
min_cv_idx = np.argmin(cv_values) if cv_values else 0
max_cv_idx = np.argmax(cv_values) if cv_values else 0
print(f"  Most stable condition: {stability_df.iloc[min_cv_idx]['Condition']}")
print(f"  Most variable condition: {stability_df.iloc[max_cv_idx]['Condition']}")

# Trend direction interpretation
print("\n📈 Trend Analysis:")
for _, row in stability_df.iterrows():
    try:
        slope = float(row['Trend Slope'])
    except:
        slope = 0
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    print(f"  {row['Condition']}: {direction} trend (slope = {slope:.4f})")

# Print all generated files
print("\n" + "="*80)
print("📁 FILES GENERATED:")
print("="*80)
print(f"  1. {os.path.join(output_path, 'temporal_features_baseline.csv')}")
print(f"  2. {os.path.join(output_path, 'temporal_features_healthy.csv')}")
print(f"  3. {os.path.join(output_path, 'temporal_features_diseased.csv')}")
print(f"  4. {os.path.join(output_path, 'temporal_features_summary.csv')}")
print(f"  5. {os.path.join(output_path, 'temporal_stability_metrics.csv')}")