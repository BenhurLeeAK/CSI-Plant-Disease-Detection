"""
FREQUENCY DOMAIN FEATURES (Beyond FFT)
Extracts Power Spectral Density, spectral features, and advanced frequency metrics
Saves output to csi_plant_data directory
"""

import numpy as np
import pandas as pd
import re
import os
from scipy import stats
from scipy.signal import welch, find_peaks, spectrogram
from scipy.fft import fft, fftfreq
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
    sampling_rate = 100  # Hz (assuming 100 packets per second)
    
    for match in matches:
        rssi = int(match[1])
        amp_str = match[2].strip()
        amp_list = [float(x) for x in amp_str.split()]
        
        if len(amp_list) >= 64:
            amp_64 = np.array(amp_list[:64])
            
            feat = {
                'label': label,
                'rssi': rssi,
            }
            
            # ========== BASIC FFT FEATURES ==========
            fft_vals = np.abs(fft(amp_64))
            fft_freqs = fftfreq(len(amp_64), d=1/sampling_rate)
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = fft_vals[:len(fft_vals)//2]
            
            feat['fft_peak'] = np.max(positive_fft)
            feat['fft_mean'] = np.mean(positive_fft)
            feat['fft_std'] = np.std(positive_fft)
            feat['fft_energy'] = np.sum(positive_fft ** 2)
            
            # Dominant frequency
            dominant_idx = np.argmax(positive_fft)
            feat['dominant_frequency'] = positive_freqs[dominant_idx]
            feat['dominant_frequency_magnitude'] = positive_fft[dominant_idx]
            
            # ========== POWER SPECTRAL DENSITY (PSD) ==========
            frequencies, psd = welch(amp_64, fs=sampling_rate, nperseg=32)
            
            feat['psd_mean'] = np.mean(psd)
            feat['psd_std'] = np.std(psd)
            feat['psd_max'] = np.max(psd)
            feat['psd_min'] = np.min(psd)
            feat['psd_skewness'] = stats.skew(psd)
            feat['psd_kurtosis'] = stats.kurtosis(psd)
            feat['psd_energy'] = np.sum(psd)
            
            # Spectral centroid (center of mass of spectrum)
            feat['spectral_centroid'] = np.sum(frequencies * psd) / (np.sum(psd) + 1e-6)
            
            # Spectral spread (variance around centroid)
            feat['spectral_spread'] = np.sqrt(
                np.sum(((frequencies - feat['spectral_centroid']) ** 2) * psd) / (np.sum(psd) + 1e-6)
            )
            
            # Spectral bandwidth (range of significant frequencies)
            cumulative_psd = np.cumsum(psd) / (np.sum(psd) + 1e-6)
            feat['spectral_bandwidth'] = frequencies[np.where(cumulative_psd >= 0.95)[0][0]] - frequencies[0]
            
            # Spectral rolloff (frequency where 85% and 95% of energy is contained)
            rolloff_idx_85 = np.where(cumulative_psd >= 0.85)[0]
            rolloff_idx_95 = np.where(cumulative_psd >= 0.95)[0]
            feat['spectral_rolloff_85'] = frequencies[rolloff_idx_85[0]] if len(rolloff_idx_85) > 0 else 0
            feat['spectral_rolloff_95'] = frequencies[rolloff_idx_95[0]] if len(rolloff_idx_95) > 0 else 0
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-6)
            feat['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-6))
            
            # Spectral flatness (geometric mean / arithmetic mean)
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-6)))
            arithmetic_mean = np.mean(psd)
            feat['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-6)
            
            # ========== ADDITIONAL FREQUENCY METRICS ==========
            # Number of peaks in PSD
            peaks, _ = find_peaks(psd, height=np.mean(psd))
            feat['num_psd_peaks'] = len(peaks)
            
            # Peak-to-average ratio
            feat['peak_to_average_ratio'] = np.max(psd) / (np.mean(psd) + 1e-6)
            
            # Frequency of maximum PSD
            feat['frequency_of_max_psd'] = frequencies[np.argmax(psd)]
            
            # Low-frequency energy (0-10 Hz)
            low_freq_mask = frequencies <= 10
            feat['low_freq_energy'] = np.sum(psd[low_freq_mask]) / (np.sum(psd) + 1e-6)
            
            # Mid-frequency energy (10-30 Hz)
            mid_freq_mask = (frequencies > 10) & (frequencies <= 30)
            feat['mid_freq_energy'] = np.sum(psd[mid_freq_mask]) / (np.sum(psd) + 1e-6)
            
            # High-frequency energy (>30 Hz)
            high_freq_mask = frequencies > 30
            feat['high_freq_energy'] = np.sum(psd[high_freq_mask]) / (np.sum(psd) + 1e-6)
            
            # Spectral slope (linear fit to log-log spectrum)
            log_freq = np.log(frequencies[1:] + 1e-6)
            log_psd = np.log(psd[1:] + 1e-6)
            if len(log_freq) > 1:
                slope, _ = np.polyfit(log_freq, log_psd, 1)
                feat['spectral_slope'] = slope
            else:
                feat['spectral_slope'] = 0
            
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

print("="*80)
print("FREQUENCY DOMAIN FEATURES EXTRACTION")
print("="*80)

# Load datasets
print("\n📂 Loading data...")
baseline_df = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
healthy_df = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
diseased_df = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

# Combine
all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)

print(f"Total samples: {len(all_data)}")
print(f"Frequency domain features extracted: {len([c for c in all_data.columns if c != 'label'])}")

# Summary statistics
print("\n📊 Frequency Domain Feature Comparison:")
feature_cols = [c for c in all_data.columns if c != 'label']

for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    print(f"\n{name} (n={len(subset)}):")
    for feat in ['spectral_centroid', 'spectral_entropy', 'spectral_flatness', 
                 'low_freq_energy', 'mid_freq_energy', 'high_freq_energy']:
        if feat in subset.columns:
            print(f"  {feat}: {subset[feat].mean():.4f} ± {subset[feat].std():.4f}")

# Save to CSV in the correct directory
output_file = os.path.join(output_path, 'frequency_domain_features.csv')
all_data.to_csv(output_file, index=False)
print(f"\n✅ Saved to '{output_file}'")

# Also save a summary CSV
summary_file = os.path.join(output_path, 'frequency_domain_features_summary.csv')
summary_data = []
for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    row = {'Condition': name, 'Count': len(subset)}
    for feat in ['spectral_centroid', 'spectral_entropy', 'spectral_flatness', 
                 'low_freq_energy', 'mid_freq_energy', 'high_freq_energy', 'spectral_slope']:
        if feat in subset.columns:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_path, 'frequency_domain_features_summary.csv'), index=False)
print(f"✅ Saved summary to '{os.path.join(output_path, 'frequency_domain_features_summary.csv')}'")

# Feature correlation with labels
print("\n📈 Frequency Features Correlation with Disease Label:")
correlations = []
for col in feature_cols:
    if 'spectral' in col or 'freq' in col or 'psd' in col or 'fft' in col:
        if col in all_data.columns and not pd.isna(all_data[col]).all():
            corr = all_data[col].corr(all_data['label'])
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))

top_freq_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:10]
print("Top 10 frequency features correlated with disease label:")
for i, (feat, corr) in enumerate(top_freq_features, 1):
    print(f"  {i}. {feat}: {corr:.4f}")

# Print all generated files
print("\n" + "="*80)
print("📁 FILES GENERATED:")
print("="*80)
print(f"  1. {os.path.join(output_path, 'frequency_domain_features.csv')}")
print(f"  2. {os.path.join(output_path, 'frequency_domain_features_summary.csv')}")