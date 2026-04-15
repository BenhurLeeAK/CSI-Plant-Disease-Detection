"""
TIME-FREQUENCY FEATURES (Wavelet & Spectrogram) - COMPLETE FIXED VERSION
Extracts wavelet-based features and time-frequency representations
Saves output to csi_plant_data directory
"""

import numpy as np
import pandas as pd
import re
import os
from scipy import stats
from scipy.signal import spectrogram, hilbert
import warnings
warnings.filterwarnings('ignore')

# Define morlet wavelet manually (no scipy import needed)
def morlet_wavelet(width):
    """Create a Morlet wavelet of given width"""
    t = np.linspace(-4, 4, max(2 * int(width) + 1, 5))
    wavelet = np.exp(-t**2 / 2) * np.cos(5 * t)
    return wavelet

def cwt_custom(data, widths):
    """Continuous Wavelet Transform using convolution"""
    output = np.zeros((len(widths), len(data)))
    for i, width in enumerate(widths):
        # Create wavelet
        wavelet = morlet_wavelet(width)
        
        # Pad or truncate to match data length
        if len(wavelet) < len(data):
            padded = np.zeros(len(data))
            start = (len(data) - len(wavelet)) // 2
            padded[start:start+len(wavelet)] = wavelet
            wavelet = padded
        elif len(wavelet) > len(data):
            # Truncate wavelet
            start = (len(wavelet) - len(data)) // 2
            wavelet = wavelet[start:start+len(data)]
        
        # Convolution in frequency domain
        conv = np.real(np.fft.ifft(np.fft.fft(data) * np.fft.fft(wavelet)))
        output[i, :] = conv
    return output

# Set paths
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'
output_path = data_path

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
                'mean_amp': np.mean(amp_64),
            }
            
            # ========== SPECTROGRAM FEATURES ==========
            try:
                f, t, Sxx = spectrogram(amp_64, fs=sampling_rate, nperseg=16, noverlap=8)
                
                feat['spectrogram_mean'] = np.mean(Sxx)
                feat['spectrogram_std'] = np.std(Sxx)
                feat['spectrogram_max'] = np.max(Sxx)
                feat['spectrogram_min'] = np.min(Sxx)
                feat['spectrogram_energy'] = np.sum(Sxx)
                
                # Time-frequency entropy
                Sxx_norm = Sxx / (np.sum(Sxx) + 1e-6)
                feat['time_freq_entropy'] = -np.sum(Sxx_norm * np.log2(Sxx_norm + 1e-6))
                
                # Temporal variation at each frequency
                temporal_variation = np.std(Sxx, axis=1)
                feat['avg_temporal_variation'] = np.mean(temporal_variation)
                feat['max_temporal_variation'] = np.max(temporal_variation)
                feat['min_temporal_variation'] = np.min(temporal_variation)
                
                # Frequency variation over time
                freq_variation = np.std(Sxx, axis=0)
                feat['avg_freq_variation'] = np.mean(freq_variation)
                feat['max_freq_variation'] = np.max(freq_variation)
                feat['min_freq_variation'] = np.min(freq_variation)
                
                # Spectrogram peak frequency
                if Sxx.size > 0:
                    max_idx = np.unravel_index(np.argmax(Sxx), Sxx.shape)
                    feat['spectrogram_peak_freq'] = f[max_idx[0]] if len(f) > max_idx[0] else 0
                    feat['spectrogram_peak_time'] = t[max_idx[1]] if len(t) > max_idx[1] else 0
                else:
                    feat['spectrogram_peak_freq'] = 0
                    feat['spectrogram_peak_time'] = 0
            except Exception as e:
                feat['spectrogram_mean'] = 0
                feat['spectrogram_std'] = 0
                feat['spectrogram_max'] = 0
                feat['spectrogram_min'] = 0
                feat['spectrogram_energy'] = 0
                feat['time_freq_entropy'] = 0
                feat['avg_temporal_variation'] = 0
                feat['max_temporal_variation'] = 0
                feat['min_temporal_variation'] = 0
                feat['avg_freq_variation'] = 0
                feat['max_freq_variation'] = 0
                feat['min_freq_variation'] = 0
                feat['spectrogram_peak_freq'] = 0
                feat['spectrogram_peak_time'] = 0
            
            # ========== CONTINUOUS WAVELET TRANSFORM (CWT) ==========
            try:
                max_width = min(32, len(amp_64) // 2)
                widths = np.arange(1, max_width, 2)
                if len(widths) > 0:
                    cwt_matrix = cwt_custom(amp_64, widths)
                    
                    feat['cwt_mean'] = np.mean(np.abs(cwt_matrix))
                    feat['cwt_std'] = np.std(np.abs(cwt_matrix))
                    feat['cwt_max'] = np.max(np.abs(cwt_matrix))
                    feat['cwt_min'] = np.min(np.abs(cwt_matrix))
                    feat['cwt_energy'] = np.sum(cwt_matrix ** 2)
                    
                    # Wavelet entropy
                    cwt_norm = np.abs(cwt_matrix) / (np.sum(np.abs(cwt_matrix)) + 1e-6)
                    feat['wavelet_entropy'] = -np.sum(cwt_norm * np.log2(cwt_norm + 1e-6))
                    
                    # Scale-wise statistics
                    scale_means = np.mean(np.abs(cwt_matrix), axis=1)
                    feat['scale_mean_mean'] = np.mean(scale_means)
                    feat['scale_mean_std'] = np.std(scale_means)
                    feat['scale_mean_max'] = np.max(scale_means)
                    feat['scale_mean_min'] = np.min(scale_means)
                    
                    # Scale of maximum energy
                    scale_energy = np.sum(cwt_matrix ** 2, axis=1)
                    feat['max_energy_scale'] = np.argmax(scale_energy)
                    feat['max_energy_value'] = np.max(scale_energy)
                else:
                    feat['cwt_mean'] = 0
                    feat['cwt_std'] = 0
                    feat['cwt_max'] = 0
                    feat['cwt_min'] = 0
                    feat['cwt_energy'] = 0
                    feat['wavelet_entropy'] = 0
                    feat['scale_mean_mean'] = 0
                    feat['scale_mean_std'] = 0
                    feat['scale_mean_max'] = 0
                    feat['scale_mean_min'] = 0
                    feat['max_energy_scale'] = 0
                    feat['max_energy_value'] = 0
            except Exception as e:
                feat['cwt_mean'] = 0
                feat['cwt_std'] = 0
                feat['cwt_max'] = 0
                feat['cwt_min'] = 0
                feat['cwt_energy'] = 0
                feat['wavelet_entropy'] = 0
                feat['scale_mean_mean'] = 0
                feat['scale_mean_std'] = 0
                feat['scale_mean_max'] = 0
                feat['scale_mean_min'] = 0
                feat['max_energy_scale'] = 0
                feat['max_energy_value'] = 0
            
            # ========== HILBERT TRANSFORM (Instantaneous features) ==========
            try:
                analytic_signal = hilbert(amp_64)
                instantaneous_amplitude = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                
                feat['inst_amp_mean'] = np.mean(instantaneous_amplitude)
                feat['inst_amp_std'] = np.std(instantaneous_amplitude)
                feat['inst_amp_max'] = np.max(instantaneous_amplitude)
                feat['inst_amp_min'] = np.min(instantaneous_amplitude)
                feat['inst_amp_range'] = feat['inst_amp_max'] - feat['inst_amp_min']
                
                feat['inst_phase_mean'] = np.mean(instantaneous_phase)
                feat['inst_phase_std'] = np.std(instantaneous_phase)
                feat['inst_phase_range'] = np.max(instantaneous_phase) - np.min(instantaneous_phase)
                
                # Instantaneous frequency (derivative of phase)
                inst_frequency = np.diff(instantaneous_phase) * sampling_rate / (2 * np.pi)
                if len(inst_frequency) > 0:
                    feat['inst_freq_mean'] = np.mean(inst_frequency)
                    feat['inst_freq_std'] = np.std(inst_frequency)
                    feat['inst_freq_max'] = np.max(inst_frequency)
                    feat['inst_freq_min'] = np.min(inst_frequency)
                    feat['inst_freq_range'] = feat['inst_freq_max'] - feat['inst_freq_min']
                else:
                    feat['inst_freq_mean'] = 0
                    feat['inst_freq_std'] = 0
                    feat['inst_freq_max'] = 0
                    feat['inst_freq_min'] = 0
                    feat['inst_freq_range'] = 0
                
                # Instantaneous bandwidth
                feat['inst_bandwidth'] = np.std(inst_frequency) if len(inst_frequency) > 0 else 0
            except Exception as e:
                feat['inst_amp_mean'] = 0
                feat['inst_amp_std'] = 0
                feat['inst_amp_max'] = 0
                feat['inst_amp_min'] = 0
                feat['inst_amp_range'] = 0
                feat['inst_phase_mean'] = 0
                feat['inst_phase_std'] = 0
                feat['inst_phase_range'] = 0
                feat['inst_freq_mean'] = 0
                feat['inst_freq_std'] = 0
                feat['inst_freq_max'] = 0
                feat['inst_freq_min'] = 0
                feat['inst_freq_range'] = 0
                feat['inst_bandwidth'] = 0
            
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

print("="*80)
print("TIME-FREQUENCY FEATURES EXTRACTION (Wavelet & Spectrogram)")
print("="*80)

# Load datasets
print("\n📂 Loading data...")
baseline_df = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
healthy_df = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
diseased_df = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

# Combine
all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)

print(f"Total samples: {len(all_data)}")
print(f"Time-frequency features extracted: {len([c for c in all_data.columns if c != 'label'])}")

# Summary statistics
print("\n📊 Time-Frequency Feature Comparison:")

for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    print(f"\n{name} (n={len(subset)}):")
    for feat in ['time_freq_entropy', 'wavelet_entropy', 'inst_amp_mean', 
                 'inst_freq_mean', 'spectrogram_mean', 'cwt_mean']:
        if feat in subset.columns:
            print(f"  {feat}: {subset[feat].mean():.4f} ± {subset[feat].std():.4f}")

# Save to CSV
output_file = os.path.join(output_path, 'time_frequency_features.csv')
all_data.to_csv(output_file, index=False)
print(f"\n✅ Saved to '{output_file}'")

# Save summary CSV
summary_file = os.path.join(output_path, 'time_frequency_features_summary.csv')
summary_data = []
for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    row = {'Condition': name, 'Count': len(subset)}
    for feat in ['time_freq_entropy', 'wavelet_entropy', 'inst_amp_mean', 
                 'inst_freq_mean', 'spectrogram_mean', 'cwt_mean',
                 'avg_temporal_variation', 'avg_freq_variation']:
        if feat in subset.columns:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_path, 'time_frequency_features_summary.csv'), index=False)
print(f"✅ Saved summary to '{os.path.join(output_path, 'time_frequency_features_summary.csv')}'")

# Feature correlation with labels
print("\n📈 Time-Frequency Features Correlation with Disease Label:")
correlations = []
for col in all_data.columns:
    if col != 'label' and not pd.isna(all_data[col]).all():
        corr = all_data[col].corr(all_data['label'])
        if not pd.isna(corr):
            correlations.append((col, abs(corr)))

top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:15]
print("Top 15 time-frequency features correlated with disease label:")
for i, (feat, corr) in enumerate(top_features, 1):
    print(f"  {i}. {feat}: {corr:.4f}")

# Print all generated files
print("\n" + "="*80)
print("📁 FILES GENERATED:")
print("="*80)
print(f"  1. {os.path.join(output_path, 'time_frequency_features.csv')}")
print(f"  2. {os.path.join(output_path, 'time_frequency_features_summary.csv')}")