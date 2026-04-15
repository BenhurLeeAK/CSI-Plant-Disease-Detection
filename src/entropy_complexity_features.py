"""
ENTROPY AND COMPLEXITY FEATURES EXTRACTION
Extracts entropy-based and complexity metrics from CSI signals
Saves output to csi_plant_data directory
"""

import numpy as np
import pandas as pd
import re
import os
from scipy import stats
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

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
            
            # Normalize amplitudes to probability distribution
            amp_norm = amp_64 / (np.sum(amp_64) + 1e-6)
            
            # ========== SHANNON ENTROPY ==========
            feat['shannon_entropy'] = -np.sum(amp_norm * np.log2(amp_norm + 1e-6))
            
            # Maximum possible entropy (uniform distribution)
            max_entropy = np.log2(len(amp_64))
            feat['normalized_entropy'] = feat['shannon_entropy'] / max_entropy
            
            # ========== RÉNYI ENTROPY ==========
            alpha_values = [0.5, 1, 2, 3]
            for alpha in alpha_values:
                if alpha == 1:
                    renyi = feat['shannon_entropy']
                else:
                    renyi = (1/(1-alpha)) * np.log2(np.sum(amp_norm ** alpha) + 1e-6)
                feat[f'renyi_entropy_alpha_{alpha}'] = abs(renyi)
            
            # ========== TSALLIS ENTROPY ==========
            q_values = [0.5, 1.5, 2]
            for q in q_values:
                if q == 1:
                    tsallis = feat['shannon_entropy']
                else:
                    tsallis = (1/(1-q)) * (np.sum(amp_norm ** q) - 1)
                feat[f'tsallis_entropy_q_{q}'] = abs(tsallis)
            
            # ========== SAMPLE ENTROPY ==========
            def sample_entropy(time_series, m=2, r=0.2):
                """Calculate sample entropy"""
                N = len(time_series)
                r = r * np.std(time_series)
                
                def _maxdist(xi, xj):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                    C = 0
                    for i in range(N - m + 1):
                        for j in range(N - m + 1):
                            if i != j and _maxdist(x[i], x[j]) <= r:
                                C += 1
                    return C / ((N - m + 1) * (N - m))
                
                if N <= m:
                    return 0
                return -np.log(_phi(m+1) / (_phi(m) + 1e-6))
            
            try:
                feat['sample_entropy'] = sample_entropy(amp_64)
            except:
                feat['sample_entropy'] = 0
            
            # ========== APPROXIMATE ENTROPY ==========
            def approximate_entropy(time_series, m=2, r=0.2):
                """Calculate approximate entropy"""
                N = len(time_series)
                r = r * np.std(time_series)
                
                def _phi(m):
                    x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                    C = 0
                    for i in range(N - m + 1):
                        count = 0
                        for j in range(N - m + 1):
                            if max([abs(x[i][k] - x[j][k]) for k in range(m)]) <= r:
                                count += 1
                        C += np.log(count / (N - m + 1) + 1e-6)
                    return C / (N - m + 1)
                
                if N <= m:
                    return 0
                return _phi(m) - _phi(m+1)
            
            try:
                feat['approximate_entropy'] = approximate_entropy(amp_64)
            except:
                feat['approximate_entropy'] = 0
            
            # ========== PERMUTATION ENTROPY ==========
            def permutation_entropy(time_series, order=3, delay=1):
                """Calculate permutation entropy"""
                n = len(time_series)
                if n < order * delay:
                    return 0
                permutations = []
                for i in range(n - (order - 1) * delay):
                    perm = time_series[i:i + order * delay:delay]
                    perm_order = np.argsort(perm)
                    permutations.append(tuple(perm_order))
                
                unique, counts = np.unique(permutations, return_counts=True)
                probs = counts / len(permutations)
                entropy = -np.sum(probs * np.log2(probs + 1e-6))
                max_entropy = np.log2(np.math.factorial(order))
                return entropy / max_entropy if max_entropy > 0 else 0
            
            try:
                feat['permutation_entropy'] = permutation_entropy(amp_64)
            except:
                feat['permutation_entropy'] = 0
            
            # ========== SINGULAR VALUE DECOMPOSITION ENTROPY ==========
            def hankel_matrix(x, L):
                K = len(x) - L + 1
                if K <= 0:
                    return np.zeros((L, 1))
                H = np.zeros((L, K))
                for i in range(L):
                    H[i, :] = x[i:i+K]
                return H
            
            L = min(32, len(amp_64)//2)
            if L > 1:
                H = hankel_matrix(amp_64, L)
                try:
                    svd_vals = np.linalg.svd(H, compute_uv=False)
                    svd_norm = svd_vals / (np.sum(svd_vals) + 1e-6)
                    feat['svd_entropy'] = -np.sum(svd_norm * np.log2(svd_norm + 1e-6))
                except:
                    feat['svd_entropy'] = 0
            else:
                feat['svd_entropy'] = 0
            
            # ========== FRACTAL DIMENSION (Higuchi) ==========
            def higuchi_fd(signal, kmax=10):
                """Calculate Higuchi fractal dimension"""
                N = len(signal)
                if N < kmax + 1:
                    return 0
                Lk = []
                for k in range(1, kmax + 1):
                    Lmk = []
                    for m in range(k):
                        L = 0
                        for i in range(1, int((N - m) / k)):
                            L += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                        if int((N - m) / k) > 0:
                            L = L * (N - 1) / (k * ((N - m) // k))
                        else:
                            L = 0
                        Lmk.append(L)
                    if np.mean(Lmk) > 0:
                        Lk.append(np.log(np.mean(Lmk)))
                    else:
                        Lk.append(0)
                x = np.log(1.0 / np.arange(1, kmax + 1))
                y = Lk
                valid_idx = ~(np.isinf(y) | np.isnan(y) | (y == 0))
                if np.sum(valid_idx) > 1:
                    slope = np.polyfit(x[valid_idx], y[valid_idx], 1)[0]
                    return 2 - slope
                return 0
            
            try:
                feat['fractal_dimension'] = higuchi_fd(amp_64)
            except:
                feat['fractal_dimension'] = 0
            
            # ========== LEMPEL-ZIV COMPLEXITY ==========
            def lz_complexity(signal):
                """Calculate Lempel-Ziv complexity"""
                # Binarize signal using median threshold
                threshold = np.median(signal)
                binary = (signal > threshold).astype(int).astype(str)
                binary_str = ''.join(binary)
                
                n = len(binary_str)
                if n == 0:
                    return 0
                c = 1
                l = 1
                i = 0
                k = 1
                while True:
                    if i + k > n:
                        break
                    sub = binary_str[i:i+k]
                    if sub in binary_str[:i]:
                        k += 1
                    else:
                        c += 1
                        i += k
                        k = 1
                return c / n  # Normalized by length
            
            try:
                feat['lz_complexity'] = lz_complexity(amp_64)
            except:
                feat['lz_complexity'] = 0
            
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

print("="*80)
print("ENTROPY AND COMPLEXITY FEATURES EXTRACTION")
print("="*80)

# Load datasets
print("\n📂 Loading data...")
baseline_df = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
healthy_df = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
diseased_df = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

# Combine
all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)

print(f"Total samples: {len(all_data)}")
print(f"Entropy/complexity features extracted: {len([c for c in all_data.columns if c != 'label'])}")

# Summary statistics
print("\n📊 Entropy & Complexity Feature Comparison:")

for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    print(f"\n{name} (n={len(subset)}):")
    for feat in ['shannon_entropy', 'sample_entropy', 'permutation_entropy', 
                 'fractal_dimension', 'lz_complexity', 'svd_entropy']:
        if feat in subset.columns:
            print(f"  {feat}: {subset[feat].mean():.4f} ± {subset[feat].std():.4f}")

# Save to CSV
output_file = os.path.join(output_path, 'entropy_complexity_features.csv')
all_data.to_csv(output_file, index=False)
print(f"\n✅ Saved to '{output_file}'")

# Save summary CSV
summary_file = os.path.join(output_path, 'entropy_complexity_features_summary.csv')
summary_data = []
for condition, name in [(0, 'Baseline'), (1, 'Healthy'), (2, 'Diseased')]:
    subset = all_data[all_data['label'] == condition]
    row = {'Condition': name, 'Count': len(subset)}
    for feat in ['shannon_entropy', 'sample_entropy', 'permutation_entropy', 
                 'fractal_dimension', 'lz_complexity', 'svd_entropy',
                 'normalized_entropy']:
        if feat in subset.columns:
            row[f'{feat}_mean'] = subset[feat].mean()
            row[f'{feat}_std'] = subset[feat].std()
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_path, 'entropy_complexity_features_summary.csv'), index=False)
print(f"✅ Saved summary to '{os.path.join(output_path, 'entropy_complexity_features_summary.csv')}'")

# Feature correlation with labels
print("\n📈 Entropy Features Correlation with Disease Label:")
correlations = []
for col in all_data.columns:
    if col != 'label' and not pd.isna(all_data[col]).all():
        corr = all_data[col].corr(all_data['label'])
        if not pd.isna(corr):
            correlations.append((col, abs(corr)))

top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:15]
print("Top 15 entropy/complexity features correlated with disease label:")
for i, (feat, corr) in enumerate(top_features, 1):
    print(f"  {i}. {feat}: {corr:.4f}")

# Print all generated files
print("\n" + "="*80)
print("📁 FILES GENERATED:")
print("="*80)
print(f"  1. {os.path.join(output_path, 'entropy_complexity_features.csv')}")
print(f"  2. {os.path.join(output_path, 'entropy_complexity_features_summary.csv')}")