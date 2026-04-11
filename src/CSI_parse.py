import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set the correct path to your data files
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'

def parse_csi_file(filename):
    """Parse CSI data from your text files - handles variable length packets"""
    full_path = f"{data_path}/{filename}"
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return [], [], []
    
    # Find all CSI packets
    pattern = r'\[CSI #(\d+)\].*?RSSI:(-?\d+)\nAmp:\s*([\d\.\s]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    packets = []
    rssi_values = []
    amplitudes = []  # Store as list of lists (not numpy array yet)
    
    for match in matches:
        packet_num = int(match[0])
        rssi = int(match[1])
        amp_str = match[2].strip()
        amp_list = [float(x) for x in amp_str.split()]
        amplitudes.append(amp_list)
        rssi_values.append(rssi)
        packets.append(packet_num)
    
    # Find the most common length to use for padding/truncating
    if len(amplitudes) == 0:
        return np.array([]), np.array([]), []
    
    # Get lengths of all packets
    lengths = [len(a) for a in amplitudes]
    common_length = max(set(lengths), key=lengths.count)  # Most common length
    
    print(f"  {filename}: {len(amplitudes)} packets, lengths: {set(lengths)} -> using {common_length} subcarriers")
    
    # Pad or truncate to common length
    padded_amplitudes = []
    for amp in amplitudes:
        if len(amp) < common_length:
            # Pad with zeros
            padded = amp + [0] * (common_length - len(amp))
        else:
            # Truncate to common length
            padded = amp[:common_length]
        padded_amplitudes.append(padded)
    
    return np.array(padded_amplitudes), np.array(rssi_values), packets

# Load your datasets
print("Loading datasets...")
print(f"Looking for files in: {data_path}")
print("-" * 40)

baseline_amps, baseline_rssi, _ = parse_csi_file('baseline_no_plant.txt')
healthy_30_amps, healthy_30_rssi, _ = parse_csi_file('with_plant_30cm.txt')
healthy_45_amps, healthy_45_rssi, _ = parse_csi_file('with_plant_45cm.txt')
diseased_30_amps, diseased_30_rssi, _ = parse_csi_file('with_disease_plant_30cm.txt')
diseased_45_amps, diseased_45_rssi, _ = parse_csi_file('with_disease_plant_45cm.txt')

print("-" * 40)
print("\nDataset shapes:")
print(f"Baseline: {baseline_amps.shape if len(baseline_amps) > 0 else 'Empty'}")
print(f"Healthy 30cm: {healthy_30_amps.shape if len(healthy_30_amps) > 0 else 'Empty'}")
print(f"Healthy 45cm: {healthy_45_amps.shape if len(healthy_45_amps) > 0 else 'Empty'}")
print(f"Diseased 30cm: {diseased_30_amps.shape if len(diseased_30_amps) > 0 else 'Empty'}")
print(f"Diseased 45cm: {diseased_45_amps.shape if len(diseased_45_amps) > 0 else 'Empty'}")

def extract_features(amplitudes, rssi_values):
    """Extract meaningful features from CSI data"""
    if len(amplitudes) == 0:
        return pd.DataFrame()
    
    n_packets = amplitudes.shape[0]
    n_subcarriers = amplitudes.shape[1]
    
    features = {}
    
    # Basic statistics per packet
    features['mean_amp'] = np.mean(amplitudes, axis=1)
    features['std_amp'] = np.std(amplitudes, axis=1)
    features['max_amp'] = np.max(amplitudes, axis=1)
    features['min_amp'] = np.min(amplitudes, axis=1)
    features['median_amp'] = np.median(amplitudes, axis=1)
    features['rssi'] = rssi_values
    
    # Subcarrier regions (if enough subcarriers)
    if n_subcarriers >= 64:
        features['region1_mean'] = np.mean(amplitudes[:, :16], axis=1)
        features['region2_mean'] = np.mean(amplitudes[:, 16:32], axis=1)
        features['region3_mean'] = np.mean(amplitudes[:, 32:48], axis=1)
        features['region4_mean'] = np.mean(amplitudes[:, 48:64], axis=1)
    elif n_subcarriers >= 32:
        features['region1_mean'] = np.mean(amplitudes[:, :16], axis=1)
        features['region2_mean'] = np.mean(amplitudes[:, 16:32], axis=1)
    
    # Energy and power
    features['total_energy'] = np.sum(amplitudes**2, axis=1)
    features['peak_to_peak'] = features['max_amp'] - features['min_amp']
    
    # Statistical moments (handle potential NaN)
    from scipy import stats as spstats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features['skewness'] = spstats.skew(amplitudes, axis=1)
        features['kurtosis'] = spstats.kurtosis(amplitudes, axis=1)
    
    # Replace NaN with 0
    features['skewness'] = np.nan_to_num(features['skewness'])
    features['kurtosis'] = np.nan_to_num(features['kurtosis'])
    
    return pd.DataFrame(features)

# Extract features for all conditions
print("\nExtracting features...")
baseline_feat = extract_features(baseline_amps, baseline_rssi)
healthy30_feat = extract_features(healthy_30_amps, healthy_30_rssi)
healthy45_feat = extract_features(healthy_45_amps, healthy_45_rssi)
diseased30_feat = extract_features(diseased_30_amps, diseased_30_rssi)
diseased45_feat = extract_features(diseased_45_amps, diseased_45_rssi)

# Add labels
if len(baseline_feat) > 0:
    baseline_feat['condition'] = 'Baseline'
if len(healthy30_feat) > 0:
    healthy30_feat['condition'] = 'Healthy_30cm'
if len(healthy45_feat) > 0:
    healthy45_feat['condition'] = 'Healthy_45cm'
if len(diseased30_feat) > 0:
    diseased30_feat['condition'] = 'Diseased_30cm'
if len(diseased45_feat) > 0:
    diseased45_feat['condition'] = 'Diseased_45cm'

# Combine all data
all_data = pd.concat([df for df in [baseline_feat, healthy30_feat, healthy45_feat, 
                                      diseased30_feat, diseased45_feat] if len(df) > 0], 
                     ignore_index=True)

print(f"\nTotal samples: {len(all_data)}")
if len(all_data) > 0:
    print("\nSamples per condition:")
    print(all_data['condition'].value_counts())

# Statistical comparison function
def compare_conditions(data1, data2, metric, name1, name2):
    """Compare two conditions using t-test"""
    if len(data1) == 0 or len(data2) == 0:
        return None
    
    t_stat, p_value = ttest_ind(data1[metric], data2[metric])
    
    # Calculate effect size (Cohen's d)
    from numpy import mean, std
    from math import sqrt
    d = (mean(data1[metric]) - mean(data2[metric])) / sqrt((std(data1[metric], ddof=1)**2 + std(data2[metric], ddof=1)**2) / 2)
    
    return {
        'comparison': f'{name1} vs {name2}',
        'metric': metric,
        'mean1': mean(data1[metric]),
        'mean2': mean(data2[metric]),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'significant': p_value < 0.05
    }

print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Compare Baseline vs Healthy (30cm)
if len(baseline_feat) > 0 and len(healthy30_feat) > 0:
    result = compare_conditions(baseline_feat, healthy30_feat, 'mean_amp', 'Baseline', 'Healthy_30cm')
    if result:
        print(f"\n📊 Baseline vs Healthy 30cm:")
        print(f"   Baseline mean amplitude: {result['mean1']:.2f}")
        print(f"   Healthy mean amplitude: {result['mean2']:.2f}")
        print(f"   Reduction: {(1 - result['mean2']/result['mean1'])*100:.1f}%")
        print(f"   T-test: t={result['t_stat']:.3f}, p={result['p_value']:.6f}")
        print(f"   Cohen's d: {result['cohens_d']:.3f} {'(Large effect)' if abs(result['cohens_d']) > 0.8 else '(Medium effect)' if abs(result['cohens_d']) > 0.5 else '(Small effect)'}")

# Compare Healthy vs Diseased (30cm)
if len(healthy30_feat) > 0 and len(diseased30_feat) > 0:
    result = compare_conditions(healthy30_feat, diseased30_feat, 'mean_amp', 'Healthy_30cm', 'Diseased_30cm')
    if result:
        print(f"\n📊 Healthy vs Diseased 30cm:")
        print(f"   Healthy mean amplitude: {result['mean1']:.2f}")
        print(f"   Diseased mean amplitude: {result['mean2']:.2f}")
        print(f"   Difference: {(1 - result['mean2']/result['mean1'])*100:.1f}% lower for diseased")
        print(f"   T-test: t={result['t_stat']:.3f}, p={result['p_value']:.6f}")
        print(f"   Cohen's d: {result['cohens_d']:.3f}")

# Compare 30cm vs 45cm for diseased
if len(diseased30_feat) > 0 and len(diseased45_feat) > 0:
    result = compare_conditions(diseased30_feat, diseased45_feat, 'mean_amp', 'Diseased_30cm', 'Diseased_45cm')
    if result:
        print(f"\n📊 Diseased 30cm vs 45cm:")
        print(f"   30cm mean amplitude: {result['mean1']:.2f}")
        print(f"   45cm mean amplitude: {result['mean2']:.2f}")
        print(f"   T-test: t={result['t_stat']:.3f}, p={result['p_value']:.6f}")

# Machine Learning Classification
if len(baseline_feat) > 0 and len(healthy30_feat) > 0 and len(diseased30_feat) > 0:
    print("\n" + "="*60)
    print("MACHINE LEARNING CLASSIFICATION")
    print("="*60)
    
    feature_cols = ['mean_amp', 'std_amp', 'max_amp', 'min_amp', 'median_amp', 
                    'rssi', 'total_energy', 'skewness', 'kurtosis']
    
    # Use only columns that exist
    feature_cols = [col for col in feature_cols if col in baseline_feat.columns]
    
    X = []
    y = []
    
    X.extend(baseline_feat[feature_cols].values)
    y.extend([0] * len(baseline_feat))
    
    X.extend(healthy30_feat[feature_cols].values)
    y.extend([1] * len(healthy30_feat))
    
    X.extend(diseased30_feat[feature_cols].values)
    y.extend([2] * len(diseased30_feat))
    
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Random Forest Classifier (Baseline vs Healthy vs Diseased)")
    print(f"   Test Accuracy: {acc*100:.2f}%")
    
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
    print(f"   5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n📈 Top 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

# Summary for paper
print("\n" + "="*60)
print("SUMMARY FOR RESEARCH PAPER")
print("="*60)

if len(baseline_feat) > 0 and len(healthy30_feat) > 0:
    baseline_mean = baseline_feat['mean_amp'].mean()
    healthy_mean = healthy30_feat['mean_amp'].mean()
    print(f"\n• Plant presence reduced CSI amplitude by {(1 - healthy_mean/baseline_mean)*100:.1f}%")

if len(healthy30_feat) > 0 and len(diseased30_feat) > 0:
    diseased_mean = diseased30_feat['mean_amp'].mean()
    print(f"• Diseased plant showed {(1 - diseased_mean/healthy_mean)*100:.1f}% lower amplitude than healthy plant")

if len(baseline_feat) > 0 and len(healthy30_feat) > 0:
    baseline_rssi_mean = baseline_rssi.mean()
    healthy_rssi_mean = healthy_30_rssi.mean()
    print(f"• RSSI dropped by {baseline_rssi_mean - healthy_rssi_mean:.1f} dB with plant presence")

print("\n✅ Analysis complete! Results ready for paper.")