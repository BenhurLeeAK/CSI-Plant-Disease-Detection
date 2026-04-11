"""
STEP 1: DATA EXPLORATION AND UNDERSTANDING
Fixed for variable length packets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set path
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'

def load_raw_data(filename):
    """Load raw CSI data without any processing"""
    full_path = f"{data_path}/{filename}"
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract all packets
    pattern = r'\[CSI #(\d+)\].*?RSSI:(-?\d+)\nAmp:\s*([\d\.\s]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    packets = []
    for match in matches:
        packets.append({
            'packet_num': int(match[0]),
            'rssi': int(match[1]),
            'amplitude': [float(x) for x in match[2].strip().split()]
        })
    
    return packets

# Load all datasets
print("="*80)
print("CSI PLANT DISEASE DETECTION - DATA EXPLORATION")
print("="*80)

datasets = {
    'Baseline': 'baseline_no_plant.txt',
    'Healthy_30cm': 'with_plant_30cm.txt',
    'Healthy_45cm': 'with_plant_45cm.txt',
    'Diseased_30cm': 'with_disease_plant_30cm.txt',
    'Diseased_45cm': 'with_disease_plant_45cm.txt'
}

raw_data = {}
for name, file in datasets.items():
    print(f"\nLoading {name}...")
    raw_data[name] = load_raw_data(file)
    print(f"  Packets: {len(raw_data[name])}")
    
    # Check amplitude lengths
    lengths = [len(p['amplitude']) for p in raw_data[name]]
    print(f"  Amplitude lengths: {set(lengths)}")
    print(f"  RSSI range: {min([p['rssi'] for p in raw_data[name]])} to {max([p['rssi'] for p in raw_data[name]])}")
    print(f"  Sample amplitude (first 20 values): {raw_data[name][0]['amplitude'][:20]}")

# Statistical summary (handle variable lengths)
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

summary = []
for name, data in raw_data.items():
    # Handle variable lengths - only use packets with standard length (64)
    # For Diseased_45cm, filter to only 64-subcarrier packets
    filtered_data = [p for p in data if len(p['amplitude']) == 64]
    
    if len(filtered_data) == 0:
        print(f"⚠️ {name}: No packets with 64 subcarriers, using first 64 values from all packets")
        amplitudes = np.array([p['amplitude'][:64] for p in data])
        rssi_values = np.array([p['rssi'] for p in data])
    else:
        amplitudes = np.array([p['amplitude'] for p in filtered_data])
        rssi_values = np.array([p['rssi'] for p in filtered_data])
        print(f"  {name}: Using {len(filtered_data)}/{len(data)} packets with 64 subcarriers")
    
    summary.append({
        'Dataset': name,
        'Packets_Total': len(data),
        'Packets_Used': len(amplitudes),
        'Subcarriers': amplitudes.shape[1],
        'Mean_Amp': np.mean(amplitudes),
        'Std_Amp': np.std(amplitudes),
        'Median_Amp': np.median(amplitudes),
        'Mean_RSSI': np.mean(rssi_values),
        'Std_RSSI': np.std(rssi_values),
        'Min_RSSI': np.min(rssi_values),
        'Max_RSSI': np.max(rssi_values)
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# Visualize basic distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. RSSI Distribution
ax = axes[0, 0]
for name, data in raw_data.items():
    rssi = [p['rssi'] for p in data]
    ax.hist(rssi, bins=30, alpha=0.5, label=name)
ax.set_xlabel('RSSI (dBm)')
ax.set_ylabel('Frequency')
ax.set_title('RSSI Distribution by Condition')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Mean Amplitude per Packet (first 100 packets, using 64 subcarriers)
ax = axes[0, 1]
for name, data in raw_data.items():
    # Use only packets with at least 64 subcarriers
    valid_packets = [p for p in data if len(p['amplitude']) >= 64]
    if valid_packets:
        means = [np.mean(p['amplitude'][:64]) for p in valid_packets[:100]]
        ax.plot(means, alpha=0.7, label=name)
ax.set_xlabel('Packet Number')
ax.set_ylabel('Mean Amplitude (first 64 subcarriers)')
ax.set_title('Mean Amplitude Trend')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Amplitude Heatmap (first packet of each condition, truncated to 64)
ax = axes[0, 2]
first_packets = []
labels = []
for name, data in raw_data.items():
    if len(data) > 0:
        # Take first 64 values
        first_packets.append(data[0]['amplitude'][:64])
        labels.append(name)
heatmap_data = np.array(first_packets)
im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Subcarrier Index')
ax.set_title('Amplitude Heatmap (First Packet, first 64 subcarriers)')
plt.colorbar(im, ax=ax)

# 4. Box plot of mean amplitudes (using 64 subcarriers)
ax = axes[1, 0]
box_data = []
box_labels = []
for name, data in raw_data.items():
    valid_packets = [p for p in data if len(p['amplitude']) >= 64]
    if valid_packets:
        box_data.append([np.mean(p['amplitude'][:64]) for p in valid_packets])
        box_labels.append(name)
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Mean Amplitude (first 64 subcarriers)')
ax.set_title('Mean Amplitude Distribution')
ax.tick_params(axis='x', rotation=45)

# 5. Subcarrier profile (average of all packets, using 64 subcarriers)
ax = axes[1, 1]
for name, data in raw_data.items():
    valid_packets = [p for p in data if len(p['amplitude']) >= 64]
    if valid_packets:
        all_amps = np.array([p['amplitude'][:64] for p in valid_packets])
        mean_profile = np.mean(all_amps, axis=0)
        std_profile = np.std(all_amps, axis=0)
        ax.plot(mean_profile, label=name, linewidth=1.5)
        ax.fill_between(range(64), mean_profile - std_profile, mean_profile + std_profile, alpha=0.1)
ax.set_xlabel('Subcarrier Index')
ax.set_ylabel('Average Amplitude')
ax.set_title('Average CSI Amplitude Profile (±1σ)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6. Packet length distribution
ax = axes[1, 2]
length_data = []
length_labels = []
for name, data in raw_data.items():
    lengths = [len(p['amplitude']) for p in data]
    length_data.append(lengths)
    length_labels.append(name)
bp = ax.boxplot(length_data, labels=length_labels)
ax.set_ylabel('Number of Subcarriers')
ax.set_title('Packet Length Distribution')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 500)

plt.tight_layout()
plt.savefig('01_data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional Analysis: Check Diseased_45cm packet lengths
print("\n" + "="*80)
print("DETAILED ANALYSIS: Diseased_45cm Packet Lengths")
print("="*80)
diseased_45 = raw_data['Diseased_45cm']
length_counts = Counter([len(p['amplitude']) for p in diseased_45])
print(f"Packet length distribution:")
for length, count in sorted(length_counts.items()):
    print(f"  {length} subcarriers: {count} packets ({count/len(diseased_45)*100:.1f}%)")

# Check if there's a pattern (e.g., alternating lengths)
print(f"\nFirst 20 packet lengths: {[len(p['amplitude']) for p in diseased_45[:20]]}")

print("\n✅ Data exploration complete! Check '01_data_exploration.png'")