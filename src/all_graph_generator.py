"""
COMPLETE GRAPH GENERATOR FOR CSI PLANT DISEASE PAPER
Generates all publication-ready graphs using extracted features
Saves output to csi_plant_data directory
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for publication
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Set paths
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'
output_path = data_path

print("="*80)
print("COMPLETE GRAPH GENERATOR FOR CSI PLANT DISEASE PAPER")
print("="*80)

# Load the extracted features
print("\n📂 Loading feature files...")

# Load basic data (using original CSI data for raw features)
def load_raw_csi_data(filename, label, max_packets=None):
    """Load raw CSI data for plotting"""
    import re
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
        
        if len(amp_list) >= 32:
            amp_32 = np.array(amp_list[:32])
            feat = {
                'label': label,
                'rssi': rssi,
                'mean_amp': np.mean(amp_32),
                'std_amp': np.std(amp_32),
                'max_amp': np.max(amp_32),
                'min_amp': np.min(amp_32),
            }
            for i, val in enumerate(amp_32):
                feat[f'subcarrier_{i}'] = val
            features.append(feat)
            
            if max_packets and len(features) >= max_packets:
                break
    
    return pd.DataFrame(features)

# Load data
print("Loading baseline data...")
baseline_df = load_raw_csi_data('baseline_no_plant.txt', 0, max_packets=799)
print("Loading healthy data...")
healthy_df = load_raw_csi_data('with_plant_30cm.txt', 1, max_packets=799)
print("Loading diseased data...")
diseased_df = load_raw_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)

all_data = pd.concat([baseline_df, healthy_df, diseased_df], ignore_index=True)
print(f"Loaded {len(all_data)} samples")

# Prepare features for ML
feature_cols = ['rssi', 'mean_amp', 'std_amp', 'max_amp', 'min_amp']
X = all_data[feature_cols].values
y = all_data['label'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier for some plots
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# Colors and labels
colors = ['gray', 'green', 'red']
labels = ['Baseline', 'Healthy', 'Diseased']

print("\n📊 Generating Graph 1: Subcarrier-wise Statistical Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, df) in enumerate(zip(labels, [baseline_df, healthy_df, diseased_df])):
    subcarrier_means = [df[f'subcarrier_{j}'].mean() for j in range(32)]
    subcarrier_stds = [df[f'subcarrier_{j}'].std() for j in range(32)]
    ax.plot(subcarrier_means, color=colors[i], label=name, linewidth=2)
    ax.fill_between(range(32), 
                    np.array(subcarrier_means) - np.array(subcarrier_stds),
                    np.array(subcarrier_means) + np.array(subcarrier_stds),
                    alpha=0.2, color=colors[i])
ax.set_xlabel('Subcarrier Index')
ax.set_ylabel('Mean Amplitude')
ax.set_title('Subcarrier-wise Amplitude Profile (±1σ)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_1_subcarrier_profile.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_1_subcarrier_profile.png")

print("\n📊 Generating Graph 2: Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = all_data[['rssi', 'mean_amp', 'std_amp', 'max_amp', 'min_amp']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_2_correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_2_correlation_matrix.png")

print("\n📊 Generating Graph 3: PCA 2D Projection")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(10, 8))
for i, label in enumerate(np.unique(y)):
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], 
               label=labels[i], alpha=0.6, s=30)
    centroid = np.mean(X_pca[mask], axis=0)
    ax.scatter(centroid[0], centroid[1], c=colors[i], marker='X', 
               s=200, edgecolors='black', linewidths=2)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA Visualization of CSI Features')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_3_pca_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_3_pca_visualization.png")

print("\n📊 Generating Graph 4: Learning Curves")
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_scaled, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)
fig, ax = plt.subplots(figsize=(10, 6))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation Accuracy')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curves (Random Forest)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_4_learning_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_4_learning_curves.png")

print("\n📊 Generating Graph 5: Precision-Recall Curves")
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_scaled, y)
y_prob = clf_rf.predict_proba(X_scaled)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, class_name) in enumerate(zip(axes, labels)):
    precision, recall, _ = precision_recall_curve(y == i, y_prob[:, i])
    ap_score = average_precision_score(y == i, y_prob[:, i])
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{class_name} (AP = {ap_score:.3f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_5_precision_recall_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_5_precision_recall_curves.png")

print("\n📊 Generating Graph 6: Normalized Feature Boxplot")
scaler_mm = MinMaxScaler()
X_norm = scaler_mm.fit_transform(all_data[feature_cols])
df_plot = pd.DataFrame(X_norm, columns=feature_cols)
df_plot['condition'] = all_data['label'].map({0: 'Baseline', 1: 'Healthy', 2: 'Diseased'})
df_melted = df_plot.melt(id_vars=['condition'], var_name='Feature', value_name='Normalized Value')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_melted, x='Feature', y='Normalized Value', hue='condition', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Normalized Feature Distribution by Condition')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_6_feature_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_6_feature_boxplot.png")

print("\n📊 Generating Graph 7: Violin Plots for Key Features")
key_features = ['rssi', 'mean_amp']
titles = ['RSSI (dBm)', 'Mean Amplitude']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat, title in zip(axes, key_features, titles):
    data = [baseline_df[feat], healthy_df[feat], diseased_df[feat]]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_7_violin_plots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_7_violin_plots.png")

print("\n📊 Generating Graph 8: RSSI vs Mean Amplitude Scatter")
fig, ax = plt.subplots(figsize=(10, 8))
for i, (name, df) in enumerate(zip(labels, [baseline_df, healthy_df, diseased_df])):
    ax.scatter(df['rssi'], df['mean_amp'], c=colors[i], label=name, alpha=0.5, s=10)
ax.set_xlabel('RSSI (dBm)')
ax.set_ylabel('Mean Amplitude')
ax.set_title('RSSI vs Mean Amplitude Relationship')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_8_rssi_vs_amplitude.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_8_rssi_vs_amplitude.png")

print("\n📊 Generating Graph 9: Class Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
class_counts = [len(baseline_df), len(healthy_df), len(diseased_df)]
bars = ax.bar(labels, class_counts, color=colors)
ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Class Distribution')
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
            str(count), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_9_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_9_class_distribution.png")

print("\n📊 Generating Graph 10: Feature Importance")
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_importance.fit(X_scaled, y)
importances = rf_importance.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(feature_cols)), importances[indices][::-1], color='steelblue')
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels([feature_cols[i] for i in indices[::-1]])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'graph_10_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: graph_10_feature_importance.png")

print("\n" + "="*80)
print("✅ ALL 10 GRAPHS GENERATED SUCCESSFULLY!")
print("="*80)
print("\n📁 FILES GENERATED (saved in csi_plant_data folder):")
print("  1. graph_1_subcarrier_profile.png")
print("  2. graph_2_correlation_matrix.png")
print("  3. graph_3_pca_visualization.png")
print("  4. graph_4_learning_curves.png")
print("  5. graph_5_precision_recall_curves.png")
print("  6. graph_6_feature_boxplot.png")
print("  7. graph_7_violin_plots.png")
print("  8. graph_8_rssi_vs_amplitude.png")
print("  9. graph_9_class_distribution.png")
print(" 10. graph_10_feature_importance.png")