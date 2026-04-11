import matplotlib.pyplot as plt
import numpy as np

# Your actual results from the analysis
models = ['XGBoost\n(Tuned)', 'Random Forest\n(Tuned)', 'Gradient\nBoosting', 'SVM', 'Weighted\nVoting', 'Stacking\nEnsemble']
accuracies = [86.89, 85.14, 84.5, 79.85, 86.86, 86.48]
cv_accuracy = 88.23
cv_std = 1.49

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Model Comparison
ax1 = axes[0]
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
bars = ax1.bar(models, accuracies, color=colors)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim(70, 95)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center', fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Cross-Validation Results
ax2 = axes[1]
metrics = ['Accuracy', 'Balanced\nAccuracy', 'F1-Macro', 'F1-Weighted', 'MCC', 'Kappa']
values = [88.23, 87.07, 87.21, 88.21, 81.63, 81.59]
bars = ax2.bar(metrics, values, color='steelblue')
ax2.set_ylabel('Score (%)')
ax2.set_title('10-Fold Cross-Validation Results')
ax2.set_ylim(70, 95)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=10)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('final_results_corrected.png', dpi=300)
plt.show()

print("\n📊 Summary of Your Results:")
print("="*50)
print(f"Best CV Accuracy: 88.23% ± 1.49%")
print(f"Best Model: XGBoost with 86.89% CV accuracy")
print(f"Ensemble Accuracy: 86.86%")
print(f"Healthy Plant Detection F1: 87.5%")
print(f"Diseased Plant Detection F1: 87.5%")