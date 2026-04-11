\# Complete Results Summary - CSI Plant Disease Detection



\## 📊 Traditional Machine Learning Results



| Model | Accuracy | F1-Score | MCC | Notes |

|-------|----------|----------|-----|-------|

| \*\*XGBoost (Tuned)\*\* | \*\*86.23%\*\* | 0.840 | 0.750 | Best overall |

| \*\*Ensemble (XGB+RF+GB)\*\* | \*\*86.86%\*\* | 0.850 | - | Soft voting |

| Random Forest (Tuned) | 85.14% | - | - | |

| Gradient Boosting | 84.50% | - | - | |

| SVM | 79.85% | - | - | |



\## 🧠 Deep Learning Results (PyTorch)



| Model | Test Accuracy | Notes |

|-------|---------------|-------|

| \*\*MLP (3 layers)\*\* | \*\*79.85%\*\* | Best DL model |

| 1D CNN | 77.60% | 3 convolutional layers |

| LSTM | 76.85% | 2 LSTM layers |

| CNN-LSTM Hybrid | 75.59% | Combined architecture |



\## 🎯 Per-Class Performance (XGBoost)



| Class | Precision | Recall | F1-Score |

|-------|-----------|--------|----------|

| Baseline | 83.1% | 83.1% | 83.1% |

| Healthy | \*\*87.5%\*\* | \*\*87.5%\*\* | \*\*87.5%\*\* |

| Diseased | \*\*87.5%\*\* | \*\*87.5%\*\* | \*\*87.5%\*\* |



\## ⚡ Real-Time Performance



| Metric | Value |

|--------|-------|

| Inference Time | 50.35 ± 6.65 ms |

| Processing Speed | 20 FPS |

| Hardware Cost | < $20 |



\## 📈 Cross-Validation Results (10-fold)



| Metric | Mean | Std |

|--------|------|-----|

| Accuracy | 86.23% | ±1.71% |

| Balanced Accuracy | 84.94% | ±1.64% |

| F1-Score | 86.20% | ±1.71% |



\## 🔬 Statistical Significance



| Comparison | p-value | Significant |

|------------|---------|-------------|

| Healthy vs Diseased | < 0.001 | Yes |

| XGBoost vs Random Forest | 0.0101 | Yes |

| XGBoost vs SVM | < 0.0001 | Yes |



\## 📁 Dataset Summary



| Condition | Samples | Mean Amp | Mean RSSI |

|-----------|---------|----------|-----------|

| Baseline | 1,283 | 13.48 | -80.33 |

| Healthy (30cm) | 1,505 | 14.27 | -80.51 |

| Healthy (45cm) | 799 | 15.48 | -79.02 |

| Diseased (30cm) | 1,370 | 17.45 | -69.29 |

| Diseased (45cm) | 1,493 | 17.41 | -74.77 |



\## 💡 Key Findings



1\. \*\*Traditional ML outperforms Deep Learning\*\* by 6.38% (86.23% vs 79.85%)

2\. \*\*XGBoost is the best model\*\* with 86.23% CV accuracy

3\. \*\*Real-time capable\*\* at 20 FPS (50ms per prediction)

4\. \*\*Cost-effective\*\* at < $20 hardware

5\. \*\*Distance robust\*\* - consistent detection at 30-45cm

