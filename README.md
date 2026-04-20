CSI-Based Plant Disease Detection using ESP32





\[!\[Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

\[!\[ESP-IDF](https://img.shields.io/badge/ESP--IDF-5.3-green.svg)](https://docs.espressif.com/projects/esp-idf/)

\[!\[License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



🎯 Overview



This project demonstrates a low-cost, non-invasive plant disease detection system using WiFi Channel State Information (CSI) captured by dual ESP32 microcontrollers. The system achieves 86.23% cross-validation accuracy in distinguishing between healthy and diseased plants, with real-time inference capability at 20 FPS.



Hardware Cost: < $20 | Accuracy: 86.23% | Real-time: 20 FPS



📊 Key Results



| Metric    		          | Value 	       |

|-------------------------|----------------|

| 10-Fold CV Accuracy 	  | 86.23% ± 1.71% |

| Healthy Plant F1-Score  | 85.9% 	       |

| Diseased Plant F1-Score | 85.8% 	       |

| Inference Speed	        | 20 FPS (50ms)  |

| Hardware Cost		        | < $20	         |





🏗️ System Architecture

┌───────────────┐        WiFi 2.4 GHz        	┌───────────────┐        ┌───────────────┐

│   ESP32 TX     │ ─────────────────────────► │     PLANT     │ ─────► │   ESP32 RX    │

│ (Transmitter)  │                            │  (Diseased)  	│        │  (Receiver)   │

└───────────────┘                            	└───────────────┘        └──────┬────────┘

&#x20;                                                                        │

&#x20;                                                                         ▼

&#x20;                                                                   ┌───────────────┐

&#x20;                                                                   │   ML Model    	│

&#x20;                                                                   │   XGBoost     	│

&#x20;                                                                   └───────────────┘







🔧 Hardware Requirements



| Component 	        	| Specification 	      | Cost |

|-----------------------|-----------------------|------|

| ESP32 DevKit (x2) 	  | ESP32-D0WD-V3 	      | \~$15 |

| USB Cables (x2) 	    | USB-A to Micro-USB   	| \~$5  |





📦 Software Requirements



```bash

Python packages

pip install numpy pandas scikit-learn xgboost lightgbm catboost

pip install matplotlib seaborn scipy

pip install torch  



ESP-IDF framework (for firmware)

Download from: https://docs.espressif.com/projects/esp-idf/

## 🔬 Feature Extraction Details (April 15, 2026)

Today we implemented comprehensive feature extraction pipelines to capture multiple aspects of CSI signals for plant disease detection.

### 📊 Feature Categories

| Category | Features | Purpose |
|----------|----------|---------|
| **Advanced Statistical** | 27 features | Mean, std, skewness, kurtosis, percentiles, crest factor, shape factor, impulse factor, margin factor |
| **Frequency Domain** | 28 features | FFT, PSD, spectral centroid, spectral entropy, spectral flatness, low/mid/high frequency energy |
| **Subcarrier Interaction** | 44 features | Adjacent differences, symmetry analysis, peak detection, cross-correlation, subcarrier group ratios |
| **Temporal Features** | Rolling stats, derivatives, autocorrelation, change detection, stability metrics |
| **Time-Frequency** | Spectrogram, wavelet transform (CWT), Hilbert transform, instantaneous features |
| **Entropy & Complexity** | Shannon entropy, Rényi entropy, Tsallis entropy, sample entropy, permutation entropy, SVD entropy, fractal dimension, Lempel-Ziv complexity |

### 📈 Key Findings from Feature Analysis

| Feature | Correlation with Disease | Insight |
|---------|------------------------|---------|
| RSSI | 0.449 | Strongest individual feature |
| SVD Entropy | 0.439 | Diseased plants show lower complexity |
| Sample Entropy | 0.416 | Diseased plants have reduced signal complexity |
| Mean Amplitude | 0.354 | Diseased plants show higher amplitude |
| Spectral Entropy | 0.430 | Diseased plants have more ordered frequency content |

### 📁 Generated Files

All feature extraction results are available in the `results/` folder:

- `advanced_statistical_features.csv` - 27 statistical features
- `frequency_domain_features.csv` - 28 frequency features
- `subcarrier_interaction_features.csv` - 44 subcarrier features
- `temporal_features_*.csv` - Temporal features by condition
- `time_frequency_features.csv` - Wavelet & spectrogram features
- `entropy_complexity_features.csv` - Entropy & complexity metrics

### 📊 Publication-Ready Graphs

10 graphs generated for the paper (available in `results/`):

1. **Subcarrier Profile** - Amplitude patterns across 32 subcarriers
2. **Correlation Matrix** - Feature relationships
3. **PCA Visualization** - 2D projection showing class separation
4. **Learning Curves** - Model training progress
5. **Precision-Recall Curves** - Per-class performance
6. **Feature Boxplot** - Normalized feature distributions
7. **Violin Plots** - RSSI and Mean Amplitude distributions
8. **RSSI vs Amplitude** - Relationship scatter plot
9. **Class Distribution** - Dataset balance
10. **Feature Importance** - Top discriminative features

### 🚀 How to Reproduce

```bash
# Run all feature extraction scripts
python src/advanced_statistical_features.py
python src/frequency_domain_features.py
python src/subcarrier_interaction_features.py
python src/temporal_features.py
python src/time_frequency_features.py
python src/entropy_complexity_features.py

# Generate all graphs
python src/all_graph_generator.py


