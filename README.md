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



