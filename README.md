# 🚀 Explainable AI for Industrial Saw Cutting: Chatter Characterization and Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![XAI](https://img.shields.io/badge/XAI-SHAP-green)
![Engineering](https://img.shields.io/badge/Domain-Mechanical%20Engineering-red)

## 📌 Project Overview
This project aims not only to detect the **chatter vibration** problem in industrial saw cutting processes but also to decode the physical characteristics of this vibration using **Explainable AI (XAI)**. 

Using raw sensor data (motor current and accelerometer) obtained from 21 different cutting experiments, an XGBoost model was developed and integrated with SHAP analysis. This allowed for the mathematical and physical verification of the root causes of signal dynamics and instabilities during chatter. The project moves beyond "black-box" models by validating machine learning decisions with raw physical signals.


### 📊 Experimental Dataset Summary
The dataset consists of 21 experimental runs with varying cutting speeds and feed rates. The experiments are categorized into "Stable" (Numbered files) and "Chatter" (Lettered files) based on physical observations.

The model was trained using raw signals arriving at a sampling interval of **100 ms**. The raw signal data used in this study is located at:  `/Users/rodigoze/Documents/GitHub/saw-cutting-chatter-detection-xai/saw-cutting-chatter/01_Raw_Data/01_raw_signals`

| File ID | Cutting Speed (m/min) | Feed Rate (mm/z) | Status |
| :--- | :---: | :---: | :--- |
| **Cutting 1** | 110 | 0.055 | Stable |
| **Cutting 2** | 110 | 0.060 | Stable |
| **Cutting 3** | 110 | 0.065 | Stable |
| **Cutting 4** | 110 | 0.070 | Stable |
| **Cutting 5** | 120 | 0.060 | Stable |
| **Cutting 6** | 120 | 0.065 | Stable |
| **Cutting 7** | 120 | 0.070 | Stable |
| **Cutting 8** | 115 | 0.055 | Stable |
| **Cutting 9** | 115 | 0.060 | Stable |
| **Cutting 10**| 115 | 0.065 | Stable |
| **Cutting A** | 125 | 0.050 | Chatter |
| **Cutting B** | 125 | 0.055 | Chatter |
| **Cutting C** | 125 | 0.060 | Chatter |
| **Cutting D** | 125 | 0.065 | Chatter |
| **Cutting E** | 125 | 0.070 | Chatter |
| **Cutting F** | 130 | 0.055 | Chatter |
| **Cutting G** | 130 | 0.060 | Chatter |
| **Cutting H** | 130 | 0.065 | Chatter |
| **Cutting I** | 130 | 0.070 | Chatter |
| **Cutting J** | 120 | 0.055 | Chatter |
| **Cutting K** | 115 | 0.070 | Chatter |

