# 🚀 Explainable AI (XAI) for Industrial Saw Cutting Chatter Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![XAI](https://img.shields.io/badge/XAI-SHAP-green)
![Domain](https://img.shields.io/badge/Domain-Mechanical%20Engineering-red)

## 📌 Project Overview
This repository contains the code, models, and analytical frameworks for detecting and explaining **Chatter Vibration (Tırlama)** in industrial saw cutting processes. 

By bridging the gap between **Mechanical Engineering** (signal processing, physical dynamics) and **Data Science** (Machine Learning, Explainable AI), this project doesn't just predict chatter—it explains *why* the model makes its decisions using physical raw sensor data.

## 🎯 Key Objectives
1. **Robust Detection:** Develop an XGBoost classifier evaluated via Leave-One-Out Cross-Validation (LOOCV) to prevent overfitting and ensure real-world generalization across different cutting conditions.
2. **Explainable AI (XAI):** Utilize SHAP (SHapley Additive exPlanations) to crack the "black box" of the ML model, revealing feature impacts down to the millisecond.
3. **Physical Validation:** Prove that the AI's mathematical findings perfectly align with raw, unfiltered physical sensor data (Current & Acceleration).

---

## 🧠 The Physics meets AI: Key Findings
Through extensive SHAP analysis, we uncovered a fascinating inverse relationship between the machine's spindle motor and its structural vibration:
* **High-Frequency Motor Load (`IMotor_HighFreq_RMS`):** Acts as the primary driver for instability. As the tool struggles and digs into the material during chatter, current fluctuations spike, pushing the model toward a "Chatter (1)" prediction.
* **Vibration Resolution (`VibRes_Accel_std`):** Surprisingly, the model identifies specific high standard deviations in acceleration as a stabilizing factor ("Stable (0)"), capturing the natural harmonic resonance of a healthy cut before it degrades into chaotic chatter.

---

## 📂 Project Architecture & Notebook Flow

The project is structured in a logical sequence, progressing from raw data to final physical validation:

* **`01_Data_Preprocessing.ipynb`**: Cleans and synchronizes raw sensor data. Extracts the 20% - 70% "steady-state" cutting region to eliminate entry/exit shock transients.
* **`02_Model_Training_and_LOOCV.ipynb`**: Trains the XGBoost model. Implements rigorous LOOCV, ensuring no data leakage occurs between different cutting experiments (e.g., varying speeds and feed rates). Includes threshold optimization.
* **`03_XAI_SHAP_Analysis.ipynb`**: Deep dive into the model's brain using SHAP Summary plots, Waterfall plots for local predictions, and Dependence plots to observe feature interactions.
* **`04_Physical_Validation_Dual_Sensors.ipynb`**: The final proof. Maps the model's calculated statistical features back to the **Raw Physical Signals**. Features dual-axis standardized plotting and peak/valley (direction change) counting to physically prove the chatter frequency.

---

## 📊 Visualizations & Analyses
This repository generates production-ready, thesis-grade visualizations including:
1. **SHAP Summary Plots:** Global feature importance.
2. **Dual-Sensor Dynamics (Twin-Axis):** Simultaneous plotting of Motor Load (Red) vs Vibration (Blue) with highlighted chatter zones.
3. **Raw Signal Direction Changes:** Algorithmic counting of peaks/valleys in raw current data to demonstrate chatter severity.
4. **KDE Density & Scatter Maps:** Clustering of stable vs. chatter data points in the feature space.

---

## ⚙️ Installation & Usage

### Prerequisites
Make sure you have Python 3.8+ installed. 

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/gozerodi/saw-cutting-chatter-detection-xai.git](https://github.com/gozerodi/saw-cutting-chatter-detection-xai.git)
   cd saw-cutting-chatter-detection-xai
