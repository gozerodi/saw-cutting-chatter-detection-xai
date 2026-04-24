# 🚀 Industrial Saw Cutting Chatter Detection via Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-green)
![Domain](https://img.shields.io/badge/Domain-Mechanical%20Engineering-red)

## 📌 Project Overview
This repository contains a comprehensive, 10-step data science and engineering pipeline designed to detect and explain **Chatter Vibration (Tırlama)** in industrial saw cutting processes. 

By bridging physical sensor dynamics (motor load, multi-axis acceleration) with advanced Machine Learning (XGBoost) and Explainable AI (SHAP), this project moves beyond simple "black-box" predictions. It successfully identifies the exact physical features that cause instability and provides mathematical proof of the machine's behavior during chatter.

## 🎯 Key Methodologies & Achievements
1. **Precise Signal Isolation:** Algorithmic separation of the active cutting region (`sep_cutting`) from transient entry/exit shocks to ensure pure steady-state analysis.
2. **Advanced Feature Engineering:** Calculating complex statistical metrics (High-Frequency RMS, Standard Deviations, Ratio to Past) from raw physical signals.
3. **Robust Detection (LOOCV):** Developing an XGBoost classifier evaluated via Leave-One-Out Cross-Validation (LOOCV) to guarantee real-world generalization across varying cutting speeds and feed rates.
    * **Model Accuracy:** `[BURAYA_ACCURACY_YAZIN]%`
    * **F1-Score:** `[BURAYA_F1_SCORE_YAZIN]`
4. **Physical Validation:** Proving the AI's logic by projecting SHAP values back onto the raw, unfiltered motor current and vibration signals.

---

## 📂 Project Architecture (The 10-Step Pipeline)

The project is structured into 10 sequential Jupyter Notebooks, demonstrating a complete end-to-end engineering workflow:

### Phase 1: Data Processing & Feature Engineering
* **`01_sep_cutting.ipynb`**: Imports raw sensor data and applies logic to isolate the actual steady-state cutting area, trimming out machine idle times and entry/exit transients.
* **`02_calculations.ipynb`**: Performs intensive feature engineering on the isolated cutting data. Calculates critical metrics such as `IMotor_HighFreq_RMS`, `VibRes_Accel_std`, and other axis-specific standard deviations.
* *(Notebooks 03-04 focus on further data merging, scaling, and preparing the final `06_Training_Data.csv`)*

### Phase 2: Machine Learning & Evaluation
* **`05_XGBoost_Model.ipynb`**: The core predictive engine. Trains the XGBoost classifier using LOOCV. Includes hyperparameter tuning, confusion matrix generation, threshold optimization, and extracts the final accuracy/performance metrics.

### Phase 3: Explainable AI (XAI) & Physical Proof
* **`06 to 09_SHAP_Analyses.ipynb`**: Deep dive into the "brain" of the model. Utilizes SHAP Summary plots, Waterfall plots for localized predictions, and Dependence plots to uncover how feature interactions drive the model's decisions.
* **`10_Raw_Signal_Dynamics_and_Physical_Validation.ipynb`**: The final physical proof. Maps the most critical features identified by SHAP (e.g., `IMotor_HighFreq_RMS` vs `VibRes_Accel_std`) back to the raw time-series data. Features dual-axis standardized plotting and peak/valley direction changes to physically demonstrate the severity of chatter.

---

## 🧠 The Physics meets AI: Key Findings
Through our SHAP and Raw Signal analyses, we uncovered a fundamental dynamic between the machine's motor and its structural vibration:
* **Chatter Driver (`IMotor_HighFreq_RMS`):** Acts as the primary indicator for instability. When the tool chatters, the motor draws erratic, high-frequency current to compensate for the dynamic loads.
* **Stabilization Marker (`VibRes_Accel_std`):** The model identifies specific standard deviations in acceleration as a stabilizing factor, capturing the natural harmonic resonance of a healthy cut before it collapses into chaotic chatter.

---

## ⚙️ Installation & Usage

### Prerequisites
Make sure you have Python installed. The required libraries include:
```bash
pip install pandas numpy matplotlib seaborn xgboost shap scikit-learn
