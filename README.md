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

| File ID | Cutting Speed (m/min) | Feed Rate (mm/z) | Chatter Onset (%) |
| :--- | :---: | :---: | :--- |
| **Cutting 1** | 100 | 0.055 | Stable |
| **Cutting 2** | 110 | 0.060 | Stable |
| **Cutting 3** | 110 | 0.065 | Stable |
| **Cutting 4** | 110 | 0.070 | Stable |
| **Cutting 5** | 120 | 0.060 | Stable |
| **Cutting 6** | 120 | 0.065 | Stable |
| **Cutting 7** | 120 | 0.070 | Stable |
| **Cutting 8** | 115 | 0.055 | Stable |
| **Cutting 9** | 115 | 0.060 | Stable |
| **Cutting 10**| 115 | 0.065 | Stable |
| **Cutting 11**| 115 | 0.070 | Stable |
| **Cutting A** | 120 | 0.055 | 45 |
| **Cutting B** | 125 | 0.055 | 9.5 |
| **Cutting C** | 125 | 0.060 | 24 |
| **Cutting D** | 125 | 0.065 | 22 |
| **Cutting E** | 125 | 0.070 | 31 |
| **Cutting F** | 130 | 0.055 | 39 |
| **Cutting G** | 130 | 0.060 | 30 |
| **Cutting H** | 130 | 0.065 | 38 |
| **Cutting I** | 130 | 0.070 | 31 |


## 📂 Project Architecture (The 10-Step Pipeline)

The project is structured into 10 sequential Jupyter Notebooks, demonstrating a complete end-to-end engineering workflow from raw sensor processing to final physical validation:

### Phase 1: Data Processing & Feature Engineering

The foundation of the project relies on extracting meaningful physical features from massive raw sensor logs. This phase executes a precise, programmatic cleaning and transformation pipeline:

* **`01_sep_cutting_lines.ipynb` (Signal Isolation):** Industrial raw data contains significant idle times, machine positioning, and entry/exit shocks. This script algorithmically isolates the pure steady-state cutting region by filtering specific kinematic conditions (e.g., `Z_Kafa_HMI < 0` and `Kafa.Act.Pos < 444`). This ensures the model strictly learns from actual cutting dynamics.

* **`02_sep_nec_columns.ipynb` (Targeting Sensors):** Reduces the massive dataset by extracting only the critical sensor streams required for the analysis: **Motor Load Percentage (`IMotor`)**, **Torque Force applied to the saw head (`Kafa_Act_Trq`)**, and Tri-axial Acceleration (`X`, `Y`, `Z_Accelerometer`).

* **`03_calculations.ipynb` (Rolling Window Feature Extraction & Labeling):** *The core of the data engineering process.* Raw signals are naturally extremely noisy. To capture the true dynamic trend, the data is processed using a **Rolling Window algorithm** (10-row windows sliding row-by-row). Within each window, complex statistical features are computed:
    * **Standard Deviations (Std):** Measures the spread and harmonic nature of vibrations.
    * **High-Frequency RMS:** Calculates the Root Mean Square of the signal's *first derivative* to capture sudden shocks and high-frequency anomalies.
    * **Chatter Labeling Strategy:** The "Chatter" (1) and "Stable" (0) labels are not applied blindly. For experiments known to exhibit chatter (Lettered files), the `1` label is applied dynamically only after a specific "Percentage Complete" threshold is reached. These thresholds correspond to the precise moments when chatter became physically observable and audibly identified during the experimental execution.

* **`04_prep_training_data.ipynb` (Dataset Assembly):** Merges all processed, calculated, and labeled files into a single master training dataset. It performs Exploratory Data Analysis (EDA) to verify the class distribution (Stable vs. Chatter rows) before feeding it into the ML algorithm.

### Phase 2: Machine Learning & Robust Evaluation

This phase defines the core predictive engine of the project. Rather than utilizing a simple train/test split, which often leads to data leakage in time-series data, a rigorous, highly controlled validation architecture was implemented:

* **`05_XGBoost_Model.ipynb` (Model Configuration & Training):**
    * **Feature Selection:** The model is fed not just raw statistics, but complex engineered metrics including Standard Deviations, High-Frequency Noise (RMS), Trend (Direction/Acceleration), and Ratio-to-Past values, providing necessary physical context.
    * **Addressing Class Imbalance:** Since chatter (1) is a rare anomaly compared to stable cutting (0), a dynamic `scale_pos_weight` was calculated and applied to penalize the model more heavily for missing a chatter event than for generating a false alarm.
    * **Anti-Overfitting Arsenal:** The XGBoost classifier is strictly constrained to prevent the memorization of specific experiment conditions. A shallow tree depth (`max_depth=4`) and a slow learning rate (`0.05`) were utilized, combined with both L1 (`reg_alpha=0.5`) and L2 (`reg_lambda=2.0`) regularization.

* **Strict Leave-One-Out Cross-Validation (LOOCV) Architecture:**
    To guarantee generalization to completely unseen cutting conditions (different speeds and feed rates), a custom LOOCV loop was constructed:
    1. **Isolation:** In each of the 20 folds, one entire experiment file is isolated as the "Test Set". The model has absolutely no access to this data during training.
    2. **Dynamic Validation Balancing:** From the remaining 19 files, exactly **1 Stable file (Numbered) and 1 Chatter file (Lettered)** are randomly selected to act as the "Validation Set".
    3. **Training & Early Stopping:** The model is trained on the remaining 17 files. Its performance is continuously evaluated on the balanced 2-file Validation Set. Using the `clone()` function, a fresh, blank model is spawned for every fold to ensure no memory leakage occurs from previous iterations.
    4. **Inference:** The model is finalized via early stopping (halting if validation performance degrades) and then makes predictions on the unseen Test file. This process is repeated 20 times, yielding true, unbiased predictions for every single millisecond of every experiment.

### Phase 3: Performance Verification & Ground Truth Alignment

This phase focuses on validating the predictive outputs by comparing model probabilities with physical observations across the time series.

* **`06_Model_vs_GroundTruth.ipynb` (Window-based Probability Analysis):** Before proceeding to Explainable AI (XAI) analysis, the model's probability outputs were evaluated for each 1-second window (calculated via a 10-sample rolling window) and compared against the Ground Truth. This visualization demonstrates the algorithm's responsiveness to the onset of instability. By mapping the predicted probabilities alongside the true class labels, the specific alignment between probability spikes and the actual chatter zones was clearly identified and analyzed for every second of the cutting process.

