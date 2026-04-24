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


### Phase 4: Statistical Optimization & Final Model Deployment

Following the initial performance verification, this phase focuses on mathematically optimizing the model's decision boundaries and deploying the final algorithm using the insights gained from the cross-validation process.

* **`07_ROC_and_Threshold.ipynb` (Boundary Optimization):** * **ROC Analysis:** A Receiver Operating Characteristic (ROC) analysis was conducted on the accumulated LOOCV predictions. The model achieved an Area Under the Curve (AUC) score of **0.9684**. This exceptionally high AUC provides mathematical proof that the algorithm possesses an excellent ability to distinguish between stable cutting and chatter, regardless of the chosen threshold.
    * **Threshold Tuning:** To maximize the harmonic mean of Precision and Recall, an F1 Score optimization was performed across all possible thresholds. The analysis identified the optimal classification threshold at **0.4965** (effectively 0.50), confirming the default probability boundary and ensuring the model does not suffer from extreme false positive or false negative bias.

* **`08_Final_Model.ipynb` (Full Dataset Training):**
    With the hyperparameters verified and the decision threshold mathematically justified, the final XGBoost classifier was assembled. Instead of relying on an arbitrary number of trees, the optimal `n_estimators` value (**303 trees**) was extracted by averaging the early-stopping iterations across all 20 LOOCV folds. The final model was then trained on the **entire 20-experiment dataset** using this optimal tree count. This final, fully informed model serves as the basis for the subsequent Explainable AI (XAI) and physical signal analyses.

    ### Phase 5: Explainable AI (XAI) & Physical Signal Validation

This phase cracks open the "black box" of the XGBoost model, explaining the mathematical reasoning behind its predictions and validating those statistical findings against actual mechanical signal dynamics.

* **`09_XAI_and_SHAP_Analysis.ipynb` (Decoding the Model Logic):**
    * **Global Explainability:** Traditional feature importance metrics were initially evaluated, followed by a comprehensive SHAP (SHapley Additive exPlanations) analysis. SHAP summary plots were utilized to reveal exactly how each statistical feature mathematically pushes the model's decision toward either a "Stable" or "Chatter" classification.
    * **Feature Interactions:** SHAP Dependence analyses were conducted to observe the complex, non-linear interactions between different engineered parameters during the cutting process.
    * **Local Explainability:** SHAP Waterfall plots were generated to dissect individual, specific predictions, providing a highly transparent breakdown of the exact mathematical reasoning behind single classification events. *(The specific physical insights derived from these analyses are detailed in the Key Findings section).*

* **`10_Physical_Validation_and_Signal_Dynamics.ipynb` (Bridging AI and Physics):**
    * **Raw Signal Mapping:** The critical parameters highlighted during the XAI phase (specifically focusing on motor load and vibration characteristics) were mapped directly back to the raw, unfiltered sensor data.
    * **Parametric Isolation:** The dynamic behavior of these physical signals was systematically compared between stable cutting regimes and confirmed chatter zones. Furthermore, rigorous parametric analyses were performed to observe signal evolution under isolated machine settings: evaluating variations while maintaining a constant cutting speed with differing feed rates, and conversely, maintaining a constant feed rate across varying cutting speeds.

## Results & Key Findings: Decoding Chatter Dynamics

This section details the culmination of the machine learning pipeline, presenting not only the predictive accuracy of the model but also the fundamental physical truths uncovered through Explainable AI (XAI) and raw signal validation.

### 1. Model Performance (The Foundation)

Before interpreting the physical causes of chatter, the reliability of the XGBoost model was established through rigorous cross-validation. The model demonstrated exceptional capability in distinguishing between stable cutting conditions and the onset of chatter vibrations.

* **Overall Accuracy & Reliability:** Evaluated over 25,307 distinct 1-second rolling windows across all experiments, the model achieved an overall accuracy of **91.2%**, supported by a robust Area Under the Curve (AUC) score of **0.9684**.
* **Class-Specific Performance:** The algorithm demonstrated high reliability across both physical states. For stable cutting windows (Class 0), an F1-Score of **0.937** was achieved. For the critical detection of chatter anomalies (Class 1), an F1-Score of **0.857** was recorded, effectively balancing precision and recall to ensure high sensitivity without generating excessive false alarms.

**Classification Report**
| Class | Precision | Recall | F1-Score | Support (Seconds) |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Stable)** | 0.943 | 0.930 | 0.937 | 17,647 |
| **1 (Chatter)** | 0.844 | 0.870 | 0.857 | 7,660 |

**Overall Confusion Matrix**
| | Predicted Stable (0) | Predicted Chatter (1) |
| :--- | :---: | :---: |
| **Actual Stable (0)** | 16,415 | 1,232 |
| **Actual Chatter (1)** | 994 | 6,666 |

### 2. XAI Discoveries: The Inverse Dynamics of Motor Load and Vibration

To decode the mathematical reasoning behind the model's predictions, a comprehensive SHAP (SHapley Additive exPlanations) analysis was conducted. The resulting feature impact evaluations revealed a critical, counter-intuitive physical dynamic between the spindle motor and the tool's structural vibration.

* **The Primary Driver of Chatter (`IMotor_HighFreq_RMS`):** High-frequency fluctuations in the motor current were identified as the most dominant feature in the dataset. High values of this metric strongly push the model's output toward a prediction of **1 (Chatter)**. This mathematically confirms that when the tool struggles during an unstable cut, the spindle motor draws erratic, high-frequency current spikes to compensate for the chaotic dynamic loads.
* **The Stabilizing Resonance (`VibRes_Accel_std`):** High standard deviations in the resultant acceleration were found to push the model's prediction toward **0 (Stable)**. Physical validation of the raw signals revealed that during chatter episodes, the vibration signal exhibits a lower deviation from its mean compared to stable cutting, yet demonstrates a significantly higher frequency of direction changes (peaks and valleys). Consequently, a high standard deviation is identified as a marker of the natural harmonic resonance of a healthy cut, whereas chatter is characterized by a constrained but high-frequency oscillation pattern with rapid direction changes.