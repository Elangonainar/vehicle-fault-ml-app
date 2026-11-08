# Instrumental Vehicle Telematics & Fault Classification System

## a. Problem Statement
Automotive predictive maintenance relies heavily on continuous real-time telemetry from instrumental vehicle sensors. The objective of this project is to build a robust binary classification model to predict whether a vehicle requires immediate mechanical maintenance/repair (`vehicle_fault_status = 1`) or is operating under normal, healthy conditions (`vehicle_fault_status = 0`). This prevents unexpected engine failures, optimizes operational uptime, and ensures driving safety.

## b. Dataset Description
The dataset contains **1,000 instances** and **14 continuous and discrete instrumental vehicle features** collected from automotive ECU sensors:
1. `engine_rpm`: Engine speed in revolutions per minute (800 – 6,500 RPM).
2. `vehicle_speed`: Vehicle ground velocity (0 – 160 km/h).
3. `engine_load`: Percentage engine load demand (10% – 100%).
4. `throttle_position`: Throttle valve opening percentage (0% – 100%).
5. `coolant_temp`: Engine coolant operating temperature (60°C – 115°C).
6. `intake_pressure`: Intake manifold absolute pressure (20 – 250 kPa).
7. `air_fuel_ratio`: Combustion chamber air-to-fuel ratio (10.0 – 18.0).
8. `oil_pressure`: Engine lubrication oil pressure (15 – 80 PSI).
9. `brake_pedal_status`: Binary indicator of brake application (0 = Released, 1 = Engaged).
10. `steering_angle`: Steering wheel deflection angle (-180° to +180°).
11. `vibration_amplitude`: Engine block vibration amplitude (0.1 – 9.5 mm/s).
12. `battery_voltage`: Vehicle electrical system DC voltage (11.0V – 14.8V).
13. `exhaust_gas_temp`: Exhaust pipe temperature (200°C – 900°C).
14. `fuel_injection_rate`: Fuel delivery rate (2.0 – 45.0 mL/s).

* **Target Variable**: `vehicle_fault_status` (0 = Normal, 1 = Maintenance Required).

## c. Github repository link
https://github.com/Elangonainar/vehicle-fault-ml-app.git

## d. Models Used & Evaluation Metrics Comparison

All 5 classification models were trained on an 80/20 train-test split using standard scaling and evaluated on unseen test data ($N = 200$)[cite: 2, 3]:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9850** | **0.9993** | **0.9795** | **1.0000** | **0.9896** | **0.9633** |
| **Decision Tree** | 0.7850 | 0.8355 | 0.8906 | 0.7972 | 0.8413 | 0.5187 |
| **kNN** | 0.8300 | 0.8855 | 0.8344 | 0.9510 | 0.8889 | 0.5550 |
| **Naive Bayes** | 0.9050 | 0.9780 | 0.8827 | 1.0000 | 0.9377 | 0.7671 |
| **Support Vector Machine** | 0.9450 | 0.9910 | 0.9400 | 0.9860 | 0.9625 | 0.8633 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall performance (Accuracy: 98.50%, AUC: 0.9993, MCC: 0.9633)[cite: 3, 4]. Physical vehicle sensor degradation follows smooth linear logit trends (e.g., temperature, vibration, voltage drops), which Logistic Regression captures exceptionally well without overfitting[cite: 2, 3]. |
| **Decision Tree** | Obtained an accuracy of 78.50% and F1 score of 0.8413[cite: 3, 4]. While providing clear interpretability, axis-aligned splits result in stepped decision boundaries that slightly underfit continuous multi-sensor interactions[cite: 2, 3]. |
| **kNN** | Recorded an accuracy of 83.00% and an AUC of 0.8855[cite: 3, 4]. Distance-based classification performs well after standard scaling, but performance is somewhat bounded by higher-dimensional feature space curvature[cite: 2, 3]. |
| **Naive Bayes** | Achieved solid performance with 90.50% accuracy, 100% recall, and an AUC of 0.9780[cite: 3, 4]. Assuming conditional independence between automotive sensor channels works surprisingly well for rapid fault screening[cite: 2, 3]. |
| **Support Vector Machine** | Delivered strong performance with 94.50% accuracy, 0.9910 AUC score, and 0.8633 MCC[cite: 3, 4]. The RBF kernel effectively separates non-linear sensor threshold intersections[cite: 2, 3]. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the clear winner[cite: 4], achieving superior precision (97.95%), perfect recall (100.00%), and the highest Matthews Correlation Coefficient (0.9633)[cite: 3, 4]. |

---

## 🚀 Step-by-Step Execution Guide

### 1. Clone or Setup Workspace
Ensure your project directory is structured as follows:
```text
vehicle-fault-ml-app/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── train_models.py
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── scaler.pkl

### 2. Bash command to install dependencies
pip install -r requirements.txt

### 3. Bash command to train the models
cd model
python train_models.py

### 4. Bash command to run streamlit app
streamlit run app.py
