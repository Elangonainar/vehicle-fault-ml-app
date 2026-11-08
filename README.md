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
| **Logistic Regression** | **0.950** | **0.9937** | **0.9655** | **0.9655** | **0.9655** | **0.8746** |
| **Decision Tree** | 0.900 | 0.8934 | 0.9630 | 0.8966 | 0.9286 | 0.7680 |
| **kNN** | 0.900 | 0.8918 | 0.9032 | 0.9655 | 0.9333 | 0.7408 |
| **Naive Bayes** | 0.850 | 0.9624 | 0.8485 | 0.9655 | 0.9032 | 0.6005 |
| **Support Vector Machine** | 0.925 | 0.9687 | 0.9062 | 1.0000 | 0.9508 | 0.8118 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall performance (Accuracy: 95.00%, AUC: 0.9937, MCC: 0.8746)[cite: 3, 4]. Physical vehicle sensor degradation follows smooth linear logit trends, which Logistic Regression captures effectively without overfitting[cite: 2, 3]. |
| **Decision Tree** | Obtained an accuracy of 90.00% and F1 score of 92.86%[cite: 3, 4]. While providing clear interpretability, axis-aligned splits result in stepped decision boundaries that moderately capture feature thresholds[cite: 2, 3]. |
| **kNN** | Recorded an accuracy of 90.00% and an AUC of 0.8918[cite: 3, 4]. Distance-based classification performs well following standard scaling, though bounded by feature space boundaries[cite: 2, 3]. |
| **Naive Bayes** | Achieved solid performance with 85.00% accuracy, 96.55% recall, and an AUC of 0.9624[cite: 3, 4]. Assuming conditional independence between sensor channels provides a stable baseline for fault screening[cite: 2, 3]. |
| **Support Vector Machine** | Delivered strong performance with 92.50% accuracy, 100% recall, an AUC score of 0.9687, and 0.8118 MCC[cite: 3, 4]. The RBF kernel effectively separates non-linear sensor threshold intersections[cite: 2, 3]. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the clear winner[cite: 4], achieving superior accuracy (95.00%), top-tier AUC (0.9937), and the highest overall Matthews Correlation Coefficient (0.8746)[cite: 3, 4]. |

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
python model/train_models.py

### 4. Bash command to run streamlit app
streamlit run app.py
