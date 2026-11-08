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

https://vehiclefaultmlapp.streamlit.app/

## d. Models Used & Evaluation Metrics Comparison

All 5 classification models were trained on an 80/20 train-test split using standard scaling and evaluated on unseen test data ($N = 200$)[cite: 2, 3]:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9700 | 0.9979 | 0.9660 | 0.9930 | 0.9793 | 0.9260 |
| **Decision Tree** | 0.9550 | 0.9564 | 0.9653 | 0.9720 | 0.9686 | 0.8891 |
| **kNN** | 0.9000 | 0.9327 | 0.8968 | 0.9720 | 0.9329 | 0.7473 |
| **Naive Bayes** | 0.8950 | 0.9696 | 0.8812 | 0.9860 | 0.9307 | 0.7366 |
| **Support Vector Machine** | **0.9750** | **0.9975** | **0.9662** | **1.0000** | **0.9828** | **0.9389** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved high performance (Accuracy: 97.00%, AUC: 0.9979, MCC: 0.9260)[cite: 17]. Smooth linear logit boundaries capture automotive sensor degradation trends effectively[cite: 2, 3]. |
| **Decision Tree** | Secured strong performance with an accuracy of 95.50% and F1 score of 96.86%[cite: 13]. Recursive axis-aligned partitioning handles multi-sensor threshold interactions efficiently[cite: 2, 3]. |
| **kNN** | Recorded an accuracy of 90.00% and high recall of 97.20%[cite: 14]. Distance-based classification performs reliably following standard feature scaling[cite: 2, 3]. |
| **Naive Bayes** | Achieved 89.50% accuracy and an impressive 98.60% recall[cite: 15]. The conditional independence assumption provides a solid probabilistic baseline for fault detection[cite: 2, 3]. |
| **Support Vector Machine** | Delivered the **top overall performance** with 97.50% accuracy, 0.9975 AUC, perfect 100% recall, and an MCC of 0.9389[cite: 16]. The RBF kernel successfully maps complex non-linear sensor decision boundaries[cite: 2, 3]. |
| **Overall Winner for your dataset?** | **Support Vector Machine** is the clear winner[cite: 4], achieving superior accuracy (97.50%), a flawless recall of 1.0000, and the highest Matthews Correlation Coefficient (0.9389)[cite: 3, 4, 16]. |

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
