# ==============================================================================
# Description: Fits all 5 required ML models on the instrumental vehicle dataset
#              and saves the trained models as serialized artifacts (.pkl).
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

def train_and_save_models():
    # 1. Load Telematics Dataset
    data_path = '../test_data.csv' if os.path.exists('../test_data.csv') else 'test_data.csv'
    df = pd.read_csv(data_path)
    
    if 'vehicle_fault_status' not in df.columns:
        raise ValueError("Target column 'vehicle_fault_status' missing from dataset.")

    X = df.drop(columns=['vehicle_fault_status'])
    y = df['vehicle_fault_status']

    # 2. Train / Test Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Standard Preprocessing Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure output model directory exists
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')

    # 4. Instantiate all 5 Classification Models
    models = {
        'logistic_regression.pkl': LogisticRegression(random_state=42),
        'decision_tree.pkl': DecisionTreeClassifier(random_state=42, max_depth=5),
        'knn.pkl': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes.pkl': GaussianNB(),
        'svm.pkl': SVC(probability=True, random_state=42)
    }

    results = []

    # 5. Fit, Evaluate, and Save Models
    for file_name, model in models.items():
        if file_name in ['logistic_regression.pkl', 'knn.pkl', 'svm.pkl']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Save model pickle artifact
        save_path = os.path.join('model', file_name)
        joblib.dump(model, save_path)

        # Calculate metrics as specified in Section 2 of PDF
        results.append({
            'ML Model Name': file_name.replace('.pkl', ''),
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        })

    # Print summary table in terminal
    res_df = pd.DataFrame(results)
    print("\n================ EVALUATION METRICS SUMMARY ================")
    print(res_df.to_string(index=False))
    print("============================================================\n")

if __name__ == '__main__':
    train_and_save_models()