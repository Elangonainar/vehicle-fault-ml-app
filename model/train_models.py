# ==============================================================================
# Description: Dynamically locates dataset files, fits all 5 required ML models,
#              evaluates them, and saves serialized artifacts (.pkl) in model/.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import os
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

def find_file(filename):
    paths = [
        filename,
        os.path.join('..', filename),
        os.path.join('model', filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), '..', filename)
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return filename # Default fallback

def train_and_save_models():
    # Dynamically locate training and test datasets
    train_file_path = find_file('vehicle_telematics_full.csv')
    test_file_path = find_file('test_data.csv')

    if os.path.exists(train_file_path):
        train_df = pd.read_csv(train_file_path)
    else:
        train_df = pd.read_csv(test_file_path)

    test_df = pd.read_csv(test_file_path)

    X_train = train_df.drop(columns=['vehicle_fault_status'])
    y_train = train_df['vehicle_fault_status']

    X_test = test_df.drop(columns=['vehicle_fault_status'])
    y_test = test_df['vehicle_fault_status']

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure model folder exists
    model_dir = 'model' if os.path.exists('model') else os.path.dirname(__file__)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # Model Dictionary
    models = {
        'logistic_regression.pkl': LogisticRegression(random_state=42),
        'decision_tree.pkl': DecisionTreeClassifier(random_state=42, max_depth=5),
        'knn.pkl': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes.pkl': GaussianNB(),
        'support_vector_machine.pkl': SVC(probability=True, random_state=42)
    }

    results = []

    # Fit, Evaluate, and Save Models
    for file_name, model in models.items():
        if file_name in ['logistic_regression.pkl', 'knn.pkl', 'support_vector_machine.pkl']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Save model pickle artifact
        save_path = os.path.join(model_dir, file_name)
        joblib.dump(model, save_path)

        # Compute Metrics
        metrics = {
            'Model': file_name.replace('.pkl', ''),
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        results.append(metrics)

    res_df = pd.DataFrame(results)
    print("\n================ EVALUATION METRICS SUMMARY ================")
    print(res_df.to_string(index=False))
    print("============================================================\n")

if __name__ == '__main__':
    train_and_save_models()
