import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Instrumental Vehicle Fault Detection App",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Instrumental Vehicle Telematics & Fault Detection System")
st.markdown("""
This interactive Streamlit app evaluates **5 Machine Learning Classification Models** on automotive sensor telematics data 
to predict vehicle maintenance requirements.
""")

st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Classification Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Support Vector Machine")
)

@st.cache_resource
def load_models_and_scaler():
    scaler = joblib.load('model/scaler.pkl')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'kNN': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Support Vector Machine': joblib.load('model/svm.pkl')
    }
    return scaler, models

scaler, models = load_models_and_scaler()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Test Data Preview")
    st.dataframe(df.head(10))
    
    if 'vehicle_fault_status' in df.columns:
        X_test = df.drop(columns=['vehicle_fault_status'])
        y_test = df['vehicle_fault_status']
        
        # Scale features if applicable
        if model_choice in ['Logistic Regression', 'kNN', 'Support Vector Machine']:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.copy()
            
        selected_model = models[model_choice]
        y_pred = selected_model.predict(X_eval)
        y_proba = selected_model.predict_proba(X_eval)[:, 1] if hasattr(selected_model, "predict_proba") else y_pred
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.markdown(f"### 📈 Evaluation Metrics for **{model_choice}**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Healthy (0)', 'Fault (1)'],
                        yticklabels=['Healthy (0)', 'Fault (1)'], ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
        with c2:
            st.subheader("📋 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose())
            
    else:
        st.error("Uploaded CSV must contain the 'vehicle_fault_status' target column.")
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar to run model inference and view evaluation metrics.")