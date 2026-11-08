# Wine Quality Classification & Streamlit Deployment

## a. Problem Statement
The objective of this project is to predict wine quality (High Quality >= 6 vs Low Quality < 6) based on physicochemical features. We implement five standard classification models, evaluate their performance across six standardized metrics, and deploy an interactive web interface using Streamlit Community Cloud.

## b. Dataset Description
* **Source:** UCI Machine Learning Repository (Wine Quality Dataset - Red Wine)
* **Instances:** 1,599 samples
* **Features (12 Total):** 
  1. fixed acidity
  2. volatile acidity
  3. citric acid
  4. residual sugar
  5. chlorides
  6. free sulfur dioxide
  7. total sulfur dioxide
  8. density
  9. pH
  10. sulphates
  11. alcohol
  12. acid_ph_ratio (Engineered feature)
* **Target Variable:** Binary Classification (`1` for Quality >= 6, `0` for Quality < 6).

## c. GitHub Repository Link
[Insert Your GitHub Repository URL Here]

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.741 | 0.812 | 0.755 | 0.765 | 0.760 | 0.478 |
| **Decision Tree** | 0.734 | 0.785 | 0.732 | 0.794 | 0.762 | 0.463 |
| **kNN** | 0.728 | 0.779 | 0.742 | 0.747 | 0.745 | 0.453 |
| **Naive Bayes** | 0.719 | 0.781 | 0.752 | 0.706 | 0.728 | 0.438 |
| **Support Vector Machine** | 0.762 | 0.835 | 0.771 | 0.794 | 0.782 | 0.521 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundary provides strong baseline performance with good AUC (0.812) and fast evaluation execution. |
| **Decision Tree** | Captures non-linear feature interactions effectively; prone to slight variance on test partitions if unpruned. |
| **kNN** | Performs decently after feature standardization, but highly sensitive to local outlier noise in border feature regions. |
| **Naive Bayes** | Struggles slightly due to moderate feature correlation assumptions (e.g., acidity vs pH), resulting in lower recall. |
| **Support Vector Machine** | Achieves highest accuracy, AUC, and MCC score by non-linearly projecting boundary space using RBF kernel. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the best performance with an **Accuracy of 76.2%**, **AUC of 0.835**, and **MCC of 0.521**. |