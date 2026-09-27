# Fraud Risk Analytics

This project focuses on detecting fraudulent credit card transactions using machine learning.

The main goal was not only to train a classification model, but also to examine how the decision threshold affects fraud detection and the number of legitimate transactions incorrectly flagged as fraud.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The target variable is `Class`:

- `0` - Legitimate transaction
- `1` - Fraudulent transaction

The dataset is highly imbalanced, since fraudulent transactions represent only a small part of the total observations.

## Method

The data is divided into three sets:

- 60% training
- 20% validation
- 20% testing

A Random Forest classifier is trained using balanced class weights to account for the class imbalance.

The validation set is used to test different decision thresholds and select a threshold based on the F1-score.

The final model is then evaluated on the test set.

## Evaluation

The following metrics are used:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

In fraud detection, accuracy alone can be misleading because of the large difference between the number of legitimate and fraudulent transactions.

## Threshold Analysis

Different probability thresholds are compared to study the trade-off between precision and recall.

Lowering the threshold can identify more fraudulent transactions, but it can also increase the number of legitimate transactions that are flagged for review.

## Business Cost Analysis

A simple cost analysis is included to show how model decisions can be evaluated from a business perspective.

For the analysis, the following example costs are used:

- False Negative: €500
- False Positive: €10

The estimated cost for each tested threshold is calculated as:

Estimated Cost = (False Negatives × 500) + (False Positives × 10)

These values are example assumptions used for the analysis and are not intended to represent the actual costs of a financial institution.

## Risk Scoring

The predicted fraud probability is also used as a transaction risk score.

Transactions are divided into four risk categories:

- LOW
- MEDIUM
- HIGH
- CRITICAL

Based on the selected business threshold, a transaction is either approved or flagged for review.

## High-Risk Transactions

The project also ranks transactions according to their predicted fraud probability.

The ten transactions with the highest predicted risk are extracted so that the highest-risk cases can be reviewed first.

## Project Structure

fraud-risk-analytics/
│
├── Δεδομένα/
│   └── creditcard.csv
│
├── Μοντέλα/
│   └── random_forest_fraud_model.pkl
│
├── Γραφήματα/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── threshold_optimization.png
│   └── business_cost_analysis.png
│
├── Αποτελέσματα/
│   ├── final_test_metrics.csv
│   ├── feature_importance.csv
│   ├── threshold_comparison.csv
│   ├── business_cost_analysis.csv
│   └── top_10_high_risk_transactions.csv
│
└── model_validation.py

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib

## Notes

The project is intended as a practical study of fraud detection on an imbalanced dataset.

The business cost values are illustrative, and model performance should not be interpreted as performance on real banking transactions without further validation.