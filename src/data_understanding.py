import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Δεδομένα/creditcard.csv")
df
print(df.head())
print(df.shape)
print(df["Class"].value_counts())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df[df.duplicated()]["Class"].value_counts())
print(df.dtypes)
print(df.describe())
print(df.groupby("Class")["Amount"].describe())
class_counts = df["Class"].value_counts()

plt.bar(["Normal", "Fraud"], class_counts.values)

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.yscale("log")
plt.show()
normal = df[df["Class"] == 0]["Amount"]
fraud = df[df["Class"] == 1]["Amount"]

print("Normal transactions:")
print(normal.describe())

print("\nFraud transactions:")
print(fraud.describe())
plt.figure(figsize=(8, 5))

plt.boxplot(
    [normal, fraud],
    tick_labels=["Normal", "Fraud"],
    showfliers=False
)

plt.title("Transaction Amount: Normal vs Fraud")
plt.ylabel("Amount (€)")
plt.xlabel("Transaction Type")

plt.show()
correlations = df.corr(numeric_only=True)["Class"].drop("Class")

correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print("\nFeatures most correlated with fraud:")
print(correlations.head(10))
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape)

print("\nTest set:")
print(X_test.shape)

print("\nFraud cases in training:")
print(y_train.value_counts())

print("\nFraud cases in test:")
print(y_test.value_counts())
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n--- RANDOM FOREST ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
from sklearn.metrics import roc_auc_score, average_precision_score

# Probabilities that each transaction is Fraud
y_prob = rf_model.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

# PR-AUC / Average Precision
pr_auc = average_precision_score(y_test, y_prob)

print("\n--- Random Forest Evaluation ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n--- Threshold Tuning ---")

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision_curve, recall_curve, thresholds_curve = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(recall_curve, precision_curve)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest - Precision-Recall Curve")
plt.grid(True)

plt.show()
import numpy as np

# Calculate F1-score for every threshold
f1_scores = (
    2 * precision_curve[:-1] * recall_curve[:-1]
    / (precision_curve[:-1] + recall_curve[:-1] + 1e-10)
)

# Find threshold with highest F1-score
best_index = np.argmax(f1_scores)

best_threshold = thresholds_curve[best_index]
best_precision = precision_curve[best_index]
best_recall = recall_curve[best_index]
best_f1 = f1_scores[best_index]

print("\n--- Optimal Threshold ---")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Precision:      {best_precision:.4f}")
print(f"Recall:         {best_recall:.4f}")
print(f"F1-score:       {best_f1:.4f}")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
