from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier


# ============================================================
# CREDIT CARD FRAUD DETECTION
# XGBoost + SMOTE + Threshold Tuning
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

THRESHOLDS = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
    0.02,
    0.01,
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION")
print("XGBoost + SMOTE + Threshold Tuning")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\nClass distribution:")

class_counts = df["Class"].value_counts()

print(class_counts)

fraud_cases = int(df["Class"].sum())
total_transactions = len(df)

fraud_rate = (
    fraud_cases / total_transactions
) * 100

print(
    f"\nFraud cases: {fraud_cases:,}"
)

print(
    f"Total transactions: {total_transactions:,}"
)

print(
    f"Fraud rate: {fraud_rate:.4f}%"
)


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Class"]
)

y = df["Class"]


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

# 70% training
# 15% validation
# 15% test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

print(
    f"Training samples:   {len(X_train):,}"
)

print(
    f"Validation samples: {len(X_val):,}"
)

print(
    f"Test samples:       {len(X_test):,}"
)

print(
    f"Training fraud cases before SMOTE: "
    f"{int(y_train.sum()):,}"
)


# ============================================================
# 7. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

print("\nApplying SMOTE...")

smote = SMOTE(
    sampling_strategy=1.0,
    random_state=RANDOM_STATE,
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train,
    )
)

print(
    f"Training samples after SMOTE: "
    f"{len(X_train_smote):,}"
)

print(
    f"Fraud cases after SMOTE: "
    f"{int(y_train_smote.sum()):,}"
)

print(
    f"Normal cases after SMOTE: "
    f"{int((y_train_smote == 0).sum()):,}"
)


# ============================================================
# 8. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# 9. TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train_smote,
    y_train_smote,
)

print("Training completed.")


# ============================================================
# 10. GENERATE PROBABILITIES
# ============================================================

print("\nGenerating predictions...")

val_probabilities = model.predict_proba(
    X_val
)[:, 1]

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. ROC-AUC
# ============================================================

validation_roc_auc = roc_auc_score(
    y_val,
    val_probabilities,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)


# ============================================================
# 12. PR-AUC
# ============================================================

validation_pr_auc = average_precision_score(
    y_val,
    val_probabilities,
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


print("\nModel performance:")

print(
    f"Validation ROC-AUC: "
    f"{validation_roc_auc:.6f}"
)

print(
    f"Test ROC-AUC:       "
    f"{test_roc_auc:.6f}"
)

print(
    f"Validation PR-AUC:  "
    f"{validation_pr_auc:.6f}"
)

print(
    f"Test PR-AUC:        "
    f"{test_pr_auc:.6f}"
)


# ============================================================
# 13. THRESHOLD TUNING
# ============================================================

print("\nThreshold analysis...")

threshold_results = []

for threshold in THRESHOLDS:

    validation_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        validation_predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_val,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        validation_predictions,
        zero_division=0,
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    RESULTS_DIR / "threshold_analysis.csv",
    index=False,
)

print("\nThreshold results:")

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SELECT THRESHOLD
# ============================================================

# Select the threshold with the highest
# validation F1-score.

best_row = threshold_df.loc[
    threshold_df["f1_score"].idxmax()
]

selected_threshold = float(
    best_row["threshold"]
)

print(
    "\nSelected threshold based on "
    f"validation F1: {selected_threshold:.2f}"
)


# ============================================================
# 15. FINAL TEST PREDICTIONS
# ============================================================

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)


# ============================================================
# 16. CONFUSION MATRIX VALUES
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()


# ============================================================
# 17. FINAL METRICS
# ============================================================

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

test_false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


# ============================================================
# 18. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Threshold:           "
    f"{selected_threshold:.2f}"
)

print(
    f"ROC-AUC:             "
    f"{test_roc_auc:.6f}"
)

print(
    f"PR-AUC:              "
    f"{test_pr_auc:.6f}"
)

print(
    f"Precision:           "
    f"{test_precision:.6f}"
)

print(
    f"Recall:              "
    f"{test_recall:.6f}"
)

print(
    f"F1-score:            "
    f"{test_f1:.6f}"
)

print(
    f"True Negatives:      "
    f"{tn:,}"
)

print(
    f"False Positives:     "
    f"{fp:,}"
)

print(
    f"False Negatives:     "
    f"{fn:,}"
)

print(
    f"True Positives:      "
    f"{tp:,}"
)

print(
    f"False Positive Rate: "
    f"{test_false_positive_rate:.6f}"
)

print(
    f"False Negative Rate: "
    f"{test_false_negative_rate:.6f}"
)


# ============================================================
# 19. CONFUSION MATRIX PLOT
# ============================================================

cm = np.array(
    [
        [tn, fp],
        [fn, tp],
    ]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

image = ax.imshow(cm)

ax.set_title(
    "XGBoost Fraud Detection\n"
    f"Threshold = {selected_threshold:.2f}"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Genuine", "Fraud"]
)

ax.set_yticklabels(
    ["Genuine", "Fraud"]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
        )

fig.colorbar(
    image,
    ax=ax,
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
)

plt.close()


# ============================================================
# 20. ROC CURVE
# ============================================================

false_positive_rates, true_positive_rates, _ = (
    roc_curve(
        y_test,
        test_probabilities,
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    false_positive_rates,
    true_positive_rates,
    label=(
        f"XGBoost ROC-AUC = "
        f"{test_roc_auc:.4f}"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "XGBoost ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=200,
)

plt.close()


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating feature importance..."
)

feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False,
    )
)

feature_importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False,
)


# ============================================================
# 22. TOP 20 FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    feature_importance
    .head(20)
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1],
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 20 XGBoost Feature Importances"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_importance.png",
    dpi=200,
)

plt.close()


# ============================================================
# 23. SAVE TRAINED MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "xgboost_fraud_model.joblib"
)

joblib.dump(
    model,
    model_path,
)

print(
    f"\nModel saved to:\n{model_path}"
)


# ============================================================
# 24. SAVE METRICS TO JSON
# ============================================================

metrics = {

    "dataset_rows": int(
        len(df)
    ),

    "dataset_columns": int(
        len(df.columns)
    ),

    "fraud_cases": int(
        fraud_cases
    ),

    "fraud_rate_percent": float(
        fraud_rate
    ),

    "training_rows_before_smote": int(
        len(X_train)
    ),

    "training_rows_after_smote": int(
        len(X_train_smote)
    ),

    "validation_rows": int(
        len(X_val)
    ),

    "test_rows": int(
        len(X_test)
    ),

    "validation_roc_auc": float(
        validation_roc_auc
    ),

    "test_roc_auc": float(
        test_roc_auc
    ),

    "validation_pr_auc": float(
        validation_pr_auc
    ),

    "test_pr_auc": float(
        test_pr_auc
    ),

    "selected_threshold": float(
        selected_threshold
    ),

    "test_precision": float(
        test_precision
    ),

    "test_recall": float(
        test_recall
    ),

    "test_f1_score": float(
        test_f1
    ),

    "true_negatives": int(
        tn
    ),

    "false_positives": int(
        fp
    ),

    "false_negatives": int(
        fn
    ),

    "true_positives": int(
        tp
    ),

    "false_positive_rate": float(
        test_false_positive_rate
    ),

    "false_negative_rate": float(
        test_false_negative_rate
    ),
}

with open(
    RESULTS_DIR / "metrics.json",
    "w",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ============================================================
# 25. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CASE STUDY 2 TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    "models/xgboost_fraud_model.joblib"
)

print(
    "results/metrics.json"
)

print(
    "results/threshold_analysis.csv"
)

print(
    "results/feature_importance.csv"
)

print(
    "results/feature_importance.png"
)

print(
    "results/confusion_matrix.png"
)

print(
    "results/roc_curve.png"
)

print("\nDone.")
