from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# -----------------------------------------------------------------------------
# Project paths and configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for directory in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES = 100_000
FRAUD_COUNT = 600
TEST_SIZE = 0.25
MODEL_FILENAME = "xgboost_smote_fraud.joblib"


# -----------------------------------------------------------------------------
# 1. Create a reproducible synthetic transaction dataset
# -----------------------------------------------------------------------------
def create_dataset(n_samples=N_SAMPLES, fraud_count=FRAUD_COUNT):
    """Create synthetic, highly imbalanced transaction data."""

    rng = np.random.default_rng(7)

    # 28 anonymized numerical transaction features.
    feature_matrix = rng.normal(0, 1, (n_samples, 28))
    amount = np.exp(rng.normal(3.2, 1.1, n_samples))
    transaction_time = rng.uniform(0, 172_800, n_samples)

    target = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, fraud_count, replace=False)
    target[fraud_indices] = 1

    # Inject synthetic fraud-related signals so the model has a learnable task.
    feature_matrix[fraud_indices, 2] += 2.2
    feature_matrix[fraud_indices, 4] -= 1.8
    feature_matrix[fraud_indices, 9] += 1.7
    feature_matrix[fraud_indices, 13] -= 2.5
    feature_matrix[fraud_indices, 16] += 1.9
    amount[fraud_indices] *= 1.8

    feature_names = [f"V{i}" for i in range(1, 29)]

    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df["Time"] = transaction_time
    df["Amount"] = amount
    df["Class"] = target

    return df


# -----------------------------------------------------------------------------
# 2. Build the XGBoost classifier
# -----------------------------------------------------------------------------
def build_model():
    """Create the XGBoost binary classifier."""

    return XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )


# -----------------------------------------------------------------------------
# 3. Train, evaluate, and save the model
# -----------------------------------------------------------------------------
def main():
    df = create_dataset()
    df.to_csv(DATA_DIR / "synthetic_credit_transactions.csv", index=False)

    target_column = "Class"
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Split BEFORE SMOTE to prevent synthetic samples from leaking into testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Oversample only the training data because fraud is highly imbalanced.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = build_model()
    model.fit(X_train_smote, y_train_smote)

    probabilities = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Threshold tuning: choose the threshold that maximizes F1 score.
    # -------------------------------------------------------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = np.argmax(f1_scores[:-1])
    threshold = float(thresholds[best_index])

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "n_total": int(len(df)),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "train_after_smote": int(len(y_train_smote)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "selected_threshold_max_f1": threshold,
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": matrix.tolist(),
    }

    # Save an uncompressed Joblib artifact so the trained XGBoost model is
    # directly usable without an additional compression layer. Protocol 5 is
    # efficient for NumPy/XGBoost objects while keeping loading straightforward.
    joblib.dump(
        model,
        MODEL_DIR / MODEL_FILENAME,
        compress=0,
        protocol=5,
    )

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    # -------------------------------------------------------------------------
    # ROC curve
    # -------------------------------------------------------------------------
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, probabilities
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {metrics['roc_auc']:.3f}",
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Credit Card Fraud ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # -------------------------------------------------------------------------
    # Precision-Recall curve
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {metrics['average_precision_pr_auc']:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Credit Card Fraud Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=160)
    plt.close()

    print("Training complete.")
    print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:   {metrics['average_precision_pr_auc']:.3f}")
    print(f"Precision:{metrics['precision']:.3f}")
    print(f"Recall:   {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1']:.3f}")
    print(f"Model:    {MODEL_DIR / MODEL_FILENAME}")


if __name__ == "__main__":
    main()
