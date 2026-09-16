# Case Study 2 — Credit Card Fraud Detection

## Objective
Detect fraudulent transactions under severe class imbalance.

## Why XGBoost?
XGBoost learns nonlinear interactions and handles heterogeneous tabular signals well.

## Why SMOTE?
SMOTE synthesizes minority-class training examples. It was applied only after the train/test split so synthetic points cannot leak information from the test set.

## Results
ROC-AUC = **0.997**  
PR-AUC = **0.930**  
Threshold = **0.9057**  
Precision = **0.900**, Recall = **0.840**, F1 = **0.869**.

## Threshold tuning
A threshold of 0.5 is not automatically optimal in imbalanced fraud detection. Lower thresholds generally catch more fraud but can increase false alerts. The selected threshold here is an educational operating point; production systems should use business loss, investigation capacity, and customer-friction costs.

## Feature importance
The XGBoost gain-based importance is saved in `results/feature_importance.csv`. Because the synthetic data injects signal into selected V-features, the resulting importance should be interpreted as a demonstration rather than a real-world claim about fraud drivers.
