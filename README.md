# Credit Card Fraud Detection with XGBoost

A reproducible educational case study for fraud detection on heavily imbalanced transaction data using XGBoost, SMOTE, threshold tuning, and feature importance.

## Model
- XGBoost classifier
- SMOTE applied only to the training data
- ROC-AUC and PR-AUC evaluation
- Decision-threshold tuning for precision/recall trade-offs
- Feature-importance interpretation

## Dataset
The included training script generates a synthetic transaction dataset with a 0.60% fraud rate and injected signal features. It is designed for learning and reproducibility, not as a reproduction of the raw IEEE-CIS/Kaggle dataset.

## Results
The repository contains the trained model, metrics, ROC/PR curves, and feature-importance analysis from the completed training run.

## Run
```bash
pip install -r requirements.txt
python src/train.py
```

See `report/` for the case-study and learning report.