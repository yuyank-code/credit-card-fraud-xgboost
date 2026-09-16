# Trained model artifact

The completed training run produced `xgboost_smote_fraud.joblib` locally. The repository keeps the project lightweight by not committing the 550+ KB binary model through the current GitHub connector path. `src/train.py` is deterministic (`random_state` values are fixed) and regenerates the trained model and all evaluation artifacts with the same configuration.

After running:

```bash
python src/train.py
```

The trained model is written to:

```text
models/xgboost_smote_fraud.joblib
```
