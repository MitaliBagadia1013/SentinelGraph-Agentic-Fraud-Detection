# Saved model artifacts

Trained model artifacts live here. The large binaries (`*.pkl`, `*.csv`) are
git-ignored — only the lightweight metadata is committed so the results in the
top-level README are reproducible and auditable.

| File | Tracked | Contents |
|------|---------|----------|
| `model_metadata_ieee_*.json` | ✅ | Dataset, feature count, and evaluation metrics (AUC, AP, Brier, precision/recall) for the run |
| `feature_columns_ieee_*.json` | ✅ | Exact ordered feature list the model was trained on |
| `feature_importance_ieee_*.csv` | ✗ (ignored) | Per-feature XGBoost importances |
| `xgboost_ieee_calibrated_*.pkl` | ✗ (ignored) | The pickled calibrated classifier |

Regenerate everything with:

```bash
python data/load_ieee_cis.py
python models/train_ieee_xgboost.py --quick
```
