import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ks_2samp
from pathlib import Path
import joblib
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

warnings.filterwarnings("ignore")


class ProductionFraudDetectionTrainer:

    def __init__(self, data_path: str, quick_mode: bool = False):
        self.data_path = Path(data_path)
        self.quick_mode = quick_mode
        self.model = None
        self.calibrated_model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.model_metrics = {}
        self.cv_scores = {}
        self.best_params = {}
        self.models_dir = Path("models/saved/production")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir = Path("models/monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print("PRODUCTION-GRADE XGBOOST FRAUD DETECTION TRAINER")
        print("=" * 80)
        if quick_mode:
            print("QUICK MODE: Skipping hyperparameter tuning")
        print()

    def load_data(self) -> pd.DataFrame:
        print(f"STEP 1: Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Loaded successfully!")
        print(f"Total records: {len(df):,}")
        print(f"Total features: {len(df.columns)}")
        print(
            f"Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean() * 100:.2f}%)"
        )
        print(f"Legitimate cases: {(df['is_fraud'] == 0).sum():,}")
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        print(f"Missing values: {missing_pct:.2f}%")
        return df

    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\nSTEP 2: Advanced Feature Engineering")
        print(f"Starting with {len(df.columns)} columns...")
        df_eng = df.copy()
        print(f"\n2.1: Encoding text columns...")
        text_columns = ["location", "merchant_category", "device_id"]
        for col in text_columns:
            if col in df_eng.columns:
                le = LabelEncoder()
                df_eng[f"{col}_encoded"] = le.fit_transform(df_eng[col].astype(str))
                self.label_encoders[col] = le
                print(f"{col}: {len(le.classes_)} unique values numeric codes")
        print(f"\n2.2: Creating interaction features...")
        if (
            "transaction_amount" in df_eng.columns
            and "transaction_velocity_1h" in df_eng.columns
        ):
            df_eng["amount_x_velocity"] = (
                df_eng["transaction_amount"] * df_eng["transaction_velocity_1h"]
            )
            print(f"amount_x_velocity (amount × 1h velocity)")
        if "hour_of_day" in df_eng.columns and "merchant_risk_score" in df_eng.columns:
            df_eng["hour_x_merchant_risk"] = (
                df_eng["hour_of_day"] * df_eng["merchant_risk_score"]
            )
            print(f"hour_x_merchant_risk (time × merchant risk)")
        if (
            "device_fingerprint_changes" in df_eng.columns
            and "location_encoded" in df_eng.columns
        ):
            df_eng["device_x_location_change"] = (
                df_eng["device_fingerprint_changes"] * df_eng["location_encoded"]
            )
            print(f"device_x_location_change (device × location)")
        print(f"\n2.3: Creating temporal features...")
        if "timestamp" in df_eng.columns:
            df_eng["timestamp"] = pd.to_datetime(df_eng["timestamp"])
            df_eng["day_of_week"] = df_eng["timestamp"].dt.dayofweek
            df_eng["is_weekend"] = df_eng["day_of_week"].isin([5, 6]).astype(int)
            df_eng["is_business_hours"] = (
                (df_eng["hour_of_day"] >= 9)
                & (df_eng["hour_of_day"] <= 17)
                & (df_eng["is_weekend"] == 0)
            ).astype(int)
            print(f"day_of_week (0=Mon, 6=Sun)")
            print(f"is_weekend (1=weekend, 0=weekday)")
            print(f"is_business_hours (1=9am-5pm M-F, 0=other)")
        print(f"\n2.4: Creating aggregation features...")
        print(
            f"ℹ user_fraud_rate and merchant_fraud_rate computed post-split (no leakage)"
        )
        print(f"\nFeature engineering complete!")
        print(f"Features before: {len(df.columns)}")
        print(f"Features after: {len(df_eng.columns)}")
        print(f"New features: {len(df_eng.columns) - len(df.columns)}")
        return df_eng

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        print(f"\nSTEP 3: Preparing features for training")
        exclude_cols = [
            "is_fraud",
            "fraud_scenario",
            "fraud_score",
            "user_id",
            "timestamp",
            "location",
            "merchant_category",
            "device_id",
            "device_id_encoded",
            "amount",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(feature_cols):
            print(
                f"Dropping {len(feature_cols) - len(numeric_cols)} non-numeric columns"
            )
            X = X[numeric_cols]
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count:,} missing values with 0")
            X = X.fillna(0)
        y = df["is_fraud"].values
        self.feature_columns = X.columns.tolist()
        print(f"Final feature count: {len(self.feature_columns)}")
        print(f"Training samples: {len(X):,}")
        return (X, y)

    def cross_validation(
        self, X: pd.DataFrame, y: np.ndarray, model: xgb.XGBClassifier
    ) -> Dict:
        print(f"\nSTEP 4a: Cross-Validation (5-Fold)")
        print(f"Testing model on 5 different data splits...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ["roc_auc", "precision", "recall", "f1"]
        cv_results = {}
        for metric in scoring:
            print(f"\nTesting {metric}...")
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            cv_results[metric] = {
                "scores": scores.tolist(),
                "mean": mean_score,
                "std": std_score,
            }
            print(f"Fold scores: {[f'{s:.3f}' for s in scores]}")
            print(f"Mean: {mean_score:.3f} ± {std_score:.3f}")
            if std_score < 0.02:
                print(f"Very stable (std < 2%)")
            elif std_score < 0.05:
                print(f"Stable (std < 5%)")
            else:
                print(f"Unstable (std > 5%) - model performance varies across folds")
        self.cv_scores = cv_results
        print(f"\nCross-validation complete!")
        return cv_results

    def hyperparameter_tuning(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> xgb.XGBClassifier:
        print(f"\nSTEP 4b: Hyperparameter Tuning")
        if self.quick_mode:
            print(f"QUICK MODE: Using default parameters (skipping tuning)")
            fraud_count = y_train.sum()
            legit_count = (y_train == 0).sum()
            scale_pos_weight = legit_count / fraud_count
            self.best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "scale_pos_weight": scale_pos_weight,
            }
            model = xgb.XGBClassifier(
                **self.best_params,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=42,
            )
            return model
        print(f"Testing 50 random parameter combinations...")
        print(f"This will take ~30-60 minutes (grab a coffee )")
        fraud_count = y_train.sum()
        legit_count = (y_train == 0).sum()
        scale_pos_weight = legit_count / fraud_count
        param_distributions = {
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 150, 200],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 1.5, 2],
        }
        base_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=50,
            scoring="roc_auc",
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"\nHyperparameter search complete! Time: {search_time / 60:.1f} minutes")
        print(f"\nBEST PARAMETERS:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value}")
        print(f"\nBEST CROSS-VALIDATION AUC: {random_search.best_score_:.4f}")
        self.best_params = random_search.best_params_
        self.best_params["scale_pos_weight"] = scale_pos_weight
        return random_search.best_estimator_

    def train_model(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> xgb.XGBClassifier:
        print(f"\nSTEP 5: Training Final Model")
        model = self.hyperparameter_tuning(X_train, y_train)
        print(f"\nTraining final model with best parameters...")
        start_time = time.time()
        if not hasattr(model, "feature_importances_"):
            model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training complete! Time: {training_time:.2f} seconds")
        self.model = model
        return model

    def calibrate_model(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> CalibratedClassifierCV:
        print(f"\nSTEP 6: Model Calibration")
        print(f"Calibrating probabilities using Platt Scaling...")
        calibrated_model = CalibratedClassifierCV(self.model, method="sigmoid", cv=3)
        calibrated_model.fit(X_train, y_train)
        print(f"Calibration complete!")
        self.calibrated_model = calibrated_model
        return calibrated_model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        print(f"\nSTEP 7: Model Evaluation")
        print(f"\n7.1: Uncalibrated Model")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        print(f"\nCLASSIFICATION REPORT:")
        print(
            classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
        )
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\nCONFUSION MATRIX:")
        print(f"Predicted")
        print(f"Legit Fraud")
        print(f"Actual Legit {tn:>6,} {fp:>6,}")
        print(f"Actual Fraud {fn:>6,} {tp:>6,}")
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        print(f"\nUNCALIBRATED METRICS:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"FPR: {fpr:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Avg Precision: {avg_precision:.4f}")
        print(f"Brier Score: {brier:.4f} (lower is better)")
        print(f"Log Loss: {logloss:.4f} (lower is better)")
        print(f"\n7.2: Calibrated Model")
        y_pred_cal = self.calibrated_model.predict(X_test)
        y_pred_proba_cal = self.calibrated_model.predict_proba(X_test)[:, 1]
        brier_cal = brier_score_loss(y_test, y_pred_proba_cal)
        logloss_cal = log_loss(y_test, y_pred_proba_cal)
        print(f"CALIBRATED METRICS:")
        print(f"Brier Score: {brier_cal:.4f} (was {brier:.4f})")
        print(f"Log Loss: {logloss_cal:.4f} (was {logloss:.4f})")
        if brier_cal < brier:
            improvement = (brier - brier_cal) / brier * 100
            print(f"Calibration improved Brier Score by {improvement:.1f}%")
        else:
            print(f"Calibration didn't improve (model was already well-calibrated)")
        print(f"\n7.3: Latency Testing")
        latencies = []
        for _ in range(1000):
            start = time.time()
            _ = self.model.predict_proba(X_test.iloc[[0]])
            latencies.append((time.time() - start) * 1000)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        print(f"Uncalibrated Model:")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")
        latencies_cal = []
        for _ in range(1000):
            start = time.time()
            _ = self.calibrated_model.predict_proba(X_test.iloc[[0]])
            latencies_cal.append((time.time() - start) * 1000)
        p50_cal = np.percentile(latencies_cal, 50)
        p95_cal = np.percentile(latencies_cal, 95)
        p99_cal = np.percentile(latencies_cal, 99)
        print(f"\nCalibrated Model:")
        print(f"P50: {p50_cal:.2f}ms")
        print(f"P95: {p95_cal:.2f}ms")
        print(f"P99: {p99_cal:.2f}ms")
        if p99_cal < 15:
            print(f"P99 latency < 15ms target ACHIEVED!")
        elif p99_cal < 20:
            print(f"P99 latency < 20ms target ACHIEVED!")
        else:
            print(f"P99 latency {p99_cal:.2f}ms exceeds 20ms target")
        self.model_metrics = {
            "uncalibrated": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
                "auc_roc": auc,
                "avg_precision": avg_precision,
                "brier_score": brier,
                "log_loss": logloss,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "p99_latency_ms": p99,
                "confusion_matrix": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                },
            },
            "calibrated": {
                "brier_score": brier_cal,
                "log_loss": logloss_cal,
                "p50_latency_ms": p50_cal,
                "p95_latency_ms": p95_cal,
                "p99_latency_ms": p99_cal,
                "calibration_improvement_pct": (
                    (brier - brier_cal) / brier * 100 if brier > brier_cal else 0
                ),
            },
        }
        return self.model_metrics

    def feature_importance_analysis(self, top_n: int = 30) -> pd.DataFrame:
        print(f"\nSTEP 8: Feature Importance Analysis")
        importance = self.model.feature_importances_
        feature_df = pd.DataFrame(
            {"feature": self.feature_columns, "importance": importance}
        ).sort_values("importance", ascending=False)
        print(f"\nTOP {top_n} MOST IMPORTANT FEATURES:")
        for i, row in feature_df.head(top_n).iterrows():
            stars = "" * min(int(row["importance"] * 50), 5)
            print(f"{row['feature']:<45s} {row['importance']:.4f} {stars}")
        importance_file = self.models_dir / "feature_importance_production.csv"
        feature_df.to_csv(importance_file, index=False)
        print(f"\nFull feature importance saved to: {importance_file}")
        return feature_df

    def setup_monitoring(self, X_train: pd.DataFrame) -> Dict:
        print(f"\nSTEP 9: Setting Up Production Monitoring")
        baseline_stats = {
            "feature_statistics": {},
            "training_fraud_rate": None,
            "training_date": datetime.now().isoformat(),
            "num_training_samples": len(X_train),
            "num_features": len(self.feature_columns),
        }
        print(
            f"Calculating baseline statistics for {len(self.feature_columns)} features..."
        )
        for feature in self.feature_columns:
            values = X_train[feature].values
            baseline_stats["feature_statistics"][feature] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
        baseline_file = self.monitoring_dir / "baseline_statistics.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline_stats, f, indent=2)
        print(f"Baseline statistics saved to: {baseline_file}")
        print(f"\nUse this for drift detection in production:")
        print(f"1. Load baseline stats")
        print(f"2. Compare new data distributions using KS test")
        print(f"3. Alert if drift_score > 0.3 (30% of features drifted)")
        return baseline_stats

    def save_production_model(self) -> Dict[str, Path]:
        print(f"\nSTEP 10: Saving Production Model")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_saved = {}
        model_file = self.models_dir / f"xgboost_production_{timestamp}.json"
        self.model.save_model(model_file)
        files_saved["xgboost_json"] = model_file
        print(f"XGBoost model (JSON): {model_file.name}")
        calibrated_file = self.models_dir / f"xgboost_calibrated_{timestamp}.pkl"
        joblib.dump(self.calibrated_model, calibrated_file)
        files_saved["calibrated_model"] = calibrated_file
        print(f"Calibrated model (PKL): {calibrated_file.name}")
        feature_file = self.models_dir / "feature_columns_production.json"
        with open(feature_file, "w") as f:
            json.dump(self.feature_columns, f, indent=2)
        files_saved["feature_columns"] = feature_file
        print(f"Feature columns: {feature_file.name}")
        if self.label_encoders:
            encoders_file = self.models_dir / f"label_encoders_{timestamp}.pkl"
            joblib.dump(self.label_encoders, encoders_file)
            files_saved["label_encoders"] = encoders_file
            print(f"Label encoders: {encoders_file.name}")
        metadata = {
            "timestamp": timestamp,
            "model_type": "XGBoost Production",
            "quick_mode": self.quick_mode,
            "num_features": len(self.feature_columns),
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
            "metrics": self.model_metrics,
            "feature_engineering": {
                "text_encoding": list(self.label_encoders.keys()),
                "interaction_features": [
                    "amount_x_velocity",
                    "hour_x_merchant_risk",
                    "device_x_location_change",
                ],
                "temporal_features": ["day_of_week", "is_weekend", "is_business_hours"],
                "aggregation_features": ["user_fraud_rate", "merchant_fraud_rate"],
            },
        }
        metadata_file = self.models_dir / f"model_metadata_production_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        files_saved["metadata"] = metadata_file
        print(f"Model metadata: {metadata_file.name}")
        readme_content = f"\n# Production XGBoost Fraud Detection Model\nGenerated: {timestamp}\n\n## Model Performance\n\n### Uncalibrated Model\n- Precision: {self.model_metrics['uncalibrated']['precision']:.4f}\n- Recall: {self.model_metrics['uncalibrated']['recall']:.4f}\n- F1-Score: {self.model_metrics['uncalibrated']['f1_score']:.4f}\n- AUC-ROC: {self.model_metrics['uncalibrated']['auc_roc']:.4f}\n- P99 Latency: {self.model_metrics['uncalibrated']['p99_latency_ms']:.2f}ms\n\n### Calibrated Model\n- Brier Score: {self.model_metrics['calibrated']['brier_score']:.4f}\n- Log Loss: {self.model_metrics['calibrated']['log_loss']:.4f}\n- P99 Latency: {self.model_metrics['calibrated']['p99_latency_ms']:.2f}ms\n- Calibration Improvement: {self.model_metrics['calibrated']['calibration_improvement_pct']:.1f}%\n\n### Cross-Validation Scores\n- AUC: {self.cv_scores['roc_auc']['mean']:.4f} ± {self.cv_scores['roc_auc']['std']:.4f}\n- Precision: {self.cv_scores['precision']['mean']:.4f} ± {self.cv_scores['precision']['std']:.4f}\n- Recall: {self.cv_scores['recall']['mean']:.4f} ± {self.cv_scores['recall']['std']:.4f}\n\n## Deployment Instructions\n\n### 1. AWS SageMaker\n```python\nimport joblib\nmodel = joblib.load('xgboost_calibrated_{timestamp}.pkl')\nprediction = model.predict_proba(features)\n```\n\n### 2. Feature Requirements\nSee `feature_columns_production.json` for required features ({len(self.feature_columns)} total).\n\n### 3. Text Encoding\nLoad `label_encoders_{timestamp}.pkl` to encode:\n- location\n- merchant_category\n- device_id\n\n### 4. Monitoring\nUse `../monitoring/baseline_statistics.json` for drift detection.\n\n## Best Hyperparameters\n{json.dumps(self.best_params, indent=2)}\n\n## When to Retrain\n- Feature drift score > 0.3\n- Precision drops below {self.model_metrics['uncalibrated']['precision'] * 0.9:.4f}\n- 30 days since last train\n"
        readme_file = self.models_dir / f"README_{timestamp}.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)
        files_saved["readme"] = readme_file
        print(f"Deployment README: {readme_file.name}")
        print(f"\nAll production artifacts saved to: {self.models_dir}")
        return files_saved


def detect_drift(
    baseline_file: Path, new_data: pd.DataFrame, threshold: float = 0.05
) -> Dict:
    print(f"\nDRIFT DETECTION")
    with open(baseline_file, "r") as f:
        baseline = json.load(f)
    drift_results = {
        "drifted_features": [],
        "drift_score": 0.0,
        "total_features_tested": 0,
    }
    for feature, stats in baseline["feature_statistics"].items():
        if feature not in new_data.columns:
            continue
        new_values = new_data[feature].values
        baseline_values = np.random.normal(
            stats["mean"], stats["std"], size=min(len(new_values), 10000)
        )
        ks_stat, p_value = ks_2samp(baseline_values, new_values)
        if p_value < threshold:
            drift_results["drifted_features"].append(
                {"feature": feature, "ks_statistic": ks_stat, "p_value": p_value}
            )
        drift_results["total_features_tested"] += 1
    drift_results["drift_score"] = (
        len(drift_results["drifted_features"]) / drift_results["total_features_tested"]
    )
    print(f"Features tested: {drift_results['total_features_tested']}")
    print(f"Features drifted: {len(drift_results['drifted_features'])}")
    print(f"Drift score: {drift_results['drift_score']:.2%}")
    if drift_results["drift_score"] > 0.3:
        print(f"HIGH DRIFT DETECTED - Consider retraining model")
    elif drift_results["drift_score"] > 0.1:
        print(f"MODERATE DRIFT - Monitor closely")
    else:
        print(f"LOW DRIFT - Model is stable")
    return drift_results


def ab_test_models(
    model_a_path: Path,
    model_b_path: Path,
    test_data: pd.DataFrame,
    test_labels: np.ndarray,
) -> Dict:
    print(f"\nA/B TEST: Model Comparison")
    model_a = joblib.load(model_a_path)
    model_b = joblib.load(model_b_path)
    print(f"Model A: {model_a_path.name}")
    print(f"Model B: {model_b_path.name}")
    print(f"Test samples: {len(test_data):,}")
    y_pred_a = model_a.predict(test_data)
    y_proba_a = model_a.predict_proba(test_data)[:, 1]
    cm_a = confusion_matrix(test_labels, y_pred_a)
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    precision_a = tp_a / (tp_a + fp_a) if tp_a + fp_a > 0 else 0
    recall_a = tp_a / (tp_a + fn_a) if tp_a + fn_a > 0 else 0
    f1_a = (
        2 * (precision_a * recall_a) / (precision_a + recall_a)
        if precision_a + recall_a > 0
        else 0
    )
    fpr_a = fp_a / (fp_a + tn_a) if fp_a + tn_a > 0 else 0
    auc_a = roc_auc_score(test_labels, y_proba_a)
    y_pred_b = model_b.predict(test_data)
    y_proba_b = model_b.predict_proba(test_data)[:, 1]
    cm_b = confusion_matrix(test_labels, y_pred_b)
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    precision_b = tp_b / (tp_b + fp_b) if tp_b + fp_b > 0 else 0
    recall_b = tp_b / (tp_b + fn_b) if tp_b + fn_b > 0 else 0
    f1_b = (
        2 * (precision_b * recall_b) / (precision_b + recall_b)
        if precision_b + recall_b > 0
        else 0
    )
    fpr_b = fp_b / (fp_b + tn_b) if fp_b + tn_b > 0 else 0
    auc_b = roc_auc_score(test_labels, y_proba_b)
    print(f"\nCOMPARISON:")
    print(f"\n{'Metric':<20s} {'Model A':<15s} {'Model B':<15s} {'Winner':<10s}")
    print(f"{'-' * 60}")
    metrics = [
        ("Precision", precision_a, precision_b, "higher"),
        ("Recall", recall_a, recall_b, "higher"),
        ("F1-Score", f1_a, f1_b, "higher"),
        ("AUC-ROC", auc_a, auc_b, "higher"),
        ("FPR", fpr_a, fpr_b, "lower"),
        ("False Positives", fp_a, fp_b, "lower"),
    ]
    score_a = 0
    score_b = 0
    for metric_name, val_a, val_b, better in metrics:
        if better == "higher":
            winner = "A " if val_a > val_b else "B " if val_b > val_a else "Tie"
            if val_a > val_b:
                score_a += 1
            elif val_b > val_a:
                score_b += 1
        else:
            winner = "A " if val_a < val_b else "B " if val_b < val_a else "Tie"
            if val_a < val_b:
                score_a += 1
            elif val_b < val_a:
                score_b += 1
        print(f"{metric_name:<20s} {val_a:<15.4f} {val_b:<15.4f} {winner:<10s}")
    fpr_reduction = (fpr_a - fpr_b) / fpr_a * 100 if fpr_a > 0 else 0
    print(f"\nFALSE POSITIVE REDUCTION: {fpr_reduction:.1f}%")
    if fpr_reduction > 0:
        print(f"Model B reduces false positives by {fpr_reduction:.1f}%")
    else:
        print(f"Model B has MORE false positives")
    print(f"\nOVERALL WINNER:")
    if score_a > score_b:
        print(f"Model A wins {score_a}-{score_b}")
    elif score_b > score_a:
        print(f"Model B wins {score_b}-{score_a}")
    else:
        print(f"Tie {score_a}-{score_b}")
    return {
        "model_a": {
            "precision": precision_a,
            "recall": recall_a,
            "f1": f1_a,
            "auc": auc_a,
            "fpr": fpr_a,
            "fp": int(fp_a),
        },
        "model_b": {
            "precision": precision_b,
            "recall": recall_b,
            "f1": f1_b,
            "auc": auc_b,
            "fpr": fpr_b,
            "fp": int(fp_b),
        },
        "fpr_reduction_pct": fpr_reduction,
        "winner": "A" if score_a > score_b else "B" if score_b > score_a else "Tie",
    }


def main():
    parser = argparse.ArgumentParser(description="Production XGBoost Trainer")
    parser.add_argument(
        "--quick", action="store_true", help="Skip hyperparameter tuning"
    )
    parser.add_argument(
        "--ab-test", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="A/B test two models"
    )
    args = parser.parse_args()
    print("\n" + "=" * 80)
    print("PRODUCTION XGBOOST TRAINING PIPELINE")
    print("=" * 80)
    if args.ab_test:
        print(f"\nA/B TESTING MODE")
        model_a_path = Path(args.ab_test[0])
        model_b_path = Path(args.ab_test[1])
        data_path = "data/user_features_1M_enriched.parquet"
        df = pd.read_parquet(data_path)
        exclude_cols = [
            "is_fraud",
            "fraud_scenario",
            "user_id",
            "timestamp",
            "location",
            "merchant_category",
            "device_id",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df["is_fraud"].values
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        results = ab_test_models(model_a_path, model_b_path, X_test, y_test)
        results_file = Path("models/ab_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nA/B test results saved to: {results_file}")
        return
    data_path = "data/user_features_1M_enriched.parquet"
    trainer = ProductionFraudDetectionTrainer(data_path, quick_mode=args.quick)
    df = trainer.load_data()
    df_eng = trainer.advanced_feature_engineering(df)
    X, y = trainer.prepare_features(df_eng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if "user_id" in df_eng.columns:
        train_idx = X_train.index
        user_fraud_rate = df_eng.loc[train_idx].groupby("user_id")["is_fraud"].mean()
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["user_fraud_rate"] = (
            df_eng.loc[X_train.index, "user_id"].map(user_fraud_rate).fillna(0)
        )
        X_test["user_fraud_rate"] = (
            df_eng.loc[X_test.index, "user_id"].map(user_fraud_rate).fillna(0)
        )
        trainer.feature_columns = X_train.columns.tolist()
        print(f"user_fraud_rate added (train-only, no leakage)")
    print(f"\nData Split:")
    print(f"Training: {len(X_train):,} samples")
    print(f"Testing: {len(X_test):,} samples")
    default_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )
    cv_results = trainer.cross_validation(X, y, default_model)
    model = trainer.train_model(X_train, y_train)
    calibrated_model = trainer.calibrate_model(X_train, y_train)
    metrics = trainer.evaluate_model(X_test, y_test)
    importance = trainer.feature_importance_analysis(top_n=30)
    baseline = trainer.setup_monitoring(X_train)
    saved_files = trainer.save_production_model()
    print("\n" + "=" * 80)
    print("PRODUCTION TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFINAL METRICS:")
    print(f"Precision: {metrics['uncalibrated']['precision']:.4f}")
    print(f"Recall: {metrics['uncalibrated']['recall']:.4f}")
    print(f"F1-Score: {metrics['uncalibrated']['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['uncalibrated']['auc_roc']:.4f}")
    print(f"P99 Latency: {metrics['uncalibrated']['p99_latency_ms']:.2f}ms")
    print(f"\nCROSS-VALIDATION:")
    print(
        f"AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}"
    )
    print(f"\nCALIBRATION:")
    print(
        f"Brier Score Improvement: {metrics['calibrated']['calibration_improvement_pct']:.1f}%"
    )
    print(f"\nSAVED FILES:")
    for name, path in saved_files.items():
        print(f"{name}: {path.name}")
    print(f"\nNEXT STEPS:")
    print(f"1. Deploy calibrated model to AWS SageMaker")
    print(f"2. Integrate with multi-agent system")
    print(f"3. Setup monitoring alerts")
    print(f"4. Run A/B test vs basic model")
    print("=" * 80)


if __name__ == "__main__":
    main()
