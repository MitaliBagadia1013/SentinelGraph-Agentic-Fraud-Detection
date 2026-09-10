import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from ieee_feature_engineering import IEEEFeatureEngineer


class IEEEXGBoostTrainer:

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.fe = IEEEFeatureEngineer()
        self.model = None
        self.calibrated_model = None

    def load_data(self, data_path: Path) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("LOADING IEEE-CIS DATASET")
        print("=" * 70)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}\nRun: python data/load_ieee_cis.py"
            )
        print(f"Loading {data_path.name}...")
        df = pd.read_parquet(data_path)
        print(f"   Rows: {len(df):,}")
        print(f"   Cols: {len(df.columns)}")
        print(f"   Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean() * 100:.2f}%)")
        return df

    def train_test_split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        print("\n" + "=" * 70)
        print("TRAIN/TEST SPLIT")
        print("=" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train):,} ({y_train.mean() * 100:.2f}% fraud)")
        print(f"Test:  {len(X_test):,} ({y_test.mean() * 100:.2f}% fraud)")
        return (X_train, X_test, y_train, y_test)

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> xgb.XGBClassifier:
        print("\n" + "=" * 70)
        print("XGBOOST TRAINING")
        print("=" * 70)
        if self.quick_mode:
            print("Quick mode: Using default hyperparameters")
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
            model.fit(X_train, y_train, verbose=False)
        else:
            print("Hyperparameter tuning (RandomizedSearchCV, 20 iterations)...")
            param_dist = {
                "max_depth": [4, 5, 6, 7],
                "learning_rate": [0.05, 0.1, 0.15],
                "n_estimators": [100, 150, 200],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.2],
            }
            base_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"\nBest params:")
            for key, val in search.best_params_.items():
                print(f"   {key}: {val}")
            print(f"\n   Best CV AUC: {search.best_score_:.4f}")
        self.model = model
        return model

    def calibrate_model(
        self, model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> CalibratedClassifierCV:
        print("\n" + "=" * 70)
        print("MODEL CALIBRATION")
        print("=" * 70)
        print("Applying Platt scaling (sigmoid calibration)...")
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        cal_pos = X_train.reset_index(drop=True).sample(frac=0.2, random_state=42).index
        X_cal = X_train.iloc[cal_pos]
        y_cal = y_train[cal_pos]
        calibrated.fit(X_cal, y_cal)
        print("Calibration complete")
        self.calibrated_model = calibrated
        return calibrated

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str = "XGBoost",
    ) -> Dict:
        print("\n" + "=" * 70)
        print(f"EVALUATION: {model_name}")
        print("=" * 70)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        print(f"\nOverall Metrics:")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"   AP (precision-recall AUC): {ap:.4f}")
        print(f"   Brier score: {brier:.4f} (lower=better, 0.0=perfect)")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        print(f"\nConfusion Matrix:")
        print(f"   TN: {tn:,}  |  FP: {fp:,}")
        print(f"   FN: {fn:,}  |  TP: {tp:,}")
        print(f"\n   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   FPR:       {fpr:.4f}")
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        idx_50_recall = np.argmin(np.abs(recalls - 0.5))
        prec_at_50_recall = precisions[idx_50_recall]
        print(f"\nPrecision @ 50% Recall: {prec_at_50_recall:.4f}")
        print(f"\nLatency Test (10K predictions):")
        X_bench = X_test.iloc[:10000]
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model.predict_proba(X_bench)
            latencies.append((time.perf_counter() - t0) * 1000 / len(X_bench))
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        print(f"   P50: {p50:.3f} ms")
        print(f"   P95: {p95:.3f} ms")
        print(f"   P99: {p99:.3f} ms")
        return {
            "auc": float(auc),
            "ap": float(ap),
            "brier": float(brier),
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr),
            "prec_at_50_recall": float(prec_at_50_recall),
            "latency_p50": float(p50),
            "latency_p95": float(p95),
            "latency_p99": float(p99),
        }

    def save_model(self, output_dir: Path, metrics: Dict = None):
        print("\n" + "=" * 70)
        print("SAVING MODEL")
        print("=" * 70)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"xgboost_ieee_calibrated_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.calibrated_model, f)
        print(f" Model: {model_path}")
        trained_features = list(
            getattr(self.model, "feature_names_in_", self.fe.feature_columns)
        )
        feature_path = output_dir / f"feature_columns_ieee_{timestamp}.json"
        with open(feature_path, "w") as f:
            json.dump(trained_features, f, indent=2)
        print(f" Features: {feature_path}")
        importance_df = pd.DataFrame(
            {"feature": trained_features, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance_path = output_dir / f"feature_importance_ieee_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f" Importance: {importance_path}")
        if metrics is not None:
            meta = {
                "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
                "trained_at": timestamp,
                "n_features": len(trained_features),
                "quick_mode": self.quick_mode,
                "metrics": metrics,
            }
            meta_path = output_dir / f"model_metadata_ieee_{timestamp}.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f" Metadata: {meta_path}")
        print(f"\nTop 20 Features:")
        for idx, row in importance_df.head(20).iterrows():
            print(f"   {row['feature']:30s}  {row['importance']:.4f}")
        return model_path


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on IEEE-CIS dataset")
    parser.add_argument("--quick", action="store_true", help="Quick mode (no tuning)")
    parser.add_argument(
        "--output", type=str, default="models/saved/production", help="Output directory"
    )
    args = parser.parse_args()
    ROOT = Path(__file__).parent.parent
    data_path = ROOT / "data" / "ieee_cis_merged.parquet"
    output_dir = ROOT / args.output
    trainer = IEEEXGBoostTrainer(quick_mode=args.quick)
    df = trainer.load_data(data_path)
    df_eng = trainer.fe.fit_transform(df)
    X, y = trainer.fe.prepare_for_training(df_eng, add_aggregations=False)
    X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
    X_train = trainer.fe.add_train_aggregations(X_train, y_train)
    X_test = trainer.fe.add_test_aggregations(X_test, X_train, y_train)
    model = trainer.train_xgboost(X_train, y_train)
    calibrated = trainer.calibrate_model(model, X_train, y_train)
    metrics = trainer.evaluate(calibrated, X_test, y_test, "Calibrated XGBoost")
    model_path = trainer.save_model(output_dir, metrics=metrics)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"\nNext steps:")
    print(f"  1. Measure the agent layer's FPR reduction:")
    print(f"     python models/calculate_agent_impact.py")
    print(f"  2. Update agent prompts for IEEE-CIS features:")
    print(f"     vim agents/prompts.py")
    print(f"  3. Test HITL UI:")
    print(f"     python hitl/ui/app.py")


if __name__ == "__main__":
    main()
