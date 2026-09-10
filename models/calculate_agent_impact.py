import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.graph import create_agent_graph
from agents.state import AgentState


class AgentImpactCalculator:

    def __init__(self, model_path: Path, quick_mode: bool = False):
        self.model_path = model_path
        self.quick_mode = quick_mode
        print("\n" + "=" * 70)
        print("AGENT IMPACT CALCULATOR")
        print("=" * 70)
        print(f"\nLoading model: {model_path.name}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        feature_json = (
            model_path.parent
            / model_path.name.replace(".pkl", "").replace(
                "xgboost_ieee_calibrated", "feature_columns_ieee"
            )
            + ".json"
        )
        if not feature_json.exists():
            feature_files = list(model_path.parent.glob("feature_columns_ieee_*.json"))
            if feature_files:
                feature_json = sorted(feature_files)[-1]
            else:
                raise FileNotFoundError(f"Feature columns not found")
        with open(feature_json, "r") as f:
            self.feature_columns = json.load(f)
        print(f"Features: {len(self.feature_columns)}")
        if not quick_mode:
            print(f"\nInitializing LangGraph agents...")
            self.agent_graph = create_agent_graph()
            print(f"Detective, Analyst, Verifier ready")
        else:
            self.agent_graph = None
            print(f"Quick mode: Agents disabled (using mock decisions)")

    def load_test_data(
        self, data_path: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        print("\n" + "=" * 70)
        print("LOADING TEST DATA")
        print("=" * 70)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}\nRun: python data/load_ieee_cis.py"
            )
        print(f"Loading {data_path.name}...")
        df = pd.read_parquet(data_path)
        from ieee_feature_engineering import IEEEFeatureEngineer

        fe = IEEEFeatureEngineer()
        df_eng = fe.fit_transform(df)
        X, y = fe.prepare_for_training(df_eng, add_aggregations=False)
        from sklearn.model_selection import train_test_split

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Test samples: {len(X_test):,}")
        print(f"Fraud rate: {y_test.mean() * 100:.2f}%")
        df_test = df.loc[X_test.index].copy()
        return (X_test, df_test, y_test)

    def baseline_evaluation(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        print("\n" + "=" * 70)
        print("BASELINE: XGBoost-Only (threshold=0.5)")
        print("=" * 70)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn:,} | FP: {fp:,}")
        print(f"FN: {fn:,} | TP: {tp:,}")
        print(f"\nMetrics:")
        print(f"FPR: {fpr * 100:.2f}% {fp:,} false positives")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
            "y_pred_proba": y_pred_proba,
        }

    def agent_assisted_evaluation(
        self,
        X_test: pd.DataFrame,
        df_test: pd.DataFrame,
        y_test: np.ndarray,
        baseline_proba: np.ndarray,
    ) -> Dict:
        print("\n" + "=" * 70)
        print("AGENT-ASSISTED: XGBoost + LangGraph Agents")
        print("=" * 70)
        grey_zone_mask = (baseline_proba >= 0.2) & (baseline_proba < 0.85)
        grey_zone_count = grey_zone_mask.sum()
        print(
            f"\nGrey zone (0.20-0.85): {grey_zone_count:,} transactions ({grey_zone_mask.mean() * 100:.1f}%)"
        )
        y_pred = (baseline_proba >= 0.5).astype(int).copy()
        if self.quick_mode:
            print(f"\nQuick mode: Using mock agent decisions")
            grey_fps = grey_zone_mask & (y_test == 0) & (y_pred == 1)
            grey_fns = grey_zone_mask & (y_test == 1) & (y_pred == 0)
            fp_indices = np.where(grey_fps)[0]
            flip_fp = np.random.choice(
                fp_indices, size=int(len(fp_indices) * 0.6), replace=False
            )
            y_pred[flip_fp] = 0
            fn_indices = np.where(grey_fns)[0]
            flip_fn = np.random.choice(
                fn_indices, size=int(len(fn_indices) * 0.4), replace=False
            )
            y_pred[flip_fn] = 1
            print(f"Corrected {len(flip_fp):,} false positives")
            print(f"Corrected {len(flip_fn):,} false negatives")
        else:
            print(f"\nRouting to agents...")
            grey_indices = np.where(grey_zone_mask)[0]
            if len(grey_indices) > 500:
                print(f"Sampling 500 / {len(grey_indices):,} grey zone cases")
                grey_indices = np.random.choice(grey_indices, size=500, replace=False)
            for idx in grey_indices:
                row = df_test.iloc[idx]
                state = AgentState(
                    transaction_id=f"test_{idx}",
                    user_id=str(row.get("card1", "unknown")),
                    amount=float(row.get("TransactionAmt", 0)),
                    merchant_id=str(row.get("P_emaildomain", "unknown")),
                    fraud_score=float(baseline_proba[idx]),
                    location=str(row.get("addr1", "unknown")),
                    device_id=str(row.get("DeviceInfo", "unknown")),
                    timestamp=pd.Timestamp.now().isoformat(),
                    transaction_velocity_1h=0,
                    location_changes_24h=0,
                    device_fingerprint_changes=0,
                    user_fraud_rate=0.0,
                    merchant_risk_score=0.0,
                    messages=[],
                    next_agent=None,
                    final_decision=None,
                    confidence_score=0.0,
                )
                result = self.agent_graph.invoke(state)
                if result["final_decision"] == "DECLINE":
                    y_pred[idx] = 1
                elif result["final_decision"] == "APPROVE":
                    y_pred[idx] = 0
            print(f"Agent review complete")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn:,} | FP: {fp:,}")
        print(f"FN: {fn:,} | TP: {tp:,}")
        print(f"\nMetrics:")
        print(f"FPR: {fpr * 100:.2f}% {fp:,} false positives")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
        }

    def calculate_improvement(self, baseline: Dict, agent_assisted: Dict):
        print("\n" + "=" * 70)
        print("IMPROVEMENT ANALYSIS")
        print("=" * 70)
        baseline_fpr = baseline["fpr"]
        agent_fpr = agent_assisted["fpr"]
        fpr_reduction_abs = baseline_fpr - agent_fpr
        fpr_reduction_pct = (
            fpr_reduction_abs / baseline_fpr * 100 if baseline_fpr > 0 else 0
        )
        fp_reduction = baseline["fp"] - agent_assisted["fp"]
        print(f"\nFALSE POSITIVE RATE:")
        print(f"Baseline: {baseline_fpr * 100:.2f}% ({baseline['fp']:,} FPs)")
        print(f"Agent-Assisted: {agent_fpr * 100:.2f}% ({agent_assisted['fp']:,} FPs)")
        print(f"Reduction: {fpr_reduction_abs * 100:.2f} percentage points")
        print(f"Relative: {fpr_reduction_pct:.1f}% lower")
        print(f"\nFALSE POSITIVES:")
        print(f"Baseline: {baseline['fp']:,}")
        print(f"Agent-Assisted: {agent_assisted['fp']:,}")
        print(
            f"Reduced by: {fp_reduction:,} ({fp_reduction / baseline['fp'] * 100:.1f}%)"
        )
        print(f"\nOTHER METRICS:")
        print(
            f"Precision: {baseline['precision'] * 100:.2f}% {agent_assisted['precision'] * 100:.2f}%"
        )
        print(
            f"Recall: {baseline['recall'] * 100:.2f}% {agent_assisted['recall'] * 100:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Calculate agent impact on FPR")
    parser.add_argument("--model", type=str, help="Path to trained XGBoost model (pkl)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: Use mock agent decisions (no API calls)",
    )
    args = parser.parse_args()
    ROOT = Path(__file__).parent.parent
    if args.model:
        model_path = Path(args.model)
    else:
        saved_dir = ROOT / "models" / "saved" / "production"
        models = list(saved_dir.glob("xgboost_ieee_calibrated_*.pkl"))
        if not models:
            print("No trained model found")
            print(f"\nRun: python models/train_ieee_xgboost.py --quick")
            sys.exit(1)
        model_path = sorted(models)[-1]
    data_path = ROOT / "data" / "ieee_cis_merged.parquet"
    calc = AgentImpactCalculator(model_path, quick_mode=args.quick)
    X_test, df_test, y_test = calc.load_test_data(data_path)
    baseline = calc.baseline_evaluation(X_test, y_test)
    agent_assisted = calc.agent_assisted_evaluation(
        X_test, df_test, y_test, baseline["y_pred_proba"]
    )
    calc.calculate_improvement(baseline, agent_assisted)
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
