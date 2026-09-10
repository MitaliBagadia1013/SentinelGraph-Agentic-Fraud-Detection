import argparse
import time
import random
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import joblib

    _MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "saved",
        "production",
    )
    _xgb_model = None
    _feature_columns = None
    _label_encoders = None
    _MODEL_READY = False
    _model_files = os.listdir(_MODELS_DIR) if os.path.isdir(_MODELS_DIR) else []
    _cal_files = [
        f
        for f in _model_files
        if f.startswith("xgboost_calibrated") and f.endswith(".pkl")
    ]
    _feat_file = os.path.join(_MODELS_DIR, "feature_columns_production.json")
    _enc_files = [
        f for f in _model_files if f.startswith("label_encoders") and f.endswith(".pkl")
    ]
    if _cal_files and os.path.exists(_feat_file):
        import json as _json

        _xgb_model = joblib.load(os.path.join(_MODELS_DIR, sorted(_cal_files)[-1]))
        _feature_columns = _json.load(open(_feat_file))
        if _enc_files:
            _label_encoders = joblib.load(
                os.path.join(_MODELS_DIR, sorted(_enc_files)[-1])
            )
        _MODEL_READY = True
        print(f"XGBoost model loaded ({len(_feature_columns)} features)")
    else:
        print("XGBoost model not found — using parquet fraud_score as proxy")
except Exception as _e:
    _MODEL_READY = False
    print(f"XGBoost load error ({_e}) — using parquet fraud_score as proxy")
try:
    from rules_engine.rules import RulesEngine

    _rules_engine = RulesEngine(enable_logging=False)
    _RULES_READY = True
    print("Rules engine loaded")
except Exception as _e:
    _RULES_READY = False
    print(f"Rules engine load error ({_e})")
try:
    from agents.graph import investigate_transaction, compile_fraud_investigation_graph

    _AGENT_GRAPH = compile_fraud_investigation_graph()
    _AGENTS_READY = True
    print("LangGraph agents loaded")
except Exception as _e:
    _AGENTS_READY = False
    _AGENT_GRAPH = None
    print(f"LangGraph agents load error ({_e}) — will skip agent layer")
PARQUET_PATH = "data/user_features_1M_enriched.parquet"
TEST_SIZE = 0.2
RANDOM_STATE = 42
META_COLS = [
    "user_id",
    "timestamp",
    "amount",
    "location",
    "device_id",
    "is_fraud",
    "fraud_score",
    "fraud_scenario",
]


def load_test_set(
    fraud_only: bool = False,
    scenario: Optional[str] = None,
    n: int = 100,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    from sklearn.model_selection import train_test_split

    y = df["is_fraud"]
    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    print(
        f"Test set: {len(test_df):,} rows ({test_df['is_fraud'].sum():,} fraud, {(test_df['is_fraud'] == 0).sum():,} legit)"
    )
    if fraud_only:
        test_df = test_df[test_df["is_fraud"] == 1]
        print(f"Filtered to fraud-only: {len(test_df):,} rows")
    if scenario:
        test_df = test_df[test_df["fraud_scenario"] == scenario]
        print(f"Filtered to scenario '{scenario}': {len(test_df):,} rows")
    if len(test_df) == 0:
        raise ValueError("No rows match the given filters")
    n = min(n, len(test_df))
    sampled = test_df.sample(n=n, random_state=seed)
    print(f"Sampled {n} rows for simulation\n")
    return sampled


def row_to_transaction(row: pd.Series) -> Dict[str, Any]:
    row_dict = row.to_dict()
    transaction = {
        "transaction_id": f"SIM-{row.name}-{int(time.time() * 1000)}",
        "user_id": str(row_dict.get("user_id", f"USER-{row.name}")),
        "amount": float(row_dict.get("amount", 0.0)),
        "merchant": str(row_dict.get("location", "unknown")),
        "timestamp": str(row_dict.get("timestamp", datetime.utcnow().isoformat())),
        "device_id": str(row_dict.get("device_id", "unknown")),
    }
    features = {k: v for k, v in row_dict.items() if k not in META_COLS}
    ground_truth = {
        "is_fraud": int(row_dict.get("is_fraud", 0)),
        "fraud_score": float(row_dict.get("fraud_score", 0.0)),
        "fraud_scenario": str(row_dict.get("fraud_scenario", "none")),
    }
    return {
        "transaction": transaction,
        "features": features,
        "ground_truth": ground_truth,
    }


_pipeline = None


def _get_pipeline(no_agents: bool = False):
    global _pipeline
    if _pipeline is None:
        from main import FraudDetectionPipeline

        _pipeline = FraudDetectionPipeline(use_agents=not no_agents)
    return _pipeline


def run_pipeline(tx_dict: Dict[str, Any], no_agents: bool = False) -> Dict[str, Any]:
    pipeline = _get_pipeline(no_agents=no_agents)
    result = pipeline.evaluate(
        transaction=tx_dict["transaction"], features=tx_dict["features"]
    )
    return result


class RealtimeSimulator:

    def __init__(self, verbose: bool = False, no_agents: bool = False):
        self.verbose = verbose
        self.no_agents = no_agents
        self.results: List[Dict[str, Any]] = []

    def run(self, test_df: pd.DataFrame) -> None:
        total = len(test_df)
        correct = 0
        latencies = []
        print(f"{'=' * 60}")
        print(f"Starting simulation: {total} transactions")
        print(f"{'=' * 60}\n")
        for i, (idx, row) in enumerate(test_df.iterrows(), 1):
            tx_dict = row_to_transaction(row)
            ground_truth = tx_dict["ground_truth"]
            t0 = time.perf_counter()
            result = run_pipeline(tx_dict, no_agents=self.no_agents)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            actual_fraud = ground_truth["is_fraud"] == 1
            predicted_fraud = result["final_decision"] == "FRAUD"
            is_correct = actual_fraud == predicted_fraud
            if is_correct:
                correct += 1
            self.results.append(
                {
                    "transaction_id": result["transaction_id"],
                    "user_id": tx_dict["transaction"]["user_id"],
                    "amount": tx_dict["transaction"]["amount"],
                    "xgboost_score": result["xgboost_score"],
                    "final_decision": result["final_decision"],
                    "recommended_action": result["recommended_action"],
                    "rules_triggered": result["rules_triggered"],
                    "layer_stopped_at": result.get("layer_stopped_at", ""),
                    "agents_run": result.get("agents_run", []),
                    "graph_risk_signals": result.get("graph_risk_signals", {}),
                    "hitl_case_id": result.get("hitl_case_id", ""),
                    "ground_truth": ground_truth["is_fraud"],
                    "fraud_scenario": ground_truth["fraud_scenario"],
                    "correct": is_correct,
                    "latency_ms": round(latency_ms, 3),
                    "latency_breakdown": result.get("latency_ms", {}),
                }
            )
            if self.verbose:
                status = "" if is_correct else ""
                fraud_label = "FRAUD" if actual_fraud else "LEGIT"
                print(
                    f"[{i:>4}/{total}] {status} GT={fraud_label:<5} PRED={result['final_decision']:<9} score={result['xgboost_score']:.3f} amount=${tx_dict['transaction']['amount']:>8.2f} {latency_ms:.1f}ms stopped={result.get('layer_stopped_at', '')} rules={result['rules_triggered']}"
                )
            elif i % 10 == 0:
                print(f"Processed {i}/{total}...", end="\r")
        print(f"\n")
        self._print_summary(latencies, correct, total, test_df)

    def _print_summary(
        self, latencies: List[float], correct: int, total: int, test_df: pd.DataFrame
    ) -> None:
        results_df = pd.DataFrame(self.results)
        accuracy = correct / total * 100
        tp = len(
            results_df[
                (results_df["ground_truth"] == 1)
                & (results_df["final_decision"] == "FRAUD")
            ]
        )
        fp = len(
            results_df[
                (results_df["ground_truth"] == 0)
                & (results_df["final_decision"] == "FRAUD")
            ]
        )
        tn = len(
            results_df[
                (results_df["ground_truth"] == 0)
                & (results_df["final_decision"] != "FRAUD")
            ]
        )
        fn = len(
            results_df[
                (results_df["ground_truth"] == 1)
                & (results_df["final_decision"] != "FRAUD")
            ]
        )
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        decisions = results_df["final_decision"].value_counts()
        escalated = len(
            results_df[results_df["recommended_action"] == "ESCALATE_TO_HITL"]
        )
        print(f"{'=' * 60}")
        print(f"SIMULATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Total transactions : {total:,}")
        print(
            f"Fraud (actual) : {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean() * 100:.1f}%)"
        )
        print(f"Legit (actual) : {(test_df['is_fraud'] == 0).sum():,}")
        print(f"")
        print(f"DECISIONS")
        for decision, count in decisions.items():
            print(f"{decision:<12}: {count:,}")
        print(f"Escalated HITL : {escalated:,}")
        print(f"")
        print(f"ACCURACY")
        print(f"Overall accuracy : {accuracy:.1f}%")
        print(f"Precision : {precision:.3f}")
        print(f"Recall : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(f"True Positives : {tp:,}")
        print(f"False Positives : {fp:,}")
        print(f"True Negatives : {tn:,}")
        print(f"False Negatives : {fn:,}")
        print(f"")
        print(f"LATENCY (real pipeline)")
        print(f"P50 : {p50:.2f}ms")
        print(f"P95 : {p95:.2f}ms")
        print(f"P99 : {p99:.2f}ms")
        print(f"")
        if "layer_stopped_at" in results_df.columns:
            print(f"PIPELINE ROUTING")
            for layer, cnt in results_df["layer_stopped_at"].value_counts().items():
                print(f"{layer:<28}: {cnt:,} ({cnt / total * 100:.1f}%)")
        print(f"{'=' * 60}")

    def save_results(self, path: str = "logs/simulation_results.json") -> None:
        import os

        os.makedirs("logs", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time transaction simulator")
    parser.add_argument(
        "--n", type=int, default=100, help="Number of transactions to simulate"
    )
    parser.add_argument(
        "--fraud-only", action="store_true", help="Only use fraud transactions"
    )
    parser.add_argument(
        "--scenario", type=str, default=None, help="Filter by fraud_scenario"
    )
    parser.add_argument("--verbose", action="store_true", help="Print each transaction")
    parser.add_argument(
        "--no-agents", action="store_true", help="Skip LLM agents (faster, no API cost)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to logs/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    test_df = load_test_set(
        fraud_only=args.fraud_only, scenario=args.scenario, n=args.n, seed=args.seed
    )
    sim = RealtimeSimulator(verbose=args.verbose, no_agents=args.no_agents)
    sim.run(test_df)
    if args.save:
        sim.save_results()


if __name__ == "__main__":
    main()
