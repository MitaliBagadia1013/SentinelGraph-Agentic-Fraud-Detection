from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")
ROOT = Path(__file__).parent
PARQUET_PATH = ROOT / "data" / "user_features_1M_enriched.parquet"
MODELS_DIR = ROOT / "models" / "saved" / "production"


class FraudDetectionPipeline:

    def __init__(
        self,
        use_agents: bool = True,
        use_graph: bool = True,
        model_path: Optional[Path] = None,
    ):
        self.use_agents = use_agents
        self.use_graph = use_graph
        logger.info("=" * 60)
        logger.info("FRAUD DETECTION PIPELINE — initialising")
        logger.info("=" * 60)
        self._graph_client = None
        if use_graph:
            try:
                from graph_db.neo4j_client import Neo4jClient, GraphConfig

                config = GraphConfig(
                    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    username=os.getenv("NEO4J_USERNAME", "neo4j"),
                    password=os.getenv("NEO4J_PASSWORD", "fraud_graph_2024"),
                    database=os.getenv("NEO4J_DATABASE", "neo4j"),
                )
                self._graph_client = Neo4jClient(config)
                if not self._graph_client.verify_connectivity():
                    logger.warning("Neo4j unreachable — graph risk signals disabled")
                    self._graph_client = None
                else:
                    logger.info("Layer 0 Neo4j connected")
            except Exception as e:
                logger.warning("Neo4j init failed (%s) — continuing without graph", e)
        try:
            from rules_engine.rules import RulesEngine

            self._rules = RulesEngine(enable_logging=False)
            logger.info("Layer 1 Rules engine ready")
        except Exception as e:
            logger.error("Rules engine init failed: %s", e)
            self._rules = None
        self._model = None
        self._feature_cols: List[str] = []
        self._label_encoders = {}
        self._load_xgboost_model(model_path)
        self._agent_graph = None
        if use_agents:
            try:
                from agents.graph import compile_fraud_investigation_graph

                self._agent_graph = compile_fraud_investigation_graph()
                logger.info("Layer 3 LangGraph agents compiled")
            except Exception as e:
                logger.warning("Agents init failed (%s) — LLM agents disabled", e)
        try:
            from hitl.queue_manager import HITLQueueManager

            self._hitl = HITLQueueManager()
            logger.info("Layer 4 HITL queue ready")
        except Exception as e:
            logger.warning("HITL init failed (%s)", e)
            self._hitl = None
        logger.info("=" * 60)
        logger.info("Pipeline ready")
        logger.info("=" * 60)

    def _load_xgboost_model(self, model_path: Optional[Path]) -> None:
        import joblib

        if model_path and Path(model_path).exists():
            candidates = [Path(model_path)]
        else:
            candidates = sorted(
                MODELS_DIR.glob("xgboost_calibrated_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not candidates:
            logger.warning(
                "No trained XGBoost model found in %s — run: python models/xgboost_trainer.py",
                MODELS_DIR,
            )
            return
        path = candidates[0]
        try:
            self._model = joblib.load(path)
            logger.info("Layer 2 XGBoost loaded: %s", path.name)
            fc_path = path.parent / "feature_columns_production.json"
            ts_part = path.stem.replace("xgboost_calibrated_", "")
            enc_path = path.parent / f"label_encoders_{ts_part}.pkl"
            if fc_path.exists():
                with open(fc_path) as f:
                    self._feature_cols = json.load(f)
                logger.info("Feature cols loaded: %d features", len(self._feature_cols))
            if enc_path.exists():
                self._label_encoders = joblib.load(enc_path)
        except Exception as e:
            logger.error("Failed to load XGBoost model: %s", e)
            self._model = None

    def _get_graph_risk(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        if not self._graph_client:
            return {}
        try:
            from graph_db.fraud_graph import FraudGraphDB, GraphConfig

            account_id = str(transaction.get("user_id", ""))
            if not account_id:
                return {}
            query = "\n MATCH (a:Account {account_id: $account_id})\n OPTIONAL MATCH (a)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(other:Account)\n WITH a, count(DISTINCT other) AS shared_device_accounts\n OPTIONAL MATCH (a)-[:HAS_EMAIL_DOMAIN]->(e:EmailDomain)<-[:HAS_EMAIL_DOMAIN]-(other2:Account)\n WITH a, shared_device_accounts, count(DISTINCT other2) AS shared_email_accounts\n OPTIONAL MATCH (a)-[:REGISTERED_FROM]->(ip:IPCountry)<-[:REGISTERED_FROM]-(other3:Account)\n RETURN shared_device_accounts,\n shared_email_accounts,\n count(DISTINCT other3) AS shared_ip_country_accounts\n "
            results = self._graph_client.run_query(query, {"account_id": account_id})
            if results:
                r = results[0]
                return {
                    "shared_device_accounts": r.get("shared_device_accounts", 0),
                    "shared_email_accounts": r.get("shared_email_accounts", 0),
                    "shared_ip_country_accounts": r.get(
                        "shared_ip_country_accounts", 0
                    ),
                    "graph_risk_score": min(
                        1.0,
                        (
                            r.get("shared_device_accounts", 0) * 0.4
                            + r.get("shared_email_accounts", 0) * 0.01
                        )
                        / 10,
                    ),
                }
        except Exception as e:
            logger.debug("Graph risk query failed: %s", e)
        return {}

    def _xgboost_score(self, features: Dict[str, Any]) -> tuple[float, float]:
        if self._model is None:
            score = float(features.get("fraud_score", 0.0))
            return (score, 0.5)
        try:
            if self._feature_cols:
                row = {c: features.get(c, 0) for c in self._feature_cols}
                X = pd.DataFrame([row])[self._feature_cols]
            else:
                X = pd.DataFrame([features]).select_dtypes(include=[np.number])
            X = X.fillna(0)
            proba = self._model.predict_proba(X)[0]
            fraud_prob = float(proba[1])
            confidence = float(max(proba))
            return (fraud_prob, confidence)
        except Exception as e:
            logger.debug("XGBoost scoring error: %s", e)
            score = float(features.get("fraud_score", 0.0))
            return (score, 0.5)

    def evaluate(
        self, transaction: Dict[str, Any], features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if features is None:
            features = transaction
        tx_id = transaction.get("transaction_id", f"TX-{id(transaction)}")
        amount = float(transaction.get("amount", 0))
        t_graph = time.perf_counter()
        graph_signals = self._get_graph_risk(transaction)
        graph_ms = (time.perf_counter() - t_graph) * 1000
        enriched = {**features, **graph_signals}
        t_rules = time.perf_counter()
        rules_output = None
        rules_triggered = []
        rules_decision = "PASS_TO_ML"
        if self._rules:
            try:
                rules_output = self._rules.evaluate(enriched)
                rules_triggered = [r.rule_name for r in rules_output.triggered_rules]
                rules_decision = rules_output.decision.value
            except Exception as e:
                logger.debug("Rules engine error: %s", e)
        rules_ms = (time.perf_counter() - t_rules) * 1000
        if rules_decision == "DECLINE":
            return self._build_result(
                tx_id,
                amount,
                xgb_score=1.0,
                confidence=1.0,
                decision="FRAUD",
                action="DECLINE",
                rules_triggered=rules_triggered,
                rules_ms=rules_ms,
                graph_ms=graph_ms,
                xgb_ms=0,
                agents_ms=0,
                total_ms=(time.perf_counter() - t0) * 1000,
                agents_ran=[],
                graph_signals=graph_signals,
                layer_stopped="rules_decline",
            )
        if rules_decision == "APPROVE":
            return self._build_result(
                tx_id,
                amount,
                xgb_score=0.0,
                confidence=1.0,
                decision="LEGIT",
                action="APPROVE",
                rules_triggered=rules_triggered,
                rules_ms=rules_ms,
                graph_ms=graph_ms,
                xgb_ms=0,
                agents_ms=0,
                total_ms=(time.perf_counter() - t0) * 1000,
                agents_ran=[],
                graph_signals=graph_signals,
                layer_stopped="rules_approve",
            )
        t_xgb = time.perf_counter()
        xgb_score, xgb_confidence = self._xgboost_score(enriched)
        xgb_ms = (time.perf_counter() - t_xgb) * 1000
        if xgb_score >= 0.85:
            return self._build_result(
                tx_id,
                amount,
                xgb_score=xgb_score,
                confidence=xgb_confidence,
                decision="FRAUD",
                action="DECLINE",
                rules_triggered=rules_triggered,
                rules_ms=rules_ms,
                graph_ms=graph_ms,
                xgb_ms=xgb_ms,
                agents_ms=0,
                total_ms=(time.perf_counter() - t0) * 1000,
                agents_ran=[],
                graph_signals=graph_signals,
                layer_stopped="xgboost_high",
            )
        if xgb_score < 0.2:
            return self._build_result(
                tx_id,
                amount,
                xgb_score=xgb_score,
                confidence=xgb_confidence,
                decision="LEGIT",
                action="APPROVE",
                rules_triggered=rules_triggered,
                rules_ms=rules_ms,
                graph_ms=graph_ms,
                xgb_ms=xgb_ms,
                agents_ms=0,
                total_ms=(time.perf_counter() - t0) * 1000,
                agents_ran=[],
                graph_signals=graph_signals,
                layer_stopped="xgboost_low",
            )
        t_agents = time.perf_counter()
        agents_ran = []
        agent_result = None
        if self.use_agents and self._agent_graph:
            try:
                from agents.graph import investigate_transaction

                rules_dict = {
                    "triggered_rules": rules_triggered,
                    "decision": rules_decision,
                }
                agent_result = investigate_transaction(
                    transaction=transaction,
                    xgboost_prediction=xgb_score,
                    xgboost_confidence=xgb_confidence,
                    rules_engine_result=rules_dict,
                    graph=self._agent_graph,
                )
                agents_ran = agent_result.get("agents_run", [])
            except Exception as e:
                logger.warning("Agent investigation error: %s", e)
        agents_ms = (time.perf_counter() - t_agents) * 1000
        if agent_result:
            decision = agent_result.get("final_decision", "UNCERTAIN")
            action = agent_result.get("recommended_action", "ESCALATE_TO_HITL")
            confidence = agent_result.get("combined_confidence", xgb_confidence)
        elif xgb_score >= 0.5:
            decision, action, confidence = (
                "UNCERTAIN",
                "ESCALATE_TO_HITL",
                xgb_confidence,
            )
        else:
            decision, action, confidence = ("LEGIT", "APPROVE", xgb_confidence)
        case_id = None
        if action == "ESCALATE_TO_HITL" and self._hitl:
            try:
                from hitl.models import HITLCase

                state_for_hitl = {
                    "transaction_id": tx_id,
                    "user_id": str(transaction.get("user_id", "")),
                    "amount": amount,
                    "merchant": str(transaction.get("merchant_id", "")),
                    "timestamp": str(transaction.get("timestamp", "")),
                    "xgboost_prediction": xgb_score,
                    "rules_engine_result": {"triggered_rules": rules_triggered},
                    "detective_findings": (
                        agent_result.get("detective_findings", {})
                        if agent_result
                        else {}
                    ),
                    "analyst_findings": (
                        agent_result.get("analyst_findings", {}) if agent_result else {}
                    ),
                    "verifier_findings": (
                        agent_result.get("verifier_findings", {})
                        if agent_result
                        else {}
                    ),
                    "graph_risk_signals": graph_signals,
                    "final_decision": decision,
                    "final_confidence": confidence,
                }
                case = HITLCase.from_investigation_state(state_for_hitl)
                case_id = self._hitl.enqueue(case)
                logger.info(
                    "HITL case enqueued: %s priority=%s", case_id, case.priority
                )
            except Exception as e:
                logger.warning("HITL enqueue failed: %s", e)
        total_ms = (time.perf_counter() - t0) * 1000
        result = self._build_result(
            tx_id,
            amount,
            xgb_score=xgb_score,
            confidence=confidence,
            decision=decision,
            action=action,
            rules_triggered=rules_triggered,
            rules_ms=rules_ms,
            graph_ms=graph_ms,
            xgb_ms=xgb_ms,
            agents_ms=agents_ms,
            total_ms=total_ms,
            agents_ran=agents_ran,
            graph_signals=graph_signals,
            layer_stopped="full_pipeline",
        )
        if case_id:
            result["hitl_case_id"] = case_id
        return result

    @staticmethod
    def _build_result(
        tx_id,
        amount,
        xgb_score,
        confidence,
        decision,
        action,
        rules_triggered,
        rules_ms,
        graph_ms,
        xgb_ms,
        agents_ms,
        total_ms,
        agents_ran,
        graph_signals,
        layer_stopped,
    ) -> Dict[str, Any]:
        return {
            "transaction_id": tx_id,
            "amount": amount,
            "final_decision": decision,
            "recommended_action": action,
            "xgboost_score": round(xgb_score, 4),
            "confidence": round(confidence, 4),
            "rules_triggered": rules_triggered,
            "graph_risk_signals": graph_signals,
            "agents_run": agents_ran,
            "layer_stopped_at": layer_stopped,
            "latency_ms": {
                "graph": round(graph_ms, 2),
                "rules": round(rules_ms, 2),
                "xgboost": round(xgb_ms, 2),
                "agents": round(agents_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    def close(self) -> None:
        if self._graph_client:
            self._graph_client.close()


def run_simulation(
    n: int = 100,
    fraud_only: bool = False,
    scenario: Optional[str] = None,
    verbose: bool = False,
    no_agents: bool = False,
    seed: int = 42,
    save: bool = False,
) -> None:
    from sklearn.model_selection import train_test_split

    logger.info("Loading test set from %s …", PARQUET_PATH)
    df_full = pd.read_parquet(PARQUET_PATH)
    _, test_idx = train_test_split(
        df_full.index, test_size=0.2, random_state=42, stratify=df_full["is_fraud"]
    )
    df = df_full.loc[test_idx].reset_index(drop=True)
    if fraud_only:
        df = df[df["is_fraud"] == 1]
    if scenario:
        df = df[df.get("fraud_scenario", pd.Series()) == scenario]
    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    logger.info(
        "Simulating %d transactions (fraud=%.1f%%) …",
        len(df),
        df["is_fraud"].mean() * 100,
    )
    pipeline = FraudDetectionPipeline(use_agents=not no_agents)
    results = []
    latencies = []
    correct = 0
    print(f"\n{'=' * 65}")
    print(f"SIMULATION — {len(df)} transactions")
    print(f"{'=' * 65}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        tx = {
            "transaction_id": f"SIM-{i:06d}",
            "user_id": str(row.get("user_id", "")),
            "amount": float(row.get("amount", 0)),
            "timestamp": str(row.get("timestamp", "")),
            "merchant_id": str(
                row.get("merchant_id", row.get("merchant_category", ""))
            ),
        }
        feats = row.to_dict()
        result = pipeline.evaluate(tx, feats)
        latency_ms = result["latency_ms"]["total_ms"]
        latencies.append(latency_ms)
        actual_fraud = int(row.get("is_fraud", 0)) == 1
        predicted_fraud = result["final_decision"] == "FRAUD"
        is_correct = actual_fraud == predicted_fraud
        if is_correct:
            correct += 1
        results.append(
            {
                "transaction_id": result["transaction_id"],
                "amount": result["amount"],
                "xgboost_score": result["xgboost_score"],
                "final_decision": result["final_decision"],
                "recommended_action": result["recommended_action"],
                "rules_triggered": result["rules_triggered"],
                "layer_stopped_at": result["layer_stopped_at"],
                "ground_truth": int(row.get("is_fraud", 0)),
                "fraud_scenario": row.get("fraud_scenario", ""),
                "correct": is_correct,
                "latency_ms": round(latency_ms, 2),
            }
        )
        if verbose:
            status = "" if is_correct else ""
            gt = "FRAUD" if actual_fraud else "LEGIT"
            print(
                f"[{i:>4}/{len(df)}] {status} GT={gt:<5} PRED={result['final_decision']:<9} score={result['xgboost_score']:.3f} ${result['amount']:>9.2f} {latency_ms:>6.1f}ms stopped={result['layer_stopped_at']} rules={result['rules_triggered']}"
            )
        elif i % 25 == 0:
            print(f"{i}/{len(df)} …", end="\r")
    pipeline.close()
    rdf = pd.DataFrame(results)
    total = len(rdf)
    tp = len(rdf[(rdf["ground_truth"] == 1) & (rdf["final_decision"] == "FRAUD")])
    fp = len(rdf[(rdf["ground_truth"] == 0) & (rdf["final_decision"] == "FRAUD")])
    tn = len(rdf[(rdf["ground_truth"] == 0) & (rdf["final_decision"] != "FRAUD")])
    fn = len(rdf[(rdf["ground_truth"] == 1) & (rdf["final_decision"] != "FRAUD")])
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    decisions = rdf["final_decision"].value_counts()
    layers = rdf["layer_stopped_at"].value_counts()
    escalated = len(rdf[rdf["recommended_action"] == "ESCALATE_TO_HITL"])
    print(f"\n{'=' * 65}")
    print(f"RESULTS")
    print(f"{'=' * 65}")
    print(
        f"Transactions : {total:,} (fraud={df['is_fraud'].sum():,} legit={(df['is_fraud'] == 0).sum():,})"
    )
    print(f"\nDECISIONS")
    for d, c in decisions.items():
        print(f"{d:<12} {c:>6,} ({c / total * 100:.1f}%)")
    print(f"HITL escalated {escalated:>6,}")
    print(f"\nPIPELINE ROUTING (layer_stopped_at)")
    for layer, cnt in layers.items():
        print(f"{layer:<25} {cnt:>6,} ({cnt / total * 100:.1f}%)")
    print(f"\nACCURACY")
    print(f"Overall {correct / total * 100:.1f}%")
    print(f"Precision {precision:.3f}")
    print(f"Recall {recall:.3f}")
    print(f"F1 Score {f1:.3f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"\nLATENCY")
    print(f"P50 {p50:>7.1f} ms")
    print(f"P95 {p95:>7.1f} ms")
    print(f"P99 {p99:>7.1f} ms")
    print(f"{'=' * 65}")
    if save:
        out = (
            ROOT
            / "logs"
            / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved {out}")


def health_check() -> None:
    print("\nHEALTH CHECK ")
    try:
        from graph_db.neo4j_client import Neo4jClient, GraphConfig

        c = Neo4jClient(
            GraphConfig(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                password=os.getenv("NEO4J_PASSWORD", "fraud_graph_2024"),
            )
        )
        ok = c.verify_connectivity()
        r = c.run_query("MATCH (n:Account) RETURN count(n) AS cnt")
        print(f"Neo4j {('' if ok else '')} accounts={(r[0]['cnt'] if r else '?')}")
        c.close()
    except Exception as e:
        print(f"Neo4j {e}")
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        keys = r.dbsize()
        print(f"Redis keys={keys:,}")
    except Exception as e:
        print(f"Redis {e}")
    try:
        import psycopg2

        conn = psycopg2.connect(
            os.getenv(
                "POSTGRES_DSN",
                "postgresql://fraud_user:fraud_pass_2024@localhost:5434/fraud_detection",
            )
        )
        conn.close()
        print(f"Postgres ")
    except Exception as e:
        print(f"Postgres {e}")
    candidates = sorted(
        MODELS_DIR.glob("xgboost_calibrated_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"XGBoost {candidates[0].name}")
    else:
        print(f"XGBoost no trained model found — run: python models/xgboost_trainer.py")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--simulate", action="store_true", help="Run simulation on test set"
    )
    parser.add_argument(
        "--health", action="store_true", help="Health check all services"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of transactions to simulate"
    )
    parser.add_argument(
        "--fraud-only", action="store_true", help="Simulate fraud transactions only"
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-agents", action="store_true", help="Skip LLM agents (faster, no API cost)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to logs/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.health:
        health_check()
    elif args.simulate:
        run_simulation(
            n=args.n,
            fraud_only=args.fraud_only,
            scenario=args.scenario,
            verbose=args.verbose,
            no_agents=args.no_agents,
            save=args.save,
            seed=args.seed,
        )
    else:
        parser.print_help()
