import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from hitl.models import AnalystDecision, HITLCase, HITLDecision
from hitl.queue_manager import HITLQueueManager

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    success: bool
    case_id: str
    decision_id: str
    actions_taken: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionHandler:

    def __init__(
        self,
        queue: HITLQueueManager,
        graph_db=None,
        retrain_log_path: str = "logs/hitl_retrain_queue.jsonl",
    ):
        self._queue = queue
        self._graph_db = graph_db
        self._retrain_log = retrain_log_path
        self._decision_store: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(retrain_log_path), exist_ok=True)
        logger.info(
            "DecisionHandler ready (graph_db=%s)",
            "connected" if graph_db else "disabled",
        )

    def process(self, decision: HITLDecision) -> DecisionResult:
        actions: List[str] = []
        error = self._validate(decision)
        if error:
            logger.warning("Decision rejected for case %s: %s", decision.case_id, error)
            return DecisionResult(
                success=False,
                case_id=decision.case_id,
                decision_id=decision.decision_id,
                error=error,
            )
        case = self._queue.get_case(decision.case_id)
        self._queue.mark_resolved(decision.case_id, decision.analyst)
        actions.append("case_resolved")
        self._save_to_audit_log(case, decision)
        actions.append("audit_log_saved")
        if decision.request_graph_update:
            graph_actions = self._update_graph(case, decision)
            actions.extend(graph_actions)
        if decision.retrain_signal:
            self._append_retrain_record(case, decision)
            actions.append("retrain_record_queued")
        logger.info(
            "Decision %s processed for case %s by %s -> %s | actions: %s",
            decision.decision_id,
            decision.case_id,
            decision.analyst,
            decision.decision,
            ", ".join(actions),
        )
        return DecisionResult(
            success=True,
            case_id=decision.case_id,
            decision_id=decision.decision_id,
            actions_taken=actions,
        )

    def get_decisions(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if case_id:
            return [d for d in self._decision_store if d.get("case_id") == case_id]
        return list(self._decision_store)

    def _validate(self, decision: HITLDecision) -> Optional[str]:
        case = self._queue.get_case(decision.case_id)
        if not case:
            return f"Case {decision.case_id} not found in queue"
        if case.status == "RESOLVED":
            return f"Case {decision.case_id} is already resolved"
        if case.status == "TIMED_OUT":
            return f"Case {decision.case_id} has timed out — cannot accept decision"
        if case.assigned_to and case.assigned_to != decision.analyst:
            return f"Case {decision.case_id} is assigned to {case.assigned_to}, not {decision.analyst}"
        if not decision.decision:
            return "Decision field is required"
        return None

    def _save_to_audit_log(self, case: HITLCase, decision: HITLDecision) -> None:
        record = {
            "decision_id": decision.decision_id,
            "case_id": decision.case_id,
            "transaction_id": decision.transaction_id or case.transaction_id,
            "user_id": case.user_id,
            "analyst": decision.analyst,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "notes": decision.notes,
            "fraud_type": decision.fraud_type,
            "tags": decision.tags,
            "block_account": decision.block_account,
            "flag_fraud_ring": decision.flag_fraud_ring,
            "xgboost_score": case.xgboost_score,
            "amount": case.amount,
            "merchant": case.merchant,
            "decided_at": decision.decided_at,
            "review_time_s": decision.review_time_seconds,
            "sla_breached": case.is_sla_breached,
        }
        self._decision_store.append(record)

    def _update_graph(self, case: HITLCase, decision: HITLDecision) -> List[str]:
        actions: List[str] = []
        if not self._graph_db:
            logger.debug(
                "graph_db not connected — skipping Neo4j update for case %s",
                case.case_id,
            )
            actions.append("graph_update_skipped_no_db")
            return actions
        try:
            is_fraud = decision.decision == "CONFIRM_FRAUD"
            risk_score = case.xgboost_score if is_fraud else 0.1
            self._graph_db.update_account_risk(
                account_id=case.user_id,
                risk_score=risk_score,
                risk_reason=f"HITL decision: {decision.decision} by {decision.analyst}",
            )
            actions.append("graph_account_risk_updated")
            if is_fraud and decision.flag_fraud_ring:
                self._graph_db.flag_fraud_ring(
                    seed_account_id=case.user_id,
                    ring_name=f"HITL-{case.case_id}",
                    confidence=decision.confidence,
                )
                actions.append("graph_fraud_ring_flagged")
        except Exception as exc:
            logger.error("Neo4j update failed for case %s: %s", case.case_id, exc)
            actions.append(f"graph_update_failed:{exc}")
        return actions

    def _append_retrain_record(self, case: HITLCase, decision: HITLDecision) -> None:
        label = 1 if decision.decision == "CONFIRM_FRAUD" else 0
        record = {
            "transaction_id": case.transaction_id,
            "label": label,
            "label_source": "HITL",
            "analyst": decision.analyst,
            "fraud_type": decision.fraud_type,
            "xgboost_score_at_time": case.xgboost_score,
            "decided_at": decision.decided_at,
            "amount": case.amount,
            "merchant": case.merchant,
        }
        try:
            with open(self._retrain_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error(
                "Failed to write retrain record for %s: %s", case.transaction_id, exc
            )
