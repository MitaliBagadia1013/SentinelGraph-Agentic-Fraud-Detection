from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
import uuid

CaseStatus = Literal["PENDING", "IN_REVIEW", "RESOLVED", "TIMED_OUT"]
AnalystDecision = Literal["CONFIRM_FRAUD", "CLEAR", "ESCALATE_FURTHER"]
Priority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class HITLCase:
    case_id: str = field(
        default_factory=lambda: f"CASE-{uuid.uuid4().hex[:10].upper()}"
    )
    transaction_id: str = ""
    user_id: str = ""
    amount: float = 0.0
    merchant: str = ""
    timestamp: str = ""
    xgboost_score: float = 0.0
    rules_triggered: List[str] = field(default_factory=list)
    detective_findings: Dict[str, Any] = field(default_factory=dict)
    analyst_findings: Dict[str, Any] = field(default_factory=dict)
    verifier_findings: Dict[str, Any] = field(default_factory=dict)
    graph_risk_signals: Dict[str, Any] = field(default_factory=dict)
    agent_final_decision: str = "UNCERTAIN"
    agent_confidence: float = 0.0
    escalation_reason: str = ""
    priority: Priority = "HIGH"
    status: CaseStatus = "PENDING"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    assigned_to: Optional[str] = None
    assigned_at: Optional[str] = None
    resolved_at: Optional[str] = None
    sla_minutes: int = 30
    is_sla_breached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_investigation_state(cls, state: Dict[str, Any]) -> "HITLCase":
        verifier = state.get("verifier_findings") or {}
        rules = state.get("rules_engine_result") or {}
        return cls(
            transaction_id=state.get("transaction_id", ""),
            user_id=state.get("user_id", ""),
            amount=state.get("amount", 0.0),
            merchant=state.get("merchant", ""),
            timestamp=state.get("timestamp", ""),
            xgboost_score=state.get("xgboost_prediction", 0.0),
            rules_triggered=rules.get("triggered_rules", []),
            detective_findings=state.get("detective_findings") or {},
            analyst_findings=state.get("analyst_findings") or {},
            verifier_findings=verifier,
            graph_risk_signals=state.get("graph_risk_signals") or {},
            agent_final_decision=state.get("final_decision", "UNCERTAIN"),
            agent_confidence=state.get("final_confidence", 0.0),
            escalation_reason=(
                verifier.get("reasoning_chain", ["No reason provided"])[-1]
                if verifier.get("reasoning_chain")
                else "Escalated by Verifier agent"
            ),
            priority=_derive_priority(
                state.get("xgboost_prediction", 0.0), state.get("amount", 0.0)
            ),
        )

    def assign(self, analyst: str) -> None:
        self.assigned_to = analyst
        self.assigned_at = datetime.utcnow().isoformat()
        self.status = "IN_REVIEW"

    def resolve(self, analyst: str) -> None:
        self.assigned_to = analyst
        self.resolved_at = datetime.utcnow().isoformat()
        self.status = "RESOLVED"

    def check_sla(self) -> bool:
        if self.status == "RESOLVED":
            return False
        created = datetime.fromisoformat(self.created_at)
        elapsed = (datetime.utcnow() - created).total_seconds() / 60
        if elapsed > self.sla_minutes:
            self.is_sla_breached = True
        return self.is_sla_breached


@dataclass
class HITLDecision:
    decision_id: str = field(
        default_factory=lambda: f"DEC-{uuid.uuid4().hex[:10].upper()}"
    )
    case_id: str = ""
    transaction_id: str = ""
    analyst: str = ""
    decision: AnalystDecision = "CLEAR"
    confidence: float = 1.0
    notes: str = ""
    block_account: bool = False
    flag_fraud_ring: bool = False
    request_graph_update: bool = True
    retrain_signal: bool = True
    fraud_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    decided_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    review_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueueStats:
    total_pending: int = 0
    total_in_review: int = 0
    total_resolved: int = 0
    total_timed_out: int = 0
    sla_breached: int = 0
    avg_review_time_s: float = 0.0
    oldest_case_age_minutes: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _derive_priority(xgb_score: float, amount: float) -> Priority:
    if xgb_score >= 0.9 or amount >= 10000:
        return "CRITICAL"
    if xgb_score >= 0.75 or amount >= 5000:
        return "HIGH"
    if xgb_score >= 0.5 or amount >= 1000:
        return "MEDIUM"
    return "LOW"
