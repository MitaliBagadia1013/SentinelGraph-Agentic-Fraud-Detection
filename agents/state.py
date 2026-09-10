from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime

FraudDecision = Literal["FRAUD", "LEGIT", "UNCERTAIN"]
ActionRecommendation = Literal["APPROVE", "DECLINE", "REVIEW", "ESCALATE_TO_HITL"]


@dataclass
class DetectiveFindings:
    agent_name: str = "Detective"
    risk_level: str = ""
    red_flags: List[str] = None
    green_flags: List[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    should_investigate_deeper: bool = True
    execution_time_ms: float = 0.0

    def __post_init__(self):
        if self.red_flags is None:
            self.red_flags = []
        if self.green_flags is None:
            self.green_flags = []


@dataclass
class AnalystFindings:
    agent_name: str = "Analyst"
    fraud_type: Optional[str] = None
    pattern_analysis: Dict[str, Any] = None
    similar_cases: List[Dict[str, Any]] = None
    anomaly_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    key_evidence: List[str] = None
    execution_time_ms: float = 0.0

    def __post_init__(self):
        if self.pattern_analysis is None:
            self.pattern_analysis = {}
        if self.similar_cases is None:
            self.similar_cases = []
        if self.key_evidence is None:
            self.key_evidence = []


@dataclass
class VerifierFindings:
    agent_name: str = "Verifier"
    final_decision: FraudDecision = "UNCERTAIN"
    confidence: float = 0.0
    reasoning_chain: List[str] = None
    false_positive_check: Dict[str, Any] = None
    recommended_action: ActionRecommendation = "REVIEW"
    execution_time_ms: float = 0.0

    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.false_positive_check is None:
            self.false_positive_check = {}


class FraudInvestigationState(TypedDict):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    timestamp: str
    transaction_features: Dict[str, Any]
    rules_engine_result: Optional[Dict[str, Any]]
    xgboost_prediction: float
    xgboost_confidence: float
    detective_findings: Optional[Dict[str, Any]]
    analyst_findings: Optional[Dict[str, Any]]
    verifier_findings: Optional[Dict[str, Any]]
    current_step: str
    should_continue: bool
    final_decision: FraudDecision
    final_confidence: float
    recommended_action: ActionRecommendation
    explanation: str
    investigation_start_time: str
    investigation_end_time: Optional[str]
    total_processing_time_ms: float
    agents_used: List[str]


def create_initial_state(
    transaction: Dict[str, Any],
    xgboost_prediction: float,
    rules_engine_result: Optional[Dict[str, Any]] = None,
) -> FraudInvestigationState:
    return FraudInvestigationState(
        transaction_id=transaction.get("transaction_id", "UNKNOWN"),
        user_id=transaction.get("user_id", "UNKNOWN"),
        amount=transaction.get("amount", 0.0),
        merchant=transaction.get("merchant_name", "UNKNOWN"),
        timestamp=transaction.get("transaction_time", datetime.now().isoformat()),
        transaction_features=transaction,
        rules_engine_result=rules_engine_result,
        xgboost_prediction=xgboost_prediction,
        xgboost_confidence=abs(xgboost_prediction - 0.5) * 2,
        detective_findings=None,
        analyst_findings=None,
        verifier_findings=None,
        current_step="detective",
        should_continue=True,
        final_decision="UNCERTAIN",
        final_confidence=0.0,
        recommended_action="REVIEW",
        explanation="",
        investigation_start_time=datetime.now().isoformat(),
        investigation_end_time=None,
        total_processing_time_ms=0.0,
        agents_used=[],
    )


def update_detective_findings(
    state: FraudInvestigationState, findings: DetectiveFindings
) -> FraudInvestigationState:
    state["detective_findings"] = asdict(findings)
    state["agents_used"].append("Detective")
    state["current_step"] = "analyst"
    if findings.confidence > 0.9:
        state["should_continue"] = False
        state["current_step"] = "complete"
    elif len(findings.red_flags) == 0 and findings.risk_level == "LOW":
        state["should_continue"] = False
        state["current_step"] = "complete"
    return state


def update_analyst_findings(
    state: FraudInvestigationState, findings: AnalystFindings
) -> FraudInvestigationState:
    state["analyst_findings"] = asdict(findings)
    state["agents_used"].append("Analyst")
    state["current_step"] = "verifier"
    return state


def update_verifier_findings(
    state: FraudInvestigationState, findings: VerifierFindings
) -> FraudInvestigationState:
    state["verifier_findings"] = asdict(findings)
    state["agents_used"].append("Verifier")
    state["current_step"] = "complete"
    state["should_continue"] = False
    state["final_decision"] = findings.final_decision
    state["final_confidence"] = findings.confidence
    state["recommended_action"] = findings.recommended_action
    state["explanation"] = "".join(findings.reasoning_chain)
    state["investigation_end_time"] = datetime.now().isoformat()
    start = datetime.fromisoformat(state["investigation_start_time"])
    end = datetime.fromisoformat(state["investigation_end_time"])
    state["total_processing_time_ms"] = (end - start).total_seconds() * 1000
    return state


def calculate_combined_confidence(state: FraudInvestigationState) -> float:
    weights = []
    scores = []
    weights.append(0.3)
    scores.append(state["xgboost_confidence"])
    if state["detective_findings"]:
        weights.append(0.2)
        scores.append(state["detective_findings"]["confidence"])
    if state["analyst_findings"]:
        weights.append(0.3)
        scores.append(state["analyst_findings"]["confidence"])
    if state["verifier_findings"]:
        weights.append(0.2)
        scores.append(state["verifier_findings"]["confidence"])
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    combined = sum((s * w for s, w in zip(scores, normalized_weights)))
    return combined


def get_investigation_summary(state: FraudInvestigationState) -> Dict[str, Any]:
    return {
        "transaction_id": state["transaction_id"],
        "amount": state["amount"],
        "merchant": state["merchant"],
        "xgboost_score": state["xgboost_prediction"],
        "agents_used": state["agents_used"],
        "final_decision": state["final_decision"],
        "confidence": state["final_confidence"],
        "action": state["recommended_action"],
        "explanation": state["explanation"],
        "processing_time_ms": state["total_processing_time_ms"],
        "detective_risk_level": (
            state["detective_findings"]["risk_level"]
            if state["detective_findings"]
            else None
        ),
        "analyst_fraud_type": (
            state["analyst_findings"]["fraud_type"]
            if state["analyst_findings"]
            else None
        ),
    }
