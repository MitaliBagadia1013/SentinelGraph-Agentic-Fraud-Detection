from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class RuleDecision(Enum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    PASS_TO_ML = "PASS_TO_ML"


class RuleSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class RuleResult:
    rule_name: str
    decision: RuleDecision
    severity: Optional[RuleSeverity]
    reason: str
    confidence: float
    execution_time_ms: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "decision": self.decision.value,
            "severity": self.severity.value if self.severity else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class RulesEngineOutput:
    decision: RuleDecision
    triggered_rules: List[RuleResult]
    total_execution_time_ms: float
    should_continue_to_ml: bool
    primary_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "total_execution_time_ms": self.total_execution_time_ms,
            "should_continue_to_ml": self.should_continue_to_ml,
            "primary_reason": self.primary_reason,
        }
