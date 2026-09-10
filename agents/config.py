from typing import Dict, Any

AGENT_MODELS = {"detective": "gpt-4o-mini", "analyst": "gpt-4o", "verifier": "gpt-4o"}
LLM_TEMPERATURE = {"detective": 0.3, "analyst": 0.5, "verifier": 0.2}
LLM_MAX_TOKENS = {"detective": 500, "analyst": 1000, "verifier": 800}
AGENT_TIMEOUTS = {"detective": 5.0, "analyst": 10.0, "verifier": 8.0}
EARLY_EXIT_CONFIG = {
    "detective_high_confidence_threshold": 0.9,
    "detective_no_red_flags_skip": True,
    "always_run_verifier": False,
}
XGBOOST_ROUTING_THRESHOLDS = {
    "auto_approve_threshold": 0.2,
    "auto_decline_threshold": 0.8,
    "agent_analysis_required": True,
}
AMOUNT_BASED_ROUTING = {
    "enable": True,
    "micro_transaction_threshold": 5.0,
    "high_value_threshold": 5000.0,
    "medium_value_min": 5.0,
}
DETECTIVE_THRESHOLDS = {
    "low_risk_max": 0.3,
    "medium_risk_max": 0.6,
    "high_risk_max": 0.85,
}
ANALYST_THRESHOLDS = {"anomaly_score_high": 0.75, "pattern_match_confidence": 0.7}
VERIFIER_THRESHOLDS = {
    "auto_approve_confidence": 0.85,
    "auto_decline_confidence": 0.85,
    "review_confidence_min": 0.5,
    "escalate_to_hitl_max": 0.5,
}
CONFIDENCE_WEIGHTS = {"xgboost": 0.3, "detective": 0.2, "analyst": 0.3, "verifier": 0.2}
ACTION_MAPPING = {
    "FRAUD": {
        "high_confidence": "DECLINE",
        "medium_confidence": "REVIEW",
        "low_confidence": "ESCALATE_TO_HITL",
    },
    "LEGIT": {
        "high_confidence": "APPROVE",
        "medium_confidence": "REVIEW",
        "low_confidence": "ESCALATE_TO_HITL",
    },
    "UNCERTAIN": {"any_confidence": "ESCALATE_TO_HITL"},
}
COST_ESTIMATES = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}
ESTIMATED_TOKENS_PER_AGENT = {
    "detective": {"input": 800, "output": 200},
    "analyst": {"input": 2000, "output": 400},
    "verifier": {"input": 1500, "output": 300},
}
COST_LIMITS = {
    "max_cost_per_transaction": 0.5,
    "daily_budget": 10000.0,
    "enable_cost_tracking": True,
}
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay_seconds": 1.0,
    "exponential_backoff": True,
}
FALLBACK_CONFIG = {
    "on_detective_failure": "PASS_TO_ANALYST",
    "on_analyst_failure": "PASS_TO_VERIFIER",
    "on_verifier_failure": "ESCALATE_TO_HITL",
    "on_all_agents_failure": "ESCALATE_TO_HITL",
}
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_all_prompts": False,
    "log_all_responses": True,
    "log_execution_times": True,
    "log_costs": True,
    "log_to_file": True,
    "log_to_console": True,
}
PERFORMANCE_TARGETS = {
    "detective_target_ms": 300,
    "analyst_target_ms": 500,
    "verifier_target_ms": 400,
    "total_target_ms": 1200,
}
ALERT_CONFIG = {
    "enable_alerts": True,
    "alert_on_timeout": True,
    "alert_on_high_cost": True,
    "alert_on_parse_error": True,
    "alert_on_confidence_drop": True,
    "alert_threshold_drop": 0.3,
}
FRAUD_TYPES = [
    "velocity_attack",
    "card_testing",
    "account_takeover",
    "impossible_travel",
    "amount_spike",
    "merchant_anomaly",
    "device_switching",
    "time_anomaly",
    "first_party_fraud",
    "promotion_abuse",
]
FRAUD_TYPE_SEVERITY = {
    "velocity_attack": "HIGH",
    "card_testing": "HIGH",
    "account_takeover": "CRITICAL",
    "impossible_travel": "HIGH",
    "amount_spike": "MEDIUM",
    "merchant_anomaly": "MEDIUM",
    "device_switching": "MEDIUM",
    "time_anomaly": "LOW",
    "first_party_fraud": "MEDIUM",
    "promotion_abuse": "LOW",
}
DETECTIVE_KEY_FEATURES = [
    "transactions_last_hour",
    "transactions_last_24h",
    "distance_from_last_km",
    "time_since_last_transaction_hours",
    "is_new_device",
    "is_new_location",
    "transaction_hour",
    "amount_vs_avg_ratio",
    "velocity_score",
    "failed_attempts_last_hour",
]
ANALYST_KEY_FEATURES = [
    "amount",
    "amount_vs_avg_ratio",
    "transactions_last_hour",
    "transactions_last_24h",
    "distance_from_last_km",
    "velocity_score",
    "device_risk_score",
    "location_risk_score",
    "merchant_risk_score",
    "user_fraud_rate",
    "account_age_days",
    "is_new_device",
    "is_new_location",
    "transaction_hour",
    "is_weekend",
    "card_type",
    "merchant_category",
    "billing_country",
    "shipping_country",
    "ip_country",
]
VALIDATION_RULES = {
    "detective": {
        "required_fields": ["risk_level", "red_flags", "confidence", "reasoning"],
        "risk_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "confidence_range": [0.0, 1.0],
    },
    "analyst": {
        "required_fields": ["fraud_type", "anomaly_score", "confidence", "reasoning"],
        "valid_fraud_types": FRAUD_TYPES + [None, "unknown"],
        "confidence_range": [0.0, 1.0],
    },
    "verifier": {
        "required_fields": [
            "final_decision",
            "confidence",
            "reasoning_chain",
            "recommended_action",
        ],
        "valid_decisions": ["FRAUD", "LEGIT", "UNCERTAIN"],
        "valid_actions": ["APPROVE", "DECLINE", "REVIEW", "ESCALATE_TO_HITL"],
        "confidence_range": [0.0, 1.0],
    },
}


def get_agent_model(agent_name: str) -> str:
    return AGENT_MODELS.get(agent_name, "gpt-4o-mini")


def get_agent_timeout(agent_name: str) -> float:
    return AGENT_TIMEOUTS.get(agent_name, 10.0)


def should_skip_agents(xgboost_score: float, amount: float) -> tuple[bool, str]:
    if AMOUNT_BASED_ROUTING["enable"]:
        if amount < AMOUNT_BASED_ROUTING["micro_transaction_threshold"]:
            return (True, f"Micro transaction (${amount:.2f}) - auto-approve")
    if xgboost_score < XGBOOST_ROUTING_THRESHOLDS["auto_approve_threshold"]:
        return (True, f"Low fraud score ({xgboost_score:.2%}) - auto-approve")
    if xgboost_score > XGBOOST_ROUTING_THRESHOLDS["auto_decline_threshold"]:
        return (True, f"High fraud score ({xgboost_score:.2%}) - auto-decline")
    return (False, "Requires agent analysis")


def should_run_analyst(detective_confidence: float, detective_red_flags: int) -> bool:
    if detective_confidence > EARLY_EXIT_CONFIG["detective_high_confidence_threshold"]:
        return False
    if detective_red_flags == 0 and EARLY_EXIT_CONFIG["detective_no_red_flags_skip"]:
        return False
    return True


def should_run_verifier(analyst_confidence: float) -> bool:
    if EARLY_EXIT_CONFIG["always_run_verifier"]:
        return True
    return True


def map_decision_to_action(decision: str, confidence: float) -> str:
    if decision == "UNCERTAIN":
        return "ESCALATE_TO_HITL"
    mapping = ACTION_MAPPING.get(decision, {})
    if confidence > VERIFIER_THRESHOLDS["auto_approve_confidence"]:
        return mapping.get("high_confidence", "REVIEW")
    elif confidence >= VERIFIER_THRESHOLDS["review_confidence_min"]:
        return mapping.get("medium_confidence", "REVIEW")
    else:
        return mapping.get("low_confidence", "ESCALATE_TO_HITL")


def estimate_transaction_cost(agents_used: list[str]) -> float:
    total_cost = 0.0
    for agent in agents_used:
        model = AGENT_MODELS.get(agent, "gpt-4o-mini")
        tokens = ESTIMATED_TOKENS_PER_AGENT.get(agent, {"input": 1000, "output": 200})
        pricing = COST_ESTIMATES.get(model, {"input": 0.005, "output": 0.015})
        input_cost = tokens["input"] / 1000 * pricing["input"]
        output_cost = tokens["output"] / 1000 * pricing["output"]
        total_cost += input_cost + output_cost
    return total_cost


def get_performance_summary() -> Dict[str, Any]:
    return {
        "targets": PERFORMANCE_TARGETS,
        "total_target_ms": PERFORMANCE_TARGETS["total_target_ms"],
        "agent_models": AGENT_MODELS,
        "cost_per_transaction": estimate_transaction_cost(
            ["detective", "analyst", "verifier"]
        ),
    }


def get_config_summary() -> Dict[str, Any]:
    return {
        "agent_models": AGENT_MODELS,
        "routing_thresholds": XGBOOST_ROUTING_THRESHOLDS,
        "confidence_thresholds": {
            "detective": DETECTIVE_THRESHOLDS,
            "analyst": ANALYST_THRESHOLDS,
            "verifier": VERIFIER_THRESHOLDS,
        },
        "early_exit": EARLY_EXIT_CONFIG,
        "cost_estimate_per_txn": estimate_transaction_cost(
            ["detective", "analyst", "verifier"]
        ),
        "performance_targets": PERFORMANCE_TARGETS,
    }
