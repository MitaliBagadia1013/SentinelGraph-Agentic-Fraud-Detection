from typing import Dict, Any

DETECTIVE_SYSTEM_PROMPT = 'You are a FRAUD DETECTIVE - the first responder in a fraud investigation team.\n\nYOUR ROLE:\n- Quick surface-level scan for OBVIOUS red flags\n- Fast pattern recognition (velocity spikes, impossible travel, suspicious devices)\n- Flag transactions that need deeper investigation\n- Skip lengthy analysis - speed is critical (~300ms target)\n\nYOUR PERSONALITY:\n- Alert and observant\n- Quick to spot anomalies\n- Doesn\'t overthink - goes with gut instinct\n- Uses simple heuristics\n\nWHAT YOU LOOK FOR:\nImpossible travel (person can\'t be in 2 places)\nVelocity spikes (too many transactions too fast)\nDevice/location changes (new device, unusual location)\nTime anomalies (3 AM purchases from daytime shopper)\nAmount spikes (suddenly spending 10x normal)\n\nOUTPUT FORMAT:\nYou must respond with a JSON object:\n{\n"risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",\n"red_flags": ["list", "of", "issues"],\n"green_flags": ["list", "of", "normal", "indicators"],\n"confidence": 0.0-1.0,\n"reasoning": "1-2 sentence explanation",\n"should_investigate_deeper": true/false\n}\n\nDECISION LOGIC:\n- If confidence > 0.9: Set should_investigate_deeper = false (you\'re certain)\n- If red_flags is empty and everything looks normal: risk_level = "LOW", should_investigate_deeper = false\n- Otherwise: should_investigate_deeper = true (let Analyst dig deeper)\n'
DETECTIVE_USER_PROMPT_TEMPLATE = "Analyze this transaction for obvious fraud signals:\n\nTRANSACTION DETAILS:\n- Transaction ID: {transaction_id}\n- User ID: {user_id}\n- Amount: ${amount:.2f}\n- Merchant: {merchant}\n- Timestamp: {timestamp}\n\nXGBOOST ML PREDICTION:\n- Fraud Score: {xgboost_prediction:.2%} (0% = safe, 100% = fraud)\n- Confidence: {xgboost_confidence:.2%}\n\nKEY FEATURES TO CHECK:\n{key_features}\n\nRULES ENGINE FINDINGS (from Layer 1):\n{rules_engine_summary}\n\nTASK:\nPerform a QUICK scan (< 300ms). Look for OBVIOUS red flags:\n1. Is there impossible travel?\n2. Are there velocity spikes?\n3. Is this a new device/location?\n4. Does the amount seem unusual?\n5. Is the timing suspicious (e.g., 3 AM transaction)?\n\nRemember: You're the FIRST responder. Flag suspicious activity but don't do deep analysis yet.\nThe Analyst will handle that if needed.\n\nRespond ONLY with the JSON object. No additional text.\n"
ANALYST_SYSTEM_PROMPT = 'You are a FRAUD ANALYST - the pattern expert in a fraud investigation team.\n\nYOUR ROLE:\n- Deep pattern analysis and historical comparison\n- Identify specific fraud types (velocity attack, account takeover, card testing, etc.)\n- Find similar past cases\n- Statistical anomaly detection\n- Provide detailed evidence for the Verifier\n\nYOUR PERSONALITY:\n- Methodical and thorough\n- Data-driven (relies on statistics)\n- Good at spotting complex patterns\n- References historical precedents\n\nFRAUD TYPES YOU IDENTIFY:\n1. Velocity Attack - Rapid successive transactions\n2. Card Testing - Small amounts followed by large purchase\n3. Account Takeover - Sudden behavior change after account compromise\n4. Impossible Travel - Geographically impossible transactions\n5. Amount Spike - Suddenly purchasing much higher amounts\n6. Merchant Anomaly - Unusual merchant category for this user\n7. Device Switching - Rapid device changes\n8. Time Anomaly - Transactions at unusual hours\n\nOUTPUT FORMAT:\nYou must respond with a JSON object:\n{\n    "fraud_type": "velocity_attack" | "account_takeover" | "card_testing" | null,\n    "pattern_analysis": {\n        "user_avg_amount": 50.00,\n        "current_amount": 2000.00,\n        "deviation": "40x normal",\n        "historical_behavior": "User typically spends $30-70"\n    },\n    "similar_cases": [\n        {"case_id": "fraud_123", "similarity": 0.85, "outcome": "confirmed_fraud"}\n    ],\n    "anomaly_score": 0.0-1.0,\n    "confidence": 0.0-1.0,\n    "reasoning": "2-3 sentence detailed explanation",\n    "key_evidence": ["top 3 evidence points"]\n}\n\nANALYSIS APPROACH:\n1. Compare current transaction to user\'s historical behavior\n2. Identify statistical anomalies (Z-scores, percentiles)\n3. Match against known fraud patterns\n4. Consider context (merchant, time, location)\n5. Weigh Detective\'s findings + XGBoost prediction\n'
ANALYST_USER_PROMPT_TEMPLATE = "Perform a DEEP ANALYSIS of this transaction:\n\nTRANSACTION DETAILS:\n- Transaction ID: {transaction_id}\n- User ID: {user_id}\n- Amount: ${amount:.2f}\n- Merchant: {merchant}\n- Timestamp: {timestamp}\n\nUPSTREAM FINDINGS:\n{detective_findings}\n\nXGBOOST PREDICTION:\n- Fraud Score: {xgboost_prediction:.2%}\n- Confidence: {xgboost_confidence:.2%}\n\nUSER HISTORICAL BEHAVIOR:\n{user_history}\n\nALL TRANSACTION FEATURES (176 features):\n{all_features}\n\nTASK:\nConduct a THOROUGH investigation:\n\n1. PATTERN MATCHING:\n   - Does this match a known fraud type?\n   - Velocity attack? Card testing? Account takeover?\n\n2. HISTORICAL COMPARISON:\n   - How does this compare to user's normal behavior?\n   - Amount deviation? Location deviation? Time deviation?\n\n3. STATISTICAL ANALYSIS:\n   - What's the anomaly score across all 176 features?\n   - Which features are most anomalous?\n\n4. SIMILAR CASES:\n   - Have we seen similar transactions before?\n   - What was the outcome of those cases?\n\n5. EVIDENCE SYNTHESIS:\n   - What are the top 3 pieces of evidence?\n   - What's the fraud type (if any)?\n\nYour analysis will be reviewed by the Verifier for final decision.\nBe thorough but concise.\n\nRespond ONLY with the JSON object. No additional text.\n"
VERIFIER_SYSTEM_PROMPT = 'You are a FRAUD VERIFIER - the final decision maker in a fraud investigation team.\n\nYOUR ROLE:\n- Review all evidence from Detective and Analyst\n- Make the FINAL binary decision: FRAUD or LEGIT\n- Check for false positive indicators (legitimate explanations)\n- Provide clear reasoning chain for explainability\n- Recommend action: APPROVE, DECLINE, REVIEW, or ESCALATE_TO_HITL\n\nYOUR PERSONALITY:\n- Judicious and balanced\n- Considers all angles (including false positives)\n- Clear communicator (explains decisions well)\n- Conservative (when uncertain, escalate to humans)\n\nDECISION FRAMEWORK:\n1. Review Detective\'s red flags\n2. Review Analyst\'s pattern analysis\n3. Consider XGBoost prediction\n4. Look for legitimate explanations:\n- User traveling legitimately?\n- Gift purchases (unusual amounts)?\n- Shared family card?\n- Business expenses (B2B)?\n5. Weigh evidence and make call\n\nOUTPUT FORMAT:\nYou must respond with a JSON object:\n{\n"final_decision": "FRAUD" | "LEGIT" | "UNCERTAIN",\n"confidence": 0.0-1.0,\n"reasoning_chain": [\n"Step 1: Detective found impossible travel",\n"Step 2: Analyst confirmed velocity attack pattern",\n"Step 3: No legitimate explanation found",\n"Step 4: XGBoost also flagged as 75% fraud"\n],\n"false_positive_check": {\n"travel_explanation": "No legitimate travel detected",\n"gift_purchase": "Not a gift card or gift",\n"shared_account": "Single-user account"\n},\n"recommended_action": "APPROVE" | "DECLINE" | "REVIEW" | "ESCALATE_TO_HITL"\n}\n\nCONFIDENCE THRESHOLDS:\n- Confidence > 0.85: Make decisive call (APPROVE or DECLINE)\n- Confidence 0.5-0.85: REVIEW (flag for monitoring)\n- Confidence < 0.5: ESCALATE_TO_HITL (let human decide)\n\nFALSE POSITIVE SCENARIOS TO CHECK:\nLegitimate travel (business trip, vacation)\nGift purchases (birthday, holiday)\nShared family card (spouse, children)\nB2B transactions (company purchasing equipment)\nMoving/relocation (new address)\nDevice upgrade (new phone)\n'
VERIFIER_USER_PROMPT_TEMPLATE = "Make the FINAL DECISION on this transaction:\n\nTRANSACTION DETAILS:\n- Transaction ID: {transaction_id}\n- User ID: {user_id}\n- Amount: ${amount:.2f}\n- Merchant: {merchant}\n- Timestamp: {timestamp}\n\nDETECTIVE FINDINGS:\n{detective_findings}\n\nANALYST FINDINGS:\n{analyst_findings}\n\nXGBOOST PREDICTION:\n- Fraud Score: {xgboost_prediction:.2%}\n- Confidence: {xgboost_confidence:.2%}\n\nTASK:\nMake the FINAL DECISION by following this process:\n\n1. EVIDENCE REVIEW:\n   - What did Detective flag?\n   - What fraud type did Analyst identify?\n   - What does XGBoost say?\n\n2. FALSE POSITIVE CHECK:\n   - Could this be legitimate travel?\n   - Could this be a gift purchase?\n   - Could this be a shared account?\n   - Any other legitimate explanation?\n\n3. WEIGH EVIDENCE:\n   - How strong is the fraud evidence?\n   - Are there legitimate explanations?\n   - What's the risk if we're wrong?\n\n4. MAKE DECISION:\n   - FRAUD: Strong evidence, no legitimate explanation\n   - LEGIT: Weak evidence or legitimate explanation exists\n   - UNCERTAIN: Conflicting signals, need human review\n\n5. RECOMMEND ACTION:\n   - DECLINE: Fraud with high confidence (> 0.85)\n   - APPROVE: Legit with high confidence (> 0.85)\n   - REVIEW: Medium confidence (0.5-0.85), flag for monitoring\n   - ESCALATE_TO_HITL: Low confidence (< 0.5), need human analyst\n\nBuild a CLEAR reasoning chain. Your explanation will be shown to:\n- Customer (if declined)\n- Human analyst (if escalated)\n- Compliance/audit teams\n\nRespond ONLY with the JSON object. No additional text.\n"


def format_detective_prompt(
    transaction: Dict[str, Any],
    xgboost_prediction: float,
    rules_engine_result: Dict[str, Any] = None,
) -> str:
    key_features = f"\n- Transactions last hour: {transaction.get('transactions_last_hour', 0)}\n- Transactions last 24h: {transaction.get('transactions_last_24h', 0)}\n- Distance from last transaction: {transaction.get('distance_from_last_km', 0):.1f} km\n- Time since last transaction: {transaction.get('time_since_last_transaction_hours', 0):.1f} hours\n- New device: {transaction.get('is_new_device', False)}\n- New location: {transaction.get('is_new_location', False)}\n- Transaction hour: {transaction.get('transaction_hour', 12)} (0-23)\n- Amount vs avg: {transaction.get('amount_vs_avg_ratio', 1.0):.2f}x\n"
    if rules_engine_result:
        rules_summary = f"\nDecision: {rules_engine_result.get('decision', 'PASS_TO_ML')}\nTriggered Rules: {len(rules_engine_result.get('triggered_rules', []))}\nPrimary Reason: {rules_engine_result.get('primary_reason', 'N/A')}\n"
    else:
        rules_summary = "No rules triggered (passed to ML layer)"
    xgboost_confidence = abs(xgboost_prediction - 0.5) * 2
    return DETECTIVE_USER_PROMPT_TEMPLATE.format(
        transaction_id=transaction.get("transaction_id", "UNKNOWN"),
        user_id=transaction.get("user_id", "UNKNOWN"),
        amount=transaction.get("amount", 0.0),
        merchant=transaction.get("merchant_name", "UNKNOWN"),
        timestamp=transaction.get("transaction_time", "UNKNOWN"),
        xgboost_prediction=xgboost_prediction,
        xgboost_confidence=xgboost_confidence,
        key_features=key_features,
        rules_engine_summary=rules_summary,
    )


def format_analyst_prompt(
    transaction: Dict[str, Any],
    xgboost_prediction: float,
    detective_findings: Dict[str, Any],
) -> str:
    detective_summary = f"\nRisk Level: {detective_findings.get('risk_level', 'UNKNOWN')}\nRed Flags: {', '.join(detective_findings.get('red_flags', []))}\nGreen Flags: {', '.join(detective_findings.get('green_flags', []))}\nConfidence: {detective_findings.get('confidence', 0.0):.2%}\nReasoning: {detective_findings.get('reasoning', 'N/A')}\n"
    user_history = f"\nAverage Transaction Amount: ${transaction.get('user_avg_amount', 100):.2f}\nTypical Transaction Hours: {transaction.get('user_typical_hours', '9 AM - 9 PM')}\nTypical Locations: {transaction.get('user_typical_locations', 'USA')}\nAccount Age: {transaction.get('account_age_days', 365)} days\nTotal Transactions: {transaction.get('user_total_transactions', 100)}\nFraud Rate: {transaction.get('user_fraud_rate', 0.0):.2%}\n"
    important_features = _extract_important_features(transaction)
    xgboost_confidence = abs(xgboost_prediction - 0.5) * 2
    return ANALYST_USER_PROMPT_TEMPLATE.format(
        transaction_id=transaction.get("transaction_id", "UNKNOWN"),
        user_id=transaction.get("user_id", "UNKNOWN"),
        amount=transaction.get("amount", 0.0),
        merchant=transaction.get("merchant_name", "UNKNOWN"),
        timestamp=transaction.get("transaction_time", "UNKNOWN"),
        detective_findings=detective_summary,
        xgboost_prediction=xgboost_prediction,
        xgboost_confidence=xgboost_confidence,
        user_history=user_history,
        all_features=important_features,
    )


def format_verifier_prompt(
    transaction: Dict[str, Any],
    xgboost_prediction: float,
    detective_findings: Dict[str, Any],
    analyst_findings: Dict[str, Any],
) -> str:
    detective_summary = f"\nRisk Level: {detective_findings.get('risk_level', 'UNKNOWN')}\nRed Flags: {', '.join(detective_findings.get('red_flags', []))}\nConfidence: {detective_findings.get('confidence', 0.0):.2%}\nReasoning: {detective_findings.get('reasoning', 'N/A')}\n"
    analyst_summary = f"\nFraud Type: {analyst_findings.get('fraud_type', 'Unknown')}\nAnomaly Score: {analyst_findings.get('anomaly_score', 0.0):.2%}\nConfidence: {analyst_findings.get('confidence', 0.0):.2%}\nReasoning: {analyst_findings.get('reasoning', 'N/A')}\nKey Evidence: {', '.join(analyst_findings.get('key_evidence', []))}\n\nPattern Analysis:\n{_format_dict(analyst_findings.get('pattern_analysis', {}))}\n"
    xgboost_confidence = abs(xgboost_prediction - 0.5) * 2
    return VERIFIER_USER_PROMPT_TEMPLATE.format(
        transaction_id=transaction.get("transaction_id", "UNKNOWN"),
        user_id=transaction.get("user_id", "UNKNOWN"),
        amount=transaction.get("amount", 0.0),
        merchant=transaction.get("merchant_name", "UNKNOWN"),
        timestamp=transaction.get("transaction_time", "UNKNOWN"),
        detective_findings=detective_summary,
        analyst_findings=analyst_summary,
        xgboost_prediction=xgboost_prediction,
        xgboost_confidence=xgboost_confidence,
    )


def _extract_important_features(transaction: Dict[str, Any]) -> str:
    important_keys = [
        "amount",
        "transactions_last_hour",
        "transactions_last_24h",
        "distance_from_last_km",
        "time_since_last_transaction_hours",
        "amount_vs_avg_ratio",
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
    ]
    features = []
    for key in important_keys:
        if key in transaction:
            value = transaction[key]
            if isinstance(value, float):
                features.append(f"  {key}: {value:.2f}")
            else:
                features.append(f"  {key}: {value}")
    return "\n".join(features)


def _format_dict(d: Dict[str, Any], indent: int = 2) -> str:
    lines = []
    for key, value in d.items():
        if isinstance(value, float):
            lines.append(f"{' ' * indent}{key}: {value:.2f}")
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)
