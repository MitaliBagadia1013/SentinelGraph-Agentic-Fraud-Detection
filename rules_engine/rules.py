import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from .rule_types import RuleDecision, RuleSeverity, RuleResult, RulesEngineOutput
from . import rule_config as config


class RulesEngine:

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.execution_stats = {
            "total_evaluated": 0,
            "declined": 0,
            "approved": 0,
            "passed_to_ml": 0,
        }

    def evaluate(self, transaction: Dict[str, Any]) -> RulesEngineOutput:
        start_time = time.perf_counter()
        triggered_rules: List[RuleResult] = []
        for rule_name in config.RULE_EXECUTION_ORDER:
            if not config.ENABLED_RULES.get(rule_name, False):
                continue
            rule_method = getattr(self, rule_name, None)
            if rule_method:
                rule_result = rule_method(transaction)
                if rule_result:
                    triggered_rules.append(rule_result)
                    if rule_result.severity == RuleSeverity.CRITICAL:
                        total_time = (time.perf_counter() - start_time) * 1000
                        return self._create_output(
                            RuleDecision.DECLINE,
                            triggered_rules,
                            total_time,
                            rule_result.reason,
                        )
                    if rule_result.decision == RuleDecision.APPROVE:
                        total_time = (time.perf_counter() - start_time) * 1000
                        return self._create_output(
                            RuleDecision.APPROVE,
                            triggered_rules,
                            total_time,
                            rule_result.reason,
                        )
        high_severity_count = sum(
            (
                1
                for r in triggered_rules
                if r.severity in [RuleSeverity.HIGH, RuleSeverity.CRITICAL]
            )
        )
        if high_severity_count >= 2:
            total_time = (time.perf_counter() - start_time) * 1000
            return self._create_output(
                RuleDecision.DECLINE,
                triggered_rules,
                total_time,
                f"Multiple high-severity violations ({high_severity_count})",
            )
        total_time = (time.perf_counter() - start_time) * 1000
        return self._create_output(
            RuleDecision.PASS_TO_ML,
            triggered_rules,
            total_time,
            "No definitive rule match - requires ML analysis",
        )

    def check_stolen_card(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        card_number = txn.get("card_number_hash")
        if card_number in config.STOLEN_CARDS_SAMPLE:
            return RuleResult(
                rule_name="check_stolen_card",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.CRITICAL,
                reason=f"Card found in stolen card database",
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"card_hash": card_number[:8] + "..."},
            )
        return None

    def check_sanctioned_country(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        country = (
            txn.get("billing_country")
            or txn.get("ip_country")
            or txn.get("country_code")
        )
        if country in config.SANCTIONED_COUNTRIES:
            return RuleResult(
                rule_name="check_sanctioned_country",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.CRITICAL,
                reason=f"Transaction from sanctioned country: {country}",
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"country": country},
            )
        return None

    def check_fraud_device(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        device_id = txn.get("device_id")
        if device_id in config.FRAUD_DEVICE_IDS_SAMPLE:
            return RuleResult(
                rule_name="check_fraud_device",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.CRITICAL,
                reason=f"Device ID in fraud blacklist",
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"device_id": device_id[:10] + "..."},
            )
        return None

    def check_tor(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        is_tor = txn.get("is_tor") or txn.get("tor_exit_node")
        if is_tor in (1, True, "1", "true", "True"):
            return RuleResult(
                rule_name="check_tor",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.CRITICAL,
                reason="Transaction routed through a TOR exit node",
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"is_tor": True},
            )
        return None

    def check_amount_exceeds_limit(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        amount = txn.get("amount", 0)
        card_limit = txn.get("card_limit", 10000)
        if amount > card_limit:
            return RuleResult(
                rule_name="check_amount_exceeds_limit",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.CRITICAL,
                reason=f"Amount ${amount:.2f} exceeds card limit ${card_limit:.2f}",
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"amount": amount, "limit": card_limit},
            )
        return None

    def check_velocity_limits(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        txn_per_hour = txn.get("transactions_last_hour", 0)
        txn_per_day = txn.get("transactions_last_24h", 0)
        failed_attempts = txn.get("failed_attempts_last_hour", 0)
        if txn_per_hour > config.VELOCITY_LIMITS["max_transactions_per_minute"]:
            return RuleResult(
                rule_name="check_velocity_limits",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.HIGH,
                reason=f"Velocity attack: {txn_per_hour} transactions/hour (limit: {config.VELOCITY_LIMITS['max_transactions_per_hour']})",
                confidence=0.95,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"txn_per_hour": txn_per_hour},
            )
        if failed_attempts > config.VELOCITY_LIMITS["max_failed_attempts_per_hour"]:
            return RuleResult(
                rule_name="check_velocity_limits",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.HIGH,
                reason=f"Too many failed attempts: {failed_attempts}",
                confidence=0.92,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"failed_attempts": failed_attempts},
            )
        return None

    def check_impossible_travel(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        current_lat = txn.get("transaction_lat")
        current_lon = txn.get("transaction_lon")
        last_lat = txn.get("last_transaction_lat")
        last_lon = txn.get("last_transaction_lon")
        if None in [current_lat, current_lon, last_lat, last_lon]:
            return None
        distance_km = self._haversine_distance(
            last_lat, last_lon, current_lat, current_lon
        )
        current_time = datetime.fromisoformat(
            txn.get("transaction_time", datetime.now().isoformat())
        )
        last_time = datetime.fromisoformat(
            txn.get("last_transaction_time", current_time.isoformat())
        )
        time_diff_hours = (current_time - last_time).total_seconds() / 3600
        if time_diff_hours > 0:
            travel_speed = distance_km / time_diff_hours
            if travel_speed > config.LOCATION_RULES["impossible_travel_speed_kmh"]:
                return RuleResult(
                    rule_name="check_impossible_travel",
                    decision=RuleDecision.DECLINE,
                    severity=RuleSeverity.HIGH,
                    reason=f"Impossible travel: {distance_km:.0f}km in {time_diff_hours:.1f}h ({travel_speed:.0f}km/h)",
                    confidence=0.98,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={
                        "distance_km": distance_km,
                        "time_hours": time_diff_hours,
                        "speed_kmh": travel_speed,
                    },
                )
        return None

    def check_new_account_risk(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        account_age_hours = txn.get("account_age_hours", 999999)
        amount = txn.get("amount", 0)
        is_first_transaction = txn.get("is_first_transaction", False)
        if (
            account_age_hours
            < config.BEHAVIORAL_RULES["min_account_age_hours_for_high_value"]
        ):
            if amount > config.AMOUNT_LIMITS["new_account_max_amount"]:
                return RuleResult(
                    rule_name="check_new_account_risk",
                    decision=RuleDecision.DECLINE,
                    severity=RuleSeverity.HIGH,
                    reason=f"New account ({account_age_hours:.1f}h old) with high-value transaction (${amount:.2f})",
                    confidence=0.88,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={"account_age_hours": account_age_hours, "amount": amount},
                )
        if (
            is_first_transaction
            and amount > config.AMOUNT_LIMITS["first_transaction_max_amount"]
        ):
            return RuleResult(
                rule_name="check_new_account_risk",
                decision=RuleDecision.DECLINE,
                severity=RuleSeverity.MEDIUM,
                reason=f"First transaction with unusually high amount: ${amount:.2f}",
                confidence=0.82,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"amount": amount},
            )
        return None

    def check_behavioral_anomalies(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        hour = txn.get("transaction_hour", 12)
        if (
            config.BEHAVIORAL_RULES["unusual_hour_start"]
            <= hour
            <= config.BEHAVIORAL_RULES["unusual_hour_end"]
        ):
            amount = txn.get("amount", 0)
            if amount > config.BEHAVIORAL_RULES["high_value_threshold"]:
                return RuleResult(
                    rule_name="check_behavioral_anomalies",
                    decision=RuleDecision.PASS_TO_ML,
                    severity=RuleSeverity.MEDIUM,
                    reason=f"High-value transaction at unusual hour: {hour}:00",
                    confidence=0.65,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={"hour": hour, "amount": amount},
                )
        return None

    def check_email_domain(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        email = txn.get("email", "") or txn.get("email_domain", "")
        domain = email.split("@")[1].lower() if "@" in email else email.lower()
        if domain:
            if domain in config.FRAUD_EMAIL_DOMAINS:
                return RuleResult(
                    rule_name="check_email_domain",
                    decision=RuleDecision.DECLINE,
                    severity=RuleSeverity.CRITICAL,
                    reason=f"Disposable email domain detected: {domain}",
                    confidence=0.9,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={"domain": domain},
                )
        return None

    def check_card_testing_pattern(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        amount = txn.get("amount", 0)
        txn_per_minute = txn.get("transactions_last_minute", 0)
        if amount in config.AMOUNT_LIMITS["suspicious_round_amounts"]:
            if txn_per_minute >= 3:
                return RuleResult(
                    rule_name="check_card_testing_pattern",
                    decision=RuleDecision.DECLINE,
                    severity=RuleSeverity.HIGH,
                    reason=f"Card testing pattern: ${amount} amount with {txn_per_minute} txn/min",
                    confidence=0.93,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={"amount": amount, "txn_per_minute": txn_per_minute},
                )
        return None

    def check_vip_user(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        user_id = txn.get("user_id")
        amount = txn.get("amount", 0)
        if user_id in config.VIP_USERS:
            if amount <= config.INSTANT_APPROVE_RULES["vip_user_max_amount"]:
                return RuleResult(
                    rule_name="check_vip_user",
                    decision=RuleDecision.APPROVE,
                    severity=None,
                    reason=f"VIP user with normal transaction amount",
                    confidence=0.99,
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    metadata={"user_id": user_id, "amount": amount},
                )
        return None

    def check_micro_transaction(self, txn: Dict[str, Any]) -> Optional[RuleResult]:
        start = time.perf_counter()
        amount = txn.get("amount", 0)
        if amount < config.AMOUNT_LIMITS["micro_transaction_threshold"]:
            return RuleResult(
                rule_name="check_micro_transaction",
                decision=RuleDecision.APPROVE,
                severity=None,
                reason=f"Micro transaction (${amount:.2f}) - low risk",
                confidence=0.95,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                metadata={"amount": amount},
            )
        return None

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r

    def _create_output(
        self,
        decision: RuleDecision,
        triggered_rules: List[RuleResult],
        total_time_ms: float,
        primary_reason: str,
    ) -> RulesEngineOutput:
        self.execution_stats["total_evaluated"] += 1
        if decision == RuleDecision.DECLINE:
            self.execution_stats["declined"] += 1
        elif decision == RuleDecision.APPROVE:
            self.execution_stats["approved"] += 1
        else:
            self.execution_stats["passed_to_ml"] += 1
        return RulesEngineOutput(
            decision=decision,
            triggered_rules=triggered_rules,
            total_execution_time_ms=total_time_ms,
            should_continue_to_ml=decision == RuleDecision.PASS_TO_ML,
            primary_reason=primary_reason,
        )

    def get_stats(self) -> Dict[str, Any]:
        total = self.execution_stats["total_evaluated"]
        if total == 0:
            return self.execution_stats
        return {
            **self.execution_stats,
            "decline_rate": self.execution_stats["declined"] / total,
            "approve_rate": self.execution_stats["approved"] / total,
            "ml_passthrough_rate": self.execution_stats["passed_to_ml"] / total,
        }
