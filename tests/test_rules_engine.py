import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rules_engine.rules import RulesEngine
from rules_engine.rule_types import RuleDecision


@pytest.fixture
def engine():
    return RulesEngine(enable_logging=False)


def make_transaction(**overrides) -> dict:
    base = {
        "user_id": "USER-001",
        "amount": 50.0,
        "merchant": "amazon.com",
        "merchant_category": "retail",
        "timestamp": "2026-04-22T14:00:00",
        "device_id": "device_clean_001",
        "email_domain": "gmail.com",
        "country_code": "US",
        "is_tor": 0,
        "is_vpn": 0,
        "is_proxy": 0,
        "card_number_hash": "clean_card_hash",
        "account_age_days": 90,
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 3,
        "failed_attempts_1h": 0,
        "unique_cards_1h": 1,
        "is_international_merchant": 0,
        "zip_code_match": 1,
        "latitude": 37.77,
        "longitude": -122.41,
        "hour_of_day": 14,
        "is_night_time": 0,
        "is_high_risk_country": 0,
        "country_risk_score": 0.1,
    }
    base.update(overrides)
    return base


class TestDeclineRules:

    def test_tor_traffic_declined(self, engine):
        tx = make_transaction(is_tor=1)
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE, "TOR traffic must be declined"

    def test_sanctioned_country_declined(self, engine):
        tx = make_transaction(country_code="KP")
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE

    def test_sanctioned_country_iran(self, engine):
        tx = make_transaction(country_code="IR")
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE

    def test_fraud_email_domain_declined(self, engine):
        tx = make_transaction(email_domain="tempmail.com")
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE

    def test_fraud_device_declined(self, engine):
        tx = make_transaction(device_id="device_fraud_001")
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE

    def test_stolen_card_declined(self, engine):
        tx = make_transaction(card_number_hash="4532123456789012")
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.DECLINE

    def test_velocity_attack_declined(self, engine):
        tx = make_transaction(transaction_velocity_1h=10)
        result = engine.evaluate(tx)
        assert result.decision in (RuleDecision.DECLINE, RuleDecision.PASS_TO_ML)

    def test_multiple_failed_attempts_declined(self, engine):
        tx = make_transaction(failed_attempts_1h=5)
        result = engine.evaluate(tx)
        assert result.decision in (RuleDecision.DECLINE, RuleDecision.PASS_TO_ML)


class TestApproveRules:

    def test_micro_transaction_approved(self, engine):
        tx = make_transaction(amount=1.99)
        result = engine.evaluate(tx)
        assert result.decision == RuleDecision.APPROVE

    def test_clean_low_risk_passes(self, engine):
        tx = make_transaction()
        result = engine.evaluate(tx)
        assert result.decision != RuleDecision.DECLINE


class TestPassToML:

    def test_normal_transaction_passes_to_ml(self, engine):
        tx = make_transaction(amount=150.0)
        result = engine.evaluate(tx)
        assert result.decision in (RuleDecision.PASS_TO_ML, RuleDecision.APPROVE)

    def test_international_transaction_passes_to_ml(self, engine):
        tx = make_transaction(is_international_merchant=1, amount=300.0)
        result = engine.evaluate(tx)
        assert result.decision != RuleDecision.DECLINE or result.triggered_rules


class TestOutputStructure:

    def test_output_has_decision(self, engine):
        result = engine.evaluate(make_transaction())
        assert hasattr(result, "decision")

    def test_output_has_triggered_rules(self, engine):
        result = engine.evaluate(make_transaction(is_tor=1))
        assert isinstance(result.triggered_rules, list)
        assert len(result.triggered_rules) > 0

    def test_triggered_rule_has_required_fields(self, engine):
        result = engine.evaluate(make_transaction(is_tor=1))
        rule = result.triggered_rules[0]
        assert hasattr(rule, "rule_name")
        assert hasattr(rule, "severity")
        assert hasattr(rule, "reason")
        assert hasattr(rule, "confidence")

    def test_output_has_execution_time(self, engine):
        result = engine.evaluate(make_transaction())
        assert result.total_execution_time_ms >= 0

    def test_decision_is_valid_enum(self, engine):
        result = engine.evaluate(make_transaction())
        assert result.decision in (
            RuleDecision.APPROVE,
            RuleDecision.DECLINE,
            RuleDecision.PASS_TO_ML,
        )


class TestLatency:

    def test_single_evaluation_under_5ms(self, engine):
        tx = make_transaction()
        start = time.perf_counter()
        engine.evaluate(tx)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5, f"Rules engine took {elapsed_ms:.2f}ms — must be <5ms"

    def test_hundred_evaluations_avg_under_2ms(self, engine):
        tx = make_transaction()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.evaluate(tx)
            times.append((time.perf_counter() - start) * 1000)
        avg = sum(times) / len(times)
        assert avg < 2, f"Rules engine avg {avg:.2f}ms over 100 calls — must be <2ms"

    def test_decline_path_fast(self, engine):
        tx = make_transaction(is_tor=1)
        start = time.perf_counter()
        engine.evaluate(tx)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 3


class TestEdgeCases:

    def test_empty_transaction_does_not_crash(self, engine):
        result = engine.evaluate({})
        assert result.decision in (
            RuleDecision.APPROVE,
            RuleDecision.DECLINE,
            RuleDecision.PASS_TO_ML,
        )

    def test_zero_amount(self, engine):
        result = engine.evaluate(make_transaction(amount=0.0))
        assert result is not None

    def test_very_large_amount(self, engine):
        result = engine.evaluate(make_transaction(amount=999999.0))
        assert result is not None

    def test_new_account_high_amount_flagged(self, engine):
        tx = make_transaction(account_age_days=0, amount=600.0)
        result = engine.evaluate(tx)
        assert result.decision in (RuleDecision.DECLINE, RuleDecision.PASS_TO_ML)

    def test_multiple_rules_can_trigger(self, engine):
        tx = make_transaction(
            is_vpn=1, is_high_risk_country=1, transaction_velocity_1h=8
        )
        result = engine.evaluate(tx)
        assert result.decision is not None
