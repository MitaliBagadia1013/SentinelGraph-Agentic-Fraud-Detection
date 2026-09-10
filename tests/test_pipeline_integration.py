import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_transaction(**overrides) -> dict:
    base = {
        "transaction_id": "TX-TEST-001",
        "user_id": "USER-001",
        "amount": 120.0,
        "merchant": "amazon.com",
        "timestamp": "2026-04-22T14:00:00",
        "device_id": "device_clean",
    }
    base.update(overrides)
    return base


def make_features(**overrides) -> dict:
    base = {
        "is_tor": 0,
        "is_vpn": 0,
        "is_proxy": 0,
        "country_code": "US",
        "email_domain": "gmail.com",
        "card_number_hash": "clean_hash",
        "device_id": "device_clean",
        "transaction_velocity_1h": 1,
        "failed_attempts_1h": 0,
        "account_age_days": 180,
        "amount": 120.0,
        "is_high_risk_country": 0,
    }
    base.update(overrides)
    return base


REQUIRED_RESULT_KEYS = {
    "transaction_id",
    "final_decision",
    "recommended_action",
    "xgboost_score",
    "rules_triggered",
    "layer_stopped_at",
    "latency_ms",
}


@pytest.fixture
def pipeline_no_deps():
    from main import FraudDetectionPipeline

    with patch("main.FraudDetectionPipeline._load_xgboost_model"):
        p = FraudDetectionPipeline(use_agents=False, use_graph=False)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.5, 0.5]]
    p._model = mock_model
    p._feature_cols = []
    return p


def make_pipeline_with_score(score: float):
    from main import FraudDetectionPipeline

    with patch("main.FraudDetectionPipeline._load_xgboost_model"):
        p = FraudDetectionPipeline(use_agents=False, use_graph=False)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[1 - score, score]]
    p._model = mock_model
    p._feature_cols = []
    return p


class TestResultShape:

    def test_result_has_all_required_keys(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(
            transaction=make_transaction(), features=make_features()
        )
        for key in REQUIRED_RESULT_KEYS:
            assert key in result, f"Missing key in result: '{key}'"

    def test_transaction_id_preserved(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(
            transaction=make_transaction(transaction_id="MY-TX-999"),
            features=make_features(),
        )
        assert result["transaction_id"] == "MY-TX-999"

    def test_xgboost_score_is_float(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(make_transaction(), make_features())
        assert isinstance(result["xgboost_score"], float)
        assert 0.0 <= result["xgboost_score"] <= 1.0

    def test_final_decision_is_valid(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(make_transaction(), make_features())
        assert result["final_decision"] in ("FRAUD", "LEGIT", "UNCERTAIN")

    def test_recommended_action_is_valid(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(make_transaction(), make_features())
        assert result["recommended_action"] in (
            "APPROVE",
            "DECLINE",
            "REVIEW",
            "ESCALATE_TO_HITL",
        )

    def test_latency_ms_dict_populated(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(make_transaction(), make_features())
        latency = result["latency_ms"]
        assert isinstance(latency, dict)
        assert "total_ms" in latency


class TestRulesHardStops:

    def test_tor_transaction_declined_before_xgboost(self):
        p = make_pipeline_with_score(0.0)
        tx = make_transaction()
        features = make_features(is_tor=1)
        result = p.evaluate(tx, features)
        assert result["final_decision"] == "FRAUD"
        assert result["layer_stopped_at"] == "rules_decline"
        p._model.predict_proba.assert_not_called()

    def test_micro_transaction_approved_before_xgboost(self):
        p = make_pipeline_with_score(0.99)
        tx = make_transaction(amount=1.5)
        features = make_features(amount=1.5)
        result = p.evaluate(tx, features)
        assert result["final_decision"] == "LEGIT"
        assert result["layer_stopped_at"] == "rules_approve"
        p._model.predict_proba.assert_not_called()


class TestXGBoostRouting:

    def test_high_score_is_fraud(self):
        p = make_pipeline_with_score(0.95)
        result = p.evaluate(make_transaction(), make_features())
        assert result["final_decision"] == "FRAUD"
        assert result["layer_stopped_at"] == "xgboost_high"

    def test_low_score_is_legit(self):
        p = make_pipeline_with_score(0.05)
        result = p.evaluate(make_transaction(), make_features())
        assert result["final_decision"] == "LEGIT"
        assert result["layer_stopped_at"] == "xgboost_low"

    def test_boundary_high_score_exactly_085(self):
        p = make_pipeline_with_score(0.85)
        result = p.evaluate(make_transaction(), make_features())
        assert result["final_decision"] == "FRAUD"

    def test_boundary_low_score_just_below_020(self):
        p = make_pipeline_with_score(0.19)
        result = p.evaluate(make_transaction(), make_features())
        assert result["final_decision"] == "LEGIT"

    def test_grey_zone_does_not_crash(self):
        p = make_pipeline_with_score(0.5)
        result = p.evaluate(make_transaction(), make_features())
        assert result["final_decision"] in ("FRAUD", "LEGIT", "UNCERTAIN")


class TestFallbackBehaviour:

    def test_no_model_does_not_crash(self):
        from main import FraudDetectionPipeline

        with patch("main.FraudDetectionPipeline._load_xgboost_model"):
            p = FraudDetectionPipeline(use_agents=False, use_graph=False)
        p._model = None
        result = p.evaluate(make_transaction(), make_features())
        assert "final_decision" in result

    def test_no_model_uses_fraud_score_fallback(self):
        from main import FraudDetectionPipeline

        with patch("main.FraudDetectionPipeline._load_xgboost_model"):
            p = FraudDetectionPipeline(use_agents=False, use_graph=False)
        p._model = None
        features = make_features(fraud_score=0.95)
        result = p.evaluate(make_transaction(), features)
        assert result["xgboost_score"] is not None


class TestFeaturesNoneFallback:

    def test_features_none_uses_transaction_dict(self, pipeline_no_deps):
        result = pipeline_no_deps.evaluate(make_transaction())
        assert "final_decision" in result


class TestPipelineLatency:

    def test_fast_path_under_20ms(self):
        import time

        p = make_pipeline_with_score(0.95)
        start = time.perf_counter()
        p.evaluate(make_transaction(), make_features())
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"Pipeline fast path took {elapsed_ms:.1f}ms"

    def test_hundred_calls_avg_under_10ms(self):
        import time

        p = make_pipeline_with_score(0.95)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            p.evaluate(make_transaction(), make_features())
            times.append((time.perf_counter() - start) * 1000)
        avg = sum(times) / len(times)
        assert avg < 10, f"Avg pipeline latency {avg:.1f}ms over 100 calls"
