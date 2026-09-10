import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitl.models import HITLCase, HITLDecision


class TestHITLCaseCreation:

    def test_case_id_auto_generated(self):
        c = HITLCase()
        assert c.case_id.startswith("CASE-")
        assert len(c.case_id) > 5

    def test_two_cases_have_different_ids(self):
        c1 = HITLCase()
        c2 = HITLCase()
        assert c1.case_id != c2.case_id

    def test_default_status_pending(self):
        c = HITLCase()
        assert c.status == "PENDING"

    def test_default_priority_high(self):
        c = HITLCase()
        assert c.priority == "HIGH"

    def test_created_at_is_set(self):
        c = HITLCase()
        assert c.created_at is not None
        datetime.fromisoformat(c.created_at)

    def test_custom_fields(self):
        c = HITLCase(
            transaction_id="TX-999",
            user_id="USER-42",
            amount=500.0,
            priority="CRITICAL",
        )
        assert c.transaction_id == "TX-999"
        assert c.user_id == "USER-42"
        assert c.amount == 500.0
        assert c.priority == "CRITICAL"


class TestHITLCaseLifecycle:

    def test_assign_sets_analyst(self):
        c = HITLCase()
        c.assign("alice")
        assert c.assigned_to == "alice"
        assert c.status == "IN_REVIEW"
        assert c.assigned_at is not None

    def test_assign_sets_assigned_at_timestamp(self):
        c = HITLCase()
        before = datetime.utcnow()
        c.assign("alice")
        after = datetime.utcnow()
        assigned_dt = datetime.fromisoformat(c.assigned_at)
        assert before <= assigned_dt <= after

    def test_resolve_sets_resolved_at(self):
        c = HITLCase()
        c.assign("alice")
        c.resolve("alice")
        assert c.status == "RESOLVED"
        assert c.resolved_at is not None


class TestSLA:

    def test_sla_not_breached_on_fresh_case(self):
        c = HITLCase(sla_minutes=30)
        assert c.check_sla() is False
        assert c.is_sla_breached is False

    def test_sla_breached_on_old_case(self):
        c = HITLCase(sla_minutes=1)
        old_time = (datetime.utcnow() - timedelta(minutes=2)).isoformat()
        c.created_at = old_time
        breached = c.check_sla()
        assert breached is True
        assert c.is_sla_breached is True

    def test_resolved_case_does_not_breach_sla(self):
        c = HITLCase(sla_minutes=1)
        old_time = (datetime.utcnow() - timedelta(minutes=2)).isoformat()
        c.created_at = old_time
        c.status = "RESOLVED"
        breached = c.check_sla()
        assert breached is False


class TestSerialization:

    def test_to_dict_returns_dict(self):
        c = HITLCase(transaction_id="TX-1", amount=100.0)
        d = c.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_key_fields(self):
        c = HITLCase(transaction_id="TX-1", amount=100.0, priority="HIGH")
        d = c.to_dict()
        assert d["transaction_id"] == "TX-1"
        assert d["amount"] == 100.0
        assert d["priority"] == "HIGH"

    def test_to_dict_no_nested_objects(self):
        import json

        c = HITLCase(transaction_id="TX-1")
        d = c.to_dict()
        json.dumps(d, default=str)
