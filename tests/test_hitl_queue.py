import time
import threading
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitl.queue_manager import HITLQueueManager
from hitl.models import HITLCase


@pytest.fixture
def queue():
    q = HITLQueueManager(sla_check_interval_seconds=1)
    yield q
    q.shutdown()


def make_case(**overrides) -> HITLCase:
    base = dict(
        transaction_id="TX-001",
        user_id="USER-001",
        amount=250.0,
        merchant="test_merchant",
        timestamp="2026-04-22T14:00:00",
        xgboost_score=0.55,
        priority="HIGH",
        agent_final_decision="UNCERTAIN",
        escalation_reason="XGBoost in grey zone",
    )
    base.update(overrides)
    return HITLCase(**base)


class TestBasicOperations:

    def test_enqueue_returns_case_id(self, queue):
        case = make_case()
        case_id = queue.enqueue(case)
        assert case_id == case.case_id

    def test_enqueue_sets_status_pending(self, queue):
        case = make_case()
        queue.enqueue(case)
        stored = queue.get_case(case.case_id)
        assert stored.status == "PENDING"

    def test_dequeue_returns_case(self, queue):
        case = make_case()
        queue.enqueue(case)
        dequeued = queue.dequeue(analyst="alice")
        assert dequeued is not None
        assert dequeued.case_id == case.case_id

    def test_dequeue_sets_status_in_review(self, queue):
        case = make_case()
        queue.enqueue(case)
        dequeued = queue.dequeue(analyst="alice")
        assert dequeued.status == "IN_REVIEW"
        assert dequeued.assigned_to == "alice"

    def test_dequeue_empty_queue_returns_none(self, queue):
        result = queue.dequeue(analyst="alice")
        assert result is None

    def test_requeue_moves_to_pending(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        success = queue.requeue(case.case_id)
        assert success is True
        assert queue.get_case(case.case_id).status == "PENDING"

    def test_requeue_clears_analyst(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        queue.requeue(case.case_id)
        assert queue.get_case(case.case_id).assigned_to is None

    def test_mark_resolved(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        success = queue.mark_resolved(case.case_id, analyst="alice")
        assert success is True
        assert queue.get_case(case.case_id).status == "RESOLVED"

    def test_get_case_unknown_id_returns_none(self, queue):
        result = queue.get_case("NONEXISTENT-CASE-ID")
        assert result is None


class TestPriorityOrdering:

    def test_critical_before_high(self, queue):
        high = make_case(priority="HIGH")
        critical = make_case(priority="CRITICAL")
        queue.enqueue(high)
        queue.enqueue(critical)
        first = queue.dequeue(analyst="alice")
        assert first.priority == "CRITICAL"

    def test_high_before_medium(self, queue):
        medium = make_case(priority="MEDIUM")
        high = make_case(priority="HIGH")
        queue.enqueue(medium)
        queue.enqueue(high)
        first = queue.dequeue(analyst="alice")
        assert first.priority == "HIGH"

    def test_medium_before_low(self, queue):
        low = make_case(priority="LOW")
        medium = make_case(priority="MEDIUM")
        queue.enqueue(low)
        queue.enqueue(medium)
        first = queue.dequeue(analyst="alice")
        assert first.priority == "MEDIUM"

    def test_full_priority_order(self, queue):
        for p in ["LOW", "HIGH", "MEDIUM", "CRITICAL"]:
            queue.enqueue(make_case(priority=p))
        order = []
        for _ in range(4):
            c = queue.dequeue(analyst="alice")
            order.append(c.priority)
            queue.mark_resolved(c.case_id, analyst="alice")
        assert order == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class TestDuplicates:

    def test_duplicate_case_id_raises(self, queue):
        case = make_case()
        queue.enqueue(case)
        with pytest.raises(ValueError, match=case.case_id):
            queue.enqueue(case)


class TestStats:

    def test_stats_pending_count(self, queue):
        for _ in range(3):
            queue.enqueue(make_case())
        stats = queue.get_stats()
        assert stats.total_pending == 3

    def test_stats_in_review_count(self, queue):
        for _ in range(2):
            queue.enqueue(make_case())
        queue.dequeue(analyst="alice")
        stats = queue.get_stats()
        assert stats.total_in_review == 1
        assert stats.total_pending == 1

    def test_stats_resolved_count(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        queue.mark_resolved(case.case_id, analyst="alice")
        stats = queue.get_stats()
        assert stats.total_resolved == 1
        assert stats.total_pending == 0

    def test_stats_empty_queue(self, queue):
        stats = queue.get_stats()
        assert stats.total_pending == 0
        assert stats.total_in_review == 0
        assert stats.total_resolved == 0


class TestListCases:

    def test_list_all(self, queue):
        for _ in range(5):
            queue.enqueue(make_case())
        cases = queue.list_cases()
        assert len(cases) == 5

    def test_list_by_status(self, queue):
        c1 = make_case()
        c2 = make_case()
        queue.enqueue(c1)
        queue.enqueue(c2)
        queue.dequeue(analyst="alice")
        pending = queue.list_cases(status="PENDING")
        in_review = queue.list_cases(status="IN_REVIEW")
        assert len(pending) == 1
        assert len(in_review) == 1

    def test_list_by_priority(self, queue):
        queue.enqueue(make_case(priority="CRITICAL"))
        queue.enqueue(make_case(priority="LOW"))
        queue.enqueue(make_case(priority="LOW"))
        critical = queue.list_cases(priority="CRITICAL")
        low = queue.list_cases(priority="LOW")
        assert len(critical) == 1
        assert len(low) == 2

    def test_list_by_analyst(self, queue):
        c1 = make_case()
        c2 = make_case()
        queue.enqueue(c1)
        queue.enqueue(c2)
        queue.dequeue(analyst="alice")
        queue.dequeue(analyst="bob")
        alice_cases = queue.list_cases(analyst="alice")
        assert len(alice_cases) == 1
        assert alice_cases[0].assigned_to == "alice"


class TestAuditLog:

    def test_enqueue_creates_audit_entry(self, queue):
        case = make_case()
        queue.enqueue(case)
        log = queue.get_audit_log(case.case_id)
        events = [e["event"] for e in log]
        assert "ENQUEUED" in events

    def test_dequeue_creates_audit_entry(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        log = queue.get_audit_log(case.case_id)
        events = [e["event"] for e in log]
        assert "ASSIGNED" in events

    def test_resolve_creates_audit_entry(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        queue.mark_resolved(case.case_id, analyst="alice")
        log = queue.get_audit_log(case.case_id)
        events = [e["event"] for e in log]
        assert "RESOLVED" in events

    def test_full_lifecycle_audit_trail(self, queue):
        case = make_case()
        queue.enqueue(case)
        queue.dequeue(analyst="alice")
        queue.requeue(case.case_id)
        queue.dequeue(analyst="bob")
        queue.mark_resolved(case.case_id, analyst="bob")
        log = queue.get_audit_log(case.case_id)
        events = [e["event"] for e in log]
        assert events == ["ENQUEUED", "ASSIGNED", "REQUEUED", "ASSIGNED", "RESOLVED"]


class TestThreadSafety:

    def test_concurrent_enqueue(self, queue):
        errors = []

        def enqueue_one():
            try:
                queue.enqueue(make_case())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=enqueue_one) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert queue.get_stats().total_pending == 50

    def test_concurrent_dequeue_no_double_assign(self, queue):
        for _ in range(10):
            queue.enqueue(make_case())
        assigned = []
        lock = threading.Lock()

        def dequeue_one():
            case = queue.dequeue(analyst=f"analyst-{threading.get_ident()}")
            if case:
                with lock:
                    assigned.append(case.case_id)

        threads = [threading.Thread(target=dequeue_one) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(assigned) == len(
            set(assigned)
        ), "Duplicate case assignments detected!"
        assert len(assigned) == 10
