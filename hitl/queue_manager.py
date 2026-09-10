import heapq
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from hitl.models import CaseStatus, HITLCase, Priority, QueueStats

logger = logging.getLogger(__name__)
_PRIORITY_RANK: Dict[Priority, int] = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


class HITLQueueManager:

    def __init__(self, sla_check_interval_seconds: int = 60):
        self._cases: Dict[str, HITLCase] = {}
        self._heap: List[tuple] = []
        self._lock = threading.Lock()
        self._audit_log: List[Dict] = []
        self._sla_interval = sla_check_interval_seconds
        self._stop_watchdog = threading.Event()
        self._watchdog = threading.Thread(
            target=self._sla_watchdog_loop, daemon=True, name="hitl-sla-watchdog"
        )
        self._watchdog.start()
        logger.info(
            "HITLQueueManager started (SLA watchdog interval: %ds)",
            sla_check_interval_seconds,
        )

    def enqueue(self, case: HITLCase) -> str:
        with self._lock:
            if case.case_id in self._cases:
                raise ValueError(f"Case {case.case_id} already in queue")
            case.status = "PENDING"
            self._cases[case.case_id] = case
            rank = _PRIORITY_RANK.get(case.priority, 2)
            heapq.heappush(self._heap, (rank, case.created_at, case.case_id))
            self._audit(
                "ENQUEUED",
                case.case_id,
                details={
                    "priority": case.priority,
                    "amount": case.amount,
                    "xgboost_score": case.xgboost_score,
                },
            )
        logger.info(
            "Enqueued case %s [%s] tx=%s",
            case.case_id,
            case.priority,
            case.transaction_id,
        )
        return case.case_id

    def dequeue(self, analyst: str) -> Optional[HITLCase]:
        with self._lock:
            while self._heap:
                rank, created_at, case_id = heapq.heappop(self._heap)
                case = self._cases.get(case_id)
                if case is None or case.status != "PENDING":
                    continue
                case.assign(analyst)
                self._audit("ASSIGNED", case_id, analyst=analyst)
                logger.info("Assigned case %s to analyst %s", case_id, analyst)
                return case
        return None

    def requeue(self, case_id: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case or case.status != "IN_REVIEW":
                return False
            prev_analyst = case.assigned_to
            case.status = "PENDING"
            case.assigned_to = None
            case.assigned_at = None
            rank = _PRIORITY_RANK.get(case.priority, 2)
            heapq.heappush(self._heap, (rank, case.created_at, case_id))
            self._audit("REQUEUED", case_id, analyst=prev_analyst)
        logger.info("Requeued case %s (was assigned to %s)", case_id, prev_analyst)
        return True

    def mark_resolved(self, case_id: str, analyst: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            case.status = "RESOLVED"
            case.resolved_at = datetime.utcnow().isoformat()
            self._audit("RESOLVED", case_id, analyst=analyst)
        logger.info("Resolved case %s by analyst %s", case_id, analyst)
        return True

    def mark_timed_out(self, case_id: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case or case.status in ("RESOLVED", "TIMED_OUT"):
                return False
            case.status = "TIMED_OUT"
            self._audit("TIMED_OUT", case_id)
        logger.warning("Case %s marked TIMED_OUT (SLA breach)", case_id)
        return True

    def get_case(self, case_id: str) -> Optional[HITLCase]:
        return self._cases.get(case_id)

    def list_cases(
        self,
        status: Optional[CaseStatus] = None,
        analyst: Optional[str] = None,
        priority: Optional[Priority] = None,
        limit: int = 100,
    ) -> List[HITLCase]:
        results = list(self._cases.values())
        if status:
            results = [c for c in results if c.status == status]
        if analyst:
            results = [c for c in results if c.assigned_to == analyst]
        if priority:
            results = [c for c in results if c.priority == priority]
        results.sort(key=lambda c: (_PRIORITY_RANK.get(c.priority, 2), c.created_at))
        return results[:limit]

    def get_stats(self) -> QueueStats:
        cases = list(self._cases.values())
        pending = [c for c in cases if c.status == "PENDING"]
        in_review = [c for c in cases if c.status == "IN_REVIEW"]
        resolved = [c for c in cases if c.status == "RESOLVED"]
        timed_out = [c for c in cases if c.status == "TIMED_OUT"]
        sla_breach = [c for c in cases if c.is_sla_breached]
        review_times = []
        for c in resolved:
            if c.assigned_at and c.resolved_at:
                assigned = datetime.fromisoformat(c.assigned_at)
                res = datetime.fromisoformat(c.resolved_at)
                review_times.append((res - assigned).total_seconds())
        avg_review = sum(review_times) / len(review_times) if review_times else 0.0
        oldest_minutes = 0.0
        if pending:
            oldest = min(pending, key=lambda c: c.created_at)
            created = datetime.fromisoformat(oldest.created_at)
            oldest_minutes = (datetime.utcnow() - created).total_seconds() / 60
        return QueueStats(
            total_pending=len(pending),
            total_in_review=len(in_review),
            total_resolved=len(resolved),
            total_timed_out=len(timed_out),
            sla_breached=len(sla_breach),
            avg_review_time_s=round(avg_review, 2),
            oldest_case_age_minutes=round(oldest_minutes, 2),
        )

    def get_audit_log(self, case_id: Optional[str] = None) -> List[Dict]:
        if case_id:
            return [e for e in self._audit_log if e.get("case_id") == case_id]
        return list(self._audit_log)

    def _audit(
        self,
        event: str,
        case_id: str,
        analyst: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        entry = {
            "event": event,
            "case_id": case_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if analyst:
            entry["analyst"] = analyst
        if details:
            entry.update(details)
        self._audit_log.append(entry)

    def _sla_watchdog_loop(self):
        while not self._stop_watchdog.wait(timeout=self._sla_interval):
            with self._lock:
                open_cases = [
                    c
                    for c in self._cases.values()
                    if c.status in ("PENDING", "IN_REVIEW")
                ]
            for case in open_cases:
                if case.check_sla():
                    logger.warning(
                        "SLA breached for case %s (priority=%s, age > %dmin)",
                        case.case_id,
                        case.priority,
                        case.sla_minutes,
                    )

    def shutdown(self):
        self._stop_watchdog.set()
        self._watchdog.join(timeout=5)
        logger.info("HITLQueueManager shut down")
