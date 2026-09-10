import json
import os
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for, flash

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hitl.queue_manager import HITLQueueManager
from hitl.models import HITLDecision

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())
queue = HITLQueueManager()
FEEDBACK_LOG = Path(__file__).parent.parent.parent / "logs" / "hitl_retrain_queue.jsonl"
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)


@app.route("/")
def dashboard():
    pending_cases = queue.list_cases(status="PENDING")
    in_review_cases = queue.list_cases(status="IN_REVIEW")
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    pending_cases.sort(key=lambda c: priority_order.get(c.priority, 99))
    return render_template_string(
        DASHBOARD_TEMPLATE, pending=pending_cases, in_review=in_review_cases
    )


@app.route("/case/<case_id>")
def case_detail(case_id):
    case = queue.get_case(case_id)
    if not case:
        flash(f"Case {case_id} not found", "error")
        return redirect(url_for("dashboard"))
    return render_template_string(CASE_DETAIL_TEMPLATE, case=case)


@app.route("/case/<case_id>/assign", methods=["POST"])
def assign_case(case_id):
    analyst = request.form.get("analyst", "analyst_001")
    case = queue.dequeue_specific(case_id, analyst=analyst)
    if case:
        flash(f"Case {case_id} assigned to {analyst}", "success")
    else:
        flash(f"Failed to assign case {case_id}", "error")
    return redirect(url_for("case_detail", case_id=case_id))


@app.route("/case/<case_id>/decide", methods=["POST"])
def decide_case(case_id):
    case = queue.get_case(case_id)
    if not case:
        flash(f"Case {case_id} not found", "error")
        return redirect(url_for("dashboard"))
    decision = request.form.get("decision")
    notes = request.form.get("notes", "")
    analyst = request.form.get("analyst", "analyst_001")
    if decision not in ["APPROVE", "DECLINE", "ESCALATE"]:
        flash("Invalid decision", "error")
        return redirect(url_for("case_detail", case_id=case_id))
    if decision == "ESCALATE":
        queue.requeue(case_id)
        flash(f"Case {case_id} escalated", "warning")
    else:
        queue.mark_resolved(case_id, analyst=analyst)
        flash(f"Case {case_id} marked as {decision}", "success")
    _log_feedback(case, decision, notes, analyst)
    return redirect(url_for("dashboard"))


def _log_feedback(case, decision, notes, analyst):
    feedback = {
        "case_id": case.case_id,
        "transaction_id": case.transaction_id,
        "user_id": case.user_id,
        "amount": case.amount,
        "xgboost_score": case.xgboost_score,
        "analyst_decision": decision,
        "analyst_notes": notes,
        "analyst_id": analyst,
        "timestamp": datetime.utcnow().isoformat(),
        "original_agent_decision": case.agent_final_decision,
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(feedback) + "\n")


DASHBOARD_TEMPLATE = '\n<!DOCTYPE html>\n<html>\n<head>\n <title>HITL Fraud Review Dashboard</title>\n <style>\n body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }\n h1 { color: #333; }\n .stats { background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n .case-list { background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n table { width: 100%; border-collapse: collapse; }\n th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }\n th { background: #f0f0f0; font-weight: bold; }\n .priority-CRITICAL { color: #d32f2f; font-weight: bold; }\n .priority-HIGH { color: #f57c00; }\n .priority-MEDIUM { color: #fbc02d; }\n .priority-LOW { color: #388e3c; }\n .btn { padding: 5px 10px; text-decoration: none; background: #2196F3; color: white; border-radius: 3px; }\n .btn:hover { background: #1976D2; }\n .flash { padding: 10px; border-radius: 5px; margin-bottom: 20px; }\n .flash.success { background: #c8e6c9; }\n .flash.error { background: #ffcdd2; }\n .flash.warning { background: #fff9c4; }\n </style>\n</head>\n<body>\n <h1> HITL Fraud Review Dashboard</h1>\n \n {% with messages = get_flashed_messages(with_categories=true) %}\n {% if messages %}\n {% for category, message in messages %}\n <div class="flash {{ category }}">{{ message }}</div>\n {% endfor %}\n {% endif %}\n {% endwith %}\n \n <div class="stats">\n <strong>Queue Status:</strong>\n Pending: {{ pending|length }} | In Review: {{ in_review|length }}\n </div>\n \n <div class="case-list">\n <h2>Pending Cases</h2>\n {% if pending %}\n <table>\n <tr>\n <th>Case ID</th>\n <th>Priority</th>\n <th>User</th>\n <th>Amount</th>\n <th>XGBoost Score</th>\n <th>Created</th>\n <th>Action</th>\n </tr>\n {% for case in pending %}\n <tr>\n <td>{{ case.case_id }}</td>\n <td class="priority-{{ case.priority }}">{{ case.priority }}</td>\n <td>{{ case.user_id }}</td>\n <td>${{ "%.2f"|format(case.amount) }}</td>\n <td>{{ "%.3f"|format(case.xgboost_score) }}</td>\n <td>{{ case.created_at[:19] }}</td>\n <td><a href="{{ url_for(\'case_detail\', case_id=case.case_id) }}"class="btn">Review</a></td>\n </tr>\n {% endfor %}\n </table>\n {% else %}\n <p>No pending cases</p>\n {% endif %}\n </div>\n \n <div class="case-list">\n <h2>In Review</h2>\n {% if in_review %}\n <table>\n <tr>\n <th>Case ID</th>\n <th>Analyst</th>\n <th>User</th>\n <th>Amount</th>\n <th>Assigned</th>\n <th>Action</th>\n </tr>\n {% for case in in_review %}\n <tr>\n <td>{{ case.case_id }}</td>\n <td>{{ case.assigned_to }}</td>\n <td>{{ case.user_id }}</td>\n <td>${{ "%.2f"|format(case.amount) }}</td>\n <td>{{ case.assigned_at[:19] if case.assigned_at else "N/A"}}</td>\n <td><a href="{{ url_for(\'case_detail\', case_id=case.case_id) }}"class="btn">View</a></td>\n </tr>\n {% endfor %}\n </table>\n {% else %}\n <p>No cases in review</p>\n {% endif %}\n </div>\n</body>\n</html>\n'
CASE_DETAIL_TEMPLATE = '\n<!DOCTYPE html>\n<html>\n<head>\n <title>Case {{ case.case_id }}</title>\n <style>\n body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }\n .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 5px; }\n h1 { color: #333; }\n .field { margin-bottom: 15px; }\n .label { font-weight: bold; color: #666; }\n .value { color: #333; }\n .decision-form { margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; }\n .btn-group { margin-top: 20px; }\n button { padding: 10px 20px; margin-right: 10px; border: none; border-radius: 3px; cursor: pointer; font-size: 14px; }\n .btn-approve { background: #4CAF50; color: white; }\n .btn-decline { background: #f44336; color: white; }\n .btn-escalate { background: #FF9800; color: white; }\n textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }\n .back { display: inline-block; margin-bottom: 20px; color: #2196F3; text-decoration: none; }\n </style>\n</head>\n<body>\n <div class="container">\n <a href="{{ url_for(\'dashboard\') }}"class="back"> Back to Dashboard</a>\n \n <h1>Case {{ case.case_id }}</h1>\n \n <div class="field">\n <span class="label">Status:</span>\n <span class="value">{{ case.status }}</span>\n </div>\n \n <div class="field">\n <span class="label">Priority:</span>\n <span class="value">{{ case.priority }}</span>\n </div>\n \n <div class="field">\n <span class="label">Transaction ID:</span>\n <span class="value">{{ case.transaction_id }}</span>\n </div>\n \n <div class="field">\n <span class="label">User ID:</span>\n <span class="value">{{ case.user_id }}</span>\n </div>\n \n <div class="field">\n <span class="label">Amount:</span>\n <span class="value">${{ "%.2f"|format(case.amount) }}</span>\n </div>\n \n <div class="field">\n <span class="label">Merchant:</span>\n <span class="value">{{ case.merchant }}</span>\n </div>\n \n <div class="field">\n <span class="label">XGBoost Score:</span>\n <span class="value">{{ "%.4f"|format(case.xgboost_score) }}</span>\n </div>\n \n <div class="field">\n <span class="label">Agent Decision:</span>\n <span class="value">{{ case.agent_final_decision }}</span>\n </div>\n \n <div class="field">\n <span class="label">Escalation Reason:</span>\n <span class="value">{{ case.escalation_reason }}</span>\n </div>\n \n <div class="field">\n <span class="label">Created:</span>\n <span class="value">{{ case.created_at }}</span>\n </div>\n \n {% if case.status == "PENDING"%}\n <form method="POST"action="{{ url_for(\'assign_case\', case_id=case.case_id) }}"style="margin-top: 20px;">\n <input type="hidden"name="analyst"value="analyst_001">\n <button type="submit"style="background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer;">\n Assign to Me\n </button>\n </form>\n {% endif %}\n \n {% if case.status == "IN_REVIEW"%}\n <div class="decision-form">\n <h2>Make Decision</h2>\n <form method="POST"action="{{ url_for(\'decide_case\', case_id=case.case_id) }}">\n <input type="hidden"name="analyst"value="{{ case.assigned_to }}">\n \n <div class="field">\n <label class="label">Notes:</label>\n <textarea name="notes"rows="4"placeholder="Reasoning for your decision..."></textarea>\n </div>\n \n <div class="btn-group">\n <button type="submit"name="decision"value="APPROVE"class="btn-approve"> APPROVE</button>\n <button type="submit"name="decision"value="DECLINE"class="btn-decline"> DECLINE</button>\n <button type="submit"name="decision"value="ESCALATE"class="btn-escalate"> ESCALATE</button>\n </div>\n </form>\n </div>\n {% endif %}\n </div>\n</body>\n</html>\n'
if __name__ == "__main__":
    print("=" * 60)
    print("HITL Fraud Review Dashboard")
    print("=" * 60)
    print(f"Opening on http://localhost:5001")
    print(f"Feedback log: {FEEDBACK_LOG}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=True)
