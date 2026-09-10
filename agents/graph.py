import logging
from typing import Dict, Any, List
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.state import (
    FraudInvestigationState,
    create_initial_state,
    update_detective_findings,
    update_analyst_findings,
    update_verifier_findings,
    calculate_combined_confidence,
    get_investigation_summary,
)
from agents.agent_nodes import (
    detective_agent_node,
    analyst_agent_node,
    verifier_agent_node,
)
from agents.config import (
    should_skip_agents,
    should_run_analyst,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def route_after_detective(state: FraudInvestigationState) -> str:
    detective_findings = state.get("detective_findings")
    if not detective_findings:
        logger.warning("No Detective findings, routing to Analyst as fallback")
        return "analyst"
    detective_confidence = detective_findings.get("confidence", 0.0)
    should_investigate = detective_findings.get("should_investigate_deeper", True)
    if detective_confidence > 0.9 and (not should_investigate):
        logger.info(
            f" EARLY EXIT: Detective {detective_confidence:.1%} confident, skipping Analyst/Verifier"
        )
        return "end"
    if should_run_analyst(state):
        logger.info(
            f" Routing to Analyst (Detective confidence: {detective_confidence:.1%})"
        )
        return "analyst"
    else:
        logger.info(f" Skipping Analyst, going straight to Verifier")
        return "verifier"


def route_after_analyst(state: FraudInvestigationState) -> str:
    analyst_findings = state.get("analyst_findings")
    if analyst_findings:
        analyst_confidence = analyst_findings.get("confidence", 0.0)
        logger.info(
            f" Analyst complete (confidence: {analyst_confidence:.1%}), routing to Verifier"
        )
    else:
        logger.warning("No Analyst findings, routing to Verifier as fallback")
    return "verifier"


def route_after_verifier(state: FraudInvestigationState) -> str:
    verifier_findings = state.get("verifier_findings")
    if verifier_findings:
        decision = verifier_findings.get("final_decision", "UNCERTAIN")
        confidence = verifier_findings.get("confidence", 0.0)
        action = verifier_findings.get("recommended_action", "REVIEW")
        logger.info(
            f" Verifier decision: {decision} (confidence: {confidence:.1%}) -> {action}"
        )
    else:
        logger.warning("No Verifier findings")
    return "end"


def start_investigation(state: FraudInvestigationState) -> FraudInvestigationState:
    transaction_id = state.get("transaction_id", "UNKNOWN")
    amount = state.get("transaction", {}).get("amount", 0.0)
    xgboost_prediction = state.get("xgboost_prediction", 0.5)
    logger.info(f"\n{'=' * 60}")
    logger.info(f" Starting Fraud Investigation: {transaction_id}")
    logger.info(f" Amount: ${amount:.2f}")
    logger.info(f" XGBoost: {xgboost_prediction:.1%} fraud probability")
    logger.info(f"{'=' * 60}\n")
    if should_skip_agents(state):
        skip_reason = state.get("skip_reason", "XGBoost decisive")
        logger.info(f" SKIPPING ALL AGENTS: {skip_reason}")
        state["agents_skipped"] = True
        state["skip_timestamp"] = datetime.now().isoformat()
    else:
        state["agents_skipped"] = False
    return state


def route_from_start(state: FraudInvestigationState) -> str:
    if state.get("agents_skipped", False):
        logger.info("Fast path: Skipping straight to end")
        return "end"
    else:
        logger.info("Starting with Detective agent")
        return "detective"


def build_fraud_investigation_graph() -> StateGraph:
    logger.info("Building Fraud Investigation Graph...")
    workflow = StateGraph(FraudInvestigationState)
    workflow.add_node("start", start_investigation)
    workflow.add_node("detective", detective_agent_node)
    workflow.add_node("analyst", analyst_agent_node)
    workflow.add_node("verifier", verifier_agent_node)
    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start", route_from_start, {"detective": "detective", "end": END}
    )
    workflow.add_conditional_edges(
        "detective",
        route_after_detective,
        {"analyst": "analyst", "verifier": "verifier", "end": END},
    )
    workflow.add_conditional_edges(
        "analyst", route_after_analyst, {"verifier": "verifier"}
    )
    workflow.add_conditional_edges("verifier", route_after_verifier, {"end": END})
    logger.info("Graph built successfully")
    return workflow


def compile_fraud_investigation_graph() -> Any:
    workflow = build_fraud_investigation_graph()
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    logger.info("Graph compiled and ready to run")
    return compiled_graph


def investigate_transaction(
    transaction: Dict[str, Any],
    xgboost_prediction: float,
    xgboost_confidence: float,
    rules_engine_result: Dict[str, Any] = None,
    graph: Any = None,
) -> Dict[str, Any]:
    start_time = datetime.now()
    state = create_initial_state(
        transaction=transaction,
        xgboost_prediction=xgboost_prediction,
        xgboost_confidence=xgboost_confidence,
        rules_engine_result=rules_engine_result,
    )
    if graph is None:
        graph = compile_fraud_investigation_graph()
    try:
        logger.info(
            f" Starting investigation for transaction {transaction.get('transaction_id', 'UNKNOWN')}"
        )
        final_state = graph.invoke(state)
        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        result = build_investigation_result(final_state, total_time_ms)
        logger.info(f" Investigation complete in {total_time_ms:.0f}ms")
        logger.info(
            f" Final decision: {result['final_decision']} ({result['combined_confidence']:.1%} confidence)"
        )
        logger.info(f" Action: {result['recommended_action']}")
        return result
    except Exception as e:
        logger.error(f" Investigation failed: {str(e)}")
        return {
            "transaction_id": transaction.get("transaction_id", "UNKNOWN"),
            "final_decision": "ERROR",
            "recommended_action": "ESCALATE_TO_HITL",
            "combined_confidence": 0.0,
            "error": str(e),
            "total_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
        }


def build_investigation_result(
    state: FraudInvestigationState, total_time_ms: float
) -> Dict[str, Any]:
    detective = state.get("detective_findings", {})
    analyst = state.get("analyst_findings", {})
    verifier = state.get("verifier_findings", {})
    agents_run = []
    if detective:
        agents_run.append("detective")
    if analyst:
        agents_run.append("analyst")
    if verifier:
        agents_run.append("verifier")
    if verifier:
        final_decision = verifier.get("final_decision", "UNCERTAIN")
        recommended_action = verifier.get("recommended_action", "REVIEW")
    elif detective:
        risk_level = detective.get("risk_level", "MEDIUM")
        final_decision = "FRAUD" if risk_level in ["HIGH", "CRITICAL"] else "LEGIT"
        recommended_action = "REVIEW"
    else:
        final_decision = "UNCERTAIN"
        recommended_action = "ESCALATE_TO_HITL"
    combined_confidence = calculate_combined_confidence(state)
    cost_per_agent = {"detective": 0.002, "analyst": 0.015, "verifier": 0.01}
    total_cost = sum((cost_per_agent.get(agent, 0) for agent in agents_run))
    result = {
        "transaction_id": state.get("transaction_id", "UNKNOWN"),
        "final_decision": final_decision,
        "recommended_action": recommended_action,
        "combined_confidence": combined_confidence,
        "detective_findings": detective,
        "analyst_findings": analyst,
        "verifier_findings": verifier,
        "xgboost_prediction": state.get("xgboost_prediction", 0.5),
        "total_time_ms": total_time_ms,
        "agents_run": agents_run,
        "cost_estimate": total_cost,
        "skipped": state.get("agents_skipped", False),
    }
    return result


def investigate_batch(
    transactions: List[Dict[str, Any]],
    xgboost_predictions: List[float],
    xgboost_confidences: List[float],
    rules_engine_results: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if rules_engine_results is None:
        rules_engine_results = [None] * len(transactions)
    logger.info(f" Processing batch of {len(transactions)} transactions")
    graph = compile_fraud_investigation_graph()
    results = []
    for i, (txn, pred, conf, rules) in enumerate(
        zip(
            transactions, xgboost_predictions, xgboost_confidences, rules_engine_results
        )
    ):
        logger.info(f"\nProcessing transaction {i + 1}/{len(transactions)}")
        result = investigate_transaction(
            transaction=txn,
            xgboost_prediction=pred,
            xgboost_confidence=conf,
            rules_engine_result=rules,
            graph=graph,
        )
        results.append(result)
    logger.info(f"\nBatch processing complete: {len(results)} transactions")
    avg_time = sum((r["total_time_ms"] for r in results)) / len(results)
    total_cost = sum((r["cost_estimate"] for r in results))
    fraud_count = sum((1 for r in results if r["final_decision"] == "FRAUD"))
    logger.info(f" Batch Statistics:")
    logger.info(f"   Average time: {avg_time:.0f}ms")
    logger.info(f"   Total cost: ${total_cost:.3f}")
    logger.info(
        f"   Fraud detected: {fraud_count}/{len(results)} ({fraud_count / len(results):.1%})"
    )
    return results


if __name__ == "__main__":
    "\n    Quick test of the graph with a mock transaction.\n"
    print("Testing Fraud Investigation Graph\n")
    test_transaction = {
        "transaction_id": "test_123",
        "user_id": "user_456",
        "amount": 500.0,
        "merchant_name": "Electronics Store",
        "transaction_time": "2024-01-15 14:30:00",
        "transactions_last_hour": 3,
        "transactions_last_24h": 8,
        "distance_from_last_km": 150.0,
        "time_since_last_transaction_hours": 0.5,
        "is_new_device": True,
        "is_new_location": False,
        "transaction_hour": 14,
        "amount_vs_avg_ratio": 5.0,
        "account_age_days": 365,
        "user_total_transactions": 100,
        "user_fraud_rate": 0.02,
    }
    result = investigate_transaction(
        transaction=test_transaction, xgboost_prediction=0.65, xgboost_confidence=0.8
    )
    print("\n" + "=" * 60)
    print("INVESTIGATION RESULTS")
    print("=" * 60)
    print(f"Transaction ID: {result['transaction_id']}")
    print(f"Final Decision: {result['final_decision']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Combined Confidence: {result['combined_confidence']:.1%}")
    print(f"Time: {result['total_time_ms']:.0f}ms")
    print(f"Cost: ${result['cost_estimate']:.4f}")
    print(f"Agents Run: {', '.join(result['agents_run'])}")
    print("=" * 60)
