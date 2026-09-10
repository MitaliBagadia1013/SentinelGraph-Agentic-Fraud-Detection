import json
import time
import logging
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
)
from agents.state import (
    FraudInvestigationState,
    DetectiveFindings,
    AnalystFindings,
    VerifierFindings,
    update_detective_findings,
    update_analyst_findings,
    update_verifier_findings,
)
from agents.prompts import (
    DETECTIVE_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    format_detective_prompt,
    format_analyst_prompt,
    format_verifier_prompt,
)
from agents.config import (
    get_agent_model,
    get_agent_timeout,
    should_run_analyst,
    RETRY_CONFIG,
    FALLBACK_CONFIG,
    VALIDATION_RULES,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LOGGING_CONFIG,
)

logger = logging.getLogger(__name__)
if LOGGING_CONFIG["log_to_console"]:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
logger.setLevel(getattr(logging, LOGGING_CONFIG.get("log_level", "INFO")))
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set.\nSet it with: export OPENAI_API_KEY='sk-your-key-here'"
            )
        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
    return _openai_client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    agent_name: str,
) -> Tuple[Optional[str], float]:
    client = get_openai_client()
    max_retries = RETRY_CONFIG["max_retries"]
    base_delay = RETRY_CONFIG["retry_delay_seconds"]
    use_backoff = RETRY_CONFIG["exponential_backoff"]
    start_time = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[{agent_name}] LLM call attempt {attempt}/{max_retries}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
            elapsed_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            if LOGGING_CONFIG["log_all_responses"]:
                logger.debug(f"[{agent_name}] Response: {content[:200]}...")
            logger.info(
                f"[{agent_name}] LLM call succeeded | attempt={attempt} | {elapsed_ms:.0f}ms | tokens={response.usage.total_tokens}"
            )
            return (content, elapsed_ms)
        except APITimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"[{agent_name}] Timeout on attempt {attempt} ({elapsed_ms:.0f}ms)"
            )
        except RateLimitError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"[{agent_name}] Rate limit on attempt {attempt}")
        except APIConnectionError as e:
            logger.warning(f"[{agent_name}] Connection error on attempt {attempt}: {e}")
        except APIStatusError as e:
            logger.error(
                f"[{agent_name}] API status error {e.status_code}: {e.message}"
            )
            if e.status_code < 500:
                break
        except Exception as e:
            logger.error(f"[{agent_name}] Unexpected error on attempt {attempt}: {e}")
        if attempt < max_retries:
            delay = base_delay * 2 ** (attempt - 1) if use_backoff else base_delay
            logger.debug(f"[{agent_name}] Retrying in {delay}s...")
            time.sleep(delay)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.error(
        f"[{agent_name}] All {max_retries} attempts failed ({elapsed_ms:.0f}ms)"
    )
    return (None, elapsed_ms)


def _parse_json_response(
    raw_response: str, agent_name: str
) -> Optional[Dict[str, Any]]:
    if not raw_response:
        logger.error(f"[{agent_name}] Empty response from LLM")
        return None
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.error(f"[{agent_name}] JSON parse error: {e}")
        logger.debug(f"[{agent_name}] Raw response: {raw_response[:500]}")
        return None
    rules = VALIDATION_RULES.get(agent_name, {})
    required_fields = rules.get("required_fields", [])
    missing = [f for f in required_fields if f not in parsed]
    if missing:
        logger.warning(f"[{agent_name}] Missing required fields: {missing}")
    if "confidence" in parsed:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    if "anomaly_score" in parsed:
        parsed["anomaly_score"] = max(0.0, min(1.0, float(parsed["anomaly_score"])))
    if agent_name == "detective" and "risk_level" in parsed:
        valid_levels = rules.get("risk_levels", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        if parsed["risk_level"] not in valid_levels:
            logger.warning(
                f"[{agent_name}] Invalid risk_level '{parsed['risk_level']}' — defaulting to MEDIUM"
            )
            parsed["risk_level"] = "MEDIUM"
    if agent_name == "verifier":
        valid_decisions = rules.get("valid_decisions", ["FRAUD", "LEGIT", "UNCERTAIN"])
        valid_actions = rules.get(
            "valid_actions", ["APPROVE", "DECLINE", "REVIEW", "ESCALATE_TO_HITL"]
        )
        if parsed.get("final_decision") not in valid_decisions:
            logger.warning(
                f"[{agent_name}] Invalid final_decision '{parsed.get('final_decision')}' — defaulting to UNCERTAIN"
            )
            parsed["final_decision"] = "UNCERTAIN"
        if parsed.get("recommended_action") not in valid_actions:
            logger.warning(
                f"[{agent_name}] Invalid recommended_action '{parsed.get('recommended_action')}' — defaulting to ESCALATE_TO_HITL"
            )
            parsed["recommended_action"] = "ESCALATE_TO_HITL"
    return parsed


def _detective_fallback(error_reason: str, elapsed_ms: float) -> DetectiveFindings:
    logger.warning(f"[Detective] Using fallback response. Reason: {error_reason}")
    return DetectiveFindings(
        agent_name="Detective",
        risk_level="MEDIUM",
        red_flags=[f"Detective agent error: {error_reason}"],
        green_flags=[],
        confidence=0.0,
        reasoning=f"Detective agent failed ({error_reason}). Passing to Analyst for evaluation.",
        should_investigate_deeper=True,
        execution_time_ms=elapsed_ms,
    )


def _analyst_fallback(error_reason: str, elapsed_ms: float) -> AnalystFindings:
    logger.warning(f"[Analyst] Using fallback response. Reason: {error_reason}")
    return AnalystFindings(
        agent_name="Analyst",
        fraud_type=None,
        pattern_analysis={"error": error_reason},
        similar_cases=[],
        anomaly_score=0.0,
        confidence=0.0,
        reasoning=f"Analyst agent failed ({error_reason}). Passing to Verifier for final decision.",
        key_evidence=[f"Analyst error: {error_reason}"],
        execution_time_ms=elapsed_ms,
    )


def _verifier_fallback(error_reason: str, elapsed_ms: float) -> VerifierFindings:
    logger.warning(f"[Verifier] Using fallback response. Reason: {error_reason}")
    return VerifierFindings(
        agent_name="Verifier",
        final_decision="UNCERTAIN",
        confidence=0.0,
        reasoning_chain=[
            f"Verifier agent failed: {error_reason}",
            "Escalating to human analyst for safety",
        ],
        false_positive_check={"error": "Agent failed — cannot determine"},
        recommended_action="ESCALATE_TO_HITL",
        execution_time_ms=elapsed_ms,
    )


def detective_agent_node(state: FraudInvestigationState) -> FraudInvestigationState:
    agent_name = "Detective"
    logger.info(
        f"[{agent_name}] Starting | tx={state['transaction_id']} | amount=${state['amount']:.2f}"
    )
    model = get_agent_model("detective")
    timeout = get_agent_timeout("detective")
    temperature = LLM_TEMPERATURE["detective"]
    max_tokens = LLM_MAX_TOKENS["detective"]
    try:
        user_prompt = format_detective_prompt(
            transaction=state["transaction_features"],
            xgboost_prediction=state["xgboost_prediction"],
            rules_engine_result=state.get("rules_engine_result"),
        )
    except Exception as e:
        logger.error(f"[{agent_name}] Prompt formatting failed: {e}")
        findings = _detective_fallback(f"Prompt error: {e}", 0.0)
        return update_detective_findings(state, findings)
    raw_response, elapsed_ms = call_llm(
        system_prompt=DETECTIVE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        agent_name=agent_name,
    )
    if raw_response is None:
        findings = _detective_fallback("LLM call failed after retries", elapsed_ms)
        return update_detective_findings(state, findings)
    parsed = _parse_json_response(raw_response, "detective")
    if parsed is None:
        findings = _detective_fallback("JSON parse failed", elapsed_ms)
        return update_detective_findings(state, findings)
    findings = DetectiveFindings(
        agent_name=agent_name,
        risk_level=parsed.get("risk_level", "MEDIUM"),
        red_flags=parsed.get("red_flags", []),
        green_flags=parsed.get("green_flags", []),
        confidence=parsed.get("confidence", 0.0),
        reasoning=parsed.get("reasoning", ""),
        should_investigate_deeper=parsed.get("should_investigate_deeper", True),
        execution_time_ms=elapsed_ms,
    )
    logger.info(
        f"[{agent_name}] Done | risk={findings.risk_level} | confidence={findings.confidence:.2%} | red_flags={len(findings.red_flags)} | {elapsed_ms:.0f}ms"
    )
    updated_state = update_detective_findings(state, findings)
    if not findings.should_investigate_deeper:
        updated_state["current_step"] = "complete"
        updated_state["should_continue"] = False
        logger.info(
            f"[{agent_name}] LLM says high confidence — skipping Analyst/Verifier"
        )
    return updated_state


def analyst_agent_node(state: FraudInvestigationState) -> FraudInvestigationState:
    agent_name = "Analyst"
    if not state.get("should_continue", True):
        logger.info(f"[{agent_name}] Skipped (should_continue=False)")
        return state
    detective_findings = state.get("detective_findings") or {}
    num_red_flags = len(detective_findings.get("red_flags", []))
    detective_confidence = detective_findings.get("confidence", 0.5)
    if not should_run_analyst(detective_confidence, num_red_flags):
        logger.info(
            f"[{agent_name}] Skipped by routing | detective_confidence={detective_confidence:.2%} | red_flags={num_red_flags}"
        )
        return state
    logger.info(f"[{agent_name}] Starting deep analysis | tx={state['transaction_id']}")
    model = get_agent_model("analyst")
    timeout = get_agent_timeout("analyst")
    temperature = LLM_TEMPERATURE["analyst"]
    max_tokens = LLM_MAX_TOKENS["analyst"]
    try:
        user_prompt = format_analyst_prompt(
            transaction=state["transaction_features"],
            xgboost_prediction=state["xgboost_prediction"],
            detective_findings=detective_findings,
        )
    except Exception as e:
        logger.error(f"[{agent_name}] Prompt formatting failed: {e}")
        findings = _analyst_fallback(f"Prompt error: {e}", 0.0)
        return update_analyst_findings(state, findings)
    raw_response, elapsed_ms = call_llm(
        system_prompt=ANALYST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        agent_name=agent_name,
    )
    if raw_response is None:
        findings = _analyst_fallback("LLM call failed after retries", elapsed_ms)
        return update_analyst_findings(state, findings)
    parsed = _parse_json_response(raw_response, "analyst")
    if parsed is None:
        findings = _analyst_fallback("JSON parse failed", elapsed_ms)
        return update_analyst_findings(state, findings)
    findings = AnalystFindings(
        agent_name=agent_name,
        fraud_type=parsed.get("fraud_type"),
        pattern_analysis=parsed.get("pattern_analysis", {}),
        similar_cases=parsed.get("similar_cases", []),
        anomaly_score=parsed.get("anomaly_score", 0.0),
        confidence=parsed.get("confidence", 0.0),
        reasoning=parsed.get("reasoning", ""),
        key_evidence=parsed.get("key_evidence", []),
        execution_time_ms=elapsed_ms,
    )
    logger.info(
        f"[{agent_name}] Done | fraud_type={findings.fraud_type} | anomaly={findings.anomaly_score:.2%} | confidence={findings.confidence:.2%} | {elapsed_ms:.0f}ms"
    )
    return update_analyst_findings(state, findings)


def verifier_agent_node(state: FraudInvestigationState) -> FraudInvestigationState:
    agent_name = "Verifier"
    if not state.get("should_continue", True):
        logger.info(f"[{agent_name}] Skipped (should_continue=False)")
        return state
    logger.info(f"[{agent_name}] Making final decision | tx={state['transaction_id']}")
    model = get_agent_model("verifier")
    timeout = get_agent_timeout("verifier")
    temperature = LLM_TEMPERATURE["verifier"]
    max_tokens = LLM_MAX_TOKENS["verifier"]
    detective_findings = state.get("detective_findings") or {}
    analyst_findings = state.get("analyst_findings") or {}
    try:
        user_prompt = format_verifier_prompt(
            transaction=state["transaction_features"],
            xgboost_prediction=state["xgboost_prediction"],
            detective_findings=detective_findings,
            analyst_findings=analyst_findings,
        )
    except Exception as e:
        logger.error(f"[{agent_name}] Prompt formatting failed: {e}")
        findings = _verifier_fallback(f"Prompt error: {e}", 0.0)
        return update_verifier_findings(state, findings)
    raw_response, elapsed_ms = call_llm(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        agent_name=agent_name,
    )
    if raw_response is None:
        findings = _verifier_fallback("LLM call failed after retries", elapsed_ms)
        return update_verifier_findings(state, findings)
    parsed = _parse_json_response(raw_response, "verifier")
    if parsed is None:
        findings = _verifier_fallback("JSON parse failed", elapsed_ms)
        return update_verifier_findings(state, findings)
    findings = VerifierFindings(
        agent_name=agent_name,
        final_decision=parsed.get("final_decision", "UNCERTAIN"),
        confidence=parsed.get("confidence", 0.0),
        reasoning_chain=parsed.get("reasoning_chain", []),
        false_positive_check=parsed.get("false_positive_check", {}),
        recommended_action=parsed.get("recommended_action", "ESCALATE_TO_HITL"),
        execution_time_ms=elapsed_ms,
    )
    logger.info(
        f"[{agent_name}] Decision: {findings.final_decision} | action={findings.recommended_action} | confidence={findings.confidence:.2%} | {elapsed_ms:.0f}ms"
    )
    return update_verifier_findings(state, findings)


def route_after_detective(state: FraudInvestigationState) -> str:
    if not state.get("should_continue", True):
        logger.info("[Router] Detective was certain — ending early")
        return "end"
    detective = state.get("detective_findings") or {}
    num_red_flags = len(detective.get("red_flags", []))
    confidence = detective.get("confidence", 0.5)
    if should_run_analyst(confidence, num_red_flags):
        logger.info("[Router] Detective -> Analyst")
        return "analyst"
    else:
        logger.info("[Router] Detective -> Verifier (skipping Analyst)")
        return "verifier"


def route_after_analyst(state: FraudInvestigationState) -> str:
    logger.info("[Router] Analyst -> Verifier")
    return "verifier"


def test_connectivity() -> Dict[str, Any]:
    logger.info("Testing OpenAI connectivity...")
    start = time.time()
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": 'Respond with {"status": "ok"}'}],
            temperature=0.0,
            max_tokens=20,
            timeout=5.0,
            response_format={"type": "json_object"},
        )
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"Connectivity test passed in {elapsed_ms:.0f}ms")
        return {
            "success": True,
            "model": "gpt-4o-mini",
            "response": response.choices[0].message.content,
            "latency_ms": elapsed_ms,
            "tokens_used": response.usage.total_tokens,
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "hint": "Set OPENAI_API_KEY environment variable",
        }
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Connectivity test failed: {e}")
        return {"success": False, "error": str(e), "latency_ms": elapsed_ms}
