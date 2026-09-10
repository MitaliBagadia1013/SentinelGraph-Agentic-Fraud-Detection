# SentinelGraph – Agentic Fraud Detection System

> A **5-layer, latency-tiered fraud decisioning pipeline** that routes each transaction to the cheapest layer that can confidently decide it — fast rules for the obvious cases, a calibrated XGBoost model for the bulk, and a **LangGraph multi-agent investigation** plus **human review** only for the ambiguous grey zone.

This is a portfolio/engineering project built to demonstrate **end-to-end ML systems design**: not just a model in a notebook, but the routing, latency budgets, graph analysis, feature store, agent orchestration, and human-in-the-loop feedback that a real fraud platform needs.

---

## Why this design?

A single model forces one tradeoff between latency, cost, and nuance. Real fraud traffic is heterogeneous: most transactions are *obviously* fine or *obviously* bad, and only a small grey zone needs expensive reasoning. This system **spends compute proportional to ambiguity**:

```
                          INCOMING TRANSACTION
                                   │
                                   ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 0 · Neo4j Graph Risk                     │
        │  "Is this account linked to a known fraud ring?"│
        └───────────────────────┬─────────────────────────┘
                                 ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 1 · Rules Engine            (<1 ms)      │
        │  Hard DECLINE / APPROVE / PASS_TO_ML            │  ← obvious cases exit here
        └───────────────────────┬─────────────────────────┘
                                 │ PASS_TO_ML
                                 ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 2 · Calibrated XGBoost   (sub-15 ms)     │
        │  419 engineered features → fraud probability    │  ← bulk decided here
        └───────────────────────┬─────────────────────────┘
              score ≥ 0.85 → DECLINE │ score < 0.20 → APPROVE
                     0.20 – 0.85 (grey zone) │
                                 ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 3 · LangGraph Agents     (~0.1–2 s)      │
        │  Detective → Analyst → Verifier                 │  ← only ambiguous cases
        │  (gpt-4o-mini)  (gpt-4o)   (gpt-4o)             │
        └───────────────────────┬─────────────────────────┘
                                 │ low confidence → escalate
                                 ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 4 · Human-in-the-Loop                    │
        │  Flask review UI · priority queue · SLA tracking│
        └───────────────────────┬─────────────────────────┘
                                 ▼
        ┌───────────────────────────────────────────────┐
        │  Layer 5 · Active-Learning Feedback             │
        │  Analyst labels → retrain queue (JSONL)         │
        └───────────────────────────────────────────────┘
```

---

## What's inside

| Layer | Component | Tech | Role |
|-------|-----------|------|------|
| 0 | **Graph risk** | Neo4j (Cypher) | Fraud-ring / shared-entity detection across accounts, devices, cards |
| 1 | **Rules engine** | Pure Python | Deterministic fast path (<1 ms) — instant DECLINE/APPROVE, else defer |
| 2 | **ML scoring** | XGBoost + scikit-learn | Calibrated gradient boosting over 419 engineered features |
| 3 | **Agentic investigation** | LangGraph + OpenAI | Detective → Analyst → Verifier, each with a distinct role & model tier |
| 4 | **Human review** | Flask + priority queue | Analyst dashboard with SLA tracking for escalated cases |
| 5 | **Feedback loop** | JSONL retrain queue | Captures analyst decisions to generate future training labels |
| — | **Feature store** | Redis (hot) + PostgreSQL (cold) | Low-latency feature reads with a cold-storage fallback |

**~16,000 lines of Python** across the pipeline, with **86 pytest tests** covering the rules engine, HITL queue (thread safety, priority ordering), and end-to-end integration with mocked agents.

---

## Results

Measured on the **IEEE-CIS Fraud Detection** dataset (590,540 real transactions, 3.5% fraud rate) with a calibrated XGBoost model evaluated on a stratified 20% hold-out (118,108 transactions the model never saw in training).

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.930** |
| Average precision (PR-AUC) | 0.677 |
| Precision @ 50% recall | 0.841 |
| Brier score (calibration) | 0.018 |
| False-positive rate @ 0.5 | 0.35% |

These land squarely in the range IEEE-CIS public kernels report (0.93–0.96 AUC) — not the inflated AUC = 1.0 that synthetic data produces, which is a data-leakage artifact rather than skill. Leakage was avoided deliberately:

- **Post-split aggregations** — per-card / per-address fraud rates are computed on the training fold only, so test labels never leak backward (and `card1_fraud_rate` still earns its place as a top-10 feature).
- **No label-derived columns** — nothing generated alongside the target is fed back as input.
- **Calibrated probabilities** (Platt scaling) so the 0.20 / 0.85 routing thresholds mean what they say.

Reproduce with `python data/load_ieee_cis.py && python models/train_ieee_xgboost.py --quick`. Full metrics are written to `models/saved/production/` as a metadata JSON on every run.

**Systems facts** (measured from the test suite and engineered pipeline):

| What | Value | Source |
|------|-------|--------|
| Rules-engine fast path | < 1 ms | `tests/test_rules_engine.py` |
| Test suite | 86 tests passing | `pytest` |
| Engineered features | 419 (IEEE-CIS) | `models/saved/production/` feature columns |

---

## Multi-agent investigation (Layer 3)

The grey zone (model score 0.20–0.85) is where most false positives hide. Instead of one expensive LLM call per transaction, a **LangGraph** workflow escalates only as far as needed:

| Agent | Model | Job |
|-------|-------|-----|
| **Detective** | `gpt-4o-mini` | Fast first pass — velocity/temporal anomalies; resolve or escalate |
| **Analyst** | `gpt-4o` | Forensic deep dive — behavioral profile + Neo4j graph links |
| **Verifier** | `gpt-4o` | Final decision + business rules; route to human if confidence is low |

Cheap model first, expensive models only on escalation — the same cost-control instinct as the layered pipeline, applied inside the agent graph.

---

## Quick start

**Prerequisites:** Python 3.9+, Docker (Redis + PostgreSQL), and an OpenAI API key for the agent layer.

```bash
# 1. Install
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env        # then add your OPENAI_API_KEY (and DB creds)

# 3. Start infrastructure (Redis + PostgreSQL)
docker-compose up -d

# 4. Train the model
#    On real data — download IEEE-CIS CSVs into data/raw/ first
#    (https://www.kaggle.com/competitions/ieee-fraud-detection/data):
python data/load_ieee_cis.py
python models/train_ieee_xgboost.py --quick
#    Or, with no Kaggle account, train on the bundled synthetic generator:
#    python data/generate_synthetic.py && python models/xgboost_trainer.py --quick

# 5. Run the pipeline over sample transactions
python main.py --simulate --n 100 --verbose
#   --no-agents  skip the LLM layer (no API key needed, much faster)
#   --health     run a health check across all layers

# 6. (Optional) Launch the human-review UI
cd hitl/ui && python app.py        # → http://localhost:5001
```

---

## Project structure

```
.
├── main.py                  # Pipeline entry point — wires all 5 layers together
├── agents/                  # LangGraph multi-agent system
│   ├── graph.py             #   Detective → Analyst → Verifier workflow
│   ├── agent_nodes.py       #   Per-agent logic + OpenAI calls
│   ├── prompts.py           #   Role-specific prompts
│   └── state.py             #   Shared graph state
├── rules_engine/            # Deterministic fast-path rules (<1 ms)
├── models/                  # XGBoost training, feature engineering, saved artifacts
├── graph_db/                # Neo4j client, loaders, fraud-ring queries
├── feature_store/           # Redis (hot) + PostgreSQL (cold) feature reads
├── hitl/                    # Human-in-the-loop queue + Flask review UI
├── data/                    # IEEE-CIS loader, synthetic generator, realtime sim
├── tests/                   # 86 pytest tests (rules, HITL, integration)
├── docker-compose.yml       # Redis + PostgreSQL
└── requirements.txt
```

---

## What this project demonstrates

- **ML systems design** — latency-tiered routing, cost-proportional compute, calibrated thresholds.
- **ML rigor** — feature engineering, *data-leakage detection and prevention*, probability calibration.
- **LLM agent orchestration** — LangGraph state machine, role specialization, escalation-based cost control.
- **Data engineering** — graph modeling in Neo4j, hot/cold feature store, leakage-safe feature pipelines.
- **Production concerns** — human-in-the-loop review, SLA-aware priority queues, active-learning feedback.
- **Software engineering** — 86 tests, containerized dependencies, clear module boundaries.

## Scope and limitations

This is a portfolio project, and it's honest about what it is and isn't:

- Runs locally on Docker Compose — there is **no cloud deployment** (no Terraform/SageMaker).
- The serving path is the batch/simulation entry point in `main.py`; a **REST API is not wired up** yet.
- **No live observability stack** (Prometheus/Grafana, automated drift detection) — monitoring is offline scripts.
- Metrics above are a single calibrated run in `--quick` mode; hyperparameter tuning (the non-`--quick` path) would push AUC higher.

---

## License

This project is for educational purposes only. It was built as a personal/portfolio exploration of agentic fraud-detection systems design. Please review the [IEEE-CIS Fraud Detection dataset terms](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) and the terms of service for OpenAI and Neo4j before reusing this code with your own data or in production.

## Author

**Mitali Bagadia**

- GitHub: github.com/MitaliBagadia1013
- LinkedIn: linkedin.com/in/mitalibagadia13
- Email: mitalibagadia13@gmail.com
