# LLMOps Starter Kit

> **Production LLM monitoring — prompt versioning, A/B testing, drift detection, and alert rules.**  
> Because deploying an LLM is the beginning, not the end.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-experiment_tracking-orange.svg)](https://mlflow.org)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-red.svg)](https://prometheus.io)
[![Groq](https://img.shields.io/badge/Groq-free_tier-purple.svg)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem This Solves

Every team that ships an LLM feature faces the same question on day 2: **is it still working?**

- Did the last prompt change make answers *better* or just *different*?
- Is response quality degrading as your document corpus grows?
- When faithfulness drops at 2 AM, who gets paged?

This kit gives you the infrastructure to answer those questions with data, not guesses.

---

## What's Inside

| Component | What it does |
|-----------|-------------|
| **Prompt Registry** | YAML-versioned prompt templates, git-trackable, diff-viewable |
| **Inference Gateway** | Single entry point for all LLM calls — Prometheus metrics on every request |
| **MLflow Tracker** | Experiment logging: prompt versions, eval metrics, A/B test outcomes |
| **Drift Detector** | Z-score rolling window — detects quality regressions across N runs |
| **Alert Engine** | Rule-based alerts: faithfulness low, latency high, relevancy degraded |
| **A/B Test Runner** | Mann-Whitney U statistical test between two prompt versions |
| **FastAPI Gateway** | REST API with `/metrics`, `/experiments`, `/drift`, `/alerts` endpoints |
| **Gradio UI** | 4-tab dashboard: Prompt Lab, A/B Test, Monitor, Alerts |
| **CI Pipeline** | GitHub Actions: unit tests on every PR, eval test on every merge to main |

---

## Architecture

```
                    YAML Prompt Files (git-versioned)
                              │
                    ┌─────────▼──────────┐
                    │   Prompt Registry  │  ← load, render, diff versions
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Inference Gateway │  ← every call logged to Prometheus
                    │  (Groq SDK direct) │    latency_ms, tokens, error_rate
                    └────┬──────────┬───┘
                         │          │
              ┌──────────▼──┐  ┌───▼──────────┐
              │  MLflow     │  │ Prometheus   │
              │  Tracker    │  │ /metrics     │
              │  (local FS) │  │ endpoint     │
              └──────────┬──┘  └─────────────┘
                         │
              ┌──────────▼────────────────────┐
              │    Metrics History             │
              │    (rolling window)            │
              │                                │
              │    Baseline (first N runs)     │
              │    Window (latest N runs)      │
              │    Z-score drift detection     │
              └──────────┬────────────────────┘
                         │
              ┌──────────▼────────────────────┐
              │    Alert Engine                │
              │                                │
              │    faithfulness < 0.70  → CRIT │
              │    latency > 3000ms     → WARN │
              │    relevancy < 0.65     → WARN │
              └───────────────────────────────┘
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/atkuri-siva-suresh/llmops-starter-kit.git
cd llmops-starter-kit

# Install torch FIRST with CPU wheel (avoids 2GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your free Groq key: GROQ_API_KEY=gsk_...
# Sign up at https://console.groq.com — no credit card
```

### 3. Pre-download embedding model (one time, ~2 min)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Model ready')"
```

### 4. Run

```bash
# Option A: Gradio UI (all 4 tabs)
python ui/app.py
# Opens at http://localhost:7862

# Option B: FastAPI inference gateway
uvicorn api.main:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
# Prometheus metrics at http://localhost:8000/metrics

# Option C: MLflow experiment tracking UI
mlflow ui --port 5000
# View all A/B test runs at http://localhost:5000

# Option D: CLI A/B test
python experiments/run_ab_test.py --prompt-a rag_qa_v1 --prompt-b rag_qa_v2

# Option E: Unit tests (no API key, ~10s)
pytest tests/ -v
```

---

## Prompt Registry

Prompts are YAML files in the `prompts/` directory. Each file is a versioned template:

```yaml
name: rag_qa_v1
version: "1.0.0"
description: "Baseline RAG QA — verbose, citation-focused"
task_type: rag_qa
variables: [context, question]
template: |
  You are a precise question-answering assistant.
  Use ONLY the context provided below to answer the question.
  ...
```

**Why YAML?**
- Human-readable and git-diffable
- Each commit = one prompt version
- `git blame prompts/rag_qa_v1.yaml` shows who changed what and when
- The Prompt Lab tab lets you render, compare diffs, and generate answers without leaving the UI

---

## A/B Testing

```bash
python experiments/run_ab_test.py \
  --prompt-a rag_qa_v1 \
  --prompt-b rag_qa_v2 \
  --n-questions 10 \
  --fail-on-regression
```

Uses **Mann-Whitney U test** (non-parametric, correct for small samples and non-normal score distributions). Reports:

- Per-question scores for both prompts
- p-value and statistical significance (p < 0.05)
- Effect size (rank-biserial correlation, range: -1 to +1)
- Winner declaration with recommendation
- MLflow run logged automatically

**CI integration:** `--fail-on-regression` exits with code 1 if Prompt B is significantly worse, blocking the merge.

---

## Drift Detection

The drift detector compares a rolling window of recent runs against the baseline:

```
Baseline: first DRIFT_WINDOW_SIZE (default: 5) evaluation runs
Window: latest DRIFT_WINDOW_SIZE runs (rolling)

z_score = (window_mean - baseline_mean) / baseline_std
|z_score| > 1.5σ → DRIFT DETECTED
```

- `direction = "degraded"` → score dropped (red alert)
- `direction = "improved"` → score rose unexpectedly (yellow — investigate)
- For latency: higher is always worse (direction logic inverted)

---

## Alert Rules

Four default rules ship out of the box:

| Rule | Metric | Condition | Severity |
|------|--------|-----------|----------|
| `faithfulness_low` | faithfulness | < 0.70 | CRITICAL |
| `relevancy_degraded` | answer_relevancy | < 0.65 | WARNING |
| `latency_high` | latency_ms | > 3000 | WARNING |
| `context_precision_low` | context_precision | < 0.60 | INFO |

Add custom rules via the Alerts tab or via API:

```bash
curl -X POST http://localhost:8000/alerts/rules \
  -H "Content-Type: application/json" \
  -d '{"name": "custom_rule", "metric_name": "faithfulness", "condition": "below", "threshold": 0.80, "severity": "warning"}'
```

---

## Prometheus Metrics

The `/metrics` endpoint (FastAPI) returns Prometheus text format:

```
# HELP llm_requests_total Total inference requests
llm_requests_total{prompt_name="rag_qa_v1",status="success"} 47.0
llm_requests_total{prompt_name="rag_qa_v2",status="success"} 47.0

# HELP llm_latency_seconds End-to-end inference latency in seconds
llm_latency_seconds_bucket{prompt_name="rag_qa_v1",le="1.0"} 12.0
llm_latency_seconds_bucket{prompt_name="rag_qa_v1",le="2.0"} 38.0

# HELP llm_tokens_total Total tokens consumed
llm_tokens_total{prompt_name="rag_qa_v1",token_type="completion"} 1840.0
llm_tokens_total{prompt_name="rag_qa_v1",token_type="prompt"} 8920.0
```

Scrape this with Prometheus + Grafana for production dashboards.

---

## Port Map

| Service | Port |
|---------|------|
| Gradio UI | 7862 |
| FastAPI Gateway | 8000 |
| MLflow UI | 5000 |

---

## Project Structure

```
llmops-starter-kit/
├── src/
│   ├── config.py       # All environment variables and thresholds
│   ├── registry.py     # YAML prompt versioning (load, render, diff)
│   ├── gateway.py      # Inference gateway with Prometheus instrumentation
│   ├── tracker.py      # MLflow experiment logging
│   ├── monitor.py      # Rolling window drift detection
│   ├── alerts.py       # Alert rules engine
│   └── evaluator.py    # Context precision, faithfulness, answer relevancy
├── api/
│   └── main.py         # FastAPI REST API (14 endpoints)
├── ui/
│   └── app.py          # Gradio 4-tab dashboard
├── prompts/
│   ├── rag_qa_v1.yaml  # Baseline: verbose, citation-focused
│   ├── rag_qa_v2.yaml  # Challenger: concise, structured
│   └── summarizer_v1.yaml
├── data/samples/       # Enterprise AI policy + RAG FAQ + 10 Q&A pairs
├── experiments/
│   └── run_ab_test.py  # CLI A/B test with Mann-Whitney U statistics
├── .github/workflows/
│   └── eval-ci.yml     # Unit tests on PR + eval on merge to main
└── tests/
    └── test_llmops.py  # 30+ unit tests (no API key needed)
```

---

## Key Design Decisions

**Why `groq` SDK directly instead of `langchain-groq`?**  
This is a gateway layer — using the native SDK keeps the dependency surface minimal, gives exact token counts from `response.usage` (LangChain wraps these inconsistently), and shows familiarity with provider SDKs beyond framework abstractions.

**Why Mann-Whitney U instead of t-test for A/B comparison?**  
LLM metric scores (faithfulness: 0.0–1.0) are bounded, non-normal distributions. Mann-Whitney is non-parametric — it makes no normality assumption and works correctly for small samples (n=5–10 questions). A t-test on 5 data points from a bounded distribution is statistically incorrect.

**Why z-score for drift detection instead of just threshold comparison?**  
Thresholds are brittle — a model that consistently scores 0.68 faithfulness shouldn't alert at 0.69. Z-score adapts to your baseline distribution: if your model is highly variable, a 0.05 drop isn't surprising; if it's rock-solid, it is. This is how production ML monitoring works (see: evidently.ai, Arize, Fiddler).

**Why YAML for prompt templates instead of a database?**  
Git is your database. YAML is human-readable, diffable, and PR-reviewable. "Who changed this prompt and when?" is answered by `git blame`. This maps to how teams like Anthropic and Glean actually manage prompts in production.

**Why Prometheus client library instead of custom counters?**  
`prometheus_client` produces a proper `/metrics` endpoint that Prometheus and Grafana can scrape directly. It adds one pip dependency and zero infrastructure requirement. In production, you point Prometheus at this endpoint — no code change needed.

---

## Running Tests

```bash
# Unit tests — no API key, no model download (~10s)
pytest tests/ -v

# Integration tests — requires GROQ_API_KEY
pytest tests/ -v -m integration
```

---

## Author

**A.S. Siva Suresh Kumar**  
Enterprise AI Architect | AI Transformation Lead  
20+ years in AI/ML | Former Head of AI, TVS Motor Company  
[LinkedIn](https://linkedin.com/in/siva-suresh-kumar-97725b78) · [GitHub](https://github.com/atkuri-siva-suresh)
