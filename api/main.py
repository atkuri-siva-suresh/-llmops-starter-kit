"""
FastAPI Inference Gateway — REST API for LLMOps Starter Kit.

Endpoints:
  POST /generate              — Run inference with a named prompt
  GET  /metrics               — Prometheus metrics (text format)
  GET  /metrics/summary       — Human-readable JSON metrics
  GET  /experiments           — Recent MLflow runs
  POST /ab-test               — Run A/B test between two prompts
  GET  /prompts               — List available prompt templates
  GET  /prompts/{name}        — Get single prompt template
  GET  /drift                 — Current drift detection report
  GET  /alerts/active         — Firing alerts
  GET  /alerts/history        — Last 50 alert events
  POST /alerts/rules          — Add alert rule
  DELETE /alerts/rules/{name} — Remove alert rule
  POST /alerts/acknowledge/{name} — Acknowledge a firing alert
  GET  /health                — Service health + stats
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import config
from src.gateway import get_gateway
from src.registry import PromptRegistry
from src.tracker import MLflowTracker
from src.monitor import get_metrics_history
from src.alerts import AlertEngine, AlertRule, AlertSeverity, get_alert_engine


app = FastAPI(
    title="LLMOps Starter Kit — Inference Gateway",
    description="Production LLM monitoring, prompt versioning, and A/B testing API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Module-level singletons
_registry = PromptRegistry()
_tracker = MLflowTracker()
_history = get_metrics_history()
_alerts = get_alert_engine()
_gateway = get_gateway()


# ── Pydantic request/response models ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt_name: str
    variables: Dict[str, str]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    answer: str
    prompt_name: str
    prompt_version: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    timestamp: str
    error: Optional[str] = None


class ABTestRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    context: str
    questions: Optional[List[str]] = None
    log_mlflow: bool = True


class AddAlertRuleRequest(BaseModel):
    name: str
    description: str = ""
    metric_name: str
    condition: str
    threshold: float
    severity: str = "warning"
    cooldown_seconds: int = 300


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service health check with gateway stats."""
    try:
        config.validate()
        api_key_ok = True
    except ValueError:
        api_key_ok = False

    return {
        "status": "healthy" if api_key_ok else "degraded",
        "api_key_configured": api_key_ok,
        "model": config.GROQ_MODEL,
        "gateway_stats": _gateway.get_stats_snapshot(),
        "prompts_loaded": len(_registry.list_prompts()),
        "active_alerts": len(_alerts.get_active_alerts()),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Run inference with a named prompt template."""
    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        prompt = _registry.get(request.prompt_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        rendered = prompt.render(**request.variables)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = _gateway.generate(
        prompt_name=request.prompt_name,
        rendered_prompt=rendered,
        prompt_version=prompt.version,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    if not result.success:
        raise HTTPException(status_code=502, detail=f"LLM error: {result.error}")

    return GenerateResponse(
        answer=result.answer,
        prompt_name=result.prompt_name,
        prompt_version=result.prompt_version,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        timestamp=result.timestamp,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-format metrics endpoint. Scrape this with Prometheus or Grafana."""
    return _gateway.get_prometheus_metrics()


@app.get("/metrics/summary")
async def metrics_summary():
    """Human-readable JSON metrics snapshot."""
    return _gateway.get_stats_snapshot()


@app.get("/prompts")
async def list_prompts():
    """List all available prompt templates."""
    try:
        prompts = _registry.load_all()
        return {
            name: {
                "version": pt.version,
                "description": pt.description,
                "task_type": pt.task_type,
                "variables": pt.variables,
            }
            for name, pt in prompts.items()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/prompts/{name}")
async def get_prompt(name: str):
    """Get a single prompt template including the full template text."""
    try:
        pt = _registry.get(name)
        return {
            "name": pt.name,
            "version": pt.version,
            "description": pt.description,
            "task_type": pt.task_type,
            "variables": pt.variables,
            "template": pt.template,
            "author": pt.author,
            "created_at": pt.created_at,
            "tags": pt.tags,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/experiments")
async def list_experiments(n: int = 20):
    """List recent MLflow experiment runs."""
    return _tracker.get_run_history(n)


@app.post("/ab-test")
async def run_ab_test(request: ABTestRequest):
    """
    Run A/B test between two prompt versions.
    Uses the built-in test question set from test_qa.json if no questions provided.
    """
    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # This is a stub for the API — the full A/B test logic is in experiments/run_ab_test.py
    # The API returns the prompt templates and test configuration
    try:
        pt_a = _registry.get(request.prompt_a)
        pt_b = _registry.get(request.prompt_b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "message": "A/B test configured. Run experiments/run_ab_test.py for full statistical results.",
        "prompt_a": {"name": pt_a.name, "version": pt_a.version},
        "prompt_b": {"name": pt_b.name, "version": pt_b.version},
        "questions_provided": len(request.questions) if request.questions else "using test_qa.json",
        "log_mlflow": request.log_mlflow,
    }


@app.get("/drift")
async def drift_report():
    """Current drift detection report for all tracked metrics."""
    if not _history.is_baseline_established():
        return {
            "status": "collecting_baseline",
            "message": _history.get_drift_status_string(),
            "summary": _history.get_summary(),
        }
    try:
        reports = _history.compute_drift()
        return {
            "status": "ok",
            "drift_detected": any(r.is_drifting for r in reports),
            "status_message": _history.get_drift_status_string(),
            "reports": [
                {
                    "metric": r.metric_name,
                    "z_score": r.z_score,
                    "is_drifting": r.is_drifting,
                    "direction": r.direction,
                    "baseline_mean": r.baseline_mean,
                    "current_window_mean": r.current_window_mean,
                }
                for r in reports
            ],
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/alerts/active")
async def active_alerts():
    """Return currently firing alerts."""
    return [
        {
            "rule_name": a.rule_name,
            "severity": a.severity.value,
            "metric": a.metric_name,
            "value": a.metric_value,
            "threshold": a.threshold,
            "message": a.message,
            "fired_at": a.fired_at,
        }
        for a in _alerts.get_active_alerts()
    ]


@app.get("/alerts/history")
async def alert_history(n: int = 50):
    """Last N alert events."""
    return [
        {
            "rule_name": a.rule_name,
            "severity": a.severity.value,
            "status": a.status.value,
            "metric": a.metric_name,
            "value": a.metric_value,
            "fired_at": a.fired_at,
            "resolved_at": a.resolved_at,
        }
        for a in _alerts.get_history(n)
    ]


@app.post("/alerts/rules")
async def add_alert_rule(request: AddAlertRuleRequest):
    """Add or replace an alert rule."""
    try:
        severity = AlertSeverity(request.severity)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity: {request.severity}. Use: info, warning, critical"
        )

    if request.condition not in ("above", "below"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid condition: {request.condition}. Use: above, below"
        )

    rule = AlertRule(
        name=request.name,
        description=request.description or f"Custom rule: {request.metric_name} {request.condition} {request.threshold}",
        metric_name=request.metric_name,
        condition=request.condition,
        threshold=request.threshold,
        severity=severity,
        cooldown_seconds=request.cooldown_seconds,
    )
    _alerts.add_rule(rule)
    return {"message": f"Rule '{request.name}' added successfully."}


@app.delete("/alerts/rules/{name}")
async def remove_alert_rule(name: str):
    """Remove an alert rule by name."""
    removed = _alerts.remove_rule(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Rule '{name}' not found.")
    return {"message": f"Rule '{name}' removed."}


@app.post("/alerts/acknowledge/{name}")
async def acknowledge_alert(name: str):
    """Acknowledge a firing alert by rule name."""
    resolved = _alerts.acknowledge_alert(name)
    if not resolved:
        return {"message": f"Alert '{name}' was not active (already resolved or doesn't exist)."}
    return {"message": f"Alert '{name}' acknowledged and resolved."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.GATEWAY_HOST, port=config.GATEWAY_PORT)
