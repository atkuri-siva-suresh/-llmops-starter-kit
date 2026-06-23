"""
Inference Gateway — single entry point for all LLM calls.

Every request is instrumented with Prometheus metrics:
  - llm_requests_total (Counter, by prompt_name + status)
  - llm_latency_seconds (Histogram, by prompt_name)
  - llm_tokens_total (Counter, by prompt_name + token_type)
  - llm_error_rate (Gauge, rolling 100-request window)
  - llm_active_requests (Gauge, in-flight count)

Uses the groq SDK directly (not LangChain) for exact token count access.
"""
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

from src.config import config


# ── Prometheus metrics (module-level — must not be inside __init__) ───────────
# Declared with try/except to survive hot-reload in Gradio dev mode
def _get_or_create_counter(name, doc, labelnames=None):
    try:
        return Counter(name, doc, labelnames=labelnames or [])
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_histogram(name, doc, labelnames=None, buckets=None):
    try:
        kwargs = {}
        if labelnames:
            kwargs["labelnames"] = labelnames
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, doc, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_gauge(name, doc):
    try:
        return Gauge(name, doc)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


REQUEST_COUNT = _get_or_create_counter(
    "llm_requests_total",
    "Total inference requests",
    labelnames=["prompt_name", "status"],
)
LATENCY_HISTOGRAM = _get_or_create_histogram(
    "llm_latency_seconds",
    "End-to-end inference latency in seconds",
    labelnames=["prompt_name"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
TOKEN_COUNTER = _get_or_create_counter(
    "llm_tokens_total",
    "Total tokens consumed",
    labelnames=["prompt_name", "token_type"],
)
ERROR_GAUGE = _get_or_create_gauge(
    "llm_error_rate",
    "Fraction of last 100 requests that failed",
)
ACTIVE_REQUESTS = _get_or_create_gauge(
    "llm_active_requests",
    "Currently in-flight inference requests",
)


@dataclass
class InferenceResult:
    prompt_name: str
    prompt_version: str
    rendered_prompt: str
    answer: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class InferenceGateway:
    """
    Single entry point for all LLM calls.
    Logs every request to Prometheus metrics.

    Usage:
        gateway = InferenceGateway()
        result = gateway.generate(
            prompt_name="rag_qa_v1",
            rendered_prompt="You are... Context: ... Question: ...",
            prompt_version="1.0.0",
        )
    """

    def __init__(self):
        # Lazy import so tests can run without GROQ_API_KEY
        self._client = None
        self._lock = threading.Lock()
        self._recent_statuses: deque = deque(maxlen=100)
        # Track totals for stats snapshot (Prometheus counters don't expose values easily)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency_ms = 0

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY)
        return self._client

    def generate(
        self,
        prompt_name: str,
        rendered_prompt: str,
        prompt_version: str = "unknown",
        temperature: float = None,
        max_tokens: int = None,
    ) -> InferenceResult:
        """
        Call Groq API, record Prometheus metrics, return InferenceResult.
        Thread-safe: Prometheus uses internal locks, deque is maxlen-bounded.
        """
        if ACTIVE_REQUESTS:
            ACTIVE_REQUESTS.inc()

        start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": rendered_prompt}],
                temperature=temperature if temperature is not None else config.GROQ_TEMPERATURE,
                max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
            )

            latency_s = time.perf_counter() - start
            latency_ms = int(latency_s * 1000)
            answer = response.choices[0].message.content or ""
            usage = response.usage

            # ── Update Prometheus ─────────────────────────────────────────────
            if REQUEST_COUNT:
                REQUEST_COUNT.labels(prompt_name=prompt_name, status="success").inc()
            if LATENCY_HISTOGRAM:
                LATENCY_HISTOGRAM.labels(prompt_name=prompt_name).observe(latency_s)
            if TOKEN_COUNTER:
                TOKEN_COUNTER.labels(prompt_name=prompt_name, token_type="prompt").inc(
                    usage.prompt_tokens
                )
                TOKEN_COUNTER.labels(prompt_name=prompt_name, token_type="completion").inc(
                    usage.completion_tokens
                )

            # ── Update local stats ────────────────────────────────────────────
            with self._lock:
                self._recent_statuses.append(True)
                self._total_requests += 1
                self._total_tokens += usage.total_tokens
                self._total_latency_ms += latency_ms
            self._update_error_rate(True)

            return InferenceResult(
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                rendered_prompt=rendered_prompt,
                answer=answer,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                model=config.GROQ_MODEL,
                timestamp=timestamp,
                error=None,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            if REQUEST_COUNT:
                REQUEST_COUNT.labels(prompt_name=prompt_name, status="error").inc()
            with self._lock:
                self._recent_statuses.append(False)
                self._total_requests += 1
                self._total_latency_ms += latency_ms
            self._update_error_rate(False)
            return InferenceResult(
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                rendered_prompt=rendered_prompt,
                answer="",
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=config.GROQ_MODEL,
                timestamp=timestamp,
                error=str(e),
            )

        finally:
            if ACTIVE_REQUESTS:
                ACTIVE_REQUESTS.dec()

    def get_prometheus_metrics(self) -> str:
        """Return raw Prometheus text format for the /metrics endpoint."""
        return generate_latest().decode("utf-8")

    def get_stats_snapshot(self) -> dict:
        """Return human-readable stats for Gradio display."""
        with self._lock:
            total = self._total_requests
            tokens = self._total_tokens
            total_latency = self._total_latency_ms
            recent = list(self._recent_statuses)

        errors = sum(1 for s in recent if not s)
        error_rate = round(errors / len(recent) * 100, 1) if recent else 0.0
        mean_latency = round(total_latency / total) if total > 0 else 0

        return {
            "total_requests": total,
            "total_tokens": tokens,
            "mean_latency_ms": mean_latency,
            "error_rate_pct": error_rate,
            "recent_window_size": len(recent),
        }

    def _update_error_rate(self, success: bool) -> None:
        """Maintain rolling 100-request error rate in the ERROR_GAUGE."""
        recent = list(self._recent_statuses)
        if recent and ERROR_GAUGE:
            rate = sum(1 for s in recent if not s) / len(recent)
            ERROR_GAUGE.set(rate)


# Module-level singleton — shared across Gradio tabs and FastAPI handlers
_gateway: Optional[InferenceGateway] = None


def get_gateway() -> InferenceGateway:
    global _gateway
    if _gateway is None:
        _gateway = InferenceGateway()
    return _gateway
