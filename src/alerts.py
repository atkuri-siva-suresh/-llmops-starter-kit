"""
Alert Rules Engine — evaluate metric thresholds and fire alert events.

Default rules cover the most critical LLM production failure modes:
  - Faithfulness below 0.70 → hallucination risk (CRITICAL)
  - Answer relevancy below 0.65 → off-topic answers (WARNING)
  - Latency above 3000ms → SLA breach (WARNING)
  - Context precision below 0.60 → retrieval failure (INFO)

Cooldown prevents alert storms: a rule can't re-fire within cooldown_seconds.
"""
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import config


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    OK = "ok"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    name: str
    description: str
    metric_name: str
    condition: str          # "below" | "above"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_seconds: int = 300


@dataclass
class AlertEvent:
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    metric_value: float
    threshold: float
    message: str
    fired_at: str
    resolved_at: Optional[str] = None

    def to_row(self) -> List:
        """Format for Gradio Dataframe row."""
        return [
            self.fired_at[:19],
            self.rule_name,
            self.metric_name,
            round(self.metric_value, 3),
            self.threshold,
            self.severity.value,
            self.status.value,
        ]


class AlertEngine:
    """
    Evaluates alert rules against incoming metrics.
    Maintains history in-memory with cooldown tracking.
    """

    DEFAULT_RULES: List[AlertRule] = [
        AlertRule(
            name="faithfulness_low",
            description="Answer faithfulness dropped below acceptable threshold — hallucination risk",
            metric_name="faithfulness",
            condition="below",
            threshold=config.DEFAULT_FAITHFULNESS_THRESHOLD,
            severity=AlertSeverity.CRITICAL,
        ),
        AlertRule(
            name="relevancy_degraded",
            description="Answer relevancy degraded — responses going off-topic",
            metric_name="answer_relevancy",
            condition="below",
            threshold=config.DEFAULT_RELEVANCY_THRESHOLD,
            severity=AlertSeverity.WARNING,
        ),
        AlertRule(
            name="latency_high",
            description="Inference latency exceeds SLA — user experience impact",
            metric_name="latency_ms",
            condition="above",
            threshold=float(config.DEFAULT_LATENCY_THRESHOLD_MS),
            severity=AlertSeverity.WARNING,
        ),
        AlertRule(
            name="context_precision_low",
            description="Retrieval precision degraded — check chunk strategy or embeddings",
            metric_name="context_precision",
            condition="below",
            threshold=0.60,
            severity=AlertSeverity.INFO,
        ),
    ]

    def __init__(self, rules: List[AlertRule] = None):
        import copy
        self.rules: List[AlertRule] = copy.deepcopy(
            rules if rules is not None else self.DEFAULT_RULES
        )
        self._history: List[AlertEvent] = []
        self._last_fired: Dict[str, float] = {}
        self._active: Dict[str, AlertEvent] = {}
        self._lock = threading.Lock()

    def evaluate(self, metrics: Dict[str, float]) -> List[AlertEvent]:
        """
        Run all enabled rules against metrics dict.
        Returns list of newly fired AlertEvents (respects cooldown).
        """
        fired = []
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                value = metrics.get(rule.metric_name)
                if value is None:
                    continue

                triggered = (
                    (rule.condition == "below" and value < rule.threshold) or
                    (rule.condition == "above" and value > rule.threshold)
                )

                if triggered:
                    last = self._last_fired.get(rule.name, 0)
                    if now_epoch - last < rule.cooldown_seconds:
                        continue  # Still in cooldown

                    msg = (
                        f"{rule.description}. "
                        f"Current: {round(value, 3)} | "
                        f"Threshold: {rule.threshold}"
                    )
                    event = AlertEvent(
                        rule_name=rule.name,
                        severity=rule.severity,
                        status=AlertStatus.FIRING,
                        metric_name=rule.metric_name,
                        metric_value=value,
                        threshold=rule.threshold,
                        message=msg,
                        fired_at=now_iso,
                    )
                    self._history.append(event)
                    self._active[rule.name] = event
                    self._last_fired[rule.name] = now_epoch
                    fired.append(event)
                else:
                    # Resolve if previously active
                    if rule.name in self._active:
                        self._active[rule.name].status = AlertStatus.RESOLVED
                        self._active[rule.name].resolved_at = now_iso
                        del self._active[rule.name]

        return fired

    def add_rule(self, rule: AlertRule) -> None:
        """Add or replace a rule by name."""
        with self._lock:
            self.rules = [r for r in self.rules if r.name != rule.name]
            self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove rule by name. Returns True if found."""
        with self._lock:
            before = len(self.rules)
            self.rules = [r for r in self.rules if r.name != name]
            return len(self.rules) < before

    def get_active_alerts(self) -> List[AlertEvent]:
        with self._lock:
            return list(self._active.values())

    def get_history(self, n: int = 50) -> List[AlertEvent]:
        with self._lock:
            return list(reversed(self._history[-n:]))

    def acknowledge_alert(self, rule_name: str) -> bool:
        """Manually resolve a firing alert."""
        with self._lock:
            if rule_name in self._active:
                self._active[rule_name].status = AlertStatus.RESOLVED
                self._active[rule_name].resolved_at = datetime.now(timezone.utc).isoformat()
                del self._active[rule_name]
                return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all active alerts. Returns count resolved."""
        names = list(self._active.keys())
        count = 0
        for name in names:
            if self.acknowledge_alert(name):
                count += 1
        return count

    def get_rules_summary(self) -> List[List]:
        """Return rules formatted for Gradio Dataframe."""
        with self._lock:
            active_names = set(self._active.keys())
            return [
                [
                    rule.name,
                    rule.metric_name,
                    rule.condition,
                    rule.threshold,
                    rule.severity.value,
                    "enabled" if rule.enabled else "disabled",
                    "FIRING" if rule.name in active_names else "OK",
                ]
                for rule in self.rules
            ]

    def get_summary_text(self) -> str:
        """One-line summary for UI status bar."""
        with self._lock:
            active = list(self._active.values())
        if not active:
            return "All clear — no alerts firing"
        by_sev: Dict[str, int] = {}
        for a in active:
            by_sev[a.severity.value] = by_sev.get(a.severity.value, 0) + 1
        parts = [f"{v} {k.upper()}" for k, v in sorted(by_sev.items())]
        return " | ".join(parts) + " firing"

    def get_history_as_table(self, n: int = 50) -> List[List]:
        """Return alert history formatted for Gradio Dataframe."""
        return [e.to_row() for e in self.get_history(n)]


# Module-level singleton
_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine
