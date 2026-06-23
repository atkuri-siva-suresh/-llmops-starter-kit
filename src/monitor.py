"""
Metrics Monitor — rolling window drift detection for LLM quality metrics.

Algorithm:
  1. First DRIFT_WINDOW_SIZE runs populate the baseline.
  2. Each subsequent run is compared against the baseline distribution.
  3. Z-score = (current_window_mean - baseline_mean) / baseline_std
  4. |z_score| > DRIFT_Z_THRESHOLD → drift alert (degraded or improved).

Thread-safe: uses threading.Lock for all state mutations.
"""
import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import config


@dataclass
class DriftReport:
    metric_name: str
    baseline_mean: float
    baseline_std: float
    current_window_mean: float
    z_score: float
    is_drifting: bool
    direction: str          # "degraded" | "improved" | "stable"
    window_size: int
    timestamp: str


TRACKED_METRICS = [
    "faithfulness",
    "context_precision",
    "answer_relevancy",
    "overall_score",
    "latency_ms",
]

# For latency, higher = worse (direction logic is inverted)
HIGHER_IS_WORSE = {"latency_ms"}


class MetricsHistory:
    """
    In-memory rolling history with automatic drift detection.

    Storage:
      - _baseline: dict[metric] → List[float] (populated once, never changes)
      - _window: dict[metric] → deque[maxlen=DRIFT_WINDOW_SIZE] (rolling)
      - _all_history: dict[metric] → List[float] (never drops, for charting)
    """

    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.DRIFT_WINDOW_SIZE
        self.z_threshold = config.DRIFT_Z_THRESHOLD
        self._lock = threading.Lock()
        self._baseline: Dict[str, List[float]] = {m: [] for m in TRACKED_METRICS}
        self._window: Dict[str, deque] = {
            m: deque(maxlen=self.window_size) for m in TRACKED_METRICS
        }
        self._all_history: Dict[str, List[float]] = {m: [] for m in TRACKED_METRICS}
        self._baseline_locked: bool = False
        self._total_runs: int = 0

    def add_run(self, metrics: Dict[str, float]) -> None:
        """
        Record one run's metrics.
        Auto-establishes baseline after window_size runs are collected.
        """
        with self._lock:
            for metric in TRACKED_METRICS:
                value = metrics.get(metric, 0.0)
                self._all_history[metric].append(value)
                self._window[metric].append(value)

            self._total_runs += 1

            # Lock baseline after first window fills
            if not self._baseline_locked and self._total_runs == self.window_size:
                for metric in TRACKED_METRICS:
                    self._baseline[metric] = list(self._all_history[metric])
                self._baseline_locked = True

    def compute_drift(self) -> List[DriftReport]:
        """
        Compare current rolling window against baseline.
        Raises RuntimeError if baseline not yet established.
        """
        if not self._baseline_locked:
            raise RuntimeError(
                f"Baseline not established yet. "
                f"Need {self.window_size} runs, have {self._total_runs}."
            )

        reports = []
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            for metric in TRACKED_METRICS:
                baseline_vals = self._baseline[metric]
                window_vals = list(self._window[metric])

                if len(baseline_vals) < 2:
                    continue

                baseline_mean = statistics.mean(baseline_vals)
                # Use stdev with fallback to avoid ZeroDivisionError on constant baseline
                try:
                    baseline_std = statistics.stdev(baseline_vals)
                except statistics.StatisticsError:
                    baseline_std = 0.0

                current_mean = statistics.mean(window_vals) if window_vals else baseline_mean

                if baseline_std > 1e-9:
                    z_score = (current_mean - baseline_mean) / baseline_std
                else:
                    z_score = 0.0

                is_drifting = abs(z_score) > self.z_threshold

                # Direction logic: for most metrics higher is better
                # For latency, higher is worse
                if metric in HIGHER_IS_WORSE:
                    direction = "degraded" if z_score > self.z_threshold else (
                        "improved" if z_score < -self.z_threshold else "stable"
                    )
                else:
                    direction = "degraded" if z_score < -self.z_threshold else (
                        "improved" if z_score > self.z_threshold else "stable"
                    )

                reports.append(DriftReport(
                    metric_name=metric,
                    baseline_mean=round(baseline_mean, 4),
                    baseline_std=round(baseline_std, 4),
                    current_window_mean=round(current_mean, 4),
                    z_score=round(z_score, 3),
                    is_drifting=is_drifting,
                    direction=direction,
                    window_size=len(window_vals),
                    timestamp=timestamp,
                ))

        return reports

    def get_chart_data(self, metric_name: str) -> Dict[str, Any]:
        """
        Return chart-ready data for Gradio gr.LinePlot.
        Includes baseline band lines for visual drift detection.
        """
        with self._lock:
            values = list(self._all_history.get(metric_name, []))
            baseline_vals = self._baseline.get(metric_name, [])

        if not values:
            return {"x": [], "y": [], "baseline_mean": None, "upper": None, "lower": None}

        if len(baseline_vals) >= 2:
            baseline_mean = statistics.mean(baseline_vals)
            try:
                baseline_std = statistics.stdev(baseline_vals)
            except statistics.StatisticsError:
                baseline_std = 0.0
            upper = baseline_mean + self.z_threshold * baseline_std
            lower = baseline_mean - self.z_threshold * baseline_std
        else:
            baseline_mean = upper = lower = None

        return {
            "x": list(range(1, len(values) + 1)),
            "y": values,
            "baseline_mean": round(baseline_mean, 4) if baseline_mean is not None else None,
            "upper_threshold": round(upper, 4) if upper is not None else None,
            "lower_threshold": round(lower, 4) if lower is not None else None,
            "metric": metric_name,
            "total_points": len(values),
        }

    def is_baseline_established(self) -> bool:
        return self._baseline_locked

    def get_summary(self) -> Dict[str, Any]:
        """Return current state summary for Gradio JSON display."""
        with self._lock:
            latest = {
                m: round(self._all_history[m][-1], 4)
                for m in TRACKED_METRICS
                if self._all_history[m]
            }
        return {
            "total_runs": self._total_runs,
            "baseline_established": self._baseline_locked,
            "baseline_window_size": self.window_size,
            "runs_until_baseline": max(0, self.window_size - self._total_runs),
            "metrics_tracked": TRACKED_METRICS,
            "latest_values": latest,
        }

    def reset(self) -> None:
        """Clear all history. Useful when switching to a new prompt from scratch."""
        with self._lock:
            self._baseline = {m: [] for m in TRACKED_METRICS}
            self._window = {m: deque(maxlen=self.window_size) for m in TRACKED_METRICS}
            self._all_history = {m: [] for m in TRACKED_METRICS}
            self._baseline_locked = False
            self._total_runs = 0

    def get_drift_status_string(self) -> str:
        """Return one-line status for UI display."""
        if not self._baseline_locked:
            remaining = self.window_size - self._total_runs
            return f"Collecting baseline... {remaining} more run(s) needed"
        try:
            reports = self.compute_drift()
            drifting = [r for r in reports if r.is_drifting]
            if not drifting:
                return "ALL METRICS STABLE"
            msgs = [f"{r.metric_name.upper()} ({r.direction})" for r in drifting]
            return "DRIFT DETECTED: " + ", ".join(msgs)
        except Exception as e:
            return f"Error computing drift: {e}"


# Module-level singleton
_history: Optional[MetricsHistory] = None


def get_metrics_history() -> MetricsHistory:
    global _history
    if _history is None:
        _history = MetricsHistory()
    return _history
