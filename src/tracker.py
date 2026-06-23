"""
MLflow Experiment Tracker — logs prompt versions, evaluation metrics, and A/B test outcomes.

Pattern: one MLflow experiment per project, one run per evaluation.
Prompt versions are stored as run TAGS (not params) to avoid polluting hyperparameter space.
"""
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.tracking

from src.config import config


class MLflowTracker:
    """
    Wraps MLflow for LLMOps-specific logging patterns.

    Usage:
        tracker = MLflowTracker()
        with tracker.start_run("eval-rag_qa_v1-2026-06-23"):
            tracker.log_prompt_version("rag_qa_v1", "1.0.0", template_text)
            tracker.log_eval_metrics({"faithfulness": 0.82, "overall_score": 0.78})
        # Run automatically ended at context manager exit
    """

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        self._active_run: Optional[mlflow.ActiveRun] = None

    @contextmanager
    def start_run(self, run_name: str, tags: Dict[str, str] = None):
        """Context manager that starts an MLflow run and ends it on exit."""
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self._active_run = run
        try:
            yield run.info.run_id
        finally:
            mlflow.end_run()
            self._active_run = None

    def log_prompt_version(
        self, prompt_name: str, version: str, template_text: str
    ) -> None:
        """Log prompt as MLflow run tag and text artifact."""
        mlflow.set_tags({
            "prompt.name": prompt_name,
            "prompt.version": version,
        })
        # Log the template text as a text artifact for auditability
        artifact_path = f"prompts/{prompt_name}_v{version}.txt"
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None, nested=True) if not mlflow.active_run() else _noop():
            mlflow.log_text(template_text, artifact_path)

    def log_eval_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log evaluation metrics. Expected keys: faithfulness, context_precision, etc."""
        mlflow.log_metrics(metrics, step=step)

    def log_inference_stats(self, stats: Dict[str, Any]) -> None:
        """Log gateway stats snapshot as MLflow params."""
        mlflow.log_params({
            str(k): str(v) for k, v in stats.items()
        })

    def log_ab_test_result(
        self,
        prompt_a: str,
        prompt_b: str,
        winner: str,
        p_value: float,
        significant: bool,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
    ) -> None:
        """Log A/B test outcome with statistical results."""
        mlflow.set_tags({
            "ab_test.prompt_a": prompt_a,
            "ab_test.prompt_b": prompt_b,
            "ab_test.winner": winner,
            "ab_test.significant": str(significant),
        })
        mlflow.log_param("ab_test.p_value", round(p_value, 6))

        # Log per-prompt metrics with prefix
        for key, val in metrics_a.items():
            mlflow.log_metric(f"a_{key}", val)
        for key, val in metrics_b.items():
            mlflow.log_metric(f"b_{key}", val)

    def get_best_run(self, metric: str = "overall_score") -> Optional[Dict[str, Any]]:
        """Find run with best value for given metric. Returns None if no runs exist."""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            if not runs:
                return None

            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "prompt_name": run.data.tags.get("prompt.name", ""),
                "prompt_version": run.data.tags.get("prompt.version", ""),
                "metric_name": metric,
                "metric_value": run.data.metrics.get(metric),
            }
        except Exception:
            return None

    def get_run_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return last N runs as list of dicts for display."""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n,
            )

            result = []
            for run in runs:
                result.append({
                    "run_id": run.info.run_id[:8],
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "prompt": run.data.tags.get("prompt.name", ""),
                    "faithfulness": round(run.data.metrics.get("faithfulness", 0), 3),
                    "context_precision": round(run.data.metrics.get("context_precision", 0), 3),
                    "answer_relevancy": round(run.data.metrics.get("answer_relevancy", 0), 3),
                    "overall_score": round(run.data.metrics.get("overall_score", 0), 3),
                    "latency_ms": int(run.data.metrics.get("latency_ms", 0)),
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).strftime("%Y-%m-%d %H:%M"),
                })
            return result
        except Exception:
            return []

    def get_run_history_as_table(self, n: int = 20) -> List[List]:
        """Return run history formatted for Gradio Dataframe."""
        history = self.get_run_history(n)
        return [
            [
                r["run_id"],
                r["run_name"],
                r["prompt"],
                r["faithfulness"],
                r["context_precision"],
                r["overall_score"],
                r["latency_ms"],
                r["start_time"],
            ]
            for r in history
        ]

    def get_metric_series(self, metric: str, n: int = 30) -> List[float]:
        """Extract a metric time series from the last N runs for drift chart."""
        history = self.get_run_history(n)
        key_map = {
            "faithfulness": "faithfulness",
            "context_precision": "context_precision",
            "answer_relevancy": "answer_relevancy",
            "overall_score": "overall_score",
            "latency_ms": "latency_ms",
        }
        key = key_map.get(metric, metric)
        return [r.get(key, 0) for r in reversed(history)]


class _noop:
    """No-op context manager for conditional 'with' blocks."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
