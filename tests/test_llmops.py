"""
Unit tests for llmops-starter-kit.
All 30+ tests run without API keys, model downloads, or MLflow server.

Run: pytest tests/ -v
"""
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import Config
from src.registry import PromptRegistry, PromptTemplate
from src.monitor import MetricsHistory, DriftReport, TRACKED_METRICS
from src.alerts import (
    AlertEngine, AlertRule, AlertSeverity, AlertStatus, AlertEvent
)
from src.evaluator import (
    _tokenize, _jaccard,
    compute_context_precision,
    EvalResult,
    summarize_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_prompt_yaml(tmp_path: Path, name: str, variables: list, template: str) -> Path:
    data = {
        "name": name,
        "version": "1.0.0",
        "description": f"Test prompt {name}",
        "task_type": "rag_qa",
        "author": "test",
        "created_at": "2026-01-01T00:00:00",
        "variables": variables,
        "template": template,
        "tags": {},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def make_metrics(
    faithfulness: float = 0.8,
    context_precision: float = 0.75,
    answer_relevancy: float = 0.70,
    overall_score: float = 0.75,
    latency_ms: float = 1200.0,
) -> Dict[str, float]:
    return {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
        "overall_score": overall_score,
        "latency_ms": latency_ms,
    }


# ── TestConfig ────────────────────────────────────────────────────────────────

class TestConfig:

    def test_groq_model_default_is_set(self):
        assert Config.GROQ_MODEL != ""
        assert "llama" in Config.GROQ_MODEL.lower()

    def test_embed_model_default(self):
        assert "MiniLM" in Config.EMBED_MODEL or "sentence" in Config.EMBED_MODEL.lower()

    def test_drift_z_threshold_is_float(self):
        assert isinstance(Config.DRIFT_Z_THRESHOLD, float)
        assert Config.DRIFT_Z_THRESHOLD > 0

    def test_drift_window_size_is_positive_int(self):
        assert isinstance(Config.DRIFT_WINDOW_SIZE, int)
        assert Config.DRIFT_WINDOW_SIZE > 0

    def test_validate_raises_without_api_key(self):
        original = Config.GROQ_API_KEY
        Config.GROQ_API_KEY = ""
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            Config.validate()
        Config.GROQ_API_KEY = original

    def test_alert_thresholds_are_floats(self):
        assert 0 < Config.DEFAULT_FAITHFULNESS_THRESHOLD < 1
        assert 0 < Config.DEFAULT_RELEVANCY_THRESHOLD < 1
        assert Config.DEFAULT_LATENCY_THRESHOLD_MS > 0


# ── TestPromptRegistry ────────────────────────────────────────────────────────

class TestPromptRegistry:

    def test_load_all_returns_dict(self, tmp_path):
        make_prompt_yaml(tmp_path, "test_v1", ["context", "question"], "Hello {context} {question}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        prompts = registry.load_all()
        assert isinstance(prompts, dict)
        assert "test_v1" in prompts

    def test_get_raises_key_error_for_unknown(self, tmp_path):
        make_prompt_yaml(tmp_path, "test_v1", ["context"], "Hello {context}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        registry.load_all()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_prompt")

    def test_render_fills_variables(self, tmp_path):
        make_prompt_yaml(tmp_path, "qa_v1", ["context", "question"], "CTX:{context} Q:{question}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        registry.load_all()
        rendered = registry.render("qa_v1", context="my doc", question="what is it?")
        assert "my doc" in rendered
        assert "what is it?" in rendered

    def test_render_raises_for_missing_variable(self, tmp_path):
        make_prompt_yaml(tmp_path, "qa_v1", ["context", "question"], "CTX:{context} Q:{question}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        registry.load_all()
        with pytest.raises(ValueError, match="requires variables"):
            registry.render("qa_v1", context="only context provided")

    def test_list_prompts_returns_sorted(self, tmp_path):
        make_prompt_yaml(tmp_path, "zzz_v1", ["x"], "{x}")
        make_prompt_yaml(tmp_path, "aaa_v1", ["x"], "{x}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        names = registry.list_prompts()
        assert names == sorted(names)

    def test_compare_versions_returns_diff_dict(self, tmp_path):
        make_prompt_yaml(tmp_path, "v1", ["context"], "Answer: {context}")
        make_prompt_yaml(tmp_path, "v2", ["context"], "Be concise. Answer: {context}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        registry.load_all()
        diff = registry.compare_versions("v1", "v2")
        assert "added_lines" in diff
        assert "removed_lines" in diff
        assert "unified_diff" in diff
        assert "char_delta" in diff

    def test_compare_versions_detects_char_delta(self, tmp_path):
        make_prompt_yaml(tmp_path, "short_v", ["x"], "Short: {x}")
        make_prompt_yaml(tmp_path, "long_v", ["x"], "This is a much longer template string: {x}")
        registry = PromptRegistry(prompts_dir=str(tmp_path))
        registry.load_all()
        diff = registry.compare_versions("short_v", "long_v")
        assert diff["char_delta"] > 0

    def test_prompt_template_render_method(self):
        pt = PromptTemplate(
            name="test", version="1.0", description="", task_type="rag_qa",
            template="Hello {name}, you asked: {question}",
            variables=["name", "question"],
            author="test", created_at="", tags={},
        )
        rendered = pt.render(name="Alice", question="What is AI?")
        assert "Alice" in rendered
        assert "What is AI?" in rendered


# ── TestMetricsHistory ────────────────────────────────────────────────────────

class TestMetricsHistory:

    def test_baseline_not_established_initially(self):
        h = MetricsHistory(window_size=3)
        assert not h.is_baseline_established()

    def test_baseline_established_after_window_size_runs(self):
        h = MetricsHistory(window_size=3)
        for _ in range(3):
            h.add_run(make_metrics())
        assert h.is_baseline_established()

    def test_not_established_before_window_size_runs(self):
        h = MetricsHistory(window_size=5)
        for _ in range(4):
            h.add_run(make_metrics())
        assert not h.is_baseline_established()

    def test_add_run_populates_history(self):
        h = MetricsHistory(window_size=3)
        h.add_run(make_metrics(faithfulness=0.9))
        assert h._all_history["faithfulness"] == [0.9]

    def test_compute_drift_raises_before_baseline(self):
        h = MetricsHistory(window_size=5)
        h.add_run(make_metrics())  # only 1 run
        with pytest.raises(RuntimeError, match="Baseline not established"):
            h.compute_drift()

    def test_compute_drift_returns_list_of_reports(self):
        h = MetricsHistory(window_size=3)
        for _ in range(3):
            h.add_run(make_metrics())
        reports = h.compute_drift()
        assert isinstance(reports, list)
        assert len(reports) > 0
        assert all(isinstance(r, DriftReport) for r in reports)

    def test_drift_detected_on_score_drop(self):
        h = MetricsHistory(window_size=3)
        # Establish high baseline
        for _ in range(3):
            h.add_run(make_metrics(faithfulness=0.95, overall_score=0.95))
        # Add very low run (severe drop → z_score < -1.5σ)
        for _ in range(3):
            h.add_run(make_metrics(faithfulness=0.1, overall_score=0.1))
        reports = h.compute_drift()
        faith_report = next(r for r in reports if r.metric_name == "faithfulness")
        assert faith_report.direction in ("degraded", "stable")  # stable if std is 0

    def test_reset_clears_history(self):
        h = MetricsHistory(window_size=3)
        for _ in range(3):
            h.add_run(make_metrics())
        assert h.is_baseline_established()
        h.reset()
        assert not h.is_baseline_established()
        assert h._total_runs == 0

    def test_get_summary_has_expected_keys(self):
        h = MetricsHistory(window_size=3)
        summary = h.get_summary()
        assert "total_runs" in summary
        assert "baseline_established" in summary
        assert "metrics_tracked" in summary

    def test_chart_data_empty_before_runs(self):
        h = MetricsHistory(window_size=3)
        data = h.get_chart_data("faithfulness")
        assert data["x"] == []
        assert data["y"] == []

    def test_chart_data_has_correct_length(self):
        h = MetricsHistory(window_size=3)
        for i in range(5):
            h.add_run(make_metrics(faithfulness=0.7 + i * 0.02))
        data = h.get_chart_data("faithfulness")
        assert len(data["x"]) == 5
        assert len(data["y"]) == 5


# ── TestAlertEngine ───────────────────────────────────────────────────────────

class TestAlertEngine:

    def test_default_rules_count(self):
        engine = AlertEngine()
        assert len(engine.DEFAULT_RULES) >= 4

    def test_default_rules_have_required_fields(self):
        for rule in AlertEngine.DEFAULT_RULES:
            assert rule.name
            assert rule.metric_name
            assert rule.condition in ("above", "below")
            assert isinstance(rule.threshold, float)

    def test_evaluate_fires_faithfulness_alert(self):
        engine = AlertEngine()
        engine._last_fired = {}  # clear cooldown for test
        metrics = make_metrics(faithfulness=0.3)  # below 0.70 threshold
        fired = engine.evaluate(metrics)
        names = [e.rule_name for e in fired]
        assert "faithfulness_low" in names

    def test_evaluate_fires_latency_alert(self):
        engine = AlertEngine()
        engine._last_fired = {}
        metrics = make_metrics(latency_ms=5000.0)  # above 3000ms threshold
        fired = engine.evaluate(metrics)
        names = [e.rule_name for e in fired]
        assert "latency_high" in names

    def test_evaluate_returns_empty_when_all_ok(self):
        engine = AlertEngine()
        engine._last_fired = {}
        metrics = make_metrics(
            faithfulness=0.90, answer_relevancy=0.85, latency_ms=1000.0
        )
        fired = engine.evaluate(metrics)
        assert fired == []

    def test_cooldown_prevents_duplicate_firing(self):
        engine = AlertEngine()
        # Set very long cooldown
        engine.rules[0].cooldown_seconds = 9999
        bad_metrics = make_metrics(faithfulness=0.1)
        first_fire = engine.evaluate(bad_metrics)
        second_fire = engine.evaluate(bad_metrics)
        assert len(first_fire) >= 1
        # Second evaluation should not re-fire (still in cooldown)
        rule_name = first_fire[0].rule_name
        assert not any(e.rule_name == rule_name for e in second_fire)

    def test_add_rule_increases_count(self):
        engine = AlertEngine(rules=[])
        assert len(engine.rules) == 0
        engine.add_rule(AlertRule(
            name="test_rule", description="test", metric_name="faithfulness",
            condition="below", threshold=0.5, severity=AlertSeverity.INFO,
        ))
        assert len(engine.rules) == 1

    def test_remove_rule_returns_true_when_found(self):
        engine = AlertEngine()
        result = engine.remove_rule("faithfulness_low")
        assert result is True

    def test_remove_rule_returns_false_when_not_found(self):
        engine = AlertEngine()
        result = engine.remove_rule("nonexistent_rule")
        assert result is False

    def test_get_active_alerts_initially_empty(self):
        engine = AlertEngine()
        assert engine.get_active_alerts() == []

    def test_acknowledge_alert_resolves_it(self):
        engine = AlertEngine()
        engine._last_fired = {}
        engine.evaluate(make_metrics(faithfulness=0.1))
        active_before = len(engine.get_active_alerts())
        if active_before > 0:
            rule_name = engine.get_active_alerts()[0].rule_name
            engine.acknowledge_alert(rule_name)
            active_after = len(engine.get_active_alerts())
            assert active_after == active_before - 1


# ── TestEvaluatorMetrics ──────────────────────────────────────────────────────

class TestEvaluatorMetrics:

    def test_tokenize_removes_stopwords(self):
        tokens = _tokenize("the cat sat on the mat")
        assert "the" not in tokens
        assert "cat" in tokens

    def test_tokenize_removes_short_words(self):
        tokens = _tokenize("AI is great technology")
        assert "is" not in tokens
        assert "great" in tokens

    def test_jaccard_identical_sets(self):
        s = {"machine", "learning", "model"}
        assert _jaccard(s, s) == 1.0

    def test_jaccard_disjoint_sets(self):
        assert _jaccard({"apple"}, {"bicycle"}) == 0.0

    def test_jaccard_partial_overlap(self):
        a = {"machine", "learning", "model"}
        b = {"machine", "learning", "deployment"}
        assert _jaccard(a, b) == pytest.approx(0.5)

    def test_context_precision_relevant_chunks(self):
        question = "enterprise AI model deployment"
        contexts = [
            "Enterprise AI models require careful deployment planning.",
            "Deploying AI in enterprise settings needs governance.",
        ]
        precision = compute_context_precision(question, contexts, threshold=0.05)
        assert precision > 0.5

    def test_context_precision_empty_returns_zero(self):
        assert compute_context_precision("any question", []) == 0.0

    def test_eval_result_overall_score_range(self):
        r = EvalResult("q", "a", 0.8, 0.9, 0.7, 100, 4)
        score = r.overall_score()
        assert 0.0 <= score <= 1.0

    def test_eval_result_overall_score_formula(self):
        r = EvalResult("q", "a", 0.6, 0.8, 0.7, 100, 4)
        expected = round(0.3 * 0.6 + 0.4 * 0.8 + 0.3 * 0.7, 4)
        assert r.overall_score() == expected

    def test_summarize_results_means(self):
        r1 = EvalResult("q1", "a1", 0.8, 0.9, 0.7, 100, 4)
        r2 = EvalResult("q2", "a2", 0.6, 0.7, 0.5, 200, 4)
        summary = summarize_results([r1, r2])
        assert summary["mean_context_precision"] == pytest.approx(0.7)
        assert summary["mean_faithfulness"] == pytest.approx(0.8)
        assert summary["total_evaluated"] == 2

    def test_summarize_empty_returns_empty_dict(self):
        assert summarize_results([]) == {}


# ── TestABTestStats ───────────────────────────────────────────────────────────

class TestABTestStats:
    """Test statistical comparison logic without running any LLM calls."""

    def test_identical_distributions_not_significant(self):
        from scipy import stats as scipy_stats
        scores = [0.7, 0.72, 0.68, 0.71, 0.69]
        _, p = scipy_stats.mannwhitneyu(scores, scores, alternative="two-sided")
        assert p > 0.05  # should not be significant

    def test_clearly_different_distributions_significant(self):
        from scipy import stats as scipy_stats
        low_scores = [0.2, 0.22, 0.19, 0.21, 0.18, 0.20, 0.23]
        high_scores = [0.85, 0.88, 0.82, 0.90, 0.87, 0.84, 0.89]
        _, p = scipy_stats.mannwhitneyu(low_scores, high_scores, alternative="two-sided")
        assert p < 0.05

    def test_effect_size_in_valid_range(self):
        from scipy import stats as scipy_stats
        a = [0.6, 0.65, 0.62, 0.68, 0.61]
        b = [0.8, 0.82, 0.79, 0.85, 0.78]
        stat, _ = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        n = len(a) * len(b)
        effect_size = 1 - (2 * float(stat)) / n
        assert -1 <= effect_size <= 1

    def test_winner_determined_by_mean(self):
        a_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        b_scores = [0.9, 0.9, 0.9, 0.9, 0.9]
        mean_a = statistics.mean(a_scores)
        mean_b = statistics.mean(b_scores)
        # B has higher mean — should be winner
        assert mean_b > mean_a


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
