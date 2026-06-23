"""
Gradio UI for LLMOps Starter Kit.

Four tabs:
  1. Prompt Lab    — Browse, render, and test prompt versions
  2. A/B Test      — Compare two prompts with statistical significance
  3. Monitor       — Metric trends, drift detection, run history
  4. Alerts        — Rule management and alert history
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr

from src.config import config
from src.registry import PromptRegistry
from src.gateway import get_gateway
from src.tracker import MLflowTracker
from src.monitor import get_metrics_history, TRACKED_METRICS
from src.alerts import AlertEngine, AlertRule, AlertSeverity, get_alert_engine
from src.evaluator import (
    compute_context_precision,
    compute_faithfulness,
    compute_answer_relevancy,
    EvalResult,
    summarize_results,
)

# ── Module-level singletons ────────────────────────────────────────────────────
_registry = PromptRegistry()
_gateway = get_gateway()
_tracker = MLflowTracker()
_history = get_metrics_history()
_alerts = get_alert_engine()

# Lazy embeddings (loaded on first A/B test run)
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)

        class _Embed:
            def embed_query(self, text: str):
                return _model.encode(text, normalize_embeddings=True).tolist()

        _embeddings = _Embed()
    return _embeddings


def _load_context() -> str:
    """Load sample documents as context."""
    parts = []
    for fname in ["ai_policy.txt", "rag_faq.txt"]:
        path = os.path.join(config.SAMPLE_DOCS_DIR, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def _load_qa_pairs():
    try:
        with open(config.TEST_QA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ── Tab 1: Prompt Lab ──────────────────────────────────────────────────────────

def refresh_prompts_fn():
    """Reload prompts from disk and return updated dropdown choices + table."""
    try:
        _registry.load_all()
        names = _registry.list_prompts()
        table = _registry.get_all_as_table()
        return (
            gr.Dropdown(choices=names, value=names[0] if names else None),
            gr.Dropdown(choices=names, value=names[1] if len(names) > 1 else names[0] if names else None),
            gr.Dropdown(choices=names, value=names[0] if names else None),
            gr.Dropdown(choices=names, value=names[1] if len(names) > 1 else names[0] if names else None),
            table,
            f"Loaded {len(names)} prompt(s): {', '.join(names)}",
        )
    except Exception as e:
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), [], f"Error: {e}"


def render_and_generate_fn(prompt_name, context, question):
    """Render prompt template and call the inference gateway."""
    if not prompt_name:
        return "Select a prompt first.", "", {}

    try:
        config.validate()
    except ValueError as e:
        return "", str(e), {}

    if not question.strip():
        return "", "Enter a question.", {}

    ctx = context.strip() or _load_context()[:3000]

    try:
        pt = _registry.get(prompt_name)
    except KeyError as e:
        return "", str(e), {}

    try:
        rendered = pt.render(context=ctx, question=question)
    except ValueError as e:
        return "", str(e), {}

    result = _gateway.generate(
        prompt_name=prompt_name,
        rendered_prompt=rendered,
        prompt_version=pt.version,
    )

    stats = {
        "prompt_name": result.prompt_name,
        "prompt_version": result.prompt_version,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "model": result.model,
        "success": result.success,
    }

    if not result.success:
        return rendered, f"Error: {result.error}", stats

    return rendered, result.answer, stats


def show_diff_fn(prompt_a, prompt_b):
    """Show unified diff between two prompt templates."""
    if not prompt_a or not prompt_b:
        return "Select both prompts to compare."
    if prompt_a == prompt_b:
        return "Select two different prompts to see a diff."
    try:
        diff = _registry.compare_versions(prompt_a, prompt_b)
        return diff["unified_diff"] or "(No differences in template text)"
    except Exception as e:
        return f"Error: {e}"


# ── Tab 2: A/B Test ────────────────────────────────────────────────────────────

def run_ab_test_fn(prompt_a, prompt_b, context_input, use_sample, qa_file, log_mlflow):
    """Full A/B test with statistical significance testing."""
    if not prompt_a or not prompt_b:
        return [], {}, "", "<p>Select both prompts.</p>"

    if prompt_a == prompt_b:
        return [], {}, "", "<p style='color:orange'>Select two DIFFERENT prompts for a meaningful A/B test.</p>"

    try:
        config.validate()
    except ValueError as e:
        return [], {"error": str(e)}, "", ""

    context = context_input.strip() if context_input and context_input.strip() else _load_context()

    if use_sample:
        qa_pairs = _load_qa_pairs()[:5]
    elif qa_file is not None:
        try:
            path = qa_file.name if hasattr(qa_file, "name") else qa_file
            with open(path, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)[:10]
        except Exception as e:
            return [], {"error": f"Could not load Q&A file: {e}"}, "", ""
    else:
        return [], {"error": "Upload a Q&A JSON file or use the sample test set."}, "", ""

    embeddings = _get_embeddings()
    context_chunks = [c.strip() for c in context.split("---") if c.strip()][:4]

    def _run_prompt(prompt_name):
        pt = _registry.get(prompt_name)
        results = []
        for qa in qa_pairs:
            q = qa["question"]
            try:
                rendered = pt.render(context=context, question=q)
            except ValueError:
                continue
            res = _gateway.generate(prompt_name, rendered, pt.version)
            if not res.success:
                continue
            cp = compute_context_precision(q, context_chunks)
            faith = compute_faithfulness(res.answer, context_chunks, _gateway)
            rel = compute_answer_relevancy(q, res.answer, _gateway, embeddings)
            results.append(EvalResult(q, res.answer, cp, faith, rel, res.latency_ms, len(context_chunks)))
        return results

    results_a = _run_prompt(prompt_a)
    results_b = _run_prompt(prompt_b)

    if not results_a or not results_b:
        return [], {"error": "One or both prompts returned no results."}, "", ""

    # Statistical significance (Mann-Whitney U)
    from scipy import stats as scipy_stats
    scores_a = [r.overall_score() for r in results_a]
    scores_b = [r.overall_score() for r in results_b]

    if len(scores_a) >= 2 and len(scores_b) >= 2:
        statistic, p_value = scipy_stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        significant = p_value < 0.05
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        winner = (
            prompt_a if (significant and mean_a > mean_b) else
            prompt_b if (significant and mean_b > mean_a) else
            "tie"
        )
        n = len(scores_a) * len(scores_b)
        effect_size = round(1 - (2 * float(statistic)) / n, 4) if n > 0 else 0.0
    else:
        p_value, significant, winner, effect_size = 1.0, False, "tie", 0.0
        mean_a = statistics.mean(scores_a) if scores_a else 0
        mean_b = statistics.mean(scores_b) if scores_b else 0

    summary_a = summarize_results(results_a)
    summary_b = summarize_results(results_b)

    # Build per-question table
    table_rows = []
    for ra, rb in zip(results_a, results_b):
        sa, sb = ra.overall_score(), rb.overall_score()
        w = prompt_a if sa > sb else (prompt_b if sb > sa else "tie")
        table_rows.append([
            ra.question[:60] + "…" if len(ra.question) > 60 else ra.question,
            round(sa, 3), round(sb, 3),
            w, round(sb - sa, 3),
        ])

    summary = {
        f"{prompt_a}_mean_overall": round(mean_a, 4),
        f"{prompt_b}_mean_overall": round(mean_b, 4),
        f"{prompt_a}_faithfulness": summary_a.get("mean_faithfulness"),
        f"{prompt_b}_faithfulness": summary_b.get("mean_faithfulness"),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "winner": winner,
        "effect_size": effect_size,
        "recommendation": (
            f"Deploy {prompt_b} (statistically better, p={p_value:.3f})" if winner == prompt_b else
            f"Keep {prompt_a} (statistically better, p={p_value:.3f})" if winner == prompt_a else
            f"Inconclusive — no significant difference (p={p_value:.3f})"
        ),
    }

    run_id_text = ""
    if log_mlflow:
        from datetime import datetime, timezone
        run_name = f"ab-{prompt_a}-vs-{prompt_b}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
        try:
            with _tracker.start_run(run_name):
                _tracker.log_ab_test_result(
                    prompt_a=prompt_a, prompt_b=prompt_b,
                    winner=winner, p_value=float(p_value), significant=significant,
                    metrics_a={"faithfulness": summary_a["mean_faithfulness"],
                               "overall_score": summary_a["mean_overall_score"]},
                    metrics_b={"faithfulness": summary_b["mean_faithfulness"],
                               "overall_score": summary_b["mean_overall_score"]},
                )
            run_id_text = f"MLflow run: {run_name} (view at http://localhost:5000)"
        except Exception as e:
            run_id_text = f"MLflow logging failed: {e}"

    # HTML badge
    if winner == prompt_b:
        badge = f"<div style='background:#16a34a;color:white;padding:12px;border-radius:8px;font-size:16px;font-weight:bold'>WINNER: {prompt_b} (p={p_value:.4f})</div>"
    elif winner == prompt_a:
        badge = f"<div style='background:#dc2626;color:white;padding:12px;border-radius:8px;font-size:16px;font-weight:bold'>KEEP: {prompt_a} wins (p={p_value:.4f})</div>"
    else:
        badge = f"<div style='background:#ca8a04;color:white;padding:12px;border-radius:8px;font-size:16px;font-weight:bold'>TIE — No significant difference (p={p_value:.4f})</div>"

    return table_rows, summary, run_id_text, badge


# ── Tab 3: Monitor ─────────────────────────────────────────────────────────────

def refresh_monitor_fn(metric_name):
    """Refresh monitor tab: get run history + drift report from MLflow."""
    chart_data = _history.get_chart_data(metric_name)
    summary = _history.get_summary()
    drift_status = _history.get_drift_status_string()
    run_table = _tracker.get_run_history_as_table(20)
    return chart_data, run_table, summary, drift_status


def populate_history_from_mlflow_fn():
    """Pull MLflow run history into in-memory MetricsHistory for drift detection."""
    history = _tracker.get_run_history(30)
    if not history:
        return "No MLflow runs found. Run A/B tests first to populate history."

    _history.reset()
    for run in reversed(history):  # oldest first
        _history.add_run({
            "faithfulness": run.get("faithfulness", 0),
            "context_precision": run.get("context_precision", 0),
            "answer_relevancy": run.get("answer_relevancy", 0),
            "overall_score": run.get("overall_score", 0),
            "latency_ms": float(run.get("latency_ms", 0)),
        })

    return f"Loaded {len(history)} runs into drift detector. {_history.get_drift_status_string()}"


def reset_baseline_fn():
    _history.reset()
    return "Baseline reset. Collect new runs to re-establish baseline."


# ── Tab 4: Alerts ──────────────────────────────────────────────────────────────

def refresh_alerts_fn():
    """Refresh alert rules and history displays."""
    rules_table = _alerts.get_rules_summary()
    history_table = _alerts.get_history_as_table(50)
    active_table = [a.to_row() for a in _alerts.get_active_alerts()]
    summary_text = _alerts.get_summary_text()
    return rules_table, history_table, active_table, summary_text


def add_alert_rule_fn(name, metric, condition, threshold, severity):
    if not name.strip():
        return "Rule name is required.", *refresh_alerts_fn()
    try:
        rule = AlertRule(
            name=name.strip(),
            description=f"Custom: {metric} {condition} {threshold}",
            metric_name=metric,
            condition=condition,
            threshold=float(threshold),
            severity=AlertSeverity(severity),
        )
        _alerts.add_rule(rule)
        return f"Rule '{name}' added.", *refresh_alerts_fn()
    except Exception as e:
        return f"Error: {e}", *refresh_alerts_fn()


def remove_alert_rule_fn(name):
    if not name.strip():
        return "Enter a rule name to remove.", *refresh_alerts_fn()
    removed = _alerts.remove_rule(name.strip())
    msg = f"Rule '{name}' removed." if removed else f"Rule '{name}' not found."
    return msg, *refresh_alerts_fn()


def acknowledge_all_fn():
    count = _alerts.acknowledge_all()
    return f"Acknowledged {count} alert(s).", *refresh_alerts_fn()


def test_alert_fn(faithfulness_val, relevancy_val, latency_val):
    """Simulate metrics to test alert rules."""
    test_metrics = {
        "faithfulness": float(faithfulness_val),
        "answer_relevancy": float(relevancy_val),
        "latency_ms": float(latency_val),
        "context_precision": 0.75,
        "overall_score": round(0.4 * float(faithfulness_val) + 0.3 * float(relevancy_val) + 0.3 * 0.75, 3),
    }
    fired = _alerts.evaluate(test_metrics)
    if fired:
        msgs = [f"{e.severity.value.upper()}: {e.message}" for e in fired]
        result = f"FIRED {len(fired)} alert(s):\n" + "\n".join(msgs)
    else:
        result = "No alerts fired for these metric values."
    return result, *refresh_alerts_fn()


# ── Build Gradio App ───────────────────────────────────────────────────────────

def create_app():
    # Pre-load prompts
    try:
        _registry.load_all()
        prompt_names = _registry.list_prompts()
    except Exception:
        prompt_names = []

    default_a = prompt_names[0] if prompt_names else None
    default_b = prompt_names[1] if len(prompt_names) > 1 else default_a
    sample_context = _load_context()[:2000] + "..." if len(_load_context()) > 2000 else _load_context()

    with gr.Blocks(title="LLMOps Starter Kit") as app:
        gr.Markdown("""
# LLMOps Starter Kit
**Production LLM monitoring — prompt versioning, A/B testing, drift detection, alert rules**
Built with Groq + MLflow + Prometheus + HuggingFace Embeddings
        """)

        with gr.Tabs():

            # ── Tab 1: Prompt Lab ─────────────────────────────────────────────
            with gr.Tab("1. Prompt Lab"):
                gr.Markdown("### Browse and test versioned prompt templates")

                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_select = gr.Dropdown(
                            choices=prompt_names, value=default_a,
                            label="Select Prompt Template",
                        )
                        context_input = gr.Textbox(
                            lines=5, label="Context (leave blank to use sample docs)",
                            placeholder="Paste document context here, or leave blank...",
                        )
                        question_input = gr.Textbox(
                            lines=2, label="Question",
                            placeholder="What is the SLA for model approval?",
                        )
                        generate_btn = gr.Button("Render & Generate", variant="primary")

                    with gr.Column(scale=1):
                        rendered_output = gr.Textbox(
                            label="Rendered Prompt (what the model sees)",
                            lines=8, interactive=False,
                        )
                        answer_output = gr.Textbox(
                            label="Generated Answer", lines=4, interactive=False
                        )
                        stats_output = gr.JSON(label="Inference Stats")

                gr.Markdown("#### Compare Two Prompt Versions")
                with gr.Row():
                    compare_a = gr.Dropdown(
                        choices=prompt_names, value=default_a, label="Prompt A"
                    )
                    compare_b = gr.Dropdown(
                        choices=prompt_names, value=default_b, label="Prompt B"
                    )
                    diff_btn = gr.Button("Show Diff")
                diff_output = gr.Textbox(
                    label="Template Diff (unified format)", lines=12, interactive=False
                )

                gr.Markdown("#### All Available Prompts")
                prompts_table = gr.Dataframe(
                    value=_registry.get_all_as_table(),
                    headers=["Name", "Version", "Task Type", "Description", "Variables"],
                    label="Prompt Registry",
                )
                refresh_prompts_btn = gr.Button("Refresh Prompts (reload from disk)")
                refresh_status = gr.Textbox(label="Status", interactive=False)

                # Wire events
                generate_btn.click(
                    fn=render_and_generate_fn,
                    inputs=[prompt_select, context_input, question_input],
                    outputs=[rendered_output, answer_output, stats_output],
                )
                diff_btn.click(
                    fn=show_diff_fn,
                    inputs=[compare_a, compare_b],
                    outputs=[diff_output],
                )
                refresh_prompts_btn.click(
                    fn=refresh_prompts_fn,
                    inputs=[],
                    outputs=[prompt_select, compare_a, ab_prompt_a := gr.State(),
                             ab_prompt_b := gr.State(), prompts_table, refresh_status],
                )

            # ── Tab 2: A/B Test ───────────────────────────────────────────────
            with gr.Tab("2. A/B Test"):
                gr.Markdown("""
### Statistical comparison of two prompt versions
Uses **Mann-Whitney U test** (non-parametric, robust for small samples).
p < 0.05 means the difference is statistically significant.
                """)
                with gr.Row():
                    with gr.Column():
                        ab_prompt_a_dd = gr.Dropdown(
                            choices=prompt_names, value=default_a, label="Prompt A (baseline)"
                        )
                        ab_prompt_b_dd = gr.Dropdown(
                            choices=prompt_names, value=default_b, label="Prompt B (challenger)"
                        )
                        ab_context = gr.Textbox(
                            lines=4, label="Context (blank = use sample docs)",
                            placeholder="Leave blank to use built-in sample documents...",
                        )
                    with gr.Column():
                        use_sample_cb = gr.Checkbox(
                            label="Use built-in 5-question test set", value=True
                        )
                        qa_file_input = gr.File(
                            label="Or upload custom Q&A JSON",
                            file_types=[".json"],
                        )
                        log_mlflow_cb = gr.Checkbox(
                            label="Log results to MLflow", value=True
                        )

                ab_run_btn = gr.Button(
                    "Run A/B Test (statistical comparison)", variant="primary"
                )

                ab_badge = gr.HTML("<p>Run the test to see results.</p>")
                ab_table = gr.Dataframe(
                    headers=["Question", f"Score A", f"Score B", "Winner", "Delta"],
                    label="Per-Question Comparison",
                    wrap=True,
                )
                ab_summary = gr.JSON(label="Statistical Summary")
                ab_mlflow_id = gr.Textbox(label="MLflow Run", interactive=False)

                ab_run_btn.click(
                    fn=run_ab_test_fn,
                    inputs=[ab_prompt_a_dd, ab_prompt_b_dd, ab_context,
                            use_sample_cb, qa_file_input, log_mlflow_cb],
                    outputs=[ab_table, ab_summary, ab_mlflow_id, ab_badge],
                )

            # ── Tab 3: Monitor ────────────────────────────────────────────────
            with gr.Tab("3. Monitor"):
                gr.Markdown("""
### Metric trends and drift detection across experiment runs

**Drift detection algorithm:** Rolling z-score against the first N-run baseline.
|z| > 1.5σ triggers a drift alert. Baseline is locked after the first 5 runs.
                """)
                with gr.Row():
                    metric_select = gr.Dropdown(
                        choices=TRACKED_METRICS,
                        value="overall_score",
                        label="Metric to Plot",
                    )
                    load_history_btn = gr.Button("Load History from MLflow")
                    reset_baseline_btn = gr.Button("Reset Baseline")

                drift_status_output = gr.Textbox(
                    label="Drift Status", interactive=False, lines=1
                )
                chart_output = gr.JSON(label="Chart Data (x, y, baseline_mean, thresholds)")

                gr.Markdown("#### Recent Experiment Runs")
                run_table_output = gr.Dataframe(
                    headers=["Run ID", "Name", "Prompt", "Faithfulness",
                             "Ctx Prec.", "Overall", "Latency (ms)", "Time"],
                    label="MLflow Run History",
                    wrap=True,
                )
                monitor_summary = gr.JSON(label="Drift Detector State")

                load_history_btn.click(
                    fn=populate_history_from_mlflow_fn,
                    inputs=[],
                    outputs=[drift_status_output],
                )
                reset_baseline_btn.click(
                    fn=reset_baseline_fn,
                    inputs=[],
                    outputs=[drift_status_output],
                )
                metric_select.change(
                    fn=refresh_monitor_fn,
                    inputs=[metric_select],
                    outputs=[chart_output, run_table_output, monitor_summary, drift_status_output],
                )
                load_history_btn.click(
                    fn=refresh_monitor_fn,
                    inputs=[metric_select],
                    outputs=[chart_output, run_table_output, monitor_summary, drift_status_output],
                )

            # ── Tab 4: Alerts ─────────────────────────────────────────────────
            with gr.Tab("4. Alerts"):
                gr.Markdown("""
### Alert Rules Engine
Define metric thresholds. Alerts fire when metrics cross thresholds.
Cooldown prevents alert storms (default: 5 minutes between re-fires).
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Active Rules")
                        rules_table_output = gr.Dataframe(
                            value=_alerts.get_rules_summary(),
                            headers=["Rule", "Metric", "Condition", "Threshold",
                                     "Severity", "Enabled", "Status"],
                            label="Alert Rules",
                        )
                        gr.Markdown("#### Add / Remove Rule")
                        new_rule_name = gr.Textbox(label="Rule Name", placeholder="my_custom_rule")
                        new_rule_metric = gr.Dropdown(
                            choices=TRACKED_METRICS + ["latency_ms"],
                            value="faithfulness",
                            label="Metric",
                        )
                        new_rule_condition = gr.Dropdown(
                            choices=["below", "above"], value="below", label="Condition"
                        )
                        new_rule_threshold = gr.Number(value=0.70, label="Threshold")
                        new_rule_severity = gr.Dropdown(
                            choices=["info", "warning", "critical"],
                            value="warning", label="Severity"
                        )
                        with gr.Row():
                            add_rule_btn = gr.Button("Add Rule", variant="primary")
                            remove_rule_name = gr.Textbox(label="Rule name to remove", scale=2)
                            remove_rule_btn = gr.Button("Remove", variant="stop")
                        rule_action_msg = gr.Textbox(label="Action Result", interactive=False)

                    with gr.Column(scale=1):
                        alert_summary_output = gr.Textbox(
                            value=_alerts.get_summary_text(),
                            label="Alert Summary", interactive=False
                        )
                        active_alerts_table = gr.Dataframe(
                            headers=["Time", "Rule", "Metric", "Value",
                                     "Threshold", "Severity", "Status"],
                            label="Firing Alerts",
                        )
                        gr.Markdown("#### Test Alert Rules")
                        gr.Markdown("Simulate metrics to verify your rules fire correctly.")
                        test_faith = gr.Slider(0, 1, value=0.65, step=0.01, label="Test Faithfulness")
                        test_rel = gr.Slider(0, 1, value=0.60, step=0.01, label="Test Relevancy")
                        test_lat = gr.Slider(0, 10000, value=3500, step=100, label="Test Latency (ms)")
                        test_btn = gr.Button("Test Alert Rules")
                        test_output = gr.Textbox(label="Test Result", interactive=False)

                gr.Markdown("#### Alert History")
                alert_history_table = gr.Dataframe(
                    headers=["Time", "Rule", "Metric", "Value",
                             "Threshold", "Severity", "Status"],
                    label="Last 50 Alert Events",
                    wrap=True,
                )
                with gr.Row():
                    refresh_alerts_btn = gr.Button("Refresh")
                    ack_all_btn = gr.Button("Acknowledge All Active", variant="stop")

                # Wire alert events
                add_rule_btn.click(
                    fn=add_alert_rule_fn,
                    inputs=[new_rule_name, new_rule_metric, new_rule_condition,
                            new_rule_threshold, new_rule_severity],
                    outputs=[rule_action_msg, rules_table_output, alert_history_table,
                             active_alerts_table, alert_summary_output],
                )
                remove_rule_btn.click(
                    fn=remove_alert_rule_fn,
                    inputs=[remove_rule_name],
                    outputs=[rule_action_msg, rules_table_output, alert_history_table,
                             active_alerts_table, alert_summary_output],
                )
                ack_all_btn.click(
                    fn=acknowledge_all_fn,
                    inputs=[],
                    outputs=[rule_action_msg, rules_table_output, alert_history_table,
                             active_alerts_table, alert_summary_output],
                )
                refresh_alerts_btn.click(
                    fn=refresh_alerts_fn,
                    inputs=[],
                    outputs=[rules_table_output, alert_history_table,
                             active_alerts_table, alert_summary_output],
                )
                test_btn.click(
                    fn=test_alert_fn,
                    inputs=[test_faith, test_rel, test_lat],
                    outputs=[test_output, rules_table_output, alert_history_table,
                             active_alerts_table, alert_summary_output],
                )

        gr.Markdown("""
---
**LLMOps Starter Kit** · Groq + MLflow + Prometheus + HuggingFace ·
[github.com/atkuri-siva-suresh/llmops-starter-kit](https://github.com/atkuri-siva-suresh/llmops-starter-kit)
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        theme=gr.themes.Soft(),
    )
