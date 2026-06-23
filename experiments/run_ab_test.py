"""
CLI A/B test runner — compares two prompt versions with statistical significance testing.

Uses Mann-Whitney U test (non-parametric, correct for small samples of non-normal scores).
Logs results to MLflow. Exits with code 1 if prompt B is significantly worse than A.

Usage:
    python experiments/run_ab_test.py --prompt-a rag_qa_v1 --prompt-b rag_qa_v2
    python experiments/run_ab_test.py --prompt-a rag_qa_v1 --prompt-b rag_qa_v2 --n-questions 10
    python experiments/run_ab_test.py --prompt-a rag_qa_v1 --prompt-b rag_qa_v2 --fail-on-regression
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer

from src.config import config
from src.gateway import InferenceGateway
from src.registry import PromptRegistry
from src.evaluator import (
    compute_context_precision,
    compute_faithfulness,
    compute_answer_relevancy,
    EvalResult,
    summarize_results,
)
from src.tracker import MLflowTracker
from src.monitor import get_metrics_history
from src.alerts import get_alert_engine


console = Console() if HAS_RICH else None


def _print(msg: str) -> None:
    if console:
        console.print(msg)
    else:
        print(msg)


def load_context(docs_dir: str) -> str:
    """Load sample documents for context."""
    parts = []
    for fname in ["ai_policy.txt", "rag_faq.txt"]:
        path = os.path.join(docs_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def load_questions(qa_path: str, n: int) -> List[Dict[str, str]]:
    """Load Q&A pairs from test_qa.json."""
    with open(qa_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs[:n]


def run_prompt_on_questions(
    prompt_name: str,
    qa_pairs: List[Dict],
    context: str,
    gateway: InferenceGateway,
    registry: PromptRegistry,
    embeddings,
) -> List[EvalResult]:
    """Run one prompt against all questions and return EvalResult list."""
    pt = registry.get(prompt_name)
    results = []

    for qa in qa_pairs:
        question = qa["question"]
        rendered = pt.render(context=context, question=question)

        result = gateway.generate(
            prompt_name=prompt_name,
            rendered_prompt=rendered,
            prompt_version=pt.version,
        )

        if not result.success:
            _print(f"[red]Error for '{question[:40]}...': {result.error}[/red]" if HAS_RICH else f"ERROR: {result.error}")
            continue

        # Simple context list for evaluation (split on separator)
        context_chunks = [c.strip() for c in context.split("---") if c.strip()][:4]

        cp = compute_context_precision(question, context_chunks)
        faith = compute_faithfulness(result.answer, context_chunks, gateway)
        rel = compute_answer_relevancy(question, result.answer, gateway, embeddings)

        results.append(EvalResult(
            question=question,
            answer=result.answer,
            context_precision=cp,
            faithfulness=faith,
            answer_relevancy=rel,
            latency_ms=result.latency_ms,
            top_k=len(context_chunks),
        ))

    return results


def compute_ab_stats(
    scores_a: List[float], scores_b: List[float]
) -> Dict[str, Any]:
    """Mann-Whitney U test for statistical significance."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "winner": "tie",
            "effect_size": 0.0,
            "note": "Insufficient data for statistical test (need >= 2 samples per group)",
        }

    statistic, p_value = scipy_stats.mannwhitneyu(
        scores_a, scores_b, alternative="two-sided"
    )

    significant = p_value < 0.05
    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)

    if significant:
        winner = "a" if mean_a > mean_b else "b"
    else:
        winner = "tie"

    # Rank-biserial correlation as effect size (range: -1 to +1)
    n = len(scores_a) * len(scores_b)
    effect_size = round(1 - (2 * statistic) / n, 4) if n > 0 else 0.0

    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "winner": winner,
        "effect_size": effect_size,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "recommendation": (
            "Deploy B (statistically better)" if winner == "b" else
            "Keep A (statistically better)" if winner == "a" else
            "Inconclusive — no significant difference (p={:.3f})".format(p_value)
        ),
    }


def run_ab_test(
    prompt_a: str,
    prompt_b: str,
    n_questions: int = 5,
    log_mlflow: bool = True,
    fail_on_regression: bool = False,
) -> int:
    """Main A/B test logic. Returns exit code (0 = OK, 1 = regression detected)."""
    _print("\n[bold blue]LLMOps Starter Kit — A/B Test Runner[/bold blue]\n" if HAS_RICH
           else "\n=== LLMOps A/B Test Runner ===\n")

    try:
        config.validate()
    except ValueError as e:
        _print(f"[red]Config error:[/red] {e}" if HAS_RICH else f"CONFIG ERROR: {e}")
        return 1

    _print(f"Comparing [cyan]{prompt_a}[/cyan] vs [cyan]{prompt_b}[/cyan] on {n_questions} questions\n"
           if HAS_RICH else f"Comparing {prompt_a} vs {prompt_b} on {n_questions} questions\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    gateway = InferenceGateway()
    registry = PromptRegistry()
    embeddings_model = SentenceTransformer(config.EMBED_MODEL)

    class _Embeddings:
        def embed_query(self, text: str) -> List[float]:
            return embeddings_model.encode(text, normalize_embeddings=True).tolist()

    embeddings = _Embeddings()

    context = load_context(config.SAMPLE_DOCS_DIR)
    qa_pairs = load_questions(config.TEST_QA_PATH, n_questions)

    _print(f"Loaded {len(qa_pairs)} questions | Context: {len(context):,} chars\n" if not HAS_RICH
           else f"Loaded [green]{len(qa_pairs)}[/green] questions | Context: {len(context):,} chars\n")

    # ── Run both prompts ──────────────────────────────────────────────────────
    _print(f"Running prompt A: {prompt_a}...")
    results_a = run_prompt_on_questions(prompt_a, qa_pairs, context, gateway, registry, embeddings)

    _print(f"Running prompt B: {prompt_b}...")
    results_b = run_prompt_on_questions(prompt_b, qa_pairs, context, gateway, registry, embeddings)

    if not results_a or not results_b:
        _print("[red]ERROR:[/red] One or both prompts returned no results." if HAS_RICH
               else "ERROR: One or both prompts returned no results.")
        return 1

    # ── Statistical comparison ────────────────────────────────────────────────
    scores_a = [r.overall_score() for r in results_a]
    scores_b = [r.overall_score() for r in results_b]
    stats = compute_ab_stats(scores_a, scores_b)

    summary_a = summarize_results(results_a)
    summary_b = summarize_results(results_b)

    # ── Display results ───────────────────────────────────────────────────────
    if HAS_RICH:
        table = Table(title="Per-Question Comparison", show_header=True,
                      header_style="bold white on dark_blue")
        table.add_column("Question", max_width=40, style="dim")
        table.add_column(f"Overall A\n({prompt_a})", justify="center")
        table.add_column(f"Overall B\n({prompt_b})", justify="center")
        table.add_column("Winner", justify="center")

        for ra, rb in zip(results_a, results_b):
            sa, sb = ra.overall_score(), rb.overall_score()
            winner = "A" if sa > sb else "B" if sb > sa else "="
            color = "green" if winner == "B" else "red" if winner == "A" else "yellow"
            table.add_row(
                ra.question[:39] + "…" if len(ra.question) > 39 else ra.question,
                f"{sa:.3f}", f"{sb:.3f}",
                f"[{color}]{winner}[/{color}]",
            )
        console.print(table)

        console.print(Panel(
            f"Prompt A mean: [cyan]{stats['mean_a']}[/cyan]\n"
            f"Prompt B mean: [cyan]{stats['mean_b']}[/cyan]\n"
            f"p-value: [yellow]{stats['p_value']}[/yellow] "
            f"({'significant' if stats['significant'] else 'not significant'})\n"
            f"Effect size (rank-biserial r): [cyan]{stats['effect_size']}[/cyan]\n"
            f"Winner: [bold]{'A' if stats['winner'] == 'a' else 'B' if stats['winner'] == 'b' else 'TIE'}[/bold]\n"
            f"Recommendation: [bold green]{stats['recommendation']}[/bold green]",
            title="Statistical Summary",
            border_style="green" if stats['winner'] == "b" else "red" if stats['winner'] == "a" else "yellow",
        ))
    else:
        print(f"\nResults: A={stats['mean_a']:.3f} | B={stats['mean_b']:.3f}")
        print(f"p-value={stats['p_value']:.6f} | Winner={stats['winner'].upper()}")
        print(f"Recommendation: {stats['recommendation']}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    if log_mlflow:
        run_name = f"ab-{prompt_a}-vs-{prompt_b}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
        tracker = MLflowTracker()
        with tracker.start_run(run_name):
            tracker.log_ab_test_result(
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                winner=stats["winner"],
                p_value=stats["p_value"],
                significant=stats["significant"],
                metrics_a={
                    "faithfulness": summary_a["mean_faithfulness"],
                    "context_precision": summary_a["mean_context_precision"],
                    "answer_relevancy": summary_a["mean_answer_relevancy"],
                    "overall_score": summary_a["mean_overall_score"],
                },
                metrics_b={
                    "faithfulness": summary_b["mean_faithfulness"],
                    "context_precision": summary_b["mean_context_precision"],
                    "answer_relevancy": summary_b["mean_answer_relevancy"],
                    "overall_score": summary_b["mean_overall_score"],
                },
            )
        _print(f"\n[green]MLflow run logged:[/green] {run_name}" if HAS_RICH
               else f"\nMLflow run logged: {run_name}")
        _print("Run [cyan]mlflow ui --port 5000[/cyan] to view results." if HAS_RICH
               else "Run: mlflow ui --port 5000")

    # ── Exit code for CI ──────────────────────────────────────────────────────
    if fail_on_regression and stats["significant"] and stats["winner"] == "a":
        _print("\n[red bold]CI FAILURE:[/red bold] Prompt B is significantly WORSE than A. Blocking merge." if HAS_RICH
               else "\nCI FAILURE: Prompt B is significantly WORSE than A.")
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMOps A/B Test Runner")
    parser.add_argument("--prompt-a", required=True, help="Baseline prompt name (e.g., rag_qa_v1)")
    parser.add_argument("--prompt-b", required=True, help="Challenger prompt name (e.g., rag_qa_v2)")
    parser.add_argument("--n-questions", type=int, default=5, help="Number of test questions (max 10)")
    parser.add_argument("--log-mlflow", action="store_true", default=True, help="Log results to MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit with code 1 if B is significantly worse than A (for CI)")
    args = parser.parse_args()

    exit_code = run_ab_test(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        n_questions=min(args.n_questions, 10),
        log_mlflow=args.log_mlflow and not args.no_mlflow,
        fail_on_regression=args.fail_on_regression,
    )
    sys.exit(exit_code)
