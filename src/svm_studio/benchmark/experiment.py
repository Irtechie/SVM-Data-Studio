"""Top-level experiment orchestrator for the benchmark pipeline.

``run_experiment`` is the single public entry point.  It wires:

  DatasetLoader → LLMLabeler → SVMEvaluator
    → [optional techniques]
    → Visualizer → ReportGenerator → DB save

Returns an ``ExperimentResult`` containing every artefact so the Streamlit
page can render everything without calling multiple modules itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .dataset_loader import DatasetLoader, StandardDataset
from .db import save_run
from .llm_labeler import LLMLabeler, LabeledDataset
from .report_generator import generate_report, report_to_pdf_bytes
from .svm_evaluator import SVMEvaluator, EvalResult
from .visualizer import (
    plot_accuracy_comparison,
    plot_confidence_distribution,
    plot_confusion_matrices,
    plot_cv_fold_comparison,
    plot_disagreement_table,
    plot_per_class_metrics,
)

_OPTIONAL_TECHNIQUES = frozenset([
    "uncertainty_sampling",
    "universum_svm",
    "itemset_mining",
    "episode_mining",
])


@dataclass
class ExperimentResult:
    """All artefacts from one benchmark run."""
    dataset: StandardDataset
    labeled: LabeledDataset
    eval_result: EvalResult
    report_markdown: str
    run_id: int | None                     # None if DB save was skipped

    # charts
    fig_accuracy: go.Figure
    fig_confusion: go.Figure
    fig_per_class: go.Figure
    fig_confidence: go.Figure
    fig_disagreement: go.Figure
    fig_cv_folds: go.Figure

    # optional technique results (populated only when technique is enabled)
    ubs_result: Any = None                 # UBSResult or None
    universum_result: Any = None           # BenchUniversumResult or None
    itemset_result: Any = None             # BenchItemsetResult or None
    episode_result: Any = None             # BenchEpisodeResult or None
    optional_summaries: dict[str, str] = field(default_factory=dict)


def run_experiment(
    source: str,
    name: str,
    llm_model: str = "gemma-4-31B-it-Q4_K_M.gguf",
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    optional_techniques: list[str] | None = None,
    max_examples: int | None = 200,
    kernel: str = "rbf",
    save_to_db: bool = True,
    db_path: Path | None = None,
    progress_callback: Any = None,
    dataset_loader_kwargs: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run a full benchmark experiment.

    Parameters
    ----------
    source : str
        Dataset source — ``"sklearn"``, ``"openml"``, ``"ucimlrepo"``,
        ``"huggingface"``, or ``"csv"``.
    name : str
        Dataset identifier (function name, ID, file path, …).
    llm_model : str
        Model name forwarded to the LLM backend.
    llm_base_url : str or None
        LLM endpoint base URL for local/remote models.
    optional_techniques : list[str], optional
        Any subset of ``"uncertainty_sampling"``, ``"universum_svm"``,
        ``"itemset_mining"``, ``"episode_mining"``.
    max_examples : int or None
        Cap on labeling calls (reduce for fast tests).
    kernel : str
        SVM kernel for evaluation.
    save_to_db : bool
        Persist results to SQLite run history.
    progress_callback : callable(stage: str, current: int, total: int)
        Optional progress hook called at each stage.
    """
    optional_techniques = [t for t in (optional_techniques or []) if t in _OPTIONAL_TECHNIQUES]

    def _progress(stage: str, current: int = 0, total: int = 0) -> None:
        if progress_callback:
            try:
                progress_callback(stage, current, total)
            except Exception:
                pass

    # ── 1. Load dataset ───────────────────────────────────────────────────
    _progress("Loading dataset", 0, 1)
    loader = DatasetLoader(**(dataset_loader_kwargs or {}))
    dataset = loader.load(source, name)
    _progress("Loading dataset", 1, 1)

    # ── 2. LLM labeling ───────────────────────────────────────────────────
    labeler = LLMLabeler(model=llm_model, base_url=llm_base_url, api_key=llm_api_key)
    n = min(max_examples, len(dataset.X)) if max_examples else len(dataset.X)
    _progress("LLM labeling", 0, n)

    def _label_progress(current: int, total: int) -> None:
        _progress("LLM labeling", current, total)

    labeled = labeler.label(dataset, max_examples=max_examples, progress_callback=_label_progress)
    _progress("LLM labeling", n, n)

    # ── 3. SVM evaluation ─────────────────────────────────────────────────
    _progress("SVM evaluation", 0, 1)
    evaluator = SVMEvaluator(kernel=kernel)
    eval_result = evaluator.evaluate(dataset, labeled)
    _progress("SVM evaluation", 1, 1)

    # ── 4. Optional techniques ────────────────────────────────────────────
    ubs_result = None
    universum_result = None
    itemset_result = None
    episode_result = None
    optional_summaries: dict[str, str] = {}

    if "uncertainty_sampling" in optional_techniques:
        _progress("Uncertainty sampling", 0, 1)
        from .optional.uncertainty_sampling import run_uncertainty_sampling
        ubs_result = run_uncertainty_sampling(dataset, labeled, labeler)
        optional_summaries["Uncertainty-Based Sampling"] = _ubs_summary(ubs_result)
        _progress("Uncertainty sampling", 1, 1)

    if "universum_svm" in optional_techniques:
        _progress("Universum SVM", 0, 1)
        from .optional.universum_svm import run_bench_universum_svm
        universum_result = run_bench_universum_svm(dataset, labeled, kernel=kernel)
        optional_summaries["Universum SVM"] = _universum_summary(universum_result)
        _progress("Universum SVM", 1, 1)

    if "itemset_mining" in optional_techniques and dataset.data_type in ("tabular", "image_features"):
        _progress("Itemset mining", 0, 1)
        from .optional.itemset_mining import run_bench_itemset_mining
        itemset_result = run_bench_itemset_mining(dataset, labeled)
        optional_summaries["Itemset Mining"] = _itemset_summary(itemset_result)
        _progress("Itemset mining", 1, 1)

    if "episode_mining" in optional_techniques:
        _progress("Episode mining", 0, 1)
        from .optional.episode_mining import run_bench_episode_mining
        episode_result = run_bench_episode_mining(dataset, labeled)
        optional_summaries["Episode Mining"] = _episode_summary(episode_result)
        _progress("Episode mining", 1, 1)

    # ── 5. Visualizations ─────────────────────────────────────────────────
    _progress("Generating charts", 0, 1)
    confidence_scores = [r.confidence for r in labeled.records]
    fig_accuracy = plot_accuracy_comparison(eval_result)
    fig_confusion = plot_confusion_matrices(eval_result)
    fig_per_class = plot_per_class_metrics(eval_result)
    fig_confidence = plot_confidence_distribution(confidence_scores)
    fig_disagreement = plot_disagreement_table(eval_result.disagreements)
    fig_cv_folds = plot_cv_fold_comparison(eval_result)
    _progress("Generating charts", 1, 1)

    # ── 6. Report generation ──────────────────────────────────────────────
    _progress("Generating report", 0, 1)
    report_md = generate_report(
        eval_result,
        optional_results=optional_summaries,
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        dataset_description=dataset.description,
        class_names_str=", ".join(dataset.class_names),
    )
    _progress("Generating report", 1, 1)

    # ── 7. Save to DB ─────────────────────────────────────────────────────
    run_id = None
    if save_to_db:
        run_id = save_run(eval_result, optional_techniques, report_md, db_path)

    return ExperimentResult(
        dataset=dataset,
        labeled=labeled,
        eval_result=eval_result,
        report_markdown=report_md,
        run_id=run_id,
        fig_accuracy=fig_accuracy,
        fig_confusion=fig_confusion,
        fig_per_class=fig_per_class,
        fig_confidence=fig_confidence,
        fig_disagreement=fig_disagreement,
        fig_cv_folds=fig_cv_folds,
        ubs_result=ubs_result,
        universum_result=universum_result,
        itemset_result=itemset_result,
        episode_result=episode_result,
        optional_summaries=optional_summaries,
    )


# ── summary helpers for optional technique results ─────────────────────────

def _ubs_summary(r: Any) -> str:
    return (
        f"Re-queried {r.n_uncertain} uncertain examples; {r.relabeled_count} changed label.\n"
        f"Before: accuracy={r.before_accuracy:.4f}, macro F1={r.before_macro_f1:.4f}\n"
        f"After:  accuracy={r.after_accuracy:.4f}, macro F1={r.after_macro_f1:.4f}\n"
        f"Delta accuracy: {r.delta_accuracy:+.4f}"
    )


def _universum_summary(r: Any) -> str:
    u = r.universum_result
    return (
        f"Strategy: {u.universum_strategy}, size: {u.universum_size}, "
        f"universum weight: label noise rate {r.label_noise_rate:.3f}\n"
        f"Standard SVM: accuracy={u.standard_accuracy:.4f}, macro F1={u.standard_macro_f1:.4f}\n"
        f"Universum SVM: accuracy={u.universum_accuracy:.4f}, macro F1={u.universum_macro_f1:.4f}\n"
        f"Delta: {u.accuracy_delta:+.4f}"
    )


def _itemset_summary(r: Any) -> str:
    n_all = len(r.all_patterns)
    n_err = len(r.error_patterns)
    n_excl = len(r.error_exclusive)
    return (
        f"Total frequent patterns: {n_all}, min_support={r.min_support_used}\n"
        f"Patterns in error examples: {n_err}\n"
        f"Patterns exclusive to error examples: {n_excl}\n"
        + (
            f"Top error-exclusive patterns:\n{r.error_exclusive.head(5).to_string(index=False)}"
            if not r.error_exclusive.empty else "No error-exclusive patterns found."
        )
    )


def _episode_summary(r: Any) -> str:
    if r.sequence_source.startswith("skipped"):
        return f"Episode mining skipped: {r.sequence_source}"
    return (
        f"Sequence source: {r.sequence_source}, min_support={r.min_support_used}\n"
        f"Total episodes: {len(r.all_episodes)}\n"
        f"Episodes in error examples: {len(r.error_episodes)}\n"
        f"Episodes exclusive to error examples: {len(r.error_exclusive)}"
    )
