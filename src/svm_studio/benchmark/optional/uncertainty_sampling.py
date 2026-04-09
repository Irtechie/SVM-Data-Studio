"""Uncertainty-Based Sampling for the benchmark pipeline.

After initial SVM training on LLM labels, finds the N examples closest to the
decision boundary (smallest margin distance) and re-queries the LLM with a
more detailed uncertainty-retry prompt.  Optionally retrains the SVM on the
corrected labels and reports before/after metrics.

Re-uses margin-sampling logic already validated in ``advanced_svm.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ...custom_analysis import _build_pipeline, prepare_custom_classification_data
from ...datasets import RANDOM_STATE
from ..dataset_loader import StandardDataset
from ..llm_labeler import LLMLabeler, LabeledDataset


@dataclass
class UBSResult:
    """Result of one uncertainty-based sampling pass."""
    n_uncertain: int
    relabeled_count: int                  # how many actually changed
    before_accuracy: float
    after_accuracy: float
    before_macro_f1: float
    after_macro_f1: float
    delta_accuracy: float
    changed_indices: list[int]
    relabeled_frame: pd.DataFrame          # index, old_label, new_label, confidence


def run_uncertainty_sampling(
    dataset: StandardDataset,
    labeled: LabeledDataset,
    labeler: LLMLabeler,
    n_uncertain: int = 20,
    retrain: bool = True,
    test_size: float = 0.20,
    kernel: str = "rbf",
    svc_params: dict[str, Any] | None = None,
) -> UBSResult:
    """Run one round of uncertainty-based sampling.

    Parameters
    ----------
    n_uncertain : int
        Number of uncertain examples to re-query.
    retrain : bool
        If True, refit the SVM after relabeling and compare before/after accuracy.
    """
    n = len(labeled.y_llm)
    X = dataset.X.iloc[:n].reset_index(drop=True)
    y_llm = labeled.y_llm
    y_true = labeled.y_true

    X_train, X_test, yl_train, yt_test = train_test_split(
        X, y_llm, test_size=test_size, stratify=y_true,
        random_state=RANDOM_STATE,
    )

    # ── Fit initial SVM on LLM labels ─────────────────────────────────────
    pipe = _build_pipeline(X_train)
    pipe.set_params(svc__kernel=kernel)
    if svc_params:
        pipe.set_params(**{f"svc__{k}": v for k, v in svc_params.items()})
    pipe.fit(X_train, yl_train)

    before_pred = pipe.predict(X_test)
    before_acc = float(accuracy_score(yt_test, before_pred))
    before_f1 = float(f1_score(yt_test, before_pred, average="macro", zero_division=0))

    # ── Find most uncertain training examples ─────────────────────────────
    svc: SVC = pipe.named_steps["svc"]
    X_train_t = pipe[:-1].transform(X_train)
    try:
        df_scores = svc.decision_function(X_train_t)
        if df_scores.ndim == 1:
            margins = np.abs(df_scores)
        else:
            sorted_scores = np.sort(df_scores, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    except Exception:
        margins = np.random.default_rng(RANDOM_STATE).random(len(X_train))

    n_to_query = min(n_uncertain, len(X_train))
    uncertain_positions = np.argsort(margins)[:n_to_query].tolist()
    uncertain_global_idx = X_train.index[uncertain_positions].tolist()

    # ── Re-query LLM for uncertain examples ───────────────────────────────
    updated = labeler.relabel_uncertain(dataset, labeled, uncertain_global_idx)

    # Rows that actually changed label
    changed = [
        i for i in uncertain_global_idx
        if updated.records[i].llm_label != labeled.records[i].llm_label
    ]
    relabeled_rows = []
    for i in uncertain_global_idx:
        relabeled_rows.append({
            "index": i,
            "old_label": labeled.records[i].llm_label,
            "new_label": updated.records[i].llm_label,
            "confidence": updated.records[i].confidence,
        })
    relabeled_frame = pd.DataFrame(relabeled_rows)

    if retrain:
        # Rebuild y_llm for the training split with updated labels
        yl_train_new = updated.y_llm.iloc[X_train.index.tolist()].values
        pipe2 = _build_pipeline(X_train)
        pipe2.set_params(svc__kernel=kernel)
        if svc_params:
            pipe2.set_params(**{f"svc__{k}": v for k, v in svc_params.items()})
        pipe2.fit(X_train, yl_train_new)
        after_pred = pipe2.predict(X_test)
        after_acc = float(accuracy_score(yt_test, after_pred))
        after_f1 = float(f1_score(yt_test, after_pred, average="macro", zero_division=0))
    else:
        after_acc = before_acc
        after_f1 = before_f1

    return UBSResult(
        n_uncertain=n_to_query,
        relabeled_count=len(changed),
        before_accuracy=before_acc,
        after_accuracy=after_acc,
        before_macro_f1=before_f1,
        after_macro_f1=after_f1,
        delta_accuracy=after_acc - before_acc,
        changed_indices=changed,
        relabeled_frame=relabeled_frame,
    )
