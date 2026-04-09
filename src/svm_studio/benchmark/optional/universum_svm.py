"""Universum SVM optional technique for the benchmark pipeline.

Wraps ``run_universum_svm`` from ``advanced_svm.py`` and applies it to
LLM-labeled data to test whether synthetic midpoint examples improve
robustness to label noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ...advanced_svm import run_universum_svm, UniversumResult
from ..dataset_loader import StandardDataset
from ..llm_labeler import LabeledDataset


@dataclass
class BenchUniversumResult:
    """Universum SVM result in benchmark context."""
    universum_result: UniversumResult
    label_noise_rate: float             # 1 - agreement_rate
    comparison: pd.DataFrame            # re-exposed for convenience


def run_bench_universum_svm(
    dataset: StandardDataset,
    labeled: LabeledDataset,
    kernel: str = "rbf",
    universum_size: int = 100,
    universum_strategy: str = "midpoint",
    universum_C: float = 0.5,
    test_size: float = 0.20,
    svc_params: dict[str, Any] | None = None,
) -> BenchUniversumResult:
    """Compare standard vs Universum SVM on LLM-labeled data.

    The Universum SVM typically helps when labels are noisy (as LLM labels
    often are), because the synthetic midpoint examples provide structural
    regularisation that counteracts memorising flipped labels.
    """
    n = len(labeled.y_llm)
    X = dataset.X.iloc[:n].reset_index(drop=True)
    y_llm = labeled.y_llm.reset_index(drop=True)

    # Build a temporary frame combining X + LLM labels for advanced_svm API
    frame = X.copy()
    target_col = "__llm_label__"
    frame[target_col] = y_llm.values
    feature_cols = list(X.columns)

    result = run_universum_svm(
        frame=frame,
        target_column=target_col,
        feature_columns=feature_cols,
        kernel=kernel,
        universum_size=universum_size,
        universum_strategy=universum_strategy,
        universum_C=universum_C,
        test_size=test_size,
        svc_params=svc_params,
    )

    return BenchUniversumResult(
        universum_result=result,
        label_noise_rate=1.0 - labeled.agreement_rate,
        comparison=result.comparison,
    )
