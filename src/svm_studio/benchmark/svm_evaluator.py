"""SVM evaluation of LLM label quality.

Trains two SVMs on the same feature set:
  - LLM-labeled SVM   → uses LLM-assigned labels from ``LLMLabeler``
  - Control SVM       → uses ground-truth labels

Reports 5-fold stratified CV, a 20% held-out test set, confusion matrices,
per-class metrics, and a disagreement table (examples where LLM ≠ truth).

Re-uses ``_build_pipeline`` from ``custom_analysis.py`` for consistent
feature preprocessing across the whole workbench.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ..custom_analysis import _build_pipeline, prepare_custom_classification_data
from ..datasets import RANDOM_STATE
from .dataset_loader import StandardDataset
from .llm_labeler import LabeledDataset


@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float


@dataclass
class SVMRunMetrics:
    """Metrics for a single SVM (LLM-trained or control)."""
    label: str                          # "llm" or "control"
    test_accuracy: float
    test_macro_f1: float
    test_macro_precision: float
    test_macro_recall: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    cv_folds: list[FoldMetrics]
    confusion: np.ndarray
    class_report: pd.DataFrame          # rows = classes, cols = p/r/f1/support


@dataclass
class EvalResult:
    """Full evaluation result comparing LLM-trained vs control SVM."""
    dataset_name: str
    llm_model: str
    feature_columns: list[str]
    class_names: list[str]
    n_train: int
    n_test: int
    llm_metrics: SVMRunMetrics
    control_metrics: SVMRunMetrics
    labeling_cost: float                # control_test_acc - llm_test_acc
    llm_agreement_rate: float           # fraction where y_llm == y_true
    disagreements: pd.DataFrame         # rows where LLM label ≠ truth
    most_common_error: str              # "true_label → llm_label"
    worst_class: str                    # class with highest LLM error rate
    best_class: str                     # class with lowest LLM error rate


class SVMEvaluator:
    """Compare SVM performance under LLM labels vs ground-truth labels.

    Parameters
    ----------
    kernel : str
        SVM kernel to use for both models.  Defaults to ``"rbf"``.
    test_size : float
        Fraction of data held out for the final test evaluation.
    n_folds : int
        Number of stratified CV folds.
    svc_params : dict, optional
        Extra params forwarded to SVC (e.g. ``{"C": 5, "gamma": "scale"}``).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        test_size: float = 0.20,
        n_folds: int = 5,
        svc_params: dict[str, Any] | None = None,
    ) -> None:
        self.kernel = kernel
        self.test_size = test_size
        self.n_folds = n_folds
        self.svc_params = svc_params or {}

    def evaluate(
        self,
        dataset: StandardDataset,
        labeled: LabeledDataset,
    ) -> EvalResult:
        """Run the full dual-SVM evaluation.

        Parameters
        ----------
        dataset : StandardDataset
            Source dataset (provides X and ground-truth y).
        labeled : LabeledDataset
            Output of ``LLMLabeler.label()`` (provides y_llm).
        """
        # Align lengths: labeled may cover a subset of the dataset
        n = len(labeled.y_llm)
        X = dataset.X.iloc[:n].reset_index(drop=True)
        y_true = labeled.y_true
        y_llm = labeled.y_llm

        # Stratify on true labels to avoid empty folds
        X_train, X_test, yt_train, yt_test, yl_train, yl_test = train_test_split(
            X, y_true, y_llm,
            test_size=self.test_size,
            stratify=y_true,
            random_state=RANDOM_STATE,
        )

        feature_cols = list(X.columns)
        llm_metrics = self._run_svm(X_train, X_test, yl_train, yt_test, "llm", dataset.class_names)
        control_metrics = self._run_svm(X_train, X_test, yt_train, yt_test, "control", dataset.class_names)

        disagreements = self._build_disagreements(X, y_true, y_llm)
        most_common_error, worst_class, best_class = self._error_analysis(y_true, y_llm, dataset.class_names)

        return EvalResult(
            dataset_name=dataset.name,
            llm_model=labeled.llm_model,
            feature_columns=feature_cols,
            class_names=dataset.class_names,
            n_train=len(X_train),
            n_test=len(X_test),
            llm_metrics=llm_metrics,
            control_metrics=control_metrics,
            labeling_cost=control_metrics.test_accuracy - llm_metrics.test_accuracy,
            llm_agreement_rate=labeled.agreement_rate,
            disagreements=disagreements,
            most_common_error=most_common_error,
            worst_class=worst_class,
            best_class=best_class,
        )

    # ── internal helpers ───────────────────────────────────────────────────

    def _run_svm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        label: str,
        class_names: list[str],
    ) -> SVMRunMetrics:
        pipe = _build_pipeline(X_train)
        pipe.set_params(svc__kernel=self.kernel)
        for k, v in self.svc_params.items():
            pipe.set_params(**{f"svc__{k}": v})

        # 5-fold CV on training data
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        fold_metrics = [
            FoldMetrics(fold=i, accuracy=float(cv_scores[i]), macro_f1=0.0,
                        macro_precision=0.0, macro_recall=0.0)
            for i in range(len(cv_scores))
        ]

        # Final fit on all training data, score on held-out test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        labels_present = sorted(set(y_test.unique()) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        report_dict = classification_report(
            y_test, y_pred, labels=labels_present, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T.drop(
            ["accuracy", "macro avg", "weighted avg"], errors="ignore"
        )

        return SVMRunMetrics(
            label=label,
            test_accuracy=float(accuracy_score(y_test, y_pred)),
            test_macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            test_macro_precision=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            test_macro_recall=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            cv_mean_accuracy=float(cv_scores.mean()),
            cv_std_accuracy=float(cv_scores.std()),
            cv_folds=fold_metrics,
            confusion=cm,
            class_report=report_df,
        )

    @staticmethod
    def _build_disagreements(
        X: pd.DataFrame, y_true: pd.Series, y_llm: pd.Series
    ) -> pd.DataFrame:
        mask = y_llm != y_true
        if not mask.any():
            return pd.DataFrame()
        disagreed = X[mask].copy()
        disagreed["true_label"] = y_true[mask].values
        disagreed["llm_label"] = y_llm[mask].values
        # Keep at most 5 feature columns so the table isn't unwieldy in the UI
        feat_cols = [c for c in X.columns[:5]]
        cols = feat_cols + ["true_label", "llm_label"]
        return disagreed[[c for c in cols if c in disagreed.columns]].reset_index(drop=True)

    @staticmethod
    def _error_analysis(
        y_true: pd.Series, y_llm: pd.Series, class_names: list[str]
    ) -> tuple[str, str, str]:
        errors = y_true[y_llm != y_true]
        llm_errors = y_llm[y_llm != y_true]

        # Most common error pair
        if len(errors):
            pairs = pd.Series([f"{t} → {l}" for t, l in zip(errors, llm_errors)])
            most_common_error = pairs.value_counts().index[0]
        else:
            most_common_error = "none"

        # Per-class error rates
        error_rates: dict[str, float] = {}
        for cls in class_names:
            cls_mask = y_true == cls
            if cls_mask.sum() == 0:
                continue
            error_rates[cls] = float((y_llm[cls_mask] != y_true[cls_mask]).mean())

        if error_rates:
            worst_class = max(error_rates, key=lambda c: error_rates[c])
            best_class = min(error_rates, key=lambda c: error_rates[c])
        else:
            worst_class = best_class = "n/a"

        return most_common_error, worst_class, best_class
