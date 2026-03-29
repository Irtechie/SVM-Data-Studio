from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .datasets import RANDOM_STATE, SvmDataset

TEST_SIZE = 0.25

KERNEL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "linear": {
        "svc__kernel": ["linear"],
        "svc__C": [0.1, 1.0, 10.0, 30.0],
    },
    "rbf": {
        "svc__kernel": ["rbf"],
        "svc__C": [0.5, 1.0, 5.0, 10.0],
        "svc__gamma": ["scale", 0.1, 0.01],
    },
    "poly": {
        "svc__kernel": ["poly"],
        "svc__C": [0.5, 1.0, 5.0],
        "svc__degree": [2, 3],
        "svc__gamma": ["scale"],
    },
}


@dataclass
class KernelRun:
    kernel: str
    estimator: BaseEstimator
    cv_accuracy: float
    cv_std: float
    test_accuracy: float
    macro_f1: float
    best_params: dict[str, Any]
    support_vector_count: int


@dataclass
class SvmStudyResult:
    dataset: SvmDataset
    kernel_runs: list[KernelRun]
    selected_kernel: str
    selected_estimator: BaseEstimator
    confusion: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    report: dict[str, Any]
    test_accuracy: float
    macro_f1: float
    support_vector_count: int


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC()),
        ]
    )


def run_svm_study(dataset: SvmDataset) -> SvmStudyResult:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=TEST_SIZE,
        stratify=dataset.y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kernel_runs: list[KernelRun] = []

    for kernel in dataset.candidate_kernels:
        search = GridSearchCV(
            estimator=_build_pipeline(),
            param_grid=KERNEL_GRIDS[kernel],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        svc = best_estimator.named_steps["svc"]

        kernel_runs.append(
            KernelRun(
                kernel=kernel,
                estimator=best_estimator,
                cv_accuracy=float(search.best_score_),
                cv_std=float(search.cv_results_["std_test_score"][search.best_index_]),
                test_accuracy=float(accuracy_score(y_test, y_pred)),
                macro_f1=float(f1_score(y_test, y_pred, average="macro")),
                best_params=dict(search.best_params_),
                support_vector_count=int(svc.n_support_.sum()),
            )
        )

    selected_run = max(
        kernel_runs,
        key=lambda run: (run.cv_accuracy, run.test_accuracy, run.macro_f1),
    )
    y_pred = selected_run.estimator.predict(X_test)

    return SvmStudyResult(
        dataset=dataset,
        kernel_runs=kernel_runs,
        selected_kernel=selected_run.kernel,
        selected_estimator=selected_run.estimator,
        confusion=confusion_matrix(y_test, y_pred),
        y_test=y_test,
        y_pred=y_pred,
        report=classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        support_vector_count=selected_run.support_vector_count,
    )


def run_all_svm_studies(datasets: list[SvmDataset]) -> list[SvmStudyResult]:
    return [run_svm_study(dataset) for dataset in datasets]


def kernel_runs_frame(results: list[SvmStudyResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        for run in result.kernel_runs:
            rows.append(
                {
                    "dataset_key": result.dataset.key,
                    "dataset_title": result.dataset.title,
                    "level": result.dataset.level,
                    "kernel": run.kernel,
                    "cv_accuracy": run.cv_accuracy,
                    "cv_std": run.cv_std,
                    "test_accuracy": run.test_accuracy,
                    "macro_f1": run.macro_f1,
                    "support_vector_count": run.support_vector_count,
                    "best_params": str(run.best_params),
                }
            )

    return pd.DataFrame(rows)


def selected_runs_frame(results: list[SvmStudyResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "dataset_key": result.dataset.key,
                "dataset_title": result.dataset.title,
                "level": result.dataset.level,
                "selected_kernel": result.selected_kernel,
                "test_accuracy": result.test_accuracy,
                "macro_f1": result.macro_f1,
                "support_vector_count": result.support_vector_count,
            }
        )

    return pd.DataFrame(rows)
