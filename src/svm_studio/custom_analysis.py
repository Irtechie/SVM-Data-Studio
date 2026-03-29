from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .datasets import RANDOM_STATE
from .svm_analysis import KERNEL_GRIDS


@dataclass
class CustomKernelRun:
    kernel: str
    estimator: Pipeline
    cv_accuracy: float
    cv_std: float
    test_accuracy: float
    macro_f1: float
    best_params: dict[str, Any]
    support_vector_count: int


@dataclass
class CustomSvmResult:
    target_column: str
    feature_columns: list[str]
    selected_kernel: str
    selected_params: dict[str, Any]
    selected_estimator: Pipeline
    test_accuracy: float
    macro_f1: float
    support_vector_count: int
    class_labels: list[str]
    confusion: np.ndarray
    kernel_results: pd.DataFrame
    classification_report: pd.DataFrame
    feature_importance: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray


def prepare_custom_classification_data(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")

    working = frame[feature_columns + [target_column]].copy()
    working = working.dropna(subset=[target_column])
    if working.empty:
        raise ValueError("No rows remain after removing missing target values.")

    X = working[feature_columns].copy()
    y = working[target_column].astype(str).copy()
    _validate_target(y)
    return X, y


def _split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def _build_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    numeric_columns, categorical_columns = _split_feature_types(feature_frame)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("svc", SVC()),
        ]
    )


def fit_custom_svm_estimator(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    kernel: str,
    best_params: dict[str, Any] | None = None,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X, y = prepare_custom_classification_data(frame, target_column, feature_columns)
    pipeline = _build_pipeline(X)
    params = dict(best_params or {})
    params["svc__kernel"] = kernel
    pipeline.set_params(**params)
    pipeline.fit(X, y)
    return pipeline, X, y


def _validate_target(target: pd.Series) -> None:
    class_count = target.nunique()
    if class_count < 2:
        raise ValueError("The target column needs at least two classes for SVM classification.")
    if class_count > 25:
        raise ValueError("The target column has too many unique values. Use a categorical classification target.")
    if target.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two rows so the train/test split can stay stratified.")


def run_custom_svm_analysis(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    kernels: list[str],
    test_size: float = 0.25,
) -> CustomSvmResult:
    if not kernels:
        raise ValueError("Choose at least one kernel.")
    X, y = prepare_custom_classification_data(frame, target_column, feature_columns)
    class_labels = sorted(y.unique().tolist())

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        raise ValueError("The selected target distribution is too small or too imbalanced for the requested split.") from exc

    min_class_size = int(y_train.value_counts().min())
    cv_splits = min(5, min_class_size)
    if cv_splits < 2:
        raise ValueError("The training split is too small for cross-validation. Add data or use fewer classes.")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    kernel_runs: list[CustomKernelRun] = []

    for kernel in kernels:
        search = GridSearchCV(
            estimator=_build_pipeline(X_train),
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
            CustomKernelRun(
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
    importance = permutation_importance(
        selected_run.estimator,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    kernel_results = pd.DataFrame(
        [
            {
                "kernel": run.kernel,
                "cv_accuracy": run.cv_accuracy,
                "cv_std": run.cv_std,
                "test_accuracy": run.test_accuracy,
                "macro_f1": run.macro_f1,
                "support_vector_count": run.support_vector_count,
                "best_params": str(run.best_params),
            }
            for run in kernel_runs
        ]
    )
    report_frame = (
        pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                labels=class_labels,
                output_dict=True,
                zero_division=0,
            )
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )
    importance_frame = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return CustomSvmResult(
        target_column=target_column,
        feature_columns=feature_columns,
        selected_kernel=selected_run.kernel,
        selected_params=selected_run.best_params,
        selected_estimator=selected_run.estimator,
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        support_vector_count=selected_run.support_vector_count,
        class_labels=class_labels,
        confusion=confusion_matrix(y_test, y_pred, labels=class_labels),
        kernel_results=kernel_results,
        classification_report=report_frame,
        feature_importance=importance_frame,
        y_test=y_test.reset_index(drop=True),
        y_pred=y_pred,
    )
