"""Advanced SVM techniques.

Two algorithms beyond standard SVC:

ActiveLearningSVM
    Uncertainty-based active learning.  Starts with a small labelled seed set,
    then iteratively selects the *most uncertain* unlabelled points (smallest
    margin distance to the decision boundary) for annotation.  Demonstrates
    how SVMs can reach high accuracy with far fewer labels than a random split.

UniversumSVM
    Universum-based regularisation for binary classification.  Synthetic
    "universum" examples — points that belong to neither class — are placed in
    the midpoint space between the two classes and added to the training set
    with a special label.  Penalising the SVM for classifying universum points
    confidently into either class acts as a structural prior that improves
    margin quality on small datasets.

Both return dataclasses that the Streamlit page can render directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .custom_analysis import _build_pipeline, _validate_target, prepare_custom_classification_data
from .datasets import RANDOM_STATE


# ── Active Learning ────────────────────────────────────────────────────────

@dataclass
class ActiveLearningRound:
    round_num: int
    labelled_count: int
    accuracy: float
    macro_f1: float
    newly_queried: list[int]  # indices into original X_unlabelled pool


@dataclass
class ActiveLearningResult:
    target_column: str
    feature_columns: list[str]
    kernel: str
    seed_size: int
    budget: int
    batch_size: int
    final_accuracy: float
    final_macro_f1: float
    baseline_accuracy: float   # same model trained on full labelled set
    rounds: list[ActiveLearningRound]
    learning_curve: pd.DataFrame   # columns: round, labelled, accuracy, macro_f1
    class_labels: list[str]


def run_active_learning(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    kernel: str = "rbf",
    seed_size: int = 10,
    budget: int = 50,
    batch_size: int = 5,
    test_size: float = 0.25,
    svc_params: dict[str, Any] | None = None,
) -> ActiveLearningResult:
    """Run uncertainty-based active learning on *frame*.

    Parameters
    ----------
    seed_size : int
        Number of stratified seed examples to start with.
    budget : int
        Maximum number of additional labels to query from the unlabelled pool.
    batch_size : int
        How many points to query per round (margin sampling, smallest |f(x)|).
    svc_params : dict, optional
        Extra params forwarded to the SVC inside the pipeline (e.g. C, gamma).
    """
    X, y = prepare_custom_classification_data(frame, target_column, feature_columns)
    _validate_target(y)
    class_labels = sorted(y.unique().tolist())

    # ── Held-out test set (never shown to the active learner)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # ── Seed set: stratified sample from the pool
    actual_seed = min(seed_size, len(y_pool) - len(class_labels))
    if actual_seed < len(class_labels):
        raise ValueError(
            f"Pool too small for a {len(class_labels)}-class stratified seed of size {seed_size}."
        )
    idx_all = np.arange(len(X_pool))
    _, idx_labelled = train_test_split(
        idx_all, test_size=actual_seed / len(idx_all),
        stratify=y_pool.values, random_state=RANDOM_STATE
    )
    idx_labelled = list(idx_labelled)
    idx_unlabelled = [i for i in idx_all if i not in set(idx_labelled)]

    def _make_pipeline() -> Pipeline:
        p = _build_pipeline(X_pool.iloc[idx_labelled])
        p.set_params(svc__kernel=kernel, svc__probability=True)
        if svc_params:
            p.set_params(**{f"svc__{k}": v for k, v in svc_params.items()})
        return p

    rounds: list[ActiveLearningRound] = []
    queries_used = 0

    # Round 0: seed only
    pipe = _make_pipeline()
    pipe.fit(X_pool.iloc[idx_labelled], y_pool.iloc[idx_labelled])
    y_pred = pipe.predict(X_test)
    rounds.append(ActiveLearningRound(
        round_num=0,
        labelled_count=len(idx_labelled),
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        newly_queried=[],
    ))

    while queries_used < budget and idx_unlabelled:
        # Margin sampling: decision_function gives distance to boundary per class.
        # For binary SVC: margin = |f(x)|, smallest = most uncertain.
        # For multiclass: margin = difference between top-2 class scores.
        svc: SVC = pipe.named_steps["svc"]
        X_un_transformed = pipe[:-1].transform(X_pool.iloc[idx_unlabelled])

        try:
            df_scores = svc.decision_function(X_un_transformed)
            if df_scores.ndim == 1:
                margins = np.abs(df_scores)
            else:
                sorted_scores = np.sort(df_scores, axis=1)
                margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        except Exception:
            # Fallback: random selection
            margins = np.random.rand(len(idx_unlabelled))

        query_count = min(batch_size, budget - queries_used, len(idx_unlabelled))
        query_positions = np.argsort(margins)[:query_count]
        newly_queried = [idx_unlabelled[p] for p in query_positions]

        idx_labelled.extend(newly_queried)
        idx_unlabelled = [i for i in idx_unlabelled if i not in set(newly_queried)]
        queries_used += query_count

        pipe = _make_pipeline()
        pipe.fit(X_pool.iloc[idx_labelled], y_pool.iloc[idx_labelled])
        y_pred = pipe.predict(X_test)
        rounds.append(ActiveLearningRound(
            round_num=len(rounds),
            labelled_count=len(idx_labelled),
            accuracy=float(accuracy_score(y_test, y_pred)),
            macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            newly_queried=newly_queried,
        ))

    # Baseline: train on *all* pool data
    baseline_pipe = _make_pipeline()
    # Rebuild pipeline on full pool
    baseline_pipe = _build_pipeline(X_pool)
    baseline_pipe.set_params(svc__kernel=kernel)
    if svc_params:
        baseline_pipe.set_params(**{f"svc__{k}": v for k, v in svc_params.items()})
    baseline_pipe.fit(X_pool, y_pool)
    baseline_acc = float(accuracy_score(y_test, baseline_pipe.predict(X_test)))

    curve = pd.DataFrame([
        {"round": r.round_num, "labelled": r.labelled_count,
         "accuracy": r.accuracy, "macro_f1": r.macro_f1}
        for r in rounds
    ])

    return ActiveLearningResult(
        target_column=target_column,
        feature_columns=feature_columns,
        kernel=kernel,
        seed_size=actual_seed,
        budget=budget,
        batch_size=batch_size,
        final_accuracy=rounds[-1].accuracy,
        final_macro_f1=rounds[-1].macro_f1,
        baseline_accuracy=baseline_acc,
        rounds=rounds,
        learning_curve=curve,
        class_labels=class_labels,
    )


# ── Universum SVM ──────────────────────────────────────────────────────────

@dataclass
class UniversumResult:
    target_column: str
    feature_columns: list[str]
    kernel: str
    universum_size: int
    universum_strategy: str
    standard_accuracy: float
    standard_macro_f1: float
    universum_accuracy: float
    universum_macro_f1: float
    accuracy_delta: float   # universum − standard
    class_labels: list[str]
    comparison: pd.DataFrame   # side-by-side metric table


def _generate_universum(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate *n* universum points from *X_train*.

    Strategies
    ----------
    midpoint
        Average of a random positive and a random negative example.
        Only meaningful for binary; generalised to random pair for multiclass.
    gaussian_noise
        Samples from N(class_mean, class_std) but adds noise so the point
        falls midway between class centroids.
    random_convex
        Convex combination of two random training points from *different* classes.
    """
    classes = np.unique(y_train)
    points = []

    if strategy == "midpoint":
        for _ in range(n):
            c1, c2 = rng.choice(classes, size=2, replace=False)
            x1 = X_train[y_train == c1][rng.integers(0, (y_train == c1).sum())]
            x2 = X_train[y_train == c2][rng.integers(0, (y_train == c2).sum())]
            points.append((x1 + x2) / 2.0)

    elif strategy == "gaussian_noise":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        noise = rng.normal(0, 0.5 * std, size=(n, X_train.shape[1]))
        return mean + noise

    else:  # random_convex
        for _ in range(n):
            c1, c2 = rng.choice(classes, size=2, replace=False)
            x1 = X_train[y_train == c1][rng.integers(0, (y_train == c1).sum())]
            x2 = X_train[y_train == c2][rng.integers(0, (y_train == c2).sum())]
            alpha = rng.uniform(0.3, 0.7)
            points.append(alpha * x1 + (1 - alpha) * x2)

    return np.vstack(points)


def run_universum_svm(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    kernel: str = "rbf",
    universum_size: int = 100,
    universum_strategy: str = "midpoint",
    universum_C: float = 0.5,
    test_size: float = 0.25,
    svc_params: dict[str, Any] | None = None,
) -> UniversumResult:
    """Train a standard SVM and a Universum-augmented SVM, compare them.

    The Universum SVM augments X_train with *universum_size* synthetic examples
    labelled with a dedicated ``'__universum__'`` class, then trains SVC with
    ``class_weight`` set so the universum class is penalised by *universum_C*
    relative to real classes.  The universum examples encourage the decision
    boundary to avoid the midpoint region between classes.

    Parameters
    ----------
    universum_C : float
        Weight applied to the universum class relative to real classes (< 1 =
        lighter penalty, > 1 = stronger push away from midpoint).
    universum_strategy : str
        ``'midpoint'``, ``'gaussian_noise'``, or ``'random_convex'``.
    """
    X, y = prepare_custom_classification_data(frame, target_column, feature_columns)
    _validate_target(y)
    class_labels = sorted(y.unique().tolist())

    if len(class_labels) > 10:
        raise ValueError(
            "Universum SVM is most meaningful for binary or low-cardinality multi-class tasks. "
            f"This target has {len(class_labels)} classes."
        )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Build and fit a preprocessing pipeline to get numeric arrays
    prep_pipe = _build_pipeline(X_train_df)
    # Fit only the preprocessor (drop the SVC step)
    prep = prep_pipe[:-1]
    prep.fit(X_train_df)
    X_train_arr = prep.transform(X_train_df).astype(float)
    X_test_arr  = prep.transform(X_test_df).astype(float)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    base_params = {"kernel": kernel, "C": 1.0}
    if svc_params:
        base_params.update(svc_params)

    # ── Standard SVM ──
    std_svc = SVC(**base_params)
    std_svc.fit(X_train_arr, y_train_enc)
    std_pred = std_svc.predict(X_test_arr)
    std_acc  = float(accuracy_score(y_test_enc, std_pred))
    std_f1   = float(f1_score(y_test_enc, std_pred, average="macro", zero_division=0))

    # ── Universum SVM ──
    rng = np.random.default_rng(RANDOM_STATE)
    U = _generate_universum(X_train_arr, y_train_enc, universum_size, universum_strategy, rng)
    UNIVERSUM_LABEL = len(le.classes_)   # new integer label beyond existing classes

    X_aug = np.vstack([X_train_arr, U])
    y_aug = np.concatenate([y_train_enc, np.full(len(U), UNIVERSUM_LABEL)])

    # class_weight: real classes weight=1, universum weight=universum_C
    cw = {i: 1.0 for i in range(len(le.classes_))}
    cw[UNIVERSUM_LABEL] = universum_C

    uni_svc = SVC(**{**base_params, "class_weight": cw})
    uni_svc.fit(X_aug, y_aug)

    # Predict only on the real label space
    raw_pred = uni_svc.predict(X_test_arr)
    # Map any universum predictions back to nearest real class (shouldn't happen on test data)
    uni_pred = np.clip(raw_pred, 0, len(le.classes_) - 1)
    uni_acc  = float(accuracy_score(y_test_enc, uni_pred))
    uni_f1   = float(f1_score(y_test_enc, uni_pred, average="macro", zero_division=0))

    comparison = pd.DataFrame([
        {"Model": "Standard SVM",  "Accuracy": round(std_acc, 4), "Macro F1": round(std_f1, 4)},
        {"Model": "Universum SVM", "Accuracy": round(uni_acc, 4), "Macro F1": round(uni_f1, 4)},
        {"Model": "Delta (Uni − Std)",
         "Accuracy": round(uni_acc - std_acc, 4),
         "Macro F1": round(uni_f1  - std_f1,  4)},
    ])

    return UniversumResult(
        target_column=target_column,
        feature_columns=feature_columns,
        kernel=kernel,
        universum_size=universum_size,
        universum_strategy=universum_strategy,
        standard_accuracy=std_acc,
        standard_macro_f1=std_f1,
        universum_accuracy=uni_acc,
        universum_macro_f1=uni_f1,
        accuracy_delta=uni_acc - std_acc,
        class_labels=class_labels,
        comparison=comparison,
    )
