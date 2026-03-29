from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"


@dataclass(frozen=True)
class SvmDataset:
    key: str
    title: str
    level: str
    description: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    candidate_kernels: tuple[str, ...]
    is_two_dimensional: bool = False


@dataclass(frozen=True)
class DemoFrameSource:
    key: str
    title: str
    group: str
    description: str
    loader: str
    path: str | None = None


def load_svm_datasets() -> list[SvmDataset]:
    iris = load_iris()
    breast = load_breast_cancer()
    digits = load_digits()
    digit_feature_names = [f"pixel_{row}_{col}" for row in range(8) for col in range(8)]

    return [
        SvmDataset(
            key="simple_iris",
            title="Iris Petal View",
            level="simple",
            description="Classic 3-class flower dataset restricted to petal length and width for clear boundary plots.",
            X=iris.data[:, [2, 3]],
            y=iris.target,
            feature_names=[iris.feature_names[2], iris.feature_names[3]],
            target_names=[str(name) for name in iris.target_names],
            candidate_kernels=("linear", "rbf", "poly"),
            is_two_dimensional=True,
        ),
        SvmDataset(
            key="medium_breast_cancer",
            title="Breast Cancer Wisconsin",
            level="medium",
            description="Well-known binary medical classification dataset with 30 continuous features.",
            X=breast.data,
            y=breast.target,
            feature_names=list(breast.feature_names),
            target_names=[str(name) for name in breast.target_names],
            candidate_kernels=("linear", "rbf", "poly"),
            is_two_dimensional=False,
        ),
        SvmDataset(
            key="complex_digits",
            title="Digits",
            level="complex",
            description="Multi-class handwritten digit recognition with 64 image-intensity features.",
            X=digits.data,
            y=digits.target,
            feature_names=digit_feature_names,
            target_names=[str(label) for label in digits.target_names],
            candidate_kernels=("linear", "rbf"),
            is_two_dimensional=False,
        ),
    ]


def load_demo_sources() -> list[DemoFrameSource]:
    sources = [
        DemoFrameSource(
            key="simple_iris",
            title="Iris Petal View",
            group="svm",
            description="Classic 2-feature SVM boundary demo built from the Iris dataset.",
            loader="svm",
        ),
        DemoFrameSource(
            key="medium_breast_cancer",
            title="Breast Cancer Wisconsin",
            group="svm",
            description="Well-known binary medical classification dataset with 30 continuous features.",
            loader="svm",
        ),
        DemoFrameSource(
            key="complex_digits",
            title="Digits",
            group="svm",
            description="Handwritten digit classification with 64 image-intensity features.",
            loader="svm",
        ),
    ]

    external_sources = [
        DemoFrameSource(
            key="itemset_uci_onlineretail",
            title="Online Retail Baskets",
            group="itemset",
            description="Prepared invoice-level spreadsheet from the UCI Online Retail dataset for frequent itemset mining.",
            loader="csv",
            path="itemset_uci_onlineretail.csv",
        ),
        DemoFrameSource(
            key="episode_uci_msnbc",
            title="MSNBC Journey Sequences",
            group="episode",
            description="Prepared clickstream spreadsheet from the UCI MSNBC dataset for episode mining.",
            loader="csv",
            path="episode_uci_msnbc.csv",
        ),
        DemoFrameSource(
            key="cancer_uci",
            title="Cancer UCI",
            group="classification",
            description="Cancer classification spreadsheet exported from the Wisconsin Diagnostic Breast Cancer dataset.",
            loader="csv",
            path="cancer_uci.csv",
        ),
        DemoFrameSource(
            key="fraud_openml",
            title="Fraud OpenML",
            group="classification",
            description="Repo-safe stratified sample from OpenML credit card fraud dataset 1597.",
            loader="csv",
            path="fraud_openml.csv",
        ),
    ]

    for source in external_sources:
        if source.path and (EXTERNAL_DATA_DIR / source.path).exists():
            sources.append(source)

    return sources


def load_demo_frame_by_title(title: str) -> pd.DataFrame:
    source_lookup = {source.title: source for source in load_demo_sources()}
    source = source_lookup[title]

    if source.loader == "svm":
        dataset_lookup = {dataset.title: dataset for dataset in load_svm_datasets()}
        dataset = dataset_lookup[title]
        frame = pd.DataFrame(dataset.X, columns=dataset.feature_names)
        frame["target"] = [dataset.target_names[int(value)] for value in dataset.y]
        return frame

    if source.loader == "csv" and source.path:
        return pd.read_csv(EXTERNAL_DATA_DIR / source.path)

    raise ValueError(f"Unsupported demo source: {title}")
