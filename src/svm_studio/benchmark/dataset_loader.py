"""Unified dataset loader for the benchmark pipeline.

Supports multiple sources under a single ``DatasetLoader.load(source, name)``
interface.  Returns a ``StandardDataset`` that every downstream module expects.

Sources
-------
``"sklearn"``   Built-in or fetchable sklearn datasets by function name.
``"openml"``    OpenML datasets by name or numeric ID (via ``fetch_openml``).
``"ucimlrepo"`` UCI ML Repository datasets by name or numeric ID.
``"huggingface"`` HuggingFace datasets by dataset/config string.
``"csv"``       Local CSV path.  ``name`` is the file path; optionally pass
                ``target_column`` as a keyword.

Guard imports keep the module importable even when optional packages are absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── optional package guards ────────────────────────────────────────────────
try:
    import ucimlrepo  # type: ignore[import-untyped]
    _UCI_AVAILABLE = True
except ImportError:
    _UCI_AVAILABLE = False

try:
    import datasets as hf_datasets  # type: ignore[import-untyped]
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


@dataclass
class StandardDataset:
    """Normalised dataset object consumed by all benchmark modules."""
    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    class_names: list[str]
    description: str
    task_type: str          # "classification" | "regression"
    data_type: str          # "tabular" | "text" | "image_features" | "sequential"
    n_examples: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_examples = len(self.X)
        self.n_classes = len(self.class_names)


class DatasetLoader:
    """Load datasets from any supported source into a ``StandardDataset``.

    Parameters
    ----------
    random_state : int
        Used when subsetting large datasets.
    max_rows : int or None
        Cap the number of rows loaded (avoids blowing out memory on very large
        datasets like rcv1 or kddcup99).  None = no limit.
    """

    _SKLEARN_LOADERS = {
        "iris": ("load_iris", "tabular"),
        "wine": ("load_wine", "tabular"),
        "breast_cancer": ("load_breast_cancer", "tabular"),
        "digits": ("load_digits", "image_features"),
        "20newsgroups": ("fetch_20newsgroups", "text"),
        "olivetti_faces": ("fetch_olivetti_faces", "image_features"),
        "california_housing": ("fetch_california_housing", "tabular"),
        "covtype": ("fetch_covtype", "tabular"),
        "kddcup99": ("fetch_kddcup99", "tabular"),
        "rcv1": ("fetch_rcv1", "text"),
        "lfw_people": ("fetch_lfw_people", "image_features"),
    }

    def __init__(self, random_state: int = 42, max_rows: int | None = 5_000) -> None:
        self.random_state = random_state
        self.max_rows = max_rows

    # ── public API ────────────────────────────────────────────────────────

    def load(self, source: str, name: str, **kwargs: Any) -> StandardDataset:
        """Load a dataset.

        Parameters
        ----------
        source : str
            One of ``"sklearn"``, ``"openml"``, ``"ucimlrepo"``,
            ``"huggingface"``, ``"csv"``.
        name : str
            Dataset identifier — function name for sklearn, dataset name or
            numeric ID for openml/ucimlrepo, ``owner/dataset`` for
            HuggingFace, or file path for csv.
        **kwargs
            Extra arguments forwarded to the specific loader
            (e.g. ``target_column="label"`` for CSV, ``subset="train"`` for
            HuggingFace, ``config="default"`` for HuggingFace).
        """
        source = source.lower().strip()
        dispatch = {
            "sklearn": self._load_sklearn,
            "openml": self._load_openml,
            "ucimlrepo": self._load_ucimlrepo,
            "huggingface": self._load_huggingface,
            "hf": self._load_huggingface,
            "csv": self._load_csv,
        }
        if source not in dispatch:
            raise ValueError(f"Unknown source {source!r}. Options: {list(dispatch)}")
        return dispatch[source](name, **kwargs)

    # ── sklearn ───────────────────────────────────────────────────────────

    def _load_sklearn(self, name: str, **kwargs: Any) -> StandardDataset:
        import sklearn.datasets as sk_ds

        key = name.lower().replace("-", "_")
        if key not in self._SKLEARN_LOADERS:
            # Try fetching by name directly (covers fetch_openml alias, etc.)
            raise ValueError(
                f"Unknown sklearn dataset {name!r}. "
                f"Known names: {list(self._SKLEARN_LOADERS)}"
            )

        loader_name, data_type = self._SKLEARN_LOADERS[key]
        loader_fn = getattr(sk_ds, loader_name)

        # text datasets need different handling
        if loader_name == "fetch_20newsgroups":
            bunch = loader_fn(subset=kwargs.get("subset", "all"), remove=("headers", "footers", "quotes"))
            X = pd.DataFrame({"text": bunch.data})
            y = pd.Series([bunch.target_names[i] for i in bunch.target], name="category")
            feature_names = ["text"]
            class_names = list(bunch.target_names)
            desc = "20 Newsgroups — 18,846 news articles across 20 topic categories."
        elif loader_name == "fetch_rcv1":
            bunch = loader_fn(subset=kwargs.get("subset", "train"))
            X_arr = bunch.data.toarray() if hasattr(bunch.data, "toarray") else np.array(bunch.data)
            X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])
            y = pd.Series(
                [bunch.target_names[np.argmax(row)] for row in bunch.target.toarray()],
                name="category"
            )
            feature_names = list(X.columns)
            class_names = list(bunch.target_names)
            desc = "RCV1 — Reuters news corpus for text categorisation."
            data_type = "text"
        elif loader_name == "fetch_california_housing":
            bunch = loader_fn()
            X = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
            # Bin target into quintiles for classification
            y_cont = pd.Series(bunch.target, name="price_band")
            y = pd.cut(y_cont, bins=5, labels=["very_low", "low", "medium", "high", "very_high"])
            y = y.astype(str)
            feature_names = list(bunch.feature_names)
            class_names = ["very_low", "low", "medium", "high", "very_high"]
            desc = "California Housing — median house prices (binned into 5 price bands for classification)."
        elif loader_name in ("fetch_covtype",):
            bunch = loader_fn()
            X = pd.DataFrame(bunch.data, columns=[f"feat_{i}" for i in range(bunch.data.shape[1])])
            y = pd.Series([str(c) for c in bunch.target], name="cover_type")
            feature_names = list(X.columns)
            class_names = sorted(y.unique().tolist())
            desc = "Forest Cover Type — 581,012 examples, 54 features, 7 cover type classes."
        elif loader_name == "fetch_kddcup99":
            bunch = loader_fn(percent10=True)
            X_arr = bunch.data
            if hasattr(X_arr[0], "__len__"):
                X = pd.DataFrame(list(X_arr), columns=[f"feat_{i}" for i in range(len(X_arr[0]))])
            else:
                X = pd.DataFrame(X_arr.reshape(-1, 1), columns=["feat_0"])
            y = pd.Series([str(t.decode() if isinstance(t, bytes) else t) for t in bunch.target], name="attack_type")
            feature_names = list(X.columns)
            class_names = sorted(y.unique().tolist())
            desc = "KDD Cup 99 — network intrusion detection (10% subset)."
        elif loader_name in ("fetch_olivetti_faces", "fetch_lfw_people"):
            bunch = loader_fn()
            X = pd.DataFrame(bunch.data, columns=[f"pixel_{i}" for i in range(bunch.data.shape[1])])
            y = pd.Series([str(c) for c in bunch.target], name="person_id")
            feature_names = list(X.columns)
            class_names = sorted(y.unique().tolist())
            desc = f"{name} — face images as flattened pixel feature vectors."
        else:
            # Standard load_xxx datasets (iris, wine, breast_cancer, digits)
            bunch = loader_fn()
            X = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
            if hasattr(bunch, "target_names"):
                y = pd.Series([bunch.target_names[i] for i in bunch.target], name="target")
                class_names = list(bunch.target_names)
            else:
                y = pd.Series([str(t) for t in bunch.target], name="target")
                class_names = sorted(y.unique().tolist())
            feature_names = list(bunch.feature_names)
            desc = getattr(bunch, "DESCR", f"sklearn {name} dataset")[:300]

        task_type = (
            "regression"
            if key in ("california_housing",) and "price_band" not in locals()
            else "classification"
        )

        ds = StandardDataset(
            name=name, X=X, y=y,
            feature_names=feature_names, class_names=class_names,
            description=desc, task_type="classification", data_type=data_type,
        )
        return self._cap_rows(ds)

    # ── OpenML ────────────────────────────────────────────────────────────

    def _load_openml(self, name: str, **kwargs: Any) -> StandardDataset:
        from sklearn.datasets import fetch_openml

        as_frame = True
        try:
            # name can be a string name or numeric ID
            dataset_id = int(name)
            bunch = fetch_openml(data_id=dataset_id, as_frame=as_frame, parser="auto")
        except ValueError:
            bunch = fetch_openml(name=name, as_frame=as_frame, parser="auto")

        X = bunch.data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = bunch.target
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")
        y = y.astype(str)

        feature_names = list(X.columns)
        class_names = sorted(y.unique().tolist())
        desc = getattr(bunch, "DESCR", f"OpenML dataset {name}")[:300]

        ds = StandardDataset(
            name=str(name), X=X, y=y,
            feature_names=feature_names, class_names=class_names,
            description=desc, task_type="classification", data_type="tabular",
        )
        return self._cap_rows(ds)

    # ── UCI ML Repository ─────────────────────────────────────────────────

    def _load_ucimlrepo(self, name: str, **kwargs: Any) -> StandardDataset:
        if not _UCI_AVAILABLE:
            raise ImportError(
                "ucimlrepo is not installed. Run: pip install ucimlrepo"
            )
        try:
            dataset_id = int(name)
            dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
        except ValueError:
            dataset = ucimlrepo.fetch_ucirepo(name=name)

        X = dataset.data.features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y_raw = dataset.data.targets
        if isinstance(y_raw, pd.DataFrame):
            y = y_raw.iloc[:, 0].astype(str)
        else:
            y = pd.Series(y_raw, name="target").astype(str)

        feature_names = list(X.columns)
        class_names = sorted(y.unique().tolist())
        desc = str(getattr(dataset.metadata, "abstract", f"UCI dataset {name}"))[:300]

        ds = StandardDataset(
            name=str(name), X=X, y=y,
            feature_names=feature_names, class_names=class_names,
            description=desc, task_type="classification", data_type="tabular",
        )
        return self._cap_rows(ds)

    # ── HuggingFace ───────────────────────────────────────────────────────

    def _load_huggingface(self, name: str, **kwargs: Any) -> StandardDataset:
        if not _HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library not installed. Run: pip install datasets"
            )
        config = kwargs.get("config", None)
        split = kwargs.get("subset", kwargs.get("split", "train"))

        hf_ds = hf_datasets.load_dataset(name, config, split=split, trust_remote_code=False)
        df = hf_ds.to_pandas()

        # Infer label/text columns
        label_col = kwargs.get("label_column") or _infer_label_column(df)
        text_col = kwargs.get("text_column") or _infer_text_column(df)

        if label_col is None or text_col is None:
            raise ValueError(
                f"Cannot auto-detect text/label columns for HuggingFace dataset {name!r}. "
                f"Pass text_column= and label_column= explicitly."
            )

        y = df[label_col].astype(str)
        X = df[[text_col]].rename(columns={text_col: "text"})
        feature_names = ["text"]
        class_names = sorted(y.unique().tolist())

        ds = StandardDataset(
            name=name, X=X, y=y,
            feature_names=feature_names, class_names=class_names,
            description=f"HuggingFace dataset: {name} (split={split})",
            task_type="classification", data_type="text",
        )
        return self._cap_rows(ds)

    # ── CSV ───────────────────────────────────────────────────────────────

    def _load_csv(self, name: str, **kwargs: Any) -> StandardDataset:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        target_col = kwargs.get("target_column") or _infer_label_column(df)
        if target_col is None:
            raise ValueError(
                "Cannot auto-detect the target column. Pass target_column= explicitly."
            )

        y = df[target_col].astype(str)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        class_names = sorted(y.unique().tolist())

        ds = StandardDataset(
            name=path.stem, X=X, y=y,
            feature_names=feature_cols, class_names=class_names,
            description=f"Local CSV: {path.name}",
            task_type="classification", data_type="tabular",
        )
        return self._cap_rows(ds)

    # ── helpers ───────────────────────────────────────────────────────────

    def _cap_rows(self, ds: StandardDataset) -> StandardDataset:
        """Randomly subsample to ``max_rows`` without replacement."""
        if self.max_rows is None or len(ds.X) <= self.max_rows:
            return ds
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(ds.X), size=self.max_rows, replace=False)
        idx = sorted(idx.tolist())
        return StandardDataset(
            name=ds.name,
            X=ds.X.iloc[idx].reset_index(drop=True),
            y=ds.y.iloc[idx].reset_index(drop=True),
            feature_names=ds.feature_names,
            class_names=ds.class_names,
            description=ds.description,
            task_type=ds.task_type,
            data_type=ds.data_type,
        )


# ── column inference helpers ───────────────────────────────────────────────

_LABEL_HINTS = re.compile(
    r"^(label|target|class|category|y|output|answer|tag|sentiment|intent)$", re.I
)
_TEXT_HINTS = re.compile(r"^(text|sentence|content|review|document|body|tweet|message)$", re.I)


def _infer_label_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _LABEL_HINTS.match(col):
            return col
    # Fall back to last column if it looks categorical
    last = df.columns[-1]
    if df[last].nunique() <= 50:
        return last
    return None


def _infer_text_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _TEXT_HINTS.match(col):
            return col
    # Find first object column with long strings
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 20:
                return col
    return None
