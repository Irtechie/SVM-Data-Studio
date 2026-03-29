from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .datasets import SvmDataset

BIN_LABELS = ["low", "medium", "high"]


@dataclass(frozen=True)
class TransactionDataset:
    name: str
    level: str
    description: str
    transactions: list[frozenset[str]]
    min_support: float


@dataclass(frozen=True)
class FrequentItemset:
    dataset_name: str
    level: str
    items: tuple[str, ...]
    length: int
    support: float
    count: int


def _quantile_bins(values: np.ndarray) -> list[str]:
    ranks = pd.Series(values).rank(method="first")
    categories = pd.qcut(ranks, q=3, labels=BIN_LABELS)
    return [str(label) for label in categories]


def _item_value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    return text if text else "<blank>"


def build_transactions_from_frame(
    frame: pd.DataFrame,
    columns: list[str],
    target_column: str | None = None,
) -> list[frozenset[str]]:
    selected_columns = list(columns)
    if target_column and target_column not in selected_columns:
        selected_columns.append(target_column)
    if not selected_columns:
        raise ValueError("Choose at least one column for itemset mining.")

    working = frame[selected_columns].copy()
    encoded_columns: dict[str, list[str]] = {}

    for column in selected_columns:
        series = working[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            filled = series.fillna(series.median() if series.notna().any() else 0)
            encoded_columns[column] = [f"{column}={label}" for label in _quantile_bins(filled.to_numpy())]
        else:
            encoded_columns[column] = [f"{column}={_item_value(value)}" for value in series]

    transactions: list[frozenset[str]] = []
    for row_index in range(len(working)):
        transactions.append(
            frozenset(encoded_columns[column][row_index] for column in selected_columns)
        )

    return transactions


def build_transaction_dataset(dataset: SvmDataset) -> TransactionDataset:
    if dataset.key == "simple_iris":
        frame = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    elif dataset.key == "medium_breast_cancer":
        selected = [0, 1, 2, 6]
        frame = pd.DataFrame(
            dataset.X[:, selected],
            columns=[dataset.feature_names[index] for index in selected],
        )
    else:
        images = dataset.X.reshape((-1, 8, 8))
        frame = pd.DataFrame(
            {
                "overall_ink": images.mean(axis=(1, 2)),
                "top_half_ink": images[:, :4, :].mean(axis=(1, 2)),
                "left_half_ink": images[:, :, :4].mean(axis=(1, 2)),
                "center_ink": images[:, 2:6, 2:6].mean(axis=(1, 2)),
            }
        )

    bucketed = {column: _quantile_bins(frame[column].to_numpy()) for column in frame.columns}
    transactions: list[frozenset[str]] = []

    for row_index in range(len(frame)):
        items = {f"{column}={bucketed[column][row_index]}" for column in frame.columns}
        items.add(f"class={dataset.target_names[int(dataset.y[row_index])]}")
        transactions.append(frozenset(items))

    min_support = {
        "simple": 0.20,
        "medium": 0.18,
        "complex": 0.10,
    }[dataset.level]

    return TransactionDataset(
        name=dataset.title,
        level=dataset.level,
        description=f"Discretized attribute transactions derived from the {dataset.title} dataset.",
        transactions=transactions,
        min_support=min_support,
    )


def _count_support(transactions: list[frozenset[str]], candidate: frozenset[str]) -> int:
    return sum(1 for transaction in transactions if candidate.issubset(transaction))


def mine_itemsets(
    transactions: list[frozenset[str]],
    dataset_name: str,
    level: str,
    min_support: float,
) -> list[FrequentItemset]:
    total = len(transactions)
    singleton_counts = Counter(item for transaction in transactions for item in transaction)
    current_level = {
        frozenset([item])
        for item, count in singleton_counts.items()
        if (count / total) >= min_support
    }
    support_cache: dict[frozenset[str], int] = {
        frozenset([item]): count
        for item, count in singleton_counts.items()
        if (count / total) >= min_support
    }
    discovered: list[FrequentItemset] = []

    size = 1
    while current_level:
        for itemset in sorted(current_level, key=lambda value: tuple(sorted(value))):
            count = support_cache[itemset]
            discovered.append(
                FrequentItemset(
                    dataset_name=dataset_name,
                    level=level,
                    items=tuple(sorted(itemset)),
                    length=len(itemset),
                    support=count / total,
                    count=count,
                )
            )

        size += 1
        current_list = sorted(current_level, key=lambda value: tuple(sorted(value)))
        candidates: set[frozenset[str]] = set()
        current_lookup = set(current_level)

        for index, left in enumerate(current_list):
            for right in current_list[index + 1 :]:
                candidate = left | right
                if len(candidate) != size:
                    continue
                if any(frozenset(subset) not in current_lookup for subset in combinations(candidate, size - 1)):
                    continue
                candidates.add(candidate)

        next_level: set[frozenset[str]] = set()
        for candidate in candidates:
            count = _count_support(transactions, candidate)
            if (count / total) >= min_support:
                support_cache[candidate] = count
                next_level.add(candidate)

        current_level = next_level

    return sorted(
        discovered,
        key=lambda itemset: (-itemset.support, -itemset.length, itemset.items),
    )


def mine_itemsets_for_datasets(datasets: list[SvmDataset]) -> list[FrequentItemset]:
    mined: list[FrequentItemset] = []

    for dataset in datasets:
        transaction_dataset = build_transaction_dataset(dataset)
        mined.extend(
            mine_itemsets(
                transactions=transaction_dataset.transactions,
                dataset_name=transaction_dataset.name,
                level=transaction_dataset.level,
                min_support=transaction_dataset.min_support,
            )
        )

    return mined


def mine_itemsets_from_frame(
    frame: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
    min_support: float,
    target_column: str | None = None,
) -> list[FrequentItemset]:
    transactions = build_transactions_from_frame(frame, columns, target_column=target_column)
    return mine_itemsets(
        transactions=transactions,
        dataset_name=dataset_name,
        level="custom",
        min_support=min_support,
    )


def itemsets_to_frame(itemsets: list[FrequentItemset]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": itemset.dataset_name,
                "level": itemset.level,
                "itemset": " + ".join(itemset.items),
                "length": itemset.length,
                "support": itemset.support,
                "count": itemset.count,
            }
            for itemset in itemsets
        ]
    )
