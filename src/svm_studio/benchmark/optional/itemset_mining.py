"""Itemset mining optional technique for the benchmark pipeline.

Finds frequent feature combinations in the full dataset and correlates them
with LLM labeling errors — revealing WHY the LLM got certain examples wrong.

Re-uses ``mine_itemsets_from_frame`` from the existing itemset_mining module.
Only applies to tabular / image_features datasets.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...itemset_mining import mine_itemsets_from_frame, itemsets_to_frame
from ..dataset_loader import StandardDataset
from ..llm_labeler import LabeledDataset


@dataclass
class BenchItemsetResult:
    """Itemset mining result in benchmark context."""
    all_patterns: pd.DataFrame          # frequent itemsets across full dataset
    error_patterns: pd.DataFrame        # patterns in disagreement examples
    correct_patterns: pd.DataFrame      # patterns in correctly labeled examples
    overlap_patterns: pd.DataFrame      # patterns present in BOTH groups
    error_exclusive: pd.DataFrame       # patterns ONLY in error examples
    min_support_used: float


def run_bench_itemset_mining(
    dataset: StandardDataset,
    labeled: LabeledDataset,
    min_support: float = 0.15,
    max_columns: int = 10,
) -> BenchItemsetResult:
    """Mine frequent feature combinations and compare error vs correct groups.

    Parameters
    ----------
    min_support : float
        Minimum support threshold for Apriori mining.
    max_columns : int
        Cap on feature columns to mine (avoids combinatorial explosion on
        high-dimensional datasets like digits).
    """
    n = len(labeled.y_llm)
    X = dataset.X.iloc[:n].reset_index(drop=True)
    y_true = labeled.y_true
    y_llm = labeled.y_llm

    error_mask = y_llm != y_true
    correct_mask = ~error_mask

    cols = list(X.columns[:max_columns])

    # Mine across full dataset
    all_patterns = _mine(X[cols], dataset.name, min_support)

    # Mine errors subset (lower support threshold since subset is smaller)
    error_support = max(0.05, min_support * 0.5)
    if error_mask.sum() >= 5:
        error_patterns = _mine(X[cols][error_mask].reset_index(drop=True), dataset.name + "_errors", error_support)
    else:
        error_patterns = pd.DataFrame()

    # Mine correct subset
    if correct_mask.sum() >= 5:
        correct_patterns = _mine(X[cols][correct_mask].reset_index(drop=True), dataset.name + "_correct", error_support)
    else:
        correct_patterns = pd.DataFrame()

    # Overlap and exclusive patterns
    overlap_patterns, error_exclusive = _compare_patterns(error_patterns, correct_patterns)

    return BenchItemsetResult(
        all_patterns=all_patterns,
        error_patterns=error_patterns,
        correct_patterns=correct_patterns,
        overlap_patterns=overlap_patterns,
        error_exclusive=error_exclusive,
        min_support_used=min_support,
    )


def _mine(frame: pd.DataFrame, name: str, min_support: float) -> pd.DataFrame:
    """Mine and return a results frame, returning empty DF on failure."""
    if frame.empty or len(frame) < 5:
        return pd.DataFrame()
    try:
        result = mine_itemsets_from_frame(
            frame=frame,
            columns=list(frame.columns),
            dataset_name=name,
            min_support=min_support,
        )
        return itemsets_to_frame(result)
    except Exception:
        return pd.DataFrame()


def _compare_patterns(
    error_df: pd.DataFrame,
    correct_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find itemsets present in both groups vs exclusively in errors."""
    if error_df.empty or correct_df.empty:
        return pd.DataFrame(), error_df

    error_items = set(error_df["itemset"].astype(str)) if "itemset" in error_df.columns else set()
    correct_items = set(correct_df["itemset"].astype(str)) if "itemset" in correct_df.columns else set()

    overlap_strs = error_items & correct_items
    exclusive_strs = error_items - correct_items

    overlap = error_df[error_df["itemset"].astype(str).isin(overlap_strs)] if "itemset" in error_df.columns else pd.DataFrame()
    exclusive = error_df[error_df["itemset"].astype(str).isin(exclusive_strs)] if "itemset" in error_df.columns else error_df

    return overlap.reset_index(drop=True), exclusive.reset_index(drop=True)
