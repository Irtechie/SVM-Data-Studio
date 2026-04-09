"""Episode mining optional technique for the benchmark pipeline.

Finds frequent ordered event sequences and checks whether LLM mislabels
cluster around specific sequence patterns.

Only meaningful for sequential / temporal datasets (data_type == "sequential").
Re-uses the existing episode_mining module.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...episode_mining import (
    build_episode_dataset_from_event_columns,
    build_episode_dataset_from_sequence_column,
    episodes_to_frame,
    mine_episodes,
)
from ..dataset_loader import StandardDataset
from ..llm_labeler import LabeledDataset


@dataclass
class BenchEpisodeResult:
    """Episode mining result in benchmark context."""
    all_episodes: pd.DataFrame
    error_episodes: pd.DataFrame        # episodes from mislabeled examples
    correct_episodes: pd.DataFrame      # episodes from correctly labeled examples
    error_exclusive: pd.DataFrame       # episodes ONLY in error group
    min_support_used: float
    sequence_source: str                # "column" or "events" or "skipped"


def run_bench_episode_mining(
    dataset: StandardDataset,
    labeled: LabeledDataset,
    sequence_column: str | None = None,
    event_columns: list[str] | None = None,
    min_support: float = 0.20,
    max_span: int = 4,
    max_length: int = 3,
) -> BenchEpisodeResult:
    """Mine ordered event sequences and correlate with LLM errors.

    Pass either ``sequence_column`` (a delimited event string column) or
    ``event_columns`` (ordered left-to-right event columns).
    If neither is given, auto-detection is attempted from column names.

    Returns a result indicating skipped mining if no suitable columns exist.
    """
    n = len(labeled.y_llm)
    X = dataset.X.iloc[:n].reset_index(drop=True)
    y_true = labeled.y_true
    y_llm = labeled.y_llm

    error_mask = y_llm != y_true
    correct_mask = ~error_mask

    # Auto-detect sequence structure if not provided
    source = "skipped"
    if sequence_column is None and event_columns is None:
        sequence_column = _find_sequence_column(X)
        if sequence_column is None:
            event_columns = _find_event_columns(X)

    if sequence_column is None and event_columns is None:
        return BenchEpisodeResult(
            all_episodes=pd.DataFrame(),
            error_episodes=pd.DataFrame(),
            correct_episodes=pd.DataFrame(),
            error_exclusive=pd.DataFrame(),
            min_support_used=min_support,
            sequence_source="skipped — no sequential structure detected",
        )

    source = "column" if sequence_column else "events"

    all_eps = _mine(X, dataset.name, sequence_column, event_columns, min_support, max_span, max_length)

    error_eps = pd.DataFrame()
    correct_eps = pd.DataFrame()
    if error_mask.sum() >= 5:
        error_eps = _mine(
            X[error_mask].reset_index(drop=True), dataset.name + "_errors",
            sequence_column, event_columns,
            max(0.05, min_support * 0.4), max_span, max_length,
        )
    if correct_mask.sum() >= 5:
        correct_eps = _mine(
            X[correct_mask].reset_index(drop=True), dataset.name + "_correct",
            sequence_column, event_columns,
            max(0.05, min_support * 0.4), max_span, max_length,
        )

    error_exclusive = _exclusive_episodes(error_eps, correct_eps)

    return BenchEpisodeResult(
        all_episodes=all_eps,
        error_episodes=error_eps,
        correct_episodes=correct_eps,
        error_exclusive=error_exclusive,
        min_support_used=min_support,
        sequence_source=source,
    )


# ── helpers ────────────────────────────────────────────────────────────────

def _mine(
    frame: pd.DataFrame,
    name: str,
    seq_col: str | None,
    evt_cols: list[str] | None,
    min_support: float,
    max_span: int,
    max_length: int,
) -> pd.DataFrame:
    if frame.empty or len(frame) < 3:
        return pd.DataFrame()
    try:
        if seq_col and seq_col in frame.columns:
            ds = build_episode_dataset_from_sequence_column(
                frame=frame, sequence_column=seq_col, name=name,
                min_support=min_support, max_span=max_span,
            )
        elif evt_cols:
            cols_present = [c for c in evt_cols if c in frame.columns]
            if len(cols_present) < 2:
                return pd.DataFrame()
            ds = build_episode_dataset_from_event_columns(
                frame=frame, event_columns=cols_present, name=name,
                min_support=min_support, max_span=max_span,
            )
        else:
            return pd.DataFrame()
        return episodes_to_frame(mine_episodes(ds, max_length=max_length))
    except Exception:
        return pd.DataFrame()


def _exclusive_episodes(error_df: pd.DataFrame, correct_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame()
    if correct_df.empty:
        return error_df
    if "episode" not in error_df.columns or "episode" not in correct_df.columns:
        return error_df
    correct_set = set(correct_df["episode"].astype(str))
    return error_df[~error_df["episode"].astype(str).isin(correct_set)].reset_index(drop=True)


def _find_sequence_column(frame: pd.DataFrame) -> str | None:
    seq_hints = {"sequence", "events", "path", "actions", "steps", "session"}
    for col in frame.columns:
        if col.lower() in seq_hints and frame[col].dtype == object:
            return col
    return None


def _find_event_columns(frame: pd.DataFrame) -> list[str] | None:
    evt_hints = {"event", "action", "step", "state", "item"}
    cols = [c for c in frame.columns if any(h in c.lower() for h in evt_hints)]
    return cols if len(cols) >= 2 else None
