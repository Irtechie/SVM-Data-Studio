from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import pandas as pd


@dataclass(frozen=True)
class EpisodeDataset:
    name: str
    level: str
    description: str
    sequences: list[list[str]]
    min_support: float
    max_span: int = 4


@dataclass(frozen=True)
class EpisodePattern:
    dataset_name: str
    level: str
    events: tuple[str, ...]
    length: int
    support: float
    count: int


def load_episode_datasets() -> list[EpisodeDataset]:
    return [
        EpisodeDataset(
            name="Website Journeys",
            level="simple",
            description="Classic e-commerce session paths with predictable purchase flow.",
            min_support=0.40,
            max_span=4,
            sequences=[
                ["Landing", "Search", "Product", "Cart", "Checkout"],
                ["Landing", "Search", "Product", "Cart", "Checkout"],
                ["Landing", "Product", "Cart", "Checkout"],
                ["Landing", "Search", "Product", "Cart"],
                ["Landing", "Search", "Product", "Wishlist", "Cart", "Checkout"],
                ["Landing", "Category", "Product", "Cart", "Checkout"],
                ["Landing", "Search", "Product", "Cart", "Payment", "Checkout"],
                ["Landing", "Search", "Product", "Cart", "Checkout"],
                ["Landing", "Category", "Product", "Wishlist", "Cart"],
                ["Landing", "Search", "Product", "Cart", "Checkout"],
            ],
        ),
        EpisodeDataset(
            name="Support Ticket Workflow",
            level="medium",
            description="Operational support-ticket events with escalation and resolution patterns.",
            min_support=0.34,
            max_span=5,
            sequences=[
                ["Open", "Diagnose", "Escalate", "Fix", "Resolve", "Close"],
                ["Open", "Diagnose", "Fix", "Resolve", "Close"],
                ["Open", "Diagnose", "Escalate", "Fix", "Resolve"],
                ["Open", "Triage", "Diagnose", "Fix", "Resolve", "Close"],
                ["Open", "Diagnose", "Escalate", "Fix", "Verify", "Resolve", "Close"],
                ["Open", "Triage", "Diagnose", "Escalate", "Fix", "Resolve"],
                ["Open", "Diagnose", "Fix", "Verify", "Resolve", "Close"],
                ["Open", "Diagnose", "Escalate", "Fix", "Resolve", "Close"],
                ["Open", "Triage", "Diagnose", "Fix", "Resolve"],
                ["Open", "Diagnose", "Escalate", "Fix", "Resolve", "Close"],
            ],
        ),
        EpisodeDataset(
            name="Industrial Maintenance",
            level="complex",
            description="Machine-failure and recovery sequences with diagnostics and validation events.",
            min_support=0.30,
            max_span=6,
            sequences=[
                ["Alert", "Inspect", "Diagnose", "OrderPart", "Repair", "Test", "Close"],
                ["Alert", "Inspect", "Diagnose", "Repair", "Test", "Close"],
                ["Alert", "Inspect", "OrderPart", "Repair", "Test", "Close"],
                ["Alert", "RemoteCheck", "Inspect", "Diagnose", "Repair", "Test", "Close"],
                ["Alert", "Inspect", "Diagnose", "OrderPart", "Repair", "Retest", "Close"],
                ["Alert", "Inspect", "Diagnose", "Repair", "Test", "Close"],
                ["Alert", "RemoteCheck", "Inspect", "Diagnose", "OrderPart", "Repair", "Test"],
                ["Alert", "Inspect", "Repair", "Test", "Close"],
                ["Alert", "Inspect", "Diagnose", "OrderPart", "Repair", "Test", "Close"],
                ["Alert", "Inspect", "Diagnose", "Repair", "Retest", "Close"],
            ],
        ),
    ]


def _clean_event(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def build_episode_dataset_from_sequence_column(
    frame: pd.DataFrame,
    sequence_column: str,
    name: str,
    min_support: float,
    max_span: int,
    separator: str = ",",
) -> EpisodeDataset:
    sequences: list[list[str]] = []

    for value in frame[sequence_column]:
        event_text = _clean_event(value)
        if not event_text:
            continue
        sequence = [event.strip() for event in event_text.split(separator) if event.strip()]
        if len(sequence) >= 2:
            sequences.append(sequence)

    if not sequences:
        raise ValueError("No valid sequences were found in the selected sequence column.")

    return EpisodeDataset(
        name=name,
        level="custom",
        description=f"Sequences built from the {sequence_column} column.",
        sequences=sequences,
        min_support=min_support,
        max_span=max_span,
    )


def build_episode_dataset_from_event_columns(
    frame: pd.DataFrame,
    event_columns: list[str],
    name: str,
    min_support: float,
    max_span: int,
) -> EpisodeDataset:
    if not event_columns:
        raise ValueError("Choose at least two ordered event columns.")

    sequences: list[list[str]] = []
    for row in frame[event_columns].itertuples(index=False, name=None):
        sequence = [event for event in (_clean_event(value) for value in row) if event]
        if len(sequence) >= 2:
            sequences.append(sequence)

    if not sequences:
        raise ValueError("No valid event sequences were found in the selected columns.")

    return EpisodeDataset(
        name=name,
        level="custom",
        description=f"Sequences built from ordered columns: {', '.join(event_columns)}.",
        sequences=sequences,
        min_support=min_support,
        max_span=max_span,
    )


def mine_episodes(dataset: EpisodeDataset, max_length: int = 3) -> list[EpisodePattern]:
    sequence_count = len(dataset.sequences)
    counter: Counter[tuple[str, ...]] = Counter()

    for sequence in dataset.sequences:
        seen: set[tuple[str, ...]] = set()
        for length in range(2, max_length + 1):
            for indexes in combinations(range(len(sequence)), length):
                if indexes[-1] - indexes[0] > dataset.max_span:
                    continue
                seen.add(tuple(sequence[index] for index in indexes))
        counter.update(seen)

    mined: list[EpisodePattern] = []
    for events, count in counter.items():
        support = count / sequence_count
        if support >= dataset.min_support:
            mined.append(
                EpisodePattern(
                    dataset_name=dataset.name,
                    level=dataset.level,
                    events=events,
                    length=len(events),
                    support=support,
                    count=count,
                )
            )

    return sorted(
        mined,
        key=lambda pattern: (-pattern.support, -pattern.length, pattern.events),
    )


def mine_all_episode_datasets(datasets: list[EpisodeDataset]) -> list[EpisodePattern]:
    patterns: list[EpisodePattern] = []

    for dataset in datasets:
        patterns.extend(mine_episodes(dataset))

    return patterns


def episodes_to_frame(patterns: list[EpisodePattern]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": pattern.dataset_name,
                "level": pattern.level,
                "episode": " -> ".join(pattern.events),
                "length": pattern.length,
                "support": pattern.support,
                "count": pattern.count,
            }
            for pattern in patterns
        ]
    )
