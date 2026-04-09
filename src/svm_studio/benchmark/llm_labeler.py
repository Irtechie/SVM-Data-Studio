"""LLM-based dataset labeler for the benchmark pipeline.

``LLMLabeler`` sends each example to an LLM, parses the JSON response, and
returns the labels together with per-example metadata (confidence, reasoning,
latency, token usage).

Backend re-uses ``chat_completion`` from ``llm_advisor.py`` so every
configured endpoint (OpenAI, Ollama, vLLM, LM Studio, llama.cpp, …) works
without extra wiring.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..llm_advisor import chat_completion
from .dataset_loader import StandardDataset
from .prompts import LABELING_PROMPT_TABULAR, LABELING_PROMPT_TEXT

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_FALLBACK_LABEL = "unknown"


@dataclass
class LabelRecord:
    """Metadata for one LLM labeling call."""
    index: int
    llm_label: str
    true_label: str
    confidence: float
    reasoning: str
    latency_ms: float
    retries: int
    fallback_used: bool


@dataclass
class LabeledDataset:
    """Output of :class:`LLMLabeler`."""
    dataset_name: str
    llm_model: str
    y_llm: pd.Series                  # LLM-assigned labels (string)
    y_true: pd.Series                 # Ground truth labels
    records: list[LabelRecord]        # Per-example metadata
    agreement_rate: float             # fraction where y_llm == y_true
    mean_confidence: float
    fallback_count: int               # examples where LLM failed → "unknown"
    total_llm_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMLabeler:
    """Label a ``StandardDataset`` using an LLM.

    Parameters
    ----------
    model : str
        Model name forwarded to ``chat_completion``.
    base_url : str or None
        LLM endpoint base URL (e.g. ``http://192.168.1.10:8000``).
    api_key : str or None
        API key for OpenAI-compatible endpoints.
    batch_size : int
        Number of examples to label before a short progress log (no actual
        batching at the API level — most local models handle one example at a
        time reliably).
    max_retries : int
        How many times to retry a failed/invalid JSON response before falling
        back to ``"unknown"``.
    """

    def __init__(
        self,
        model: str = "gemma-4-31B-it-Q4_K_M.gguf",
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 10,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries

    # ── public API ─────────────────────────────────────────────────────────

    def label(
        self,
        dataset: StandardDataset,
        max_examples: int | None = None,
        progress_callback: Any = None,
        conversation_callback: Any = None,
    ) -> LabeledDataset:
        """Label all (or up to *max_examples*) examples in *dataset*.

        Parameters
        ----------
        progress_callback : callable(current, total) or None
            Called after each example so callers can update a progress bar.
        conversation_callback : callable(idx, prompt, response, label, confidence) or None
            Called after each LLM call so callers can display the live conversation.
        """
        n = len(dataset.X) if max_examples is None else min(max_examples, len(dataset.X))
        records: list[LabelRecord] = []

        for i in range(n):
            rec = self._label_one(i, dataset, conversation_callback=conversation_callback)
            records.append(rec)
            if (i + 1) % self.batch_size == 0:
                logger.info("Labeled %d / %d examples", i + 1, n)
            if progress_callback is not None:
                try:
                    progress_callback(i + 1, n)
                except Exception:
                    pass

        y_llm = pd.Series([r.llm_label for r in records], name="llm_label")
        y_true = dataset.y.iloc[:n].reset_index(drop=True)

        agreement = float((y_llm == y_true).mean())
        mean_conf = float(sum(r.confidence for r in records) / len(records)) if records else 0.0
        fallback_count = sum(1 for r in records if r.fallback_used)

        return LabeledDataset(
            dataset_name=dataset.name,
            llm_model=self.model,
            y_llm=y_llm,
            y_true=y_true,
            records=records,
            agreement_rate=agreement,
            mean_confidence=mean_conf,
            fallback_count=fallback_count,
            total_llm_calls=sum(1 + r.retries for r in records),
            metadata={
                "n_labeled": n,
                "class_names": dataset.class_names,
                "data_type": dataset.data_type,
            },
        )

    def relabel_uncertain(
        self,
        dataset: StandardDataset,
        labeled: LabeledDataset,
        uncertain_indices: list[int],
    ) -> LabeledDataset:
        """Re-query the LLM for uncertain examples using the retry prompt.

        Returns a new ``LabeledDataset`` with updated labels for those indices.
        """
        from .prompts import LABELING_PROMPT_UNCERTAINTY_RETRY

        updated_records = list(labeled.records)
        for idx in uncertain_indices:
            old_rec = labeled.records[idx]
            row = dataset.X.iloc[idx]
            example_str = _format_example(row, dataset.data_type)
            prompt = LABELING_PROMPT_UNCERTAINTY_RETRY.format(
                dataset_name=dataset.name,
                class_names=", ".join(dataset.class_names),
                example_data=example_str,
                previous_label=old_rec.llm_label,
                previous_confidence=old_rec.confidence,
            )
            new_rec = self._call_llm(idx, prompt, dataset.class_names, dataset.y.iloc[idx])
            updated_records[idx] = new_rec

        y_llm = pd.Series([r.llm_label for r in updated_records], name="llm_label")
        y_true = labeled.y_true
        agreement = float((y_llm == y_true).mean())
        mean_conf = float(sum(r.confidence for r in updated_records) / len(updated_records))
        fallback_count = sum(1 for r in updated_records if r.fallback_used)

        return LabeledDataset(
            dataset_name=labeled.dataset_name,
            llm_model=self.model,
            y_llm=y_llm,
            y_true=y_true,
            records=updated_records,
            agreement_rate=agreement,
            mean_confidence=mean_conf,
            fallback_count=fallback_count,
            total_llm_calls=labeled.total_llm_calls + len(uncertain_indices),
            metadata=labeled.metadata,
        )

    # ── internal helpers ───────────────────────────────────────────────────

    def _label_one(self, idx: int, dataset: StandardDataset, conversation_callback: Any = None) -> LabelRecord:
        row = dataset.X.iloc[idx]
        true_label = str(dataset.y.iloc[idx])
        prompt = _build_prompt(row, dataset)
        return self._call_llm(idx, prompt, dataset.class_names, true_label, conversation_callback=conversation_callback)

    def _call_llm(
        self,
        idx: int,
        prompt: str,
        class_names: list[str],
        true_label: str,
        conversation_callback: Any = None,
    ) -> LabelRecord:
        messages = [{"role": "user", "content": prompt}]
        label = _FALLBACK_LABEL
        confidence = 0.0
        reasoning = ""
        retries = 0
        fallback_used = False

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                raw = chat_completion(
                    messages,
                    model=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                parsed = _parse_label_response(raw, class_names)
                if parsed is not None:
                    label, confidence, reasoning = parsed
                    if conversation_callback is not None:
                        try:
                            conversation_callback(idx, prompt, raw, label, confidence)
                        except Exception:
                            pass
                    break
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.warning("LLM call failed for index %d (attempt %d): %s", idx, attempt + 1, exc)
            retries += 1

        if label == _FALLBACK_LABEL:
            fallback_used = True
            logger.warning("Falling back to 'unknown' for index %d after %d attempts", idx, self.max_retries + 1)

        return LabelRecord(
            index=idx,
            llm_label=label,
            true_label=true_label,
            confidence=confidence,
            reasoning=reasoning,
            latency_ms=latency_ms,
            retries=max(0, retries - 1),
            fallback_used=fallback_used,
        )


# ── module-level helpers ───────────────────────────────────────────────────

def _build_prompt(row: pd.Series, dataset: StandardDataset) -> str:
    """Select the right prompt template based on ``data_type``."""
    if dataset.data_type == "text":
        text = str(row.get("text", row.iloc[0]))
        return LABELING_PROMPT_TEXT.format(
            dataset_name=dataset.name,
            task_description=dataset.description,
            class_names=", ".join(dataset.class_names),
            text=text[:2000],   # cap to avoid huge prompts for 30B models
        )
    # tabular / image_features / sequential all get the feature dict
    return LABELING_PROMPT_TABULAR.format(
        dataset_name=dataset.name,
        dataset_description=dataset.description,
        class_names=", ".join(dataset.class_names),
        feature_dict=_format_example(row, dataset.data_type),
    )


def _format_example(row: pd.Series, data_type: str) -> str:
    """Format a row as a readable key=value block."""
    if data_type == "text":
        return str(row.get("text", row.iloc[0]))[:1500]
    items = []
    for col, val in row.items():
        items.append(f"  {col}: {val}")
        if len(items) >= 30:  # cap to avoid overwhelming 30B model
            items.append(f"  ... ({len(row) - 30} more features)")
            break
    return "\n".join(items)


def _parse_label_response(raw: str, class_names: list[str]) -> tuple[str, float, str] | None:
    """Parse a JSON LLM response into (label, confidence, reasoning).

    Tries strict JSON parse first, then a lenient regex extraction.
    Returns ``None`` if parsing fails or the label is not in *class_names*.
    """
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")]
    raw = raw.strip()

    # Try to find the JSON object
    brace_start = raw.find("{")
    brace_end = raw.rfind("}") + 1
    if brace_start == -1 or brace_end == 0:
        return None

    json_str = raw[brace_start:brace_end]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    label = str(data.get("label", "")).strip()
    # Case-insensitive match against class names
    class_map = {c.lower(): c for c in class_names}
    label_lower = label.lower()
    if label_lower in class_map:
        label = class_map[label_lower]
    elif label not in class_names:
        # Try partial match
        matches = [c for c in class_names if label_lower in c.lower()]
        if len(matches) == 1:
            label = matches[0]
        else:
            return None  # invalid label → trigger retry

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", ""))

    return label, confidence, reasoning
