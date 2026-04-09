"""LLM-powered column advisor for SVM feature selection.

Auto-detects the API style of the configured endpoint (OpenAI-compatible or
Ollama-native) so the caller never needs a manual compatibility toggle.
Works with: OpenAI, vLLM, LM Studio, llama.cpp, TRT-LLM (OpenAI server),
Ollama (both /v1 compat and native /api/chat).
Falls back to a deterministic heuristic when no endpoint is reachable.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

try:
    from openai import OpenAI as _OpenAI  # type: ignore[import-untyped]

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ── common names that hint at a classification target ──────────────────────
_TARGET_KEYWORDS: frozenset[str] = frozenset(
    [
        "target",
        "label",
        "class",
        "outcome",
        "category",
        "diagnosis",
        "result",
        "fraud",
        "spam",
        "churn",
        "default",
        "type",
        "group",
        "status",
        "response",
        "y",
    ]
)


@dataclass
class ColumnAdvice:
    target_column: str
    feature_columns: list[str]
    rationale: str
    source: str  # "llm" | "heuristic"
    model_used: str = field(default="")
    # Alternative feature sets from multi-candidate prompt (LLM path only).
    # Caller can evaluate each and keep the best-graded set.
    candidates: list[list[str]] = field(default_factory=list)
    # Per-candidate reasoning text from the LLM (parallel list to candidates).
    candidate_reasoning: list[str] = field(default_factory=list)


# ── data context builder ──────────────────────────────────────────────────────────

def _build_data_context(
    frame: pd.DataFrame,
    max_cat_samples: int = 8,
    sample_rows: int = 10,
    include_sample: bool = True,
    include_correlations: bool = True,
) -> str:
    """Return a rich text representation of *frame* for use as LLM context.

    Includes column schema, descriptive stats, null counts, value distributions,
    an optional row sample, and optional pairwise correlations between numeric
    columns and each low-cardinality column (probable target proxy).
    """
    n_rows, n_cols = frame.shape
    lines: list[str] = [
        f"Dataset: {n_rows:,} rows, {n_cols} columns",
        "",
        "### Column details",
    ]

    null_counts = frame.isnull().sum()
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

    for col in frame.columns:
        nulls = null_counts[col]
        null_str = f", {nulls} nulls ({nulls/n_rows:.1%})" if nulls else ""
        n_unique = frame[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(frame[col]):
            s = frame[col].describe()
            lines.append(
                f"  {col} [numeric{null_str}]"
                f" | unique={n_unique}"
                f" | min={s['min']:.4g}  max={s['max']:.4g}"
                f" | mean={s['mean']:.4g}  std={s['std']:.4g}"
                f" | 25%={s['25%']:.4g}  50%={s['50%']:.4g}  75%={s['75%']:.4g}"
            )
        else:
            top_vals = (
                frame[col].astype(str).value_counts(dropna=True)
                          .head(max_cat_samples)
                          .to_dict()
            )
            top_str = ", ".join(f"{v!r}: {c}" for v, c in top_vals.items())
            lines.append(
                f"  {col} [categorical{null_str}]"
                f" | unique={n_unique}"
                f" | top values: {top_str}"
            )

    # ── pairwise correlations: numeric vs. low-cardinality columns ──
    if include_correlations and len(numeric_cols) >= 2:
        low_card = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 30]
        if low_card and numeric_cols:
            lines.append("")
            lines.append("### Numeric correlations with low-cardinality columns (point-biserial / eta)")
            for target_cand in low_card[:4]:  # cap at 4 candidates to keep context short
                corrs = {}
                for nc in numeric_cols:
                    if nc == target_cand:
                        continue
                    valid = frame[[nc, target_cand]].dropna()
                    if len(valid) < 10:
                        continue
                    try:
                        enc_arr = pd.Categorical(valid[target_cand]).codes.astype(float)
                        enc_series = pd.Series(enc_arr.tolist(), index=valid.index)
                        corr = abs(valid[nc].corr(enc_series))
                        if not pd.isna(corr):
                            corrs[nc] = round(corr, 3)
                    except Exception:
                        pass
                if corrs:
                    top = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:6]
                    top_str = "  ".join(f"{c}: {v}" for c, v in top)
                    lines.append(f"  vs '{target_cand}': {top_str}")

    # ── row sample ──
    if include_sample and sample_rows > 0:
        lines.append("")
        lines.append(f"### First {min(sample_rows, n_rows)} rows (CSV)")
        lines.append(frame.head(sample_rows).to_csv(index=False))

    return "\n".join(lines)


# Backward-compat alias used by the advisor prompt path
def _build_schema_text(frame: pd.DataFrame, max_samples: int = 5) -> str:
    return _build_data_context(frame, max_cat_samples=max_samples, sample_rows=0, include_correlations=False)


# ── heuristic fallback ─────────────────────────────────────────────────────

def _heuristic_advice(frame: pd.DataFrame) -> ColumnAdvice:
    """Pick a reasonable target + feature set without an LLM."""
    columns = list(frame.columns)
    target: str | None = None

    # 1. keyword match
    for col in columns:
        col_lower = col.lower()
        if col_lower in _TARGET_KEYWORDS or any(kw in col_lower for kw in _TARGET_KEYWORDS):
            n_unique = frame[col].nunique(dropna=True)
            if 2 <= n_unique <= 30:
                target = col
                break

    # 2. last low-cardinality column
    if target is None:
        for col in reversed(columns):
            n_unique = frame[col].nunique(dropna=True)
            if 2 <= n_unique <= 20:
                target = col
                break

    # 3. final fallback
    if target is None:
        target = columns[-1]

    # features: numeric columns excluding target
    feature_candidates = [
        col
        for col in columns
        if col != target and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not feature_candidates:
        feature_candidates = [col for col in columns if col != target]

    kw_hit = any(kw in target.lower() for kw in _TARGET_KEYWORDS)
    reason = "matches a common label-column naming pattern" if kw_hit else "has low cardinality"
    rationale = (
        f"Heuristic: '{target}' was chosen as the target because it {reason} "
        f"({frame[target].nunique()} unique values).  "
        f"Features are the {len(feature_candidates)} remaining numeric column(s)."
    )

    return ColumnAdvice(
        target_column=target,
        feature_columns=feature_candidates,
        rationale=rationale,
        source="heuristic",
        model_used="heuristic",
    )


# ── LLM path ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert data scientist preparing features for a Support Vector Machine (SVM) classifier.
You will be given a dataset schema that includes, for each column: its data type, number of unique
values, and value range or sample values.  Base every decision ONLY on what is stated in that schema.
Do not invent properties, assume domain knowledge, or guess at relationships not visible in the data.

STEP 1 — IDENTIFY THE TARGET COLUMN
Criteria (all must be met):
  - Categorical or low-cardinality discrete (2–30 unique values)
  - Name suggests a label: class, diagnosis, category, fraud, outcome, species, status, etc.
  - NOT a continuous numeric measurement, NOT an ID/key (monotonically increasing integers or UUIDs),
    NOT a timestamp or date, NOT a free-text string

STEP 2 — ELIMINATE INELIGIBLE FEATURE COLUMNS
Remove any column that meets one or more of these disqualifying criteria, and CITE which rule applies:
  - Same column as the target
  - ID / surrogate key: integer with unique count ≈ row count, or named "id", "index", "key", "rownum"
  - Timestamp / date column
  - Free-text / high-cardinality string: unique count > 50 and dtype is string/object
  - Near-zero variance: numeric with only 1 unique value
  - Direct data leak: column whose name or values obviously encode the target (e.g. "target_encoded")

STEP 3 — REASON ABOUT RELATIONSHIPS IN THE REMAINING POOL
For each eligible column, state one observable fact from the schema that explains its SVM relevance:
  - numeric range and variance (wide range → potentially separating)
  - unique value count relative to row count
  - suspected redundancy with another eligible column (same domain, similar name, overlapping range)

STEP 4 — GENERATE UP TO TEN DISTINCT FEATURE-COLUMN COMBINATIONS
Each combination must follow from Step 3 reasoning — not from guessing.  Rules:
  - Every included column must have a stated reason from Step 3
  - Every excluded column must have a stated reason from Step 2 or Step 3
  - No two candidates may be identical sets
  - Explore the space systematically:
      * Full eligible pool (baseline)
      * Full pool minus each suspected-redundant pair, one pair at a time
      * Strongest 3–5 predictors by range/variance
      * Domain sub-groups if distinguishable from column names / value patterns
      * Minimal 2-column set of the two highest-variance numerics

Respond with ONLY valid JSON in exactly this format.  No markdown, no extra text, no commentary:
{
  "target_column": "<name>",
  "candidates": [
    {"feature_columns": ["<name>", ...], "reasoning": "<cite schema evidence, not assumptions>"},
    ...
  ],
  "rationale": "<one sentence: which schema property identified this column as the target>"
}
"""


# ── backend detection ──────────────────────────────────────────────────────

def _get(url: str, timeout: float = 3.0) -> dict | None:
    """GET *url* and return parsed JSON dict, or None on any error."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        pass
    return None


def _detect_backend(base_url: str) -> tuple[str, str]:
    """Probe *base_url* and return *(style, resolved_url)*.

    *style* is ``"openai"`` (OpenAI-compatible /chat/completions) or
    ``"ollama"`` (Ollama native /api/chat).
    Works with: OpenAI, vLLM, LM Studio, llama.cpp server, TRT-LLM OpenAI
    server, Ollama (both /v1 and native).
    """
    stripped = base_url.rstrip("/")

    # ── OpenAI-compat probe: look for /models returning {"data": [...]} ──
    openai_candidates = (
        [stripped] if stripped.endswith("/v1") else [stripped + "/v1", stripped]
    )
    for candidate in openai_candidates:
        data = _get(candidate + "/models")
        if isinstance(data, dict) and "data" in data:
            return ("openai", candidate)

    # ── Ollama-native probe: look for /api/tags returning {"models": [...]} ──
    host_root = stripped.removesuffix("/v1")
    data = _get(host_root + "/api/tags")
    if isinstance(data, dict) and "models" in data:
        return ("ollama", host_root)

    # ── Unknown / offline — default to OpenAI-compat so errors are readable ──
    return ("openai", openai_candidates[0])


# ── per-backend callers ────────────────────────────────────────────────────

def _call_openai_compat(
    base_url: str | None,
    api_key: str,
    model: str,
    messages: list[dict],
) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("The 'openai' package is not installed.  Run: pip install openai")
    kwargs: dict = {"api_key": api_key or "none"}
    if base_url:
        kwargs["base_url"] = base_url
    client = _OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


def _call_ollama_native(base_url: str, model: str, messages: list[dict]) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return (data.get("message", {}).get("content") or "").strip()


# ── dispatcher + response parser ───────────────────────────────────────────

def _parse_advice(raw: str, frame: pd.DataFrame, model_tag: str) -> ColumnAdvice:
    """Extract ColumnAdvice from a raw LLM response string.

    Handles the multi-candidate format::

        {"target_column": ..., "candidates": [{"feature_columns": [...], ...}, ...],
         "rationale": ...}

    Also accepts the legacy single-set format::

        {"target_column": ..., "feature_columns": [...], "rationale": ...}
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise RuntimeError(f"LLM returned no JSON block.  Raw response: {raw[:200]!r}")
    data = json.loads(match.group())
    target = str(data["target_column"])
    rationale = str(data.get("rationale", "LLM-selected columns."))
    all_cols = set(frame.columns)

    if target not in all_cols:
        raise RuntimeError(f"LLM suggested unknown target column: {target!r}")

    def _clean(cols: list) -> list[str]:
        return [f for f in (str(c) for c in cols) if f in all_cols and f != target]

    # ── multi-candidate path ──
    if "candidates" in data and isinstance(data["candidates"], list):
        sets: list[list[str]] = []
        reasons: list[str] = []
        for entry in data["candidates"]:
            cleaned = _clean(entry.get("feature_columns", []))
            if cleaned and cleaned not in sets:
                sets.append(cleaned)
                reasons.append(str(entry.get("reasoning", "")))
        if not sets:
            raise RuntimeError("LLM candidates contained no valid feature columns.")
        return ColumnAdvice(
            target_column=target,
            feature_columns=sets[0],
            rationale=rationale,
            source="llm",
            model_used=model_tag,
            candidates=sets,
            candidate_reasoning=reasons,
        )

    # ── legacy single-set path ──
    features = _clean(data.get("feature_columns", []))
    if not features:
        raise RuntimeError("LLM did not suggest any valid feature columns.")
    return ColumnAdvice(
        target_column=target,
        feature_columns=features,
        rationale=rationale,
        source="llm",
        model_used=model_tag,
        candidates=[features],
        candidate_reasoning=[""],
    )


def _llm_advice(
    frame: pd.DataFrame,
    api_key: str,
    model: str,
    base_url: str | None,
) -> ColumnAdvice:
    """Detect the endpoint's API style and call it.  Raises RuntimeError on failure."""
    schema_text = _build_schema_text(frame)
    user_msg = f"{schema_text}\n\nPick the best TARGET and FEATURE columns for an SVM."
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if base_url:
        style, resolved = _detect_backend(base_url)
        if style == "ollama":
            raw = _call_ollama_native(resolved, model, messages)
            tag = f"{model} (ollama-native @ {resolved})"
        else:
            raw = _call_openai_compat(resolved, api_key, model, messages)
            tag = f"{model} (openai-compat @ {resolved})"
    else:
        raw = _call_openai_compat(None, api_key, model, messages)
        tag = f"{model} (openai)"

    return _parse_advice(raw, frame, tag)


# ── public API ─────────────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """\
You are a helpful data science assistant embedded in an SVM workbench application.
You help the user understand their dataset, choose features for an SVM classifier,
interpret model results, reason about data quality, and find new datasets to work with.

## Ground rules
- Ground every statement about the loaded data in the schema facts you are given.
- Do not invent column properties not present in the schema.
- Be concise but precise.  Bullet points are fine for lists.
- If asked to recommend columns, apply the same reasoning protocol as the feature
  advisor: check cardinality, exclude IDs/leaks, reason from variance and range.

## Dataset discovery knowledge
When the user wants to find a dataset, suggest specific sources based on what they
want to measure.  Always give a direct search URL or dataset URL, not a vague site name.

Well-known open repositories (link directly):
- UCI ML Repository  https://archive.ics.uci.edu  — tabular, well-labeled, classic benchmarks
- OpenML  https://www.openml.org/search?type=data  — thousands of classification/regression sets, API-accessible
- Kaggle  https://www.kaggle.com/datasets  — competitions + community uploads; wide variety
- HuggingFace Datasets  https://huggingface.co/datasets  — NLP-heavy but growing tabular section
- Google Dataset Search  https://datasetsearch.research.google.com  — meta-search across all repos
- PMLB (Penn ML Benchmarks)  https://epistasislab.github.io/pmlb  — curated SVM-ready tabular sets
- StatLib / CMU  http://lib.stat.cmu.edu/datasets  — older but clean statistical datasets
- Data.gov  https://data.gov  — US government open data; health, finance, environment
- World Bank Open Data  https://data.worldbank.org  — economic / demographic indicators
- Our World in Data  https://ourworldindata.org  — longitudinal global indicators with good labeling

SVM works well for these problem types (suggest datasets accordingly):
| Task type              | Good dataset examples / search terms                              |
|------------------------|-------------------------------------------------------------------|
| Binary classification  | medical diagnosis, fraud detection, spam, churn, fault detection  |
| Multi-class            | species identification, handwriting digits, wine cultivar, activity recognition |
| High-dimensional       | gene expression (TCGA), text TF-IDF, image pixel patches          |
| Imbalanced classes     | fraud, rare disease, defect detection (use class_weight=balanced) |
| Mixed features         | titanic-style survival, loan default, customer segmentation       |

When suggesting a dataset:
1. State what task type it suits and why it is a good SVM fit.
2. Give the direct dataset URL (UCI id, OpenML id, Kaggle path, or HuggingFace id).
3. Note the approximate size, number of features, and target cardinality if known.
4. Flag any known SVM-specific gotchas (scale sensitivity, class imbalance, leaky columns).
"""


_EXPLAIN_SYSTEM_PROMPT = """\
You are an expert data-science educator embedded in an SVM workbench.
The user just ran a machine-learning technique and got structured results.
Your job is to explain those results in **plain English** so a non-expert can
understand what happened and why.

## Structure of your answer
1. **What happened** — one-paragraph summary of the technique that was run.
2. **Key findings** — walk through the most important numbers and say what they
   mean in the context of this specific dataset (cite column names, class labels,
   and metric values).
3. **Why it makes sense** — give a grounded explanation using the dataset's domain.
   For example: "It makes sense that *glucose* and *BMI* are the strongest
   discriminators for diabetes because …".
4. **Surprises & counter-examples** — highlight anything that *contradicts*
   common intuition or popular assumptions.  Frame it like: "You might expect X,
   but the data shows Y instead — here is why that could happen."
5. **Practical take-away** — one or two sentences on what the user should do next
   or how to use this result.

## Rules
- Ground every claim in the numbers provided.  Do not invent metrics.
- Keep the total length between 200 and 600 words.
- Use Markdown formatting: bold key terms, bullet lists for findings.
- If the results are inconclusive or bad, say so honestly.
"""


def explain_result(
    technique: str,
    result_summary: str,
    dataset_context: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Ask the LLM to explain structured technique results in plain English.

    Parameters
    ----------
    technique : str
        Human-readable technique name, e.g. "Active Learning (Uncertainty Sampling)".
    result_summary : str
        Structured text with the key metrics / tables the LLM should interpret.
    dataset_context : str, optional
        Rich dataset description (from ``_build_data_context``).
    """
    user_parts = [f"## Technique: {technique}", "", result_summary]
    if dataset_context:
        user_parts.extend(["", "## Dataset context", dataset_context])

    messages: list[dict] = [
        {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    return chat_completion(
        messages, api_key=api_key, model=model, base_url=base_url,
    )


def chat_completion(
    messages: list[dict],
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Send a multi-turn chat message list to the configured LLM and return the reply.

    Uses the same backend detection and dispatch as *advise_columns*.
    Raises RuntimeError if no backend is reachable and no API key is set.
    """
    resolved_url = base_url or os.environ.get("LLM_BASE_URL", "") or None
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "") or None
    resolved_model = model or ("gemma-4-31B-it-Q4_K_M.gguf" if resolved_url else "gpt-4o-mini")

    if resolved_url:
        style, resolved = _detect_backend(resolved_url)
        if style == "ollama":
            return _call_ollama_native(resolved, resolved_model, messages)
        return _call_openai_compat(resolved, resolved_key or "ollama", resolved_model, messages)

    if resolved_key:
        return _call_openai_compat(None, resolved_key, resolved_model, messages)

    raise RuntimeError(
        "No LLM backend configured.  Set a base URL (Local/Remote LLM) or an OpenAI API key."
    )


def advise_columns(
    frame: pd.DataFrame,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ColumnAdvice:
    """Return the best column advice for fitting an SVM on *frame*.

    Resolution order:
    1. Explicit *base_url* argument
    2. ``LLM_BASE_URL`` environment variable
    3. Explicit *api_key* argument or ``OPENAI_API_KEY`` env var
    Falls back to a deterministic heuristic when no endpoint/key is available
    or when the API call fails.
    """
    resolved_url = base_url or os.environ.get("LLM_BASE_URL", "") or None
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "") or None

    if resolved_url or resolved_key:
        # Default model depends on whether we're talking to Ollama or OpenAI
        resolved_model = model or ("gemma-4-31B-it-Q4_K_M.gguf" if resolved_url else "gpt-4o-mini")
        try:
            return _llm_advice(
                frame,
                api_key=resolved_key or "ollama",
                model=resolved_model,
                base_url=resolved_url,
            )
        except Exception:
            pass  # fall through to heuristic

    return _heuristic_advice(frame)
