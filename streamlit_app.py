from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm_studio.custom_analysis import fit_custom_svm_estimator, run_custom_svm_analysis
from svm_studio.datasets import load_demo_frame_by_title, load_demo_sources
from svm_studio.episode_mining import (
    build_episode_dataset_from_event_columns,
    build_episode_dataset_from_sequence_column,
    episodes_to_frame,
    mine_episodes,
)
from svm_studio.itemset_mining import itemsets_to_frame, mine_itemsets_from_frame
from svm_studio import ui_shell
from svm_studio.visualization import apply_style


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes, separator: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), sep=separator)


@st.cache_data(show_spinner=False)
def load_demo_frame(dataset_title: str) -> pd.DataFrame:
    return load_demo_frame_by_title(dataset_title)


@st.cache_data(show_spinner=False)
def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _default_target_column(columns: list[str]) -> str:
    if "target" in columns:
        return "target"
    return columns[-1]


def _sanitize_selection(current: list[str] | None, valid_options: list[str], fallback: list[str]) -> list[str]:
    selected = [value for value in (current or []) if value in valid_options]
    return selected or fallback


def sync_workspace_state(frame: pd.DataFrame, source_name: str) -> None:
    columns = list(frame.columns)
    if not columns:
        return

    default_target = _default_target_column(columns)
    target_column = st.session_state.get("target_column")
    if target_column not in columns:
        st.session_state["target_column"] = default_target
    target_column = st.session_state["target_column"]

    feature_options = [column for column in columns if column != target_column]
    default_features = feature_options[:]
    st.session_state["feature_columns"] = _sanitize_selection(
        st.session_state.get("feature_columns"),
        feature_options,
        default_features,
    )

    spotlight_column = st.session_state.get("spotlight_column")
    if spotlight_column not in columns:
        st.session_state["spotlight_column"] = columns[0]

    valid_kernels = ["linear", "rbf", "poly"]
    st.session_state["kernels"] = _sanitize_selection(
        st.session_state.get("kernels"),
        valid_kernels,
        ["linear", "rbf"],
    )

    numeric_features = _numeric_visualization_columns(frame, st.session_state["feature_columns"])
    if len(numeric_features) >= 2:
        geometry_x = st.session_state.get("geometry_x")
        if geometry_x not in numeric_features:
            st.session_state["geometry_x"] = numeric_features[0]
        geometry_y_options = [column for column in numeric_features if column != st.session_state["geometry_x"]]
        geometry_y = st.session_state.get("geometry_y")
        if geometry_y not in geometry_y_options:
            st.session_state["geometry_y"] = geometry_y_options[0]

    if len(numeric_features) >= 3:
        plane_x = st.session_state.get("plane_x")
        if plane_x not in numeric_features:
            st.session_state["plane_x"] = numeric_features[0]
        plane_y_options = [column for column in numeric_features if column != st.session_state["plane_x"]]
        plane_y = st.session_state.get("plane_y")
        if plane_y not in plane_y_options:
            st.session_state["plane_y"] = plane_y_options[0]
        plane_z_options = [
            column for column in numeric_features if column not in {st.session_state["plane_x"], st.session_state["plane_y"]}
        ]
        plane_z = st.session_state.get("plane_z")
        if plane_z not in plane_z_options and plane_z_options:
            st.session_state["plane_z"] = plane_z_options[0]

    default_item_columns = columns[: min(6, len(columns))]
    st.session_state["item_columns"] = _sanitize_selection(
        st.session_state.get("item_columns"),
        columns,
        default_item_columns,
    )

    episode_mode = st.session_state.get("episode_mode")
    if episode_mode not in {"Delimited sequence column", "Ordered event columns"}:
        st.session_state["episode_mode"] = "Delimited sequence column"

    sequence_column = st.session_state.get("sequence_column")
    if sequence_column not in columns:
        object_candidates = [
            column for column in columns if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
        ]
        st.session_state["sequence_column"] = object_candidates[0] if object_candidates else columns[0]

    default_episode_columns = columns[: min(4, len(columns))]
    st.session_state["episode_columns"] = _sanitize_selection(
        st.session_state.get("episode_columns"),
        columns,
        default_episode_columns,
    )

    st.session_state.setdefault("source_mode", "Upload CSV")
    st.session_state.setdefault("csv_separator", ",")
    st.session_state.setdefault("test_size", 0.25)
    st.session_state.setdefault("item_support", 0.20)
    st.session_state.setdefault("episode_length", 3)
    st.session_state.setdefault("episode_span", 4)
    st.session_state.setdefault("episode_support", 0.30)
    st.session_state.setdefault("sequence_separator", ",")
    st.session_state.setdefault("include_target", False)
    st.session_state.setdefault("geometry_mode", "2D decision view")
    st.session_state["_active_source_name"] = source_name


def build_svm_run_signature(source_name: str, target_column: str, feature_columns: list[str], kernels: list[str], test_size: float) -> tuple:
    return (source_name, target_column, tuple(feature_columns), tuple(kernels), float(test_size))


def build_itemset_run_signature(source_name: str, item_columns: list[str], include_target: bool, target_column: str | None, min_support: float) -> tuple:
    return (source_name, tuple(item_columns), bool(include_target), target_column or "", float(min_support))


def build_episode_run_signature(
    source_name: str,
    mode: str,
    sequence_column: str | None,
    separator: str | None,
    ordered_columns: list[str] | None,
    max_length: int,
    max_span: int,
    min_support: float,
) -> tuple:
    return (
        source_name,
        mode,
        sequence_column or "",
        separator or "",
        tuple(ordered_columns or []),
        int(max_length),
        int(max_span),
        float(min_support),
    )


def clear_computed_results() -> None:
    for key in [
        "custom_svm_result",
        "custom_svm_error",
        "custom_svm_signature",
        "itemset_result",
        "itemset_error",
        "itemset_signature",
        "episode_result",
        "episode_error",
        "episode_signature",
    ]:
        st.session_state.pop(key, None)


def assess_mining_readiness(frame: pd.DataFrame) -> dict[str, object]:
    columns = list(frame.columns)
    object_columns = [
        column for column in columns if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
    ]
    categorical_like_columns = [
        column
        for column in columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
            or frame[column].nunique(dropna=False) <= min(24, max(6, len(frame) // 20))
        )
    ]

    sequence_like_columns: list[str] = []
    for column in object_columns:
        text = frame[column].dropna().astype(str)
        if text.empty:
            continue
        sequence_ratio = text.str.contains(r"[,;>|]", regex=True).mean()
        if sequence_ratio >= 0.25:
            sequence_like_columns.append(column)

    ordered_event_columns = [
        column
        for column in columns
        if any(token in column.lower() for token in ("event_", "step_", "stage_", "action_", "page_", "touch_"))
    ]
    if len(ordered_event_columns) < 2:
        numeric_suffix_columns = [column for column in columns if any(char.isdigit() for char in column)]
        ordered_event_columns = numeric_suffix_columns[:6] if len(numeric_suffix_columns) >= 2 else ordered_event_columns

    itemset_ready = len(categorical_like_columns) >= 2 and len(frame) >= 10
    episode_ready = bool(sequence_like_columns) or len(ordered_event_columns) >= 2

    return {
        "itemset_ready": itemset_ready,
        "episode_ready": episode_ready,
        "categorical_like_columns": categorical_like_columns,
        "sequence_like_columns": sequence_like_columns,
        "ordered_event_columns": ordered_event_columns,
    }


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-cream: #f5efe7;
            --bg-mint: #eaf3f0;
            --ink: #142126;
            --muted: #5b6d74;
            --panel: rgba(255, 252, 246, 0.78);
            --panel-strong: rgba(255, 255, 255, 0.92);
            --line: rgba(20, 33, 38, 0.09);
            --accent: #0f8b8d;
            --accent-hot: #f97316;
            --accent-warm: #f4b400;
            --shadow: 0 24px 55px rgba(17, 24, 39, 0.12);
            --radius-xl: 30px;
            --radius-lg: 24px;
            --radius-md: 18px;
        }

        html, body, [class*="css"] {
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(249, 115, 22, 0.18), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(15, 139, 141, 0.18), transparent 24%),
                linear-gradient(180deg, var(--bg-cream) 0%, var(--bg-mint) 52%, #f8f9f6 100%);
        }

        .main .block-container {
            max-width: 1420px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }

        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(24, 36, 52, 0.96) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8f4ec !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label p {
            color: #f8f4ec !important;
        }

        [data-baseweb="base-input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] > div > div,
        [data-testid="stTextInput"] > div > div {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--line) !important;
            background: rgba(255, 255, 255, 0.9) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div,
        [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: var(--radius-lg);
            border: 1px dashed rgba(255, 255, 255, 0.24);
            background: rgba(255, 255, 255, 0.05);
        }

        [data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            padding: 1rem 1.1rem 0.4rem 1.1rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            padding: 0.35rem;
            box-shadow: 0 16px 35px rgba(17, 24, 39, 0.08);
        }

        div.stButton > button,
        div.stDownloadButton > button,
        button[kind="primaryFormSubmit"] {
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--accent-hot) 0%, #ff8f3d 100%);
            color: white;
            font-weight: 700;
            letter-spacing: 0.01em;
            padding: 0.72rem 1.25rem;
            box-shadow: 0 16px 30px rgba(249, 115, 22, 0.28);
            transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        button[kind="primaryFormSubmit"]:hover {
            transform: translateY(-1px);
            filter: saturate(1.04);
            box-shadow: 0 18px 32px rgba(249, 115, 22, 0.34);
        }

        [data-testid="stTabs"] button[role="tab"] {
            height: auto;
            padding: 0.8rem 1.05rem;
            border-radius: 999px;
            border: 1px solid rgba(20, 33, 38, 0.08);
            background: rgba(255, 255, 255, 0.56);
            color: var(--muted);
            font-weight: 700;
            margin-right: 0.55rem;
        }

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(15, 139, 141, 0.95), rgba(11, 111, 130, 0.92));
            color: white;
            border-color: transparent;
            box-shadow: 0 16px 32px rgba(15, 139, 141, 0.24);
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem;
            border-radius: var(--radius-xl);
            background:
                radial-gradient(circle at 85% 15%, rgba(244, 180, 0, 0.16), transparent 24%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(12, 94, 114, 0.92) 54%, rgba(15, 139, 141, 0.88) 100%);
            box-shadow: 0 30px 65px rgba(15, 23, 42, 0.24);
            margin-bottom: 1.6rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -5% -28% 48%;
            height: 260px;
            background: radial-gradient(circle, rgba(249, 115, 22, 0.35) 0%, transparent 62%);
            filter: blur(18px);
            pointer-events: none;
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
            gap: 1.35rem;
            align-items: stretch;
        }

        .hero-kicker,
        .section-kicker,
        .stat-label,
        .panel-kicker {
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            letter-spacing: 0.16em;
        }

        .hero-kicker {
            margin: 0 0 0.65rem 0;
            font-size: 0.78rem;
            color: rgba(255, 248, 240, 0.72);
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.3rem, 4.2vw, 4.5rem);
            line-height: 0.94;
            color: #fffaf1;
        }

        .hero-copy {
            max-width: 52rem;
            margin: 0.95rem 0 0 0;
            font-size: 1rem;
            line-height: 1.65;
            color: rgba(255, 248, 240, 0.82);
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.55rem 0.88rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.12);
            color: #fffaf1;
            font-size: 0.88rem;
            backdrop-filter: blur(12px);
        }

        .hero-panel {
            border-radius: 26px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.12);
            padding: 1.2rem 1.25rem;
            backdrop-filter: blur(16px);
        }

        .panel-kicker {
            margin: 0 0 0.55rem 0;
            font-size: 0.75rem;
            color: rgba(255, 248, 240, 0.72);
        }

        .panel-number {
            margin: 0;
            font-size: 2.6rem;
            font-weight: 700;
            color: #fffaf1;
        }

        .panel-copy {
            margin: 0.55rem 0 0 0;
            color: rgba(255, 248, 240, 0.78);
            line-height: 1.55;
        }

        .section-intro {
            margin: 0.15rem 0 1rem 0;
        }

        .section-kicker {
            margin: 0 0 0.3rem 0;
            font-size: 0.78rem;
            color: var(--accent);
        }

        .section-title {
            margin: 0;
            font-size: 1.85rem;
            line-height: 1.05;
            color: var(--ink);
        }

        .section-copy {
            margin: 0.6rem 0 0 0;
            max-width: 58rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.95rem;
            margin: 1rem 0 1.6rem 0;
        }

        .stat-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1.05rem;
            box-shadow: 0 18px 40px rgba(17, 24, 39, 0.09);
            backdrop-filter: blur(14px);
        }

        .stat-label {
            margin: 0;
            font-size: 0.76rem;
            color: var(--muted);
        }

        .stat-value {
            margin: 0.45rem 0 0 0;
            font-size: 1.95rem;
            line-height: 1;
            color: var(--ink);
            font-weight: 700;
        }

        .stat-note {
            margin: 0.5rem 0 0 0;
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.45;
        }

        .callout-card {
            margin: 0.55rem 0 1.1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(15, 139, 141, 0.15);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(243, 249, 247, 0.94));
            box-shadow: 0 16px 34px rgba(17, 24, 39, 0.08);
        }

        .callout-title {
            margin: 0;
            font-size: 1rem;
            font-weight: 700;
            color: var(--ink);
        }

        .callout-copy {
            margin: 0.35rem 0 0 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .method-box {
            margin: 0.75rem 0 1.1rem 0;
            padding: 1.05rem 1.1rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 33, 38, 0.08);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 251, 249, 0.92));
            box-shadow: 0 18px 36px rgba(17, 24, 39, 0.08);
        }

        .method-box-title {
            margin: 0;
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--ink);
        }

        .method-box-copy {
            margin: 0.38rem 0 0.9rem 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .method-box-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.8rem;
        }

        .method-item {
            padding: 0.85rem 0.9rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 139, 141, 0.12);
            background: rgba(255, 255, 255, 0.9);
        }

        .method-name {
            margin: 0;
            font-size: 0.83rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .method-formula {
            margin: 0.55rem 0 0 0;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.9rem;
            line-height: 1.55;
            color: var(--ink);
            white-space: pre-wrap;
        }

        .method-note {
            margin: 0.55rem 0 0 0;
            color: var(--muted);
            line-height: 1.5;
            font-size: 0.93rem;
        }

        @media (max-width: 980px) {
            .hero-grid,
            .stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <p class="section-kicker">{html.escape(kicker)}</p>
            <h2 class="section-title">{html.escape(title)}</h2>
            <p class="section-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_grid(cards: list[tuple[str, str, str]]) -> None:
    body = "".join(
        f"""
        <div class="stat-card">
            <p class="stat-label">{html.escape(label)}</p>
            <p class="stat-value">{html.escape(value)}</p>
            <p class="stat-note">{html.escape(note)}</p>
        </div>
        """
        for label, value, note in cards
    )
    st.markdown(f'<div class="stat-grid">{body}</div>', unsafe_allow_html=True)


def render_callout(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="callout-card">
            <p class="callout-title">{html.escape(title)}</p>
            <p class="callout-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_box(title: str, copy: str, items: list[tuple[str, str, str]]) -> None:
    item_html = "".join(
        f"""
        <div class="method-item">
            <p class="method-name">{html.escape(name)}</p>
            <p class="method-formula">{html.escape(formula)}</p>
            <p class="method-note">{html.escape(note)}</p>
        </div>
        """
        for name, formula, note in items
    )
    st.markdown(
        f"""
        <section class="method-box">
            <p class="method-box-title">{html.escape(title)}</p>
            <p class="method-box-copy">{html.escape(copy)}</p>
            <div class="method-box-grid">{item_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_hero(source_name: str, frame: pd.DataFrame) -> None:
    numeric_count = int(frame.select_dtypes(include="number").shape[1])
    categorical_count = int(frame.shape[1] - numeric_count)
    missing_count = int(frame.isna().sum().sum())
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-grid">
                <div>
                    <p class="hero-kicker">SVM / ITEMSET / EPISODE WORKBENCH</p>
                    <h1 class="hero-title">SVM Data Studio</h1>
                    <p class="hero-copy">
                        Load a dataset, isolate the columns that matter, inspect the separator geometry,
                        and generate mining outputs from the same polished workspace.
                    </p>
                    <div class="chip-row">
                        <span class="chip">{html.escape(source_name)}</span>
                        <span class="chip">{frame.shape[0]} rows</span>
                        <span class="chip">{frame.shape[1]} columns</span>
                        <span class="chip">{numeric_count} numeric</span>
                        <span class="chip">{categorical_count} categorical</span>
                    </div>
                </div>
                <aside class="hero-panel">
                    <div>
                        <p class="panel-kicker">Data Snapshot</p>
                        <p class="panel-number">{frame.shape[0]:,}</p>
                        <p class="panel-copy">
                            Rows currently staged for modeling. Missing cells: {missing_count:,}. Use the tabs below
                            to move from schema review to SVM geometry and mining outputs.
                        </p>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def format_numeric_value(value: object, force_scientific: bool = False) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if numeric == 0:
            return "0"
        magnitude = abs(numeric)
        if force_scientific or magnitude < 1e-3 or magnitude >= 1e4:
            return f"{numeric:.3e}"
        if magnitude >= 100:
            return f"{numeric:.2f}"
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_parameter_mapping(params: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        label = key.replace("svc__", "")
        if isinstance(value, (int, np.integer)):
            display = format_numeric_value(value)
        elif isinstance(value, (float, np.floating)):
            display = format_numeric_value(value, force_scientific=True)
        else:
            display = str(value)
        parts.append(f"{label} = {display}")
    return " | ".join(parts) if parts else "No tuned parameters recorded"


def build_display_frame(frame: pd.DataFrame, scientific_columns: set[str] | None = None) -> pd.DataFrame:
    scientific_columns = scientific_columns or set()
    display = frame.copy()

    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            force_scientific = column in scientific_columns
            display[column] = display[column].map(lambda value: format_numeric_value(value, force_scientific=force_scientific))

    return display


def plot_confusion_figure(confusion, labels) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#18c5d8", as_cmap=True),
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    return figure


def plot_feature_importance_figure(importance_frame: pd.DataFrame) -> plt.Figure:
    top = importance_frame.head(12).sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    palette = sns.color_palette(["#103a53", "#18c5d8", "#7ef2c8", "#ffd166", "#ff7a18"], n_colors=len(top))
    sns.barplot(data=top, x="importance_mean", y="feature", hue="feature", dodge=False, palette=palette, legend=False, ax=axis)
    axis.set_title("Columns That Matter Most")
    axis.set_xlabel("Permutation importance")
    axis.set_ylabel("Feature")
    return figure


def plot_kernel_results_figure(kernel_results: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    sns.barplot(
        data=kernel_results,
        x="kernel",
        y="test_accuracy",
        hue="kernel",
        dodge=False,
        palette={"linear": "#18c5d8", "rbf": "#ff7a18", "poly": "#103a53"},
        legend=False,
        ax=axis,
    )
    axis.set_title("Kernel Comparison")
    axis.set_xlabel("Kernel")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.05)
    return figure


def plot_column_profile_figure(frame: pd.DataFrame, column: str) -> plt.Figure:
    series = frame[column]
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        sns.histplot(series.dropna(), kde=True, color="#18c5d8", edgecolor="white", linewidth=0.6, ax=axis)
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        return figure

    counts = (
        series.fillna("<missing>")
        .astype(str)
        .value_counts(dropna=False)
        .head(12)
        .sort_values(ascending=True)
    )
    palette = sns.color_palette(["#103a53", "#18c5d8", "#7ef2c8", "#ffd166", "#ff7a18"], n_colors=len(counts))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, dodge=False, palette=palette, legend=False, ax=axis)
    axis.set_title(f"{column} top categories")
    axis.set_xlabel("Count")
    axis.set_ylabel(column)
    return figure


def _numeric_visualization_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]


def _imputed_numeric_frame(model, feature_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_pipeline = model.named_steps["preprocessor"].named_transformers_["numeric"]
    imputed = numeric_pipeline.named_steps["imputer"].transform(feature_frame)
    return pd.DataFrame(imputed, columns=feature_frame.columns, index=feature_frame.index)


def plot_decision_surface_figure(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    kernel: str,
    best_params: dict[str, object],
) -> plt.Figure:
    model, X, y = fit_custom_svm_estimator(
        frame=frame,
        target_column=target_column,
        feature_columns=feature_columns,
        kernel=kernel,
        best_params=best_params,
    )
    plotted = _imputed_numeric_frame(model, X)
    labels = sorted(y.unique().tolist())
    palette = sns.color_palette(["#18c5d8", "#ff7a18", "#7ef2c8", "#295fe7", "#ffd166"], n_colors=len(labels))

    x_column, y_column = feature_columns
    x_min, x_max = plotted[x_column].min(), plotted[x_column].max()
    y_min, y_max = plotted[y_column].min(), plotted[y_column].max()
    pad_x = (x_max - x_min) * 0.08 or 1.0
    pad_y = (y_max - y_min) * 0.08 or 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, 250),
        np.linspace(y_min - pad_y, y_max + pad_y, 250),
    )
    grid = pd.DataFrame({x_column: xx.ravel(), y_column: yy.ravel()})
    prediction_codes = pd.Categorical(model.predict(grid), categories=labels).codes.reshape(xx.shape)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.contourf(
        xx,
        yy,
        prediction_codes,
        alpha=0.22,
        levels=np.arange(len(labels) + 1) - 0.5,
        cmap=sns.blend_palette(["#ecf7fb", "#b8ecf2", "#88dcea", "#3fa8d7"], as_cmap=True),
    )

    for index, label in enumerate(labels):
        mask = y == label
        axis.scatter(
            plotted.loc[mask, x_column],
            plotted.loc[mask, y_column],
            color=palette[index],
            s=48,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )

    support_points = plotted.iloc[model.named_steps["svc"].support_]
    axis.scatter(
        support_points[x_column],
        support_points[y_column],
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        s=130,
        label="support vectors",
    )
    axis.set_title(f"Decision Boundary: {x_column} vs {y_column}")
    axis.set_xlabel(x_column)
    axis.set_ylabel(y_column)
    axis.legend(loc="best", frameon=True)
    return figure


def plot_linear_plane_figure(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    best_params: dict[str, object],
) -> plt.Figure | None:
    model, X, y = fit_custom_svm_estimator(
        frame=frame,
        target_column=target_column,
        feature_columns=feature_columns,
        kernel="linear",
        best_params=best_params,
    )
    if y.nunique() != 2:
        return None

    plotted = _imputed_numeric_frame(model, X)
    labels = sorted(y.unique().tolist())
    palette = sns.color_palette(["#18c5d8", "#ff7a18"], n_colors=2)
    numeric_pipeline = model.named_steps["preprocessor"].named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    svc = model.named_steps["svc"]
    if not hasattr(svc, "coef_"):
        return None

    w_scaled = svc.coef_[0]
    b_scaled = svc.intercept_[0]
    w_original = w_scaled / scaler.scale_
    b_original = b_scaled - np.sum((w_scaled * scaler.mean_) / scaler.scale_)
    solve_index = int(np.argmax(np.abs(w_original)))
    if abs(w_original[solve_index]) < 1e-9:
        return None

    remaining = [index for index in range(3) if index != solve_index]
    first_index, second_index = remaining
    first_column = feature_columns[first_index]
    second_column = feature_columns[second_index]
    first_values = np.linspace(plotted[first_column].min(), plotted[first_column].max(), 25)
    second_values = np.linspace(plotted[second_column].min(), plotted[second_column].max(), 25)
    first_grid, second_grid = np.meshgrid(first_values, second_values)
    solved_grid = -(
        b_original
        + w_original[first_index] * first_grid
        + w_original[second_index] * second_grid
    ) / w_original[solve_index]

    coordinate_grids = [None, None, None]
    coordinate_grids[first_index] = first_grid
    coordinate_grids[second_index] = second_grid
    coordinate_grids[solve_index] = solved_grid

    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for index, label in enumerate(labels):
        mask = y == label
        axis.scatter(
            plotted.loc[mask, feature_columns[0]],
            plotted.loc[mask, feature_columns[1]],
            plotted.loc[mask, feature_columns[2]],
            color=palette[index],
            s=34,
            alpha=0.80,
            label=label,
        )

    support_points = plotted.iloc[svc.support_]
    axis.scatter(
        support_points[feature_columns[0]],
        support_points[feature_columns[1]],
        support_points[feature_columns[2]],
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        s=130,
        label="support vectors",
    )
    axis.plot_surface(
        coordinate_grids[0],
        coordinate_grids[1],
        coordinate_grids[2],
        alpha=0.30,
        color="#18c5d8",
        linewidth=0,
    )
    axis.set_title("Linear SVM Plane")
    axis.set_xlabel(feature_columns[0])
    axis.set_ylabel(feature_columns[1])
    axis.set_zlabel(feature_columns[2])
    axis.legend(loc="upper left")
    return figure


def plot_pattern_figure(frame: pd.DataFrame, label_column: str, title: str, palette: str) -> plt.Figure:
    top = frame[frame["length"] >= 2].head(10).sort_values("support", ascending=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    sns.barplot(data=top, x="support", y=label_column, hue=label_column, dodge=False, palette=palette, legend=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Support")
    axis.set_ylabel(label_column.replace("_", " ").title())
    return figure


def load_data_source() -> tuple[pd.DataFrame | None, str | None]:
    with st.sidebar:
        st.header("Data Source")
        source_mode = st.radio("Choose input", ["Upload CSV", "Built-in demo"], index=0, key="source_mode")

        if source_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            separator = st.text_input("CSV separator", key="csv_separator")
            if uploaded_file is None:
                return None, None
            frame = read_uploaded_csv(uploaded_file.getvalue(), separator)
            return frame, uploaded_file.name

        demo_sources = load_demo_sources()
        demo_options = [source.title for source in demo_sources]
        st.session_state.setdefault("demo_name", demo_options[0])
        if st.session_state["demo_name"] not in demo_options:
            st.session_state["demo_name"] = demo_options[0]
        demo_name = st.selectbox("Choose a demo dataset", demo_options, key="demo_name")
        selected_source = next(source for source in demo_sources if source.title == demo_name)
        st.caption(f"{selected_source.group.upper()} DEMO: {selected_source.description}")
        return load_demo_frame(demo_name), demo_name


def render_data_tab(frame: pd.DataFrame) -> None:
    readiness = assess_mining_readiness(frame)
    ui_shell.render_section_intro(
        "Data Atlas",
        "Read the shape of the dataset before you model it",
        "Start with structure, missingness, and value distribution. A clean read here makes the SVM and mining outputs far more trustworthy.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Scan the dataset", "Check row and column volume, missingness, and duplication before you trust downstream outputs."),
            ("Inspect the schema", "Read dtypes and uniqueness so you know which columns can model cleanly and which need care."),
            ("Spotlight one field", "Drill into a single column to understand distribution shape before selecting targets or mined inputs."),
        ],
    )
    ui_shell.render_stat_grid(
        [
            ("Rows", f"{frame.shape[0]:,}", "Total observations loaded into the workspace."),
            ("Columns", f"{frame.shape[1]:,}", "Available fields across numeric and categorical data."),
            ("Missing cells", f"{int(frame.isna().sum().sum()):,}", "Gaps that may affect modeling or mining quality."),
            ("Duplicate rows", f"{int(frame.duplicated().sum()):,}", "Potential redundancy in the uploaded sample."),
        ]
    )
    ui_shell.render_section_intro(
        "Mining Checker",
        "See whether this spreadsheet is ready for itemsets, episodes, or neither",
        "Itemsets need stable row-level items. Episodes need order, either in a sequence column or across ordered event columns.",
    )
    ui_shell.render_stat_grid(
        [
            (
                "Itemset status",
                "Ready" if readiness["itemset_ready"] else "Needs work",
                f"{len(readiness['categorical_like_columns'])} columns look usable for row-level items.",
            ),
            (
                "Episode status",
                "Ready" if readiness["episode_ready"] else "Needs work",
                f"{len(readiness['sequence_like_columns'])} sequence-like columns and {len(readiness['ordered_event_columns'])} ordered-column candidates found.",
            ),
            (
                "Sequence columns",
                f"{len(readiness['sequence_like_columns']):,}",
                "Columns whose text already looks like ordered event strings.",
            ),
            (
                "Event columns",
                f"{len(readiness['ordered_event_columns']):,}",
                "Columns whose names suggest a left-to-right event layout.",
            ),
        ]
    )
    if readiness["itemset_ready"] and readiness["episode_ready"]:
        ui_shell.render_state_panel(
            "success",
            "This file can drive both mining modes",
            "You have enough row-level item structure for frequent itemsets and enough order information for episode mining.",
            detail=(
                f"Itemset candidates: {', '.join(readiness['categorical_like_columns'][:6])} | "
                f"Episode cues: {', '.join((readiness['sequence_like_columns'] or readiness['ordered_event_columns'])[:6])}"
            ),
        )
    elif readiness["itemset_ready"]:
        ui_shell.render_state_panel(
            "info",
            "This file is itemset-ready",
            "The sheet has enough row-level structure for frequent itemset mining, but it does not clearly expose ordered event sequences yet.",
            detail="To unlock episode mining, add a delimited sequence column or ordered event columns such as event_1, event_2, event_3.",
        )
    elif readiness["episode_ready"]:
        ui_shell.render_state_panel(
            "info",
            "This file is episode-ready",
            "The sheet exposes ordered event information, but it may need cleaner categorical columns before itemset mining becomes useful.",
            detail="For itemsets, make sure each row contains stable categorical values or bucketed numeric fields that you would treat as items.",
        )
    else:
        ui_shell.render_state_panel(
            "warning",
            "This file needs reshaping before mining",
            "The checker does not see clear transaction-style columns or ordered event structure yet.",
            detail="Itemsets want row-level categories. Episodes want a sequence column or ordered event columns.",
        )

    preview, schema = st.columns([1.8, 1.2])

    with preview:
        st.subheader("Preview")
        st.dataframe(frame.head(50), use_container_width=True)

    with schema:
        st.subheader("Schema")
        schema_frame = pd.DataFrame(
            {
                "column": frame.columns,
                "dtype": frame.dtypes.astype(str),
                "missing": frame.isna().sum().to_numpy(),
                "unique": frame.nunique(dropna=False).to_numpy(),
            }
        )
        st.dataframe(schema_frame, use_container_width=True, hide_index=True)

    st.subheader("Column spotlight")
    spotlight_column = st.selectbox("Inspect one column in detail", frame.columns, key="spotlight_column")
    profile_column, details_column = st.columns([1.35, 0.65])

    with profile_column:
        st.pyplot(plot_column_profile_figure(frame, spotlight_column))

    with details_column:
        series = frame[spotlight_column]
        ui_shell.render_stat_grid(
            [
                ("Type", str(series.dtype), "Detected pandas dtype for this field."),
                ("Missing", f"{int(series.isna().sum()):,}", "Rows missing a value in this column."),
                ("Unique", f"{int(series.nunique(dropna=False)):,}", "Distinct values including missing if present."),
                ("Non-null", f"{int(series.notna().sum()):,}", "Rows available for analysis."),
            ]
        )


def render_svm_tab(frame: pd.DataFrame, source_name: str) -> None:
    ui_shell.render_section_intro(
        "Model Forge",
        "Train the classifier and surface the columns that move the margin",
        "Choose the target, choose the feature set, compare kernels, and then inspect the resulting separator from a geometric point of view.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Define the prediction target", "Pick the label column first so the rest of the page can frame features and metrics correctly."),
            ("Compare kernels", "Run linear, RBF, or polynomial fits and let the app choose the strongest holdout performer."),
            ("Inspect the boundary", "Move from metrics into confusion, importance, and geometry so the margin is not just a number."),
        ],
    )
    columns = list(frame.columns)
    target_column = st.selectbox("Target column", columns, key="target_column")
    feature_options = [column for column in columns if column != target_column]
    feature_columns = st.multiselect(
        "Feature columns",
        feature_options,
        default=feature_options,
        key="feature_columns",
    )

    ui_shell.render_callout(
        "Geometry rule of thumb",
        "One feature gives you a threshold point, two features give you a line, three features give you a plane, and anything beyond that is a higher-dimensional hyperplane that needs slicing or projection to visualize.",
    )
    ui_shell.render_method_box(
        "SVM math and scoring",
        "These are the exact ideas behind the model selection view, the margin, and the ranking of important columns.",
        [
            (
                "Hyperplane",
                "H = {x : w·x + b = 0}",
                "For a linear SVM, the separator is the set of points whose decision score is exactly zero.",
            ),
            (
                "Soft-margin objective",
                "min  1/2 ||w||^2 + C Σ ξ_i",
                "The model balances a wide margin against penalties for classification violations.",
            ),
            (
                "Decision rule",
                "f(x) = sign(w·x + b)",
                "With kernels, the dot product is replaced by a kernel similarity function.",
            ),
            (
                "Metrics",
                "accuracy = correct / N\nmacro_F1 = (1/K) Σ F1_k",
                "Accuracy scores overall correctness. Macro F1 treats each class evenly before averaging.",
            ),
            (
                "Permutation importance",
                "Δ_j = score_baseline - score_after_shuffle(j)",
                "A column matters when shuffling it produces a bigger drop in holdout performance.",
            ),
            (
                "Scientific notation",
                "1.00e-02 = 0.01",
                "Small tuned values such as gamma are easier to read in scientific notation.",
            ),
        ],
    )
    with st.form("svm_form"):
        controls = st.columns(3)
        with controls[0]:
            kernels = st.multiselect("Kernels", ["linear", "rbf", "poly"], default=["linear", "rbf"], key="kernels")
        with controls[1]:
            test_size = st.slider("Test split", min_value=0.10, max_value=0.40, value=0.25, step=0.05, key="test_size")
        with controls[2]:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            run_analysis = st.form_submit_button("Run SVM analysis", use_container_width=True)

    current_signature = build_svm_run_signature(source_name, target_column, feature_columns, kernels, test_size)

    if run_analysis:
        try:
            with st.status("Running SVM analysis", expanded=True) as status:
                try:
                    status.write("Validating the selected target and feature set.")
                    status.write("Training and comparing the requested kernels.")
                    result = run_custom_svm_analysis(
                        frame=frame,
                        target_column=target_column,
                        feature_columns=feature_columns,
                        kernels=kernels,
                        test_size=test_size,
                    )
                    status.write("Ranking feature impact and preparing evaluation views.")
                    status.update(label="SVM analysis complete", state="complete", expanded=False)
                except ValueError:
                    status.update(label="SVM analysis failed", state="error", expanded=True)
                    raise
            st.session_state["custom_svm_result"] = result
            st.session_state["custom_svm_signature"] = current_signature
            st.session_state.pop("custom_svm_error", None)
        except ValueError as exc:
            st.session_state["custom_svm_error"] = str(exc)
            st.session_state.pop("custom_svm_result", None)
            st.session_state.pop("custom_svm_signature", None)

    if "custom_svm_error" in st.session_state:
        ui_shell.render_state_panel(
            "error",
            "SVM analysis could not run",
            "The current target, feature set, or kernel setup is invalid for model training.",
            detail=st.session_state["custom_svm_error"],
        )
        return

    result = st.session_state.get("custom_svm_result")
    if result is None:
        ui_shell.render_state_panel(
            "info",
            "Analysis not started yet",
            "Choose a target, select the feature columns that matter, and run the comparison to generate metrics, geometry views, and column importance.",
        )
        return

    if st.session_state.get("custom_svm_signature") != current_signature:
        ui_shell.render_state_panel(
            "info",
            "Displaying the last completed SVM run",
            "The controls changed after the previous analysis finished, so the tables and plots below still reflect the last completed configuration.",
            detail="Run SVM analysis again to refresh the metrics, geometry, and importance outputs for the current selections.",
        )

    ui_shell.render_stat_grid(
        [
            ("Best kernel", result.selected_kernel.upper(), "Highest cross-validation score among the kernels tested."),
            ("Accuracy", f"{result.test_accuracy:.3f}", "Holdout performance on the test split."),
            ("Macro F1", f"{result.macro_f1:.3f}", "Balanced signal across classes."),
            ("Support vectors", f"{result.support_vector_count:,}", "Training points that define the margin."),
        ]
    )
    ui_shell.render_state_panel(
        "success",
        "SVM analysis ready",
        "The model comparison completed successfully and the selected boundary is now available for inspection across metrics, confusion, and geometry views.",
        detail=f"Selected kernel: {result.selected_kernel.upper()} | Accuracy: {result.test_accuracy:.3f} | Macro F1: {result.macro_f1:.3f}",
    )
    ui_shell.render_callout(
        "Selected method snapshot",
        f"Target = {result.target_column} | Features = {len(result.feature_columns)} | Parameters = {format_parameter_mapping(result.selected_params)}",
    )

    plots = st.columns(2)
    with plots[0]:
        st.pyplot(plot_kernel_results_figure(result.kernel_results))
    with plots[1]:
        st.pyplot(plot_feature_importance_figure(result.feature_importance))

    confusion_column, report_column = st.columns([1.1, 0.9])
    with confusion_column:
        st.pyplot(plot_confusion_figure(result.confusion, result.class_labels))
    with report_column:
        st.subheader("Classification report")
        st.dataframe(
            build_display_frame(result.classification_report, scientific_columns={"support"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Kernel results")
    st.dataframe(
        build_display_frame(result.kernel_results, scientific_columns={"cv_std"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download kernel results CSV",
        dataframe_bytes(result.kernel_results),
        file_name="kernel_results.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )

    st.subheader("Feature importance")
    st.dataframe(
        build_display_frame(result.feature_importance, scientific_columns={"importance_mean", "importance_std"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download feature importance CSV",
        dataframe_bytes(result.feature_importance),
        file_name="feature_importance.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )

    st.subheader("Geometry view")
    st.caption("Use this to translate the abstract hyperplane into something you can actually inspect on screen.")
    numeric_features = _numeric_visualization_columns(frame, result.feature_columns)

    if len(numeric_features) < 2:
        ui_shell.render_state_panel(
            "info",
            "Geometry view unavailable",
            "A decision boundary plot needs at least two numeric feature columns from the trained model.",
            detail="Add more numeric features to the model selection and rerun the analysis to unlock the geometry panel.",
        )
        return

    geometry_modes = ["2D decision view"]
    if len(numeric_features) >= 3 and result.selected_kernel == "linear" and len(result.class_labels) == 2:
        geometry_modes.append("3D plane view")

    geometry_mode = st.radio("Geometry mode", geometry_modes, horizontal=True, key="geometry_mode")

    if geometry_mode == "2D decision view":
        geometry_columns = st.columns(2)
        with geometry_columns[0]:
            x_column = st.selectbox("X-axis column", numeric_features, key="geometry_x")
        with geometry_columns[1]:
            y_options = [column for column in numeric_features if column != x_column]
            y_column = st.selectbox("Y-axis column", y_options, key="geometry_y")

        if len(result.feature_columns) == 2 and set(result.feature_columns) == {x_column, y_column}:
            st.caption("This is the actual decision boundary for the trained two-feature model.")
        else:
            st.caption("This is a two-feature slice using the selected kernel and tuned parameters. It is an intuition view, not the full high-dimensional boundary.")

        st.pyplot(
            plot_decision_surface_figure(
                frame=frame,
                target_column=result.target_column,
                feature_columns=[x_column, y_column],
                kernel=result.selected_kernel,
                best_params=result.selected_params,
            )
        )
        return

    geometry_columns = st.columns(3)
    with geometry_columns[0]:
        x_column = st.selectbox("X-axis column", numeric_features, key="plane_x")
    with geometry_columns[1]:
        y_options = [column for column in numeric_features if column != x_column]
        y_column = st.selectbox("Y-axis column", y_options, key="plane_y")
    with geometry_columns[2]:
        z_options = [column for column in numeric_features if column not in {x_column, y_column}]
        z_column = st.selectbox("Z-axis column", z_options, key="plane_z")

    st.caption("A true plane only exists in this view because the kernel is linear, the target is binary, and the plot is limited to three numeric columns.")
    plane_figure = plot_linear_plane_figure(
        frame=frame,
        target_column=result.target_column,
        feature_columns=[x_column, y_column, z_column],
        best_params=result.selected_params,
    )
    if plane_figure is None:
        ui_shell.render_state_panel(
            "warning",
            "Plane view could not be stabilized",
            "The current three-column combination did not produce a reliable plane fit for visualization.",
            detail="Try a different set of three numeric columns or fall back to the 2D decision view.",
        )
        return
    st.pyplot(plane_figure)


def render_itemset_tab(frame: pd.DataFrame, source_name: str) -> None:
    ui_shell.render_section_intro(
        "Pattern Mining",
        "Find the combinations of values that keep showing up together",
        "Itemset mining is useful when you care about co-occurrence rather than prediction. It works especially well after discretizing numeric fields into interpretable buckets.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Pick transaction columns", "Choose the fields that should contribute items into each row-level basket."),
            ("Set the support floor", "Tune the minimum support so the output is neither too sparse nor too noisy."),
            ("Review repeat combinations", "Use the chart and table together to spot stable co-occurrence patterns worth exporting."),
        ],
    )
    ui_shell.render_method_box(
        "Itemset math and method",
        "The itemset engine converts selected columns into transactions, discretizes numeric fields into buckets, and then mines frequent combinations with Apriori pruning.",
        [
            (
                "Transaction",
                "T_i = {column=value, ...}",
                "Each row becomes a set of items after categorical encoding or numeric binning.",
            ),
            (
                "Support",
                "support(X) = count(T_i : X ⊆ T_i) / N",
                "An itemset is frequent when it appears in a sufficient fraction of the transactions.",
            ),
            (
                "Apriori rule",
                "if Y is infrequent, every superset of Y is infrequent",
                "This lets the miner prune large parts of the search space early.",
            ),
            (
                "Scientific notation",
                "2.50e-01 = 0.25",
                "Support values can be shown in scientific notation when you want consistent numeric formatting.",
            ),
        ],
    )
    columns = list(frame.columns)
    default_columns = columns[: min(6, len(columns))]
    with st.form("itemset_form"):
        item_columns = st.multiselect("Columns to mine", columns, default=default_columns, key="item_columns")
        include_target = st.checkbox("Include the target column as an item", value=False, key="include_target")
        target_column = st.session_state.get("target_column")
        min_support = st.slider("Minimum support", min_value=0.05, max_value=0.80, value=0.20, step=0.01, key="item_support")
        run_itemsets = st.form_submit_button("Run itemset mining", use_container_width=True)

    current_signature = build_itemset_run_signature(source_name, item_columns, include_target, target_column, min_support)

    if run_itemsets:
        try:
            with st.status("Running itemset mining", expanded=True) as status:
                try:
                    status.write("Encoding selected columns into transaction items.")
                    itemset_result = itemsets_to_frame(
                        mine_itemsets_from_frame(
                            frame=frame,
                            columns=item_columns,
                            dataset_name="Uploaded data",
                            min_support=min_support,
                            target_column=target_column if include_target and target_column in frame.columns else None,
                        )
                    )
                    status.write("Mining frequent combinations and formatting the result table.")
                    status.update(label="Itemset mining complete", state="complete", expanded=False)
                except ValueError:
                    status.update(label="Itemset mining failed", state="error", expanded=True)
                    raise
            st.session_state["itemset_result"] = itemset_result
            st.session_state["itemset_signature"] = current_signature
            st.session_state.pop("itemset_error", None)
        except ValueError as exc:
            st.session_state["itemset_error"] = str(exc)
            st.session_state.pop("itemset_result", None)
            st.session_state.pop("itemset_signature", None)

    if "itemset_error" in st.session_state:
        ui_shell.render_state_panel(
            "error",
            "Itemset mining could not run",
            "The selected transaction columns could not be encoded into a valid mining run.",
            detail=st.session_state["itemset_error"],
        )
        return

    itemset_result = st.session_state.get("itemset_result")
    if itemset_result is None:
        ui_shell.render_state_panel(
            "info",
            "Itemset mining not started yet",
            "Choose the columns to turn into transactions, decide whether to include the target, and run mining to surface frequent combinations.",
        )
        return
    if itemset_result.empty:
        ui_shell.render_state_panel(
            "warning",
            "No frequent itemsets found",
            "Nothing cleared the current support threshold, so the result set is empty.",
            detail="Lower the minimum support or widen the mined columns to reveal more recurring combinations.",
        )
        return

    if st.session_state.get("itemset_signature") != current_signature:
        ui_shell.render_state_panel(
            "info",
            "Displaying the last completed itemset run",
            "The transaction controls changed after the previous mining pass finished, so the outputs below still reflect the earlier run.",
            detail="Run itemset mining again to refresh the table and chart for the current support threshold and column set.",
        )

    ui_shell.render_stat_grid(
        [
            ("Patterns", f"{len(itemset_result):,}", "Frequent itemsets returned after support filtering."),
            ("Max length", f"{int(itemset_result['length'].max()):,}", "Largest combination size discovered."),
            ("Top support", f"{itemset_result['support'].max():.3f}", "Strongest co-occurrence level in the result set."),
            ("Columns mined", f"{len(item_columns):,}", "Source columns used to construct transactions."),
        ]
    )
    ui_shell.render_state_panel(
        "success",
        "Itemset results ready",
        "Frequent combinations are available for review and export.",
        detail=f"Patterns returned: {len(itemset_result):,} | Top support: {itemset_result['support'].max():.3f}",
    )
    st.pyplot(plot_pattern_figure(itemset_result.sort_values(["support", "length"], ascending=[False, False]), "itemset", "Top Frequent Itemsets", "flare"))
    st.dataframe(
        build_display_frame(itemset_result, scientific_columns={"support"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download itemset results CSV",
        dataframe_bytes(itemset_result),
        file_name="itemset_results.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )


def render_episode_tab(frame: pd.DataFrame, source_name: str) -> None:
    ui_shell.render_section_intro(
        "Sequence Lab",
        "Mine ordered event behavior instead of static combinations",
        "Episodes capture order. Use a single delimited sequence column or map a row across ordered event columns to reveal repeated paths through a process.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Choose the sequence source", "Start with either a delimited path column or a fixed left-to-right set of event columns."),
            ("Tune support and span", "Control how often an ordered pattern must appear and how far apart the matched events may drift."),
            ("Review ordered paths", "Inspect the most stable event sequences and export the result set when the threshold is right."),
        ],
    )
    ui_shell.render_method_box(
        "Episode math and method",
        "Episode mining looks for ordered subsequences that repeat across many sequences while respecting the maximum span you choose.",
        [
            (
                "Episode",
                "E = (e_1, e_2, ..., e_k)",
                "An episode is an ordered pattern of events rather than an unordered set.",
            ),
            (
                "Support",
                "support(E) = count(S_i containing E in order) / M",
                "A sequence contributes when the events appear in the specified order.",
            ),
            (
                "Span rule",
                "max(index_last - index_first) ≤ max_span",
                "This keeps the matched events close enough together to still be meaningful.",
            ),
            (
                "Scientific notation",
                "3.00e-01 = 0.30",
                "Support thresholds can be read in either decimal or scientific form without changing the result.",
            ),
        ],
    )
    with st.form("episode_form"):
        mode = st.radio("Sequence source", ["Delimited sequence column", "Ordered event columns"], horizontal=True, key="episode_mode")
        max_length = st.slider("Maximum episode length", min_value=2, max_value=4, value=3, step=1, key="episode_length")
        max_span = st.slider("Maximum span", min_value=2, max_value=8, value=4, step=1, key="episode_span")
        min_support = st.slider("Minimum support", min_value=0.05, max_value=0.90, value=0.30, step=0.01, key="episode_support")

        if mode == "Delimited sequence column":
            sequence_column = st.selectbox("Sequence column", frame.columns, key="sequence_column")
            separator = st.text_input("Event separator", value=",", key="sequence_separator")
            run_episodes = st.form_submit_button("Run episode mining", use_container_width=True)
        else:
            ordered_columns = st.multiselect("Ordered event columns", frame.columns, key="episode_columns")
            run_episodes = st.form_submit_button("Run episode mining", use_container_width=True)

    current_signature = build_episode_run_signature(
        source_name,
        mode,
        sequence_column if mode == "Delimited sequence column" else None,
        separator if mode == "Delimited sequence column" else None,
        ordered_columns if mode == "Ordered event columns" else None,
        max_length,
        max_span,
        min_support,
    )

    if mode == "Delimited sequence column":
        if run_episodes:
            try:
                with st.status("Running episode mining", expanded=True) as status:
                    try:
                        status.write("Parsing the selected sequence column into ordered events.")
                        dataset = build_episode_dataset_from_sequence_column(
                            frame=frame,
                            sequence_column=sequence_column,
                            name=source_name,
                            min_support=min_support,
                            max_span=max_span,
                            separator=separator,
                        )
                        episode_result = episodes_to_frame(mine_episodes(dataset, max_length=max_length))
                        status.write("Mining ordered subsequences and formatting the output.")
                        status.update(label="Episode mining complete", state="complete", expanded=False)
                    except ValueError:
                        status.update(label="Episode mining failed", state="error", expanded=True)
                        raise
                st.session_state["episode_result"] = episode_result
                st.session_state["episode_signature"] = current_signature
                st.session_state.pop("episode_error", None)
            except ValueError as exc:
                st.session_state["episode_error"] = str(exc)
                st.session_state.pop("episode_result", None)
                st.session_state.pop("episode_signature", None)
    else:
        if run_episodes:
            try:
                with st.status("Running episode mining", expanded=True) as status:
                    try:
                        status.write("Building ordered sequences from the selected event columns.")
                        dataset = build_episode_dataset_from_event_columns(
                            frame=frame,
                            event_columns=ordered_columns,
                            name=source_name,
                            min_support=min_support,
                            max_span=max_span,
                        )
                        episode_result = episodes_to_frame(mine_episodes(dataset, max_length=max_length))
                        status.write("Mining ordered subsequences and formatting the output.")
                        status.update(label="Episode mining complete", state="complete", expanded=False)
                    except ValueError:
                        status.update(label="Episode mining failed", state="error", expanded=True)
                        raise
                st.session_state["episode_result"] = episode_result
                st.session_state["episode_signature"] = current_signature
                st.session_state.pop("episode_error", None)
            except ValueError as exc:
                st.session_state["episode_error"] = str(exc)
                st.session_state.pop("episode_result", None)
                st.session_state.pop("episode_signature", None)

    if "episode_error" in st.session_state:
        ui_shell.render_state_panel(
            "error",
            "Episode mining could not run",
            "The sequence input could not be converted into a valid ordered-event dataset.",
            detail=st.session_state["episode_error"],
        )
        return

    episode_result = st.session_state.get("episode_result")
    if episode_result is None:
        ui_shell.render_state_panel(
            "info",
            "Episode mining not started yet",
            "Point the page at a sequence column or a set of ordered event columns, then run the mining pass to surface repeated event paths.",
        )
        return
    if episode_result.empty:
        ui_shell.render_state_panel(
            "warning",
            "No frequent episodes found",
            "No ordered patterns cleared the current support threshold.",
            detail="Lower the minimum support, relax the span, or shorten the maximum episode length to capture more repeated sequences.",
        )
        return

    if st.session_state.get("episode_signature") != current_signature:
        ui_shell.render_state_panel(
            "info",
            "Displaying the last completed episode run",
            "The sequence controls changed after the previous mining pass finished, so the outputs below still reflect the earlier run.",
            detail="Run episode mining again to refresh the results for the current source mode, support threshold, and span settings.",
        )

    ui_shell.render_stat_grid(
        [
            ("Episodes", f"{len(episode_result):,}", "Ordered patterns that cleared the support threshold."),
            ("Max length", f"{int(episode_result['length'].max()):,}", "Longest episode discovered."),
            ("Top support", f"{episode_result['support'].max():.3f}", "Most stable ordered pattern in the result set."),
            ("Mode", "Delimited" if mode == "Delimited sequence column" else "Ordered columns", "How the sequences were constructed."),
        ]
    )
    ui_shell.render_state_panel(
        "success",
        "Episode results ready",
        "Ordered event patterns are now staged for review, visualization, and export.",
        detail=f"Episodes returned: {len(episode_result):,} | Top support: {episode_result['support'].max():.3f}",
    )
    st.pyplot(plot_pattern_figure(episode_result.sort_values(["support", "length"], ascending=[False, False]), "episode", "Top Serial Episodes", "mako"))
    st.dataframe(
        build_display_frame(episode_result, scientific_columns={"support"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download episode results CSV",
        dataframe_bytes(episode_result),
        file_name="episode_results.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="SVM Data Studio", layout="wide")
    apply_style()
    ui_shell.inject_app_css()

    with st.sidebar:
        st.markdown("### Control Deck")
        st.markdown("Tune the input source here, then move through the workspace pages to model, mine, and visualize.")

    frame, source_name = load_data_source()
    if frame is None or source_name is None:
        ui_shell.render_state_panel(
            "info",
            "No dataset staged",
            "Upload a CSV from the sidebar or load one of the built-in demo datasets to unlock the workbench pages.",
            detail="Recommended start: choose a demo dataset first to see the full workflow, then switch to your own CSV when the flow looks right.",
        )
        return

    if st.session_state.get("_active_source_name") != source_name:
        clear_computed_results()

    sync_workspace_state(frame, source_name)

    ui_shell.render_hero(source_name, frame)
    ui_shell.render_step_strip(
        "Quick start",
        [
            ("Load data", "Begin with a built-in demo for a clean walkthrough or upload your own CSV from the sidebar."),
            ("Read the dataset", "Use Data Atlas to validate structure, missingness, and column behavior before modeling."),
            ("Train the boundary", "Move into SVM Lab to compare kernels, inspect accuracy, and visualize the separator."),
            ("Mine patterns", "Finish with Itemsets or Episodes when you want combinations and ordered event behavior from the same data."),
        ],
    )

    def data_page() -> None:
        render_data_tab(frame)

    def svm_page() -> None:
        render_svm_tab(frame, source_name)

    def itemset_page() -> None:
        render_itemset_tab(frame, source_name)

    def episode_page() -> None:
        render_episode_tab(frame, source_name)

    page = st.navigation(
        [
            st.Page(data_page, title="Data Atlas"),
            st.Page(svm_page, title="SVM Lab"),
            st.Page(itemset_page, title="Itemsets"),
            st.Page(episode_page, title="Episodes"),
        ]
    )
    page.run()


if __name__ == "__main__":
    main()
