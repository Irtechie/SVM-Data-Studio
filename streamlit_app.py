from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm_studio.custom_analysis import (
    evaluate_column_set,
    fit_custom_svm_estimator,
    prepare_custom_classification_data,
    run_custom_svm_analysis,
)
from svm_studio.datasets import load_demo_frame_by_title, load_demo_sources
from svm_studio.llm_advisor import ColumnAdvice, advise_columns
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


# ── Plotly interactive charts (ui-cold-05) ─────────────────────────────────

_PLOTLY_PALETTE = {"linear": "#18c5d8", "rbf": "#ff7a18", "poly": "#103a53"}
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="transparent",
    plot_bgcolor="#f8fbfd",
    font=dict(family="Sora, Segoe UI, sans-serif", color="#091722"),
    margin=dict(l=10, r=10, t=44, b=10),
)


def plot_kernel_results_plotly(kernel_results: pd.DataFrame) -> go.Figure:
    colors = [_PLOTLY_PALETTE.get(k, "#7ef2c8") for k in kernel_results["kernel"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=kernel_results["kernel"].str.upper(),
            y=kernel_results["test_accuracy"],
            marker_color=colors,
            error_y=dict(type="data", array=kernel_results["cv_std"].tolist(), visible=True, thickness=1.5),
            hovertemplate="Kernel: %{x}<br>Test acc: %{y:.4f}<br><extra></extra>",
        )
    )
    fig.update_layout(
        title="Kernel Comparison",
        xaxis_title="Kernel",
        yaxis_title="Test accuracy",
        yaxis_range=[0, 1.05],
        showlegend=False,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_feature_importance_plotly(importance_frame: pd.DataFrame) -> go.Figure:
    top = importance_frame.head(12).sort_values("importance_mean")
    cmap = ["#103a53", "#18c5d8", "#7ef2c8", "#ffd166", "#ff7a18"]
    colors = [cmap[i % len(cmap)] for i in range(len(top))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["importance_mean"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=top["importance_std"].tolist(), visible=True, thickness=1.5),
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Permutation importance",
        yaxis_title="",
        showlegend=False,
        height=max(280, len(top) * 30 + 80),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_pca_projection_plotly(
    frame: pd.DataFrame,
    result: object,  # CustomSvmResult
) -> go.Figure:
    """PCA scatter of the preprocessed features, coloured by class and support vectors marked."""
    X, y = prepare_custom_classification_data(frame, result.target_column, result.feature_columns)
    X_transformed = result.selected_estimator.named_steps["preprocessor"].transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_transformed)
    ev1, ev2 = pca.explained_variance_ratio_[:2]

    sv_indices = result.selected_estimator.named_steps["svc"].support_
    sv_coords = coords[sv_indices]

    palette = ["#18c5d8", "#ff7a18", "#7ef2c8", "#295fe7", "#ffd166"]
    fig = go.Figure()

    for i, label in enumerate(result.class_labels):
        mask = y.values == label
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=str(label),
                marker=dict(color=palette[i % len(palette)], size=7, opacity=0.82),
                hovertemplate=f"Class: {label}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=sv_coords[:, 0],
            y=sv_coords[:, 1],
            mode="markers",
            name="Support vectors",
            marker=dict(symbol="circle-open", color="black", size=14, line=dict(width=2)),
            hovertemplate="Support vector<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="PCA Projection with Support Vectors",
        xaxis_title=f"PC1 ({ev1:.1%} variance)",
        yaxis_title=f"PC2 ({ev2:.1%} variance)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_PLOTLY_LAYOUT,
    )
    return fig


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


@st.cache_data(ttl=30, show_spinner=False)
def _probe_models(base_url: str) -> list[str]:
    """Query the endpoint for available model IDs (cached 30 s)."""
    from svm_studio.llm_advisor import fetch_available_models
    return fetch_available_models(base_url)


def _llm_url_and_model_widgets(
    url_key: str,
    model_key: str,
    url_label: str = "LLM base URL",
    default_url: str = "",
    inside_form: bool = True,
) -> tuple[str, str]:
    """Render a URL text-input and auto-derive the model from the endpoint.

    - 1 model available  → selected silently, shown as a read-only info chip
    - Multiple models    → compact selectbox to pick one
    - Unreachable / blank URL → small fallback text input

    Returns *(base_url, model)*.
    """
    base_url = st.text_input(
        url_label,
        value=default_url,
        placeholder="http://192.168.1.203:8000",
        key=url_key,
    )

    model = ""
    if base_url and base_url.startswith("http"):
        models = _probe_models(base_url)
        if len(models) == 1:
            model = models[0]
            st.caption(f"Model auto-detected: **{model}**")
            # Store in session so downstream reads via session_state still work
            st.session_state[model_key] = model
        elif len(models) > 1:
            stored = st.session_state.get(model_key, models[0])
            default_idx = models.index(stored) if stored in models else 0
            model = st.selectbox("Model", models, index=default_idx, key=model_key)
        else:
            # Unreachable — tiny fallback text input
            model = st.text_input(
                "Model name (endpoint unreachable — enter manually)",
                value=st.session_state.get(model_key, os.environ.get("LLM_MODEL", "")),
                placeholder="e.g. gemma-4-31B-it-Q4_K_M.gguf",
                key=model_key,
            )
    else:
        # No URL yet — hide the model field entirely, default from env
        model = os.environ.get("LLM_MODEL", "")

    return base_url, model


def load_data_source() -> tuple[pd.DataFrame | None, str | None]:
    with st.sidebar:
        st.header("Data Source")
        source_mode = st.radio("Choose input", ["Built-in demo", "Upload CSV"], index=0, key="source_mode")

        if source_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            separator = st.text_input("CSV separator", key="csv_separator")
            if uploaded_file is None:
                return None, None
            frame = read_uploaded_csv(uploaded_file.getvalue(), separator)
            return frame, uploaded_file.name

        demo_sources = load_demo_sources()

        # Group for display: classification first, then SVM demos, then mining
        _group_order = {"classification": 0, "svm": 1, "itemset": 2, "episode": 3}
        demo_sources_sorted = sorted(demo_sources, key=lambda s: _group_order.get(s.group, 9))
        demo_options = [source.title for source in demo_sources_sorted]

        # Group labels for the selectbox
        _group_labels = {
            "classification": "── Classification datasets ──",
            "svm": "── SVM boundary demos ──",
            "itemset": "── Itemset mining ──",
            "episode": "── Episode mining ──",
        }
        current_group = None
        grouped_options = []
        for src in demo_sources_sorted:
            grp_label = _group_labels.get(src.group, f"── {src.group} ──")
            if src.group != current_group:
                grouped_options.append(grp_label)
                current_group = src.group
            grouped_options.append(src.title)

        all_options = grouped_options
        # Filter out group headers for the actual value
        valid_titles = set(demo_options)

        # Default index: first actual dataset (skip the leading group header)
        _prev = st.session_state.get("demo_name")
        if _prev and _prev in valid_titles:
            _default_idx = all_options.index(_prev)
        else:
            _default_idx = next(i for i, o in enumerate(all_options) if o in valid_titles)

        selected = st.selectbox(
            "Choose a demo dataset",
            all_options,
            index=_default_idx,
            key="demo_name_raw",
            format_func=lambda x: x,
        )
        # If user picked a header label, use previous valid title or first valid
        if selected not in valid_titles:
            selected = st.session_state.get("demo_name") or demo_options[0]
        else:
            st.session_state["demo_name"] = selected

        demo_name = selected
        selected_source = next(source for source in demo_sources if source.title == demo_name)
        rows = getattr(selected_source, 'rows', None)
        row_str = f" · {rows:,} rows" if rows else ""
        st.caption(f"{selected_source.group.upper()}{row_str} — {selected_source.description}")
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
        st.dataframe(frame.head(50), width='stretch')

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
        st.dataframe(schema_frame, width='stretch', hide_index=True)

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


def _render_svm_prediction_formula(result) -> None:
    """Show the active kernel's decision function annotated term-by-term."""
    kernel = result.selected_kernel
    params = result.selected_params or {}
    C = params.get("C", 1.0)
    gamma = params.get("gamma", "scale")
    degree = int(params.get("degree", 3))

    gamma_str = f"{gamma:.2e}" if isinstance(gamma, float) else str(gamma)

    if kernel == "linear":
        segments: list[tuple[str, str | None]] = [
            ("f(x)", "prediction"),
            (" = sign(", None),
            ("w", "weight vector"),
            (" \u00b7 ", None),
            ("x", "feature input"),
            (" + ", None),
            ("b", "bias"),
            (")", None),
        ]
        note = f"C = {C:.2e}  |  kernel = linear  |  sign(\u00b7) outputs the class label"
    elif kernel == "rbf":
        segments = [
            ("f(x)", "prediction"),
            (" = sign(", None),
            ("\u03a3 \u03b1\u1d62 y\u1d62", "support vector weights"),
            (" \u00b7 ", None),
            ("exp(\u2212\u03b3\u2016x\u2212x\u1d62\u2016\u00b2)", "rbf kernel"),
            (" + ", None),
            ("b", "bias"),
            (")", None),
        ]
        note = (
            f"C = {C:.2e}  |  \u03b3 = {gamma_str}  |  kernel = rbf  |  "
            "distance shrinks exponentially \u2192 far points contribute less"
        )
    else:  # poly
        segments = [
            ("f(x)", "prediction"),
            (" = sign(", None),
            ("\u03a3 \u03b1\u1d62 y\u1d62", "support vector weights"),
            (" \u00b7 ", None),
            (f"(\u03b3 x\u00b7x\u1d62 + r)^{degree}", "poly kernel"),
            (" + ", None),
            ("b", "bias"),
            (")", None),
        ]
        note = (
            f"C = {C:.2e}  |  \u03b3 = {gamma_str}  |  degree = {degree}  |  kernel = poly  |  "
            "dot product raised to a power \u2192 curved boundary"
        )

    ui_shell.render_annotated_formula(
        "Prediction formula \u2014 active model",
        segments,
        note=note,
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
            run_analysis = st.form_submit_button("Run SVM analysis", width='stretch')

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

    _render_svm_prediction_formula(result)

    plots = st.columns(2)
    with plots[0]:
        st.plotly_chart(plot_kernel_results_plotly(result.kernel_results), width='stretch')
    with plots[1]:
        st.plotly_chart(plot_feature_importance_plotly(result.feature_importance), width='stretch')

    confusion_column, report_column = st.columns([1.1, 0.9])
    with confusion_column:
        st.pyplot(plot_confusion_figure(result.confusion, result.class_labels))
    with report_column:
        st.subheader("Classification report")
        st.dataframe(
            build_display_frame(result.classification_report, scientific_columns={"support"}),
            width='stretch',
            hide_index=True,
        )

    st.subheader("Kernel results")
    st.dataframe(
        build_display_frame(result.kernel_results, scientific_columns={"cv_std"}),
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "Download kernel results CSV",
        dataframe_bytes(result.kernel_results),
        file_name="kernel_results.csv",
        mime="text/csv",
        on_click="ignore",
        width='stretch',
    )

    st.subheader("Feature importance")
    st.dataframe(
        build_display_frame(result.feature_importance, scientific_columns={"importance_mean", "importance_std"}),
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "Download feature importance CSV",
        dataframe_bytes(result.feature_importance),
        file_name="feature_importance.csv",
        mime="text/csv",
        on_click="ignore",
        width='stretch',
    )

    svm_summary = (
        f"Technique: SVM Kernel Comparison\n"
        f"Target: {result.target_column}\n"
        f"Features ({len(result.feature_columns)}): {', '.join(result.feature_columns)}\n"
        f"Classes: {', '.join(str(c) for c in result.class_labels)}\n"
        f"Selected kernel: {result.selected_kernel}\n"
        f"Parameters: {format_parameter_mapping(result.selected_params)}\n"
        f"Test accuracy: {result.test_accuracy:.4f}\n"
        f"Macro F1: {result.macro_f1:.4f}\n"
        f"Support vectors: {result.support_vector_count}\n"
        f"Kernel results:\n{result.kernel_results.to_string(index=False)}\n"
        f"Feature importance (top 10):\n{result.feature_importance.head(10).to_string(index=False)}"
    )
    _llm_explain_button("svm_explain", "SVM Kernel Comparison", svm_summary, frame)

    # ── "Why did this fail?" diagnostic mode ──────────────────────────────
    _POOR_ACCURACY_THRESHOLD = 0.75
    if result.test_accuracy < _POOR_ACCURACY_THRESHOLD:
        st.divider()
        st.markdown(
            f"**Accuracy {result.test_accuracy:.3f} is below {_POOR_ACCURACY_THRESHOLD:.0%} — "
            "SVMs fail in characteristic ways. Ask the advisor to diagnose this run.**"
        )
        if st.button("Why did this fail? (LLM diagnosis)", key="svm_diagnose", width='stretch'):
            from svm_studio.llm_advisor import diagnose_bad_result_stream, _build_data_context, fetch_available_models
            base_url = (
                st.session_state.get("advisor_base_url_input")
                or st.session_state.get("chat_base_url")
                or os.environ.get("LLM_BASE_URL")
                or None
            )
            api_key = st.session_state.get("advisor_api_key_input") or st.session_state.get("chat_api_key") or None
            model = st.session_state.get("advisor_model_input") or st.session_state.get("chat_model") or None
            if not model and base_url:
                models = fetch_available_models(base_url)
                model = models[0] if models else None
            ctx = _build_data_context(frame, max_cat_samples=3, sample_rows=0, include_correlations=False)[:600] if frame is not None else None
            st.session_state.pop("svm_diagnosis", None)
            st.session_state.pop("svm_diagnosis_error", None)
            _think = st.empty()
            _think.info("Diagnosing failure — waiting for first token…")
            def _diag_stream():
                try:
                    yield from diagnose_bad_result_stream(svm_summary, ctx, api_key=api_key, model=model, base_url=base_url)
                except Exception as _e:
                    st.session_state["svm_diagnosis_error"] = str(_e)
            try:
                text = st.write_stream(_diag_stream())
                _think.empty()
                if text:
                    st.session_state["svm_diagnosis"] = text
                else:
                    st.session_state["svm_diagnosis_error"] = st.session_state.pop("svm_diagnosis_error", "Model returned empty response.")
            except Exception as exc:
                _think.empty()
                st.session_state["svm_diagnosis_error"] = f"{type(exc).__name__}: {exc}"

        diag_err = st.session_state.get("svm_diagnosis_error")
        if diag_err:
            st.error("Diagnosis failed")
            st.code(diag_err)
        diag_text = st.session_state.get("svm_diagnosis")
        if diag_text:
            with st.expander("LLM Diagnosis", expanded=True):
                st.markdown(diag_text)
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
    geometry_modes.append("PCA projection")

    geometry_mode = st.radio("Geometry mode", geometry_modes, horizontal=True, key="geometry_mode")

    if geometry_mode == "PCA projection":
        st.caption(
            "PCA collapses all trained features into two principal components so you can see whether the classes "
            "separate cleanly in the learned feature space.  Support vectors are shown as open circles."
        )
        st.plotly_chart(plot_pca_projection_plotly(frame, result), width='stretch')
        return

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


# ── LLM Auto-Advisor page (llm-feat-04 / llm-feat-05) ─────────────────────


def render_advisor_tab(frame: pd.DataFrame, source_name: str) -> None:
    ui_shell.render_section_intro(
        "LLM Auto-Advisor",
        "Let an AI recommend the best columns, then grade it mathematically",
        "The advisor reads your dataset schema and picks the most promising target and feature columns for an SVM. "
        "It scores itself with a 50 / 50 hold-out split and 10-fold cross-validation so you can judge the recommendation objectively.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Get a recommendation", "Press 'Get advice' and let the system pick the target and feature columns."),
            ("Review the rationale", "Read why the advisor chose those columns before you commit to grading them."),
            ("Grade the advice", "Run the SVM with the recommended columns to measure hold-out accuracy and CV stability."),
        ],
    )
    ui_shell.render_method_box(
        "How the grading works",
        "Two independent evaluations are run on the same column set, then averaged into one grade.",
        [
            (
                "50 / 50 hold-out",
                "Train on half, score on the other half",
                "Directly measures how well an SVM trained on the recommended columns predicts unseen labels.",
            ),
            (
                "10-fold CV",
                "StratifiedKFold(n_splits=10)",
                "Splits the whole dataset 10 ways using the best kernel found during the hold-out run.  "
                "Mean ± std tells you whether the score is stable across different random partitions.",
            ),
            (
                "Grade",
                "grade = (hold-out acc + CV mean acc) / 2",
                "A single number between 0 and 1.  Closer to 1 means the LLM chose columns that let the SVM separate classes well.",
            ),
        ],
    )

    with st.form("advisor_form"):
        backend = st.radio(
            "LLM backend",
            ["Local / Remote LLM", "OpenAI", "Heuristic (no LLM)"],
            horizontal=True,
            key="advisor_backend",
        )
        if backend == "OpenAI":
            api_key = st.text_input(
                "OpenAI API key (leave blank to use OPENAI_API_KEY env var)",
                type="password",
                key="advisor_api_key_input",
            )
            advisor_base_url: str | None = None
            advisor_model: str | None = None
        elif backend == "Local / Remote LLM":
            api_key = ""
            advisor_base_url, advisor_model = _llm_url_and_model_widgets(
                url_key="advisor_base_url_input",
                model_key="advisor_model_input",
                url_label="LLM base URL  (e.g. http://host:8000)",
                default_url=os.environ.get("LLM_BASE_URL", ""),
            )
        else:  # Heuristic
            api_key = ""
            advisor_base_url = None
            advisor_model = None
        advisor_kernels = st.multiselect(
            "Kernels to test during grading",
            ["linear", "rbf", "poly"],
            default=["linear", "rbf"],
            key="advisor_kernels",
        )
        get_advice = st.form_submit_button("Get column advice", width='stretch')

    if get_advice:
        if not advisor_kernels:
            ui_shell.render_state_panel(
                "warning",
                "No kernels selected",
                "Choose at least one kernel to test before requesting advice.",
            )
        else:
            with st.status("Analysing dataset schema …", expanded=True) as status:
                try:
                    advice = advise_columns(
                        frame,
                        api_key=api_key or None,
                        base_url=advisor_base_url or None,
                        model=advisor_model or None,
                    )
                    st.session_state["advisor_advice"] = advice
                    st.session_state.pop("advisor_grade", None)
                    st.session_state.pop("advisor_error", None)
                    status.update(label="Column advice ready", state="complete", expanded=False)
                except Exception as exc:
                    status.update(label="Advisor failed", state="error", expanded=True)
                    st.session_state["advisor_error"] = str(exc)
                    st.session_state.pop("advisor_advice", None)

    if "advisor_error" in st.session_state:
        ui_shell.render_state_panel(
            "error",
            "Column advisor could not run",
            "The advisor encountered an error.",
            detail=st.session_state["advisor_error"],
        )

    advice: ColumnAdvice | None = st.session_state.get("advisor_advice")
    if advice is None:
        ui_shell.render_state_panel(
            "info",
            "No advice generated yet",
            "Press 'Get column advice' to let the system recommend a target and feature set for this dataset.",
        )
        return

    source_label = f"via {advice.model_used}" if advice.source == "llm" else "heuristic"
    ui_shell.render_state_panel(
        "success",
        f"Column recommendation ready ({source_label})",
        advice.rationale,
        detail=(
            f"Target: {advice.target_column}  |  "
            f"Features ({len(advice.feature_columns)}): {', '.join(advice.feature_columns[:6])}"
            + (" …" if len(advice.feature_columns) > 6 else "")
        ),
    )
    ui_shell.render_stat_grid(
        [
            ("Target column", advice.target_column, "The column the advisor chose as the classification label."),
            ("Feature columns", str(len(advice.feature_columns)), "Number of columns the advisor selected as predictors."),
            ("Source", advice.source.upper(), "Whether the recommendation came from an LLM or the built-in heuristic."),
            ("Unique target values", str(frame[advice.target_column].nunique(dropna=True)), "Distinct classes in the recommended target."),
        ]
    )

    # ── candidate review ───────────────────────────────────────────────────
    # deduplicate candidates while preserving LLM rank order
    _seen: list[list[str]] = []
    for _c in (advice.candidates if advice.candidates else [advice.feature_columns]):
        if _c not in _seen:
            _seen.append(_c)

    st.markdown("---")
    st.markdown(
        f"#### Review proposed feature combinations\n"
        f"The advisor proposed **{len(_seen)} distinct feature set(s)** for target "
        f"**`{advice.target_column}`**. "
        f"Tick the ones you want to grade, or hit **Yolo** to run them all."
    )

    # Per-candidate checkboxes — stored in session_state by key so toggles survive reruns
    for _i, _cset in enumerate(_seen):
        _key = f"advisor_cand_{_i}"
        if _key not in st.session_state:
            st.session_state[_key] = True  # default: all selected
        _label = f"**Candidate {_i + 1}** — {len(_cset)} features: `{', '.join(_cset[:5])}{'…' if len(_cset) > 5 else ''}`"
        with st.expander(_label, expanded=(_i == 0)):
            st.checkbox("Include in grading run", key=_key)
            st.caption(f"Columns: {', '.join(_cset)}")
            if advice.source == "llm" and _i < len(advice.candidate_reasoning):
                _r = advice.candidate_reasoning[_i]
                if _r:
                    st.caption(f"Reasoning: {_r}")

    _selected_sets = [_seen[_i] for _i in range(len(_seen)) if st.session_state.get(f"advisor_cand_{_i}", True)]

    col_apply, col_selected, col_yolo = st.columns([1, 1, 1])
    with col_apply:
        if st.button("Apply best to SVM Lab", width='stretch'):
            st.session_state["target_column"] = advice.target_column
            feature_options = [c for c in frame.columns if c != advice.target_column]
            valid_features = [f for f in advice.feature_columns if f in feature_options]
            st.session_state["feature_columns"] = valid_features or feature_options
            st.session_state.pop("custom_svm_result", None)
            st.session_state.pop("custom_svm_signature", None)
            ui_shell.render_state_panel(
                "success",
                "Applied to SVM Lab",
                "Switch to the SVM Lab page — the target and feature columns have been updated to match the advisor's recommendation.",
            )

    def _run_grading(sets_to_grade: list[list[str]]) -> None:
        kernels_to_test = st.session_state.get("advisor_kernels", ["linear", "rbf"])
        with st.status(f"Grading {len(sets_to_grade)} feature set(s) …", expanded=True) as status:
            try:
                best_grade = None
                best_features = sets_to_grade[0]
                for idx, feature_set in enumerate(sets_to_grade, 1):
                    status.write(
                        f"Candidate {idx}/{len(sets_to_grade)}: "
                        f"{len(feature_set)} features — 50/50 hold-out + 10-fold CV …"
                    )
                    result = evaluate_column_set(
                        frame=frame,
                        target_column=advice.target_column,
                        feature_columns=feature_set,
                        kernels=kernels_to_test,
                        n_cv_folds=10,
                    )
                    if best_grade is None or result.grade > best_grade.grade:
                        best_grade = result
                        best_features = feature_set
                advice.feature_columns = best_features
                st.session_state["advisor_advice"] = advice
                st.session_state["advisor_grade"] = best_grade
                st.session_state.pop("advisor_grade_error", None)
                status.update(
                    label=f"Best of {len(sets_to_grade)} — grade {best_grade.grade:.3f}",
                    state="complete",
                    expanded=False,
                )
            except (ValueError, Exception) as exc:
                status.update(label="Grading failed", state="error", expanded=True)
                st.session_state["advisor_grade_error"] = str(exc)

    with col_selected:
        _n_sel = len(_selected_sets)
        if st.button(
            f"Grade selected ({_n_sel})",
            width='stretch',
            disabled=_n_sel == 0,
        ):
            _run_grading(_selected_sets)

    with col_yolo:
        if st.button(f"Yolo — grade all ({len(_seen)})", width='stretch'):
            _run_grading(_seen)

    if "advisor_grade_error" in st.session_state:
        ui_shell.render_state_panel(
            "error",
            "Grading could not complete",
            "The selected column set could not be evaluated.",
            detail=st.session_state["advisor_grade_error"],
        )

    grade_result = st.session_state.get("advisor_grade")
    if grade_result is None:
        return

    grade_pct = grade_result.grade
    if grade_pct >= 0.85:
        grade_label, grade_kind = "Excellent", "success"
    elif grade_pct >= 0.70:
        grade_label, grade_kind = "Good", "info"
    elif grade_pct >= 0.55:
        grade_label, grade_kind = "Fair", "warning"
    else:
        grade_label, grade_kind = "Poor", "error"

    ui_shell.render_stat_grid(
        [
            ("Hold-out accuracy", f"{grade_result.holdout_accuracy:.3f}", "SVM accuracy on the 50% held-out test set."),
            ("Hold-out macro F1", f"{grade_result.holdout_macro_f1:.3f}", "Balanced class score on the held-out half."),
            (
                f"{grade_result.n_cv_folds}-fold CV",
                f"{grade_result.cv_mean_accuracy:.3f} ± {grade_result.cv_std_accuracy:.3f}",
                "Cross-validation mean accuracy and standard deviation.",
            ),
            ("Grade", f"{grade_pct:.3f} — {grade_label}", "Average of hold-out and CV accuracy."),
        ]
    )
    ui_shell.render_state_panel(
        grade_kind,
        f"LLM column advice graded: {grade_label}",
        f"The recommended columns achieved a combined grade of {grade_pct:.3f} "
        f"({grade_result.holdout_accuracy:.3f} hold-out, {grade_result.cv_mean_accuracy:.3f} CV mean).  "
        f"Kernels tested: {', '.join(grade_result.kernels_tested).upper()}.",
    )


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
        run_itemsets = st.form_submit_button("Run itemset mining", width='stretch')

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
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "Download itemset results CSV",
        dataframe_bytes(itemset_result),
        file_name="itemset_results.csv",
        mime="text/csv",
        on_click="ignore",
        width='stretch',
    )

    # LLM explain for itemsets
    top_patterns = itemset_result.sort_values("support", ascending=False).head(15)
    itemset_summary = (
        f"Technique: Frequent Itemset Mining (Apriori)\n"
        f"Columns mined: {', '.join(item_columns)}\n"
        f"Minimum support: {min_support}\n"
        f"Total patterns found: {len(itemset_result)}\n"
        f"Max itemset length: {int(itemset_result['length'].max())}\n"
        f"Top support: {itemset_result['support'].max():.4f}\n"
        f"Top 15 patterns:\n{top_patterns.to_string(index=False)}"
    )
    _llm_explain_button("itemset_explain", "Frequent Itemset Mining", itemset_summary, frame)


def render_episode_tab(frame: pd.DataFrame, source_name: str) -> None:
    # Reset column/separator/support selections when the dataset changes
    if st.session_state.get("_episode_source") != source_name:
        st.session_state.pop("sequence_column", None)
        st.session_state.pop("sequence_separator", None)
        st.session_state.pop("episode_support", None)
        st.session_state["_episode_source"] = source_name

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
        min_support = st.slider("Minimum support", min_value=0.01, max_value=0.90, value=0.10, step=0.01, key="episode_support")

        if mode == "Delimited sequence column":
            # Auto-detect a good sequence column: prefer columns whose name suggests sequences
            # and whose values contain the separator character
            _seq_candidates = [c for c in frame.columns if any(k in c.lower() for k in ("sequence", "journey", "path", "events", "seq"))]
            _seq_default_idx = list(frame.columns).index(_seq_candidates[0]) if _seq_candidates else 0
            sequence_column = st.selectbox("Sequence column", frame.columns, index=_seq_default_idx, key="sequence_column")
            # Auto-detect separator: check whether values use ", " or "," or " "
            _sep_default = ", "
            _sample_vals = frame[sequence_column].dropna().astype(str).head(20)
            if _sample_vals.str.contains(r", ", regex=True).any():
                _sep_default = ", "
            elif _sample_vals.str.contains(",", regex=False).any():
                _sep_default = ","
            elif _sample_vals.str.contains(" ", regex=False).any():
                _sep_default = " "
            separator = st.text_input("Event separator", value=_sep_default, key="sequence_separator")
            run_episodes = st.form_submit_button("Run episode mining", width='stretch')
        else:
            ordered_columns = st.multiselect("Ordered event columns", frame.columns, key="episode_columns")
            run_episodes = st.form_submit_button("Run episode mining", width='stretch')

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
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "Download episode results CSV",
        dataframe_bytes(episode_result),
        file_name="episode_results.csv",
        mime="text/csv",
        on_click="ignore",
        width='stretch',
    )

    # LLM explain for episodes
    top_episodes = episode_result.sort_values("support", ascending=False).head(15)
    episode_summary = (
        f"Technique: Episode Mining (ordered subsequences)\n"
        f"Mode: {mode}\n"
        f"Max episode length: {max_length}, Max span: {max_span}\n"
        f"Minimum support: {min_support}\n"
        f"Total episodes found: {len(episode_result)}\n"
        f"Longest episode: {int(episode_result['length'].max())}\n"
        f"Top support: {episode_result['support'].max():.4f}\n"
        f"Top 15 episodes:\n{top_episodes.to_string(index=False)}"
    )
    _llm_explain_button("episode_explain", "Episode Mining", episode_summary, frame)


# ── Session save / restore (ui-cold-04) ───────────────────────────────────

_SESSION_SETTING_KEYS: tuple[str, ...] = (
    "source_mode",
    "csv_separator",
    "demo_name",
    "target_column",
    "feature_columns",
    "spotlight_column",
    "kernels",
    "test_size",
    "geometry_mode",
    "geometry_x",
    "geometry_y",
    "plane_x",
    "plane_y",
    "plane_z",
    "item_columns",
    "include_target",
    "item_support",
    "episode_mode",
    "episode_length",
    "episode_span",
    "episode_support",
    "sequence_column",
    "sequence_separator",
    "episode_columns",
    "advisor_kernels",
)


def session_settings_bytes() -> bytes:
    snapshot = {k: st.session_state.get(k) for k in _SESSION_SETTING_KEYS if k in st.session_state}
    return json.dumps(snapshot, default=str, indent=2).encode("utf-8")


def apply_session_settings(config: dict) -> None:
    for key, value in config.items():
        if key in _SESSION_SETTING_KEYS:
            st.session_state[key] = value


# ── Workspace export ZIP (ui-cold-02) ─────────────────────────────────────


def workspace_export_bytes(frame: pd.DataFrame | None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # session config
        zf.writestr("session_config.json", json.dumps(
            {k: st.session_state.get(k) for k in _SESSION_SETTING_KEYS if k in st.session_state},
            default=str,
            indent=2,
        ))
        # pre-computed results
        svm_result = st.session_state.get("custom_svm_result")
        if svm_result is not None:
            zf.writestr("kernel_results.csv", svm_result.kernel_results.to_csv(index=False))
            zf.writestr("feature_importance.csv", svm_result.feature_importance.to_csv(index=False))
        itemset_result = st.session_state.get("itemset_result")
        if itemset_result is not None and not itemset_result.empty:
            zf.writestr("itemset_results.csv", itemset_result.to_csv(index=False))
        episode_result = st.session_state.get("episode_result")
        if episode_result is not None and not episode_result.empty:
            zf.writestr("episode_results.csv", episode_result.to_csv(index=False))
        # LLM advisor grade
        advisor_grade = st.session_state.get("advisor_grade")
        if advisor_grade is not None:
            import dataclasses
            zf.writestr("advisor_grade.json", json.dumps(dataclasses.asdict(advisor_grade), default=str, indent=2))
        # raw data preview
        if frame is not None:
            zf.writestr("data_preview.csv", frame.head(500).to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


# ── Benchmark page (bench-13) ─────────────────────────────────────────────

def render_benchmark_tab() -> None:
    """Full LLM-vs-Ground-Truth benchmark pipeline UI."""
    from svm_studio.benchmark.experiment import run_experiment, ExperimentResult
    from svm_studio.benchmark.db import list_runs
    from svm_studio.benchmark.report_generator import report_to_pdf_bytes

    ui_shell.render_section_intro(
        "LLM Benchmark",
        "Measure how well an LLM labels your data, then let an SVM score the quality",
        "Load any dataset, let the LLM assign labels, and compare the downstream SVM "
        "performance against ground truth to understand exactly where and why the LLM fails.",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Pick a dataset", "Choose a source and dataset name — sklearn, OpenML, UCI, HuggingFace, or CSV."),
            ("Configure LLM", "Set the model endpoint (same as the rest of the app)."),
            ("Enable optional techniques", "Add Uncertainty Sampling, Universum SVM, Itemset Mining, or Episode Mining."),
            ("Run and review", "Read the charts, disagreement table, and generated report."),
        ],
    )

    # ── Dataset form ──────────────────────────────────────────────────────
    # Source must be OUTSIDE the form so changing it reruns immediately
    # and the dataset-name widget updates to match.
    _SKLEARN_NAMES = ["iris", "wine", "breast_cancer", "digits", "20newsgroups",
                      "olivetti_faces", "covtype", "kddcup99"]
    _CSV_NAMES = [
        "cancer_uci.csv", "fraud_openml.csv", "wine_uci.csv", "titanic_openml.csv",
        "banknote_auth.csv", "diabetes_pima.csv", "credit_german.csv", "churn_telecom.csv",
        "steel_plates.csv", "digits_sklearn.csv", "har_smartphone.csv",
        "sensorless_drive.csv", "spambase.csv", "adult_census.csv",
        "shuttle_nasa.csv", "fashion_mnist.csv",
    ]
    _OPENML_PRESETS = ["44 (Spambase)", "1590 (Adult Census)", "40685 (NASA Shuttle)",
                       "40996 (Fashion MNIST)", "31 (German Credit)", "37 (Pima Diabetes)",
                       "1478 (HAR Smartphone)", "1501 (Sensorless Drive)"]
    _UCI_PRESETS = ["iris", "wine", "breast_cancer", "car", "adult", "mushroom"]
    _HF_PRESETS = ["imdb", "ag_news", "emotion", "trec"]

    src_col, _ = st.columns([1, 2])
    with src_col:
        source = st.selectbox(
            "Source", ["sklearn", "openml", "ucimlrepo", "csv", "huggingface"],
            key="bench_source",
        )

    with st.form("benchmark_form"):
        st.subheader("Dataset")
        ds_col2, _ = st.columns([2, 1])
        with ds_col2:
            if source == "sklearn":
                ds_name = st.selectbox("Dataset name", _SKLEARN_NAMES, key="bench_sklearn_name")
            elif source == "csv":
                ds_name = st.selectbox(
                    "Dataset file",
                    _CSV_NAMES,
                    key="bench_csv_name",
                    help="Files in data/external/",
                )
            elif source == "openml":
                ds_name = st.selectbox(
                    "Dataset (id or name)",
                    _OPENML_PRESETS,
                    key="bench_openml_name",
                    help="Select a preset or type an OpenML dataset id/name.",
                )
                ds_name = ds_name.split(" ")[0]  # extract just the id
            elif source == "ucimlrepo":
                ds_name = st.selectbox(
                    "Dataset name", _UCI_PRESETS, key="bench_uci_name"
                )
            else:  # huggingface
                ds_name = st.selectbox(
                    "Dataset name", _HF_PRESETS, key="bench_hf_name"
                )

        max_ex = st.slider("Max examples to label (reduces LLM cost)", 10, 500, 100, step=10, key="bench_max_ex")

        st.subheader("LLM Configuration")
        bench_base_url, bench_model = _llm_url_and_model_widgets(
            url_key="bench_base_url",
            model_key="bench_model",
            default_url=os.environ.get("LLM_BASE_URL", ""),
        )
        bench_api_key = st.text_input("API key (optional)", type="password", key="bench_api_key")

        st.subheader("Optional Techniques")
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        with opt_col1:
            use_ubs = st.checkbox("Uncertainty Sampling", value=False, key="bench_ubs",
                                  help="Re-query LLM for uncertain SVM boundary examples.")
        with opt_col2:
            use_uni = st.checkbox("Universum SVM", value=False, key="bench_uni",
                                  help="Add synthetic midpoint examples to improve label-noise robustness.")
        with opt_col3:
            use_items = st.checkbox("Itemset Mining", value=False, key="bench_items",
                                    help="Find feature patterns correlated with LLM errors (tabular only).")
        with opt_col4:
            use_eps = st.checkbox("Episode Mining", value=False, key="bench_eps",
                                  help="Find sequence patterns in mislabeled examples (sequential datasets).")

        run_bench = st.form_submit_button("Run Benchmark", width='stretch', type="primary")

    if run_bench:
        optional = []
        if use_ubs:
            optional.append("uncertainty_sampling")
        if use_uni:
            optional.append("universum_svm")
        if use_items:
            optional.append("itemset_mining")
        if use_eps:
            optional.append("episode_mining")

        progress_bar = st.progress(0.0, text="Starting …")
        status_text = st.empty()
        stage_weights = {
            "Loading dataset": 0.05,
            "LLM labeling": 0.55,
            "SVM evaluation": 0.70,
            "Uncertainty sampling": 0.75,
            "Universum SVM": 0.80,
            "Itemset mining": 0.83,
            "Episode mining": 0.86,
            "Generating charts": 0.92,
            "Generating report": 1.0,
        }

        def _progress(stage: str, current: int, total: int) -> None:
            base = stage_weights.get(stage, 0.5)
            if total > 0:
                frac = base * 0.8 + (current / total) * base * 0.2
            else:
                frac = base
            progress_bar.progress(min(frac, 1.0), text=f"{stage} …")
            status_text.caption(f"{stage}: {current}/{total}" if total else stage)

        # ── Live labeling display ─────────────────────────────────────────
        st.markdown("#### Live labeling")
        live_left, live_right = st.columns([1, 2])
        with live_left:
            st.caption("Labels assigned so far")
            label_grid = st.empty()
        with live_right:
            st.caption("LLM conversation")
            convo_stream = st.empty()

        convo_log: list[tuple[int, str, str, str, float]] = []
        label_log: list[tuple[int, str, float]] = []   # (idx, label, confidence)

        LABEL_COLOURS = [
            "#18c5d8", "#ff7a18", "#7ef2c8", "#ffd166",
            "#e05c97", "#6a7fff", "#63c132", "#e8733a",
        ]
        label_colour_map: dict[str, str] = {}

        def _conversation(idx: int, prompt: str, response: str, label: str, confidence: float) -> None:
            convo_log.append((idx, prompt, response, label, confidence))
            label_log.append((idx, label, confidence))

            # assign a stable colour per unique label
            if label not in label_colour_map:
                label_colour_map[label] = LABEL_COLOURS[len(label_colour_map) % len(LABEL_COLOURS)]

            # ── left: label chips grid ────────────────────────────────────
            chips_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;'>"
            for _i, _lbl, _conf in label_log:
                col = label_colour_map.get(_lbl, "#aaa")
                chips_html += (
                    f"<span title='#{_i}: {_lbl} ({_conf:.0%})' style='"
                    f"background:{col};color:#091722;font-size:0.72rem;"
                    f"font-weight:600;padding:3px 7px;border-radius:999px;"
                    f"cursor:default;'>{_lbl}</span>"
                )
            chips_html += "</div>"
            label_grid.markdown(chips_html, unsafe_allow_html=True)

            # ── right: last 4 prompt/response pairs ───────────────────────
            stream_md = ""
            for _idx, _prompt, _resp, _label, _conf in convo_log[-4:]:
                ok = _label != "unknown"
                icon = "✅" if ok else "❓"
                stream_md += (
                    f"**{icon} #{_idx} → `{_label}`** ({_conf:.0%} confidence)\n\n"
                    f"> **Prompt (excerpt):** {_prompt[200:400].strip()}…\n\n"
                    f"> **Response:** {_resp[:300].strip()}\n\n---\n\n"
                )
            convo_stream.markdown(stream_md)

        try:
            result: ExperimentResult = run_experiment(
                source=source,
                name=ds_name,
                llm_model=bench_model or "gemma-4-31B-it-Q4_K_M.gguf",
                llm_base_url=bench_base_url or None,
                llm_api_key=bench_api_key or None,
                optional_techniques=optional,
                max_examples=max_ex,
                save_to_db=True,
                progress_callback=_progress,
                conversation_callback=_conversation,
            )
            st.session_state["bench_result"] = result
            st.session_state.pop("bench_error", None)
        except Exception as exc:
            st.session_state["bench_error"] = str(exc)
            st.session_state.pop("bench_result", None)
        finally:
            progress_bar.empty()
            status_text.empty()

    if "bench_error" in st.session_state:
        ui_shell.render_state_panel("error", "Benchmark failed", st.session_state["bench_error"])
        return

    result = st.session_state.get("bench_result")
    if result is None:
        ui_shell.render_state_panel(
            "info", "Benchmark not started",
            "Configure a dataset and LLM, then press Run Benchmark.",
        )
        return

    er = result.eval_result

    # ── Quality note: what these numbers mean ─────────────────────────────
    st.info(
        "**What 'quality' means here:** Agreement and SVM accuracy are proxies. "
        "A strategy can score well on accuracy and still generalise poorly — especially with small or imbalanced datasets. "
        "Use F1 (below) alongside accuracy. Calibration check: if LLM SVM F1 is much lower than its accuracy, "
        "the model is likely predicting the majority class and the labels are not usable."
    )

    # ── Key metrics ───────────────────────────────────────────────────────
    ui_shell.render_stat_grid([
        ("LLM Agreement", f"{er.llm_agreement_rate:.1%}", "Fraction of LLM labels matching ground truth."),
        ("LLM SVM Accuracy", f"{er.llm_metrics.test_accuracy:.3f}", "SVM trained on LLM labels, scored vs true labels."),
        ("LLM SVM Macro-F1", f"{er.llm_metrics.test_macro_f1:.3f}", "F1 across all classes — more robust than accuracy for imbalanced datasets."),
        ("Control SVM Accuracy", f"{er.control_metrics.test_accuracy:.3f}", "SVM trained on true labels."),
        ("Control SVM Macro-F1", f"{er.control_metrics.test_macro_f1:.3f}", "Control F1 — compare with LLM F1 to measure labeling quality."),
        ("Labeling Cost", f"{er.labeling_cost:+.3f}", "Control − LLM accuracy. Positive = LLM loses performance."),
    ])

    sign = "better" if er.labeling_cost <= 0 else "worse"
    # Calibration check: if F1 << accuracy, majority-class collapse is likely
    _f1_gap = er.llm_metrics.test_accuracy - er.llm_metrics.test_macro_f1
    _calib_note = f" ⚠ F1 gap {_f1_gap:.2f} — possible majority-class collapse." if _f1_gap > 0.10 else ""
    ui_shell.render_state_panel(
        "success" if er.labeling_cost <= 0.05 else "warning",
        f"LLM labels performed {sign} than ground truth",
        f"Agreement: {er.llm_agreement_rate:.1%} | LLM Acc: {er.llm_metrics.test_accuracy:.3f} "
        f"F1: {er.llm_metrics.test_macro_f1:.3f} | Control Acc: {er.control_metrics.test_accuracy:.3f} "
        f"F1: {er.control_metrics.test_macro_f1:.3f} | Cost: {er.labeling_cost:+.3f}{_calib_note}",
        detail=f"Most common error: {er.most_common_error} | Worst class: {er.worst_class} | Best class: {er.best_class}",
    )

    # ── Charts ────────────────────────────────────────────────────────────
    st.subheader("Accuracy Comparison")
    st.plotly_chart(result.fig_accuracy, width='stretch')

    st.subheader("CV Fold Stability")
    st.plotly_chart(result.fig_cv_folds, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrices")
        st.plotly_chart(result.fig_confusion, width='stretch')
    with col2:
        st.subheader("Per-Class Precision & Recall")
        st.plotly_chart(result.fig_per_class, width='stretch')

    st.subheader("LLM Confidence Distribution")
    st.plotly_chart(result.fig_confidence, width='stretch')

    st.subheader("Disagreement Table")
    st.plotly_chart(result.fig_disagreement, width='stretch')

    # ── Optional technique results ────────────────────────────────────────
    if result.ubs_result is not None:
        with st.expander("Uncertainty-Based Sampling results", expanded=True):
            ubs = result.ubs_result
            ui_shell.render_stat_grid([
                ("Re-queried", str(ubs.n_uncertain), "Uncertain examples sent back to LLM."),
                ("Changed", str(ubs.relabeled_count), "Examples where LLM changed its label."),
                ("Before accuracy", f"{ubs.before_accuracy:.3f}", "SVM accuracy before UBS."),
                ("After accuracy", f"{ubs.after_accuracy:.3f}", f"Delta: {ubs.delta_accuracy:+.3f}"),
            ])
            if not ubs.relabeled_frame.empty:
                st.dataframe(ubs.relabeled_frame, width='stretch', hide_index=True)

    if result.universum_result is not None:
        with st.expander("Universum SVM results", expanded=True):
            st.dataframe(result.universum_result.comparison, width='stretch', hide_index=True)

    if result.itemset_result is not None and not result.itemset_result.error_exclusive.empty:
        with st.expander("Itemset Mining — error-exclusive patterns", expanded=True):
            st.dataframe(result.itemset_result.error_exclusive.head(20), width='stretch', hide_index=True)

    if result.episode_result is not None and not result.episode_result.error_exclusive.empty:
        with st.expander("Episode Mining — error-exclusive patterns", expanded=True):
            st.dataframe(result.episode_result.error_exclusive.head(20), width='stretch', hide_index=True)

    # ── Report ────────────────────────────────────────────────────────────
    st.subheader("Generated Report")
    with st.expander("View report", expanded=False):
        st.markdown(result.report_markdown)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download report (Markdown)",
            data=result.report_markdown.encode("utf-8"),
            file_name=f"benchmark_{er.dataset_name}.md",
            mime="text/markdown",
            width='stretch',
            on_click="ignore",
        )
    with dl_col2:
        pdf_bytes = report_to_pdf_bytes(result.report_markdown)
        st.download_button(
            "Download report (PDF)" if pdf_bytes != result.report_markdown.encode("utf-8") else "Download report (MD fallback)",
            data=pdf_bytes,
            file_name=f"benchmark_{er.dataset_name}.pdf",
            mime="application/pdf",
            width='stretch',
            on_click="ignore",
        )

    if result.run_id is not None:
        st.caption(f"Run saved to history (ID: {result.run_id})")


# ── Run History page (bench-14) ────────────────────────────────────────────

def render_history_tab() -> None:
    """Comparison view for past benchmark runs stored in SQLite."""
    from svm_studio.benchmark.db import list_runs, compare_runs, load_run

    ui_shell.render_section_intro(
        "Run History",
        "Compare past benchmark experiments side by side",
        "Every benchmark run is saved automatically. Select two or more runs to "
        "see their metrics in a side-by-side comparison table.",
    )

    try:
        runs_df = list_runs()
    except Exception as exc:
        ui_shell.render_state_panel("error", "Could not load run history", str(exc))
        return

    if runs_df.empty:
        ui_shell.render_state_panel(
            "info", "No runs yet",
            "Complete at least one benchmark experiment to populate the history.",
        )
        return

    # Format display
    display_df = runs_df.copy()
    for col in ("llm_accuracy", "control_accuracy", "labeling_cost", "llm_agreement"):
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if v is not None else "—")

    st.dataframe(display_df, width='stretch', hide_index=True)

    # Selection for comparison
    st.subheader("Compare runs")
    all_ids = runs_df["id"].tolist()
    selected_ids = st.multiselect(
        "Select run IDs to compare",
        options=all_ids,
        default=all_ids[:2] if len(all_ids) >= 2 else all_ids,
        key="history_compare_ids",
    )
    if len(selected_ids) >= 2:
        try:
            cmp = compare_runs(selected_ids)
            st.dataframe(cmp, width='stretch', hide_index=True)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    # Report viewer
    st.subheader("View report for a run")
    view_id = st.selectbox("Run ID", options=all_ids, key="history_view_id")
    if st.button("Load report", key="history_load_report"):
        try:
            run_data = load_run(view_id)
            report_text = run_data.get("report_text", "*No report stored for this run.*")
            st.markdown(report_text)
            st.download_button(
                "Download this report (Markdown)",
                data=report_text.encode("utf-8"),
                file_name=f"report_run_{view_id}.md",
                mime="text/markdown",
                on_click="ignore",
            )
        except Exception as exc:
            st.error(f"Could not load run {view_id}: {exc}")


# ── Chat page ──────────────────────────────────────────────────────────────

def _chat_system_context(frame: pd.DataFrame | None, source_name: str | None) -> str:
    """Build a system message that gives the LLM context about the current workspace."""
    from svm_studio.llm_advisor import _CHAT_SYSTEM_PROMPT, _build_schema_text
    lines = [_CHAT_SYSTEM_PROMPT]
    if frame is not None and source_name:
        lines.append(f"\n## Current dataset: {source_name}")
        lines.append(_build_schema_text(frame))
    advice = st.session_state.get("advisor_advice")
    if advice is not None:
        lines.append(
            f"\n## Current advisor recommendation\n"
            f"Target: {advice.target_column}\n"
            f"Features ({len(advice.feature_columns)}): {', '.join(advice.feature_columns)}\n"
            f"Rationale: {advice.rationale}"
        )
    grade = st.session_state.get("advisor_grade")
    if grade is not None:
        lines.append(
            f"\n## Latest grade\n"
            f"Hold-out accuracy: {grade.holdout_accuracy:.3f}  "
            f"CV mean: {grade.cv_mean_accuracy:.3f} ± {grade.cv_std_accuracy:.3f}  "
            f"Grade: {grade.grade:.3f}"
        )
    return "\n".join(lines)


# ── SVM Visualizer page — pure Plotly, step-by-step visual explainer ───────

_VIZ_PALETTE = ["#18c5d8", "#ff7a18", "#7ef2c8", "#295fe7", "#ffd166", "#e05297", "#a476ff", "#44d49b"]


def _viz_plotly_layout(**extra) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fbfd",
        font=dict(family="Sora, Segoe UI, sans-serif", color="#091722", size=13),
        margin=dict(l=10, r=10, t=52, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    base.update(extra)
    return base


def _viz_step_1_scatter(X: pd.DataFrame, y: pd.Series, xcol: str, ycol: str, labels: list[str]) -> go.Figure:
    """Step 1: Raw data scatter — see the two classes."""
    fig = go.Figure()
    for i, lab in enumerate(labels):
        mask = y == lab
        fig.add_trace(go.Scatter(
            x=X.loc[mask, xcol], y=X.loc[mask, ycol],
            mode="markers",
            name=str(lab),
            marker=dict(color=_VIZ_PALETTE[i % len(_VIZ_PALETTE)], size=9, opacity=0.82,
                        line=dict(width=0.6, color="white")),
            hovertemplate=f"Class: {lab}<br>{xcol}: %{{x:.3f}}<br>{ycol}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Step 1 — Your data, plotted",
        xaxis_title=xcol, yaxis_title=ycol,
        **_viz_plotly_layout(),
    )
    return fig


def _viz_step_2_boundary(X: pd.DataFrame, y: pd.Series, xcol: str, ycol: str,
                         labels: list[str], model, feature_cols: list[str]) -> go.Figure:
    """Step 2: Decision boundary heatmap + data points."""
    xvals = X[xcol]
    yvals = X[ycol]
    pad_x = (xvals.max() - xvals.min()) * 0.08 or 1.0
    pad_y = (yvals.max() - yvals.min()) * 0.08 or 1.0
    xx = np.linspace(xvals.min() - pad_x, xvals.max() + pad_x, 200)
    yy = np.linspace(yvals.min() - pad_y, yvals.max() + pad_y, 200)
    xxg, yyg = np.meshgrid(xx, yy)
    grid = pd.DataFrame({xcol: xxg.ravel(), ycol: yyg.ravel()})
    # Fill other feature columns with median
    for c in feature_cols:
        if c not in grid.columns:
            grid[c] = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]
    grid = grid[feature_cols]
    pred_codes = pd.Categorical(model.predict(grid), categories=labels).codes.reshape(xxg.shape)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xx, y=yy, z=pred_codes,
        colorscale=[[0, "rgba(24,197,216,0.18)"], [0.5, "rgba(126,242,200,0.14)"], [1, "rgba(255,122,24,0.18)"]],
        showscale=False, hoverinfo="skip",
    ))
    for i, lab in enumerate(labels):
        mask = y == lab
        fig.add_trace(go.Scatter(
            x=X.loc[mask, xcol], y=X.loc[mask, ycol],
            mode="markers", name=str(lab),
            marker=dict(color=_VIZ_PALETTE[i % len(_VIZ_PALETTE)], size=9, opacity=0.85,
                        line=dict(width=0.6, color="white")),
            hovertemplate=f"Class: {lab}<br>{xcol}: %{{x:.3f}}<br>{ycol}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Step 2 — The decision boundary",
        xaxis_title=xcol, yaxis_title=ycol,
        **_viz_plotly_layout(),
    )
    return fig


def _viz_step_3_margin(X: pd.DataFrame, y: pd.Series, xcol: str, ycol: str,
                       labels: list[str], model, feature_cols: list[str]) -> go.Figure:
    """Step 3: Decision boundary + margin bands + support vectors."""
    svc = model.named_steps["svc"]
    xvals = X[xcol]
    yvals = X[ycol]
    pad_x = (xvals.max() - xvals.min()) * 0.08 or 1.0
    pad_y = (yvals.max() - yvals.min()) * 0.08 or 1.0
    xx = np.linspace(xvals.min() - pad_x, xvals.max() + pad_x, 200)
    yy = np.linspace(yvals.min() - pad_y, yvals.max() + pad_y, 200)
    xxg, yyg = np.meshgrid(xx, yy)
    grid = pd.DataFrame({xcol: xxg.ravel(), ycol: yyg.ravel()})
    for c in feature_cols:
        if c not in grid.columns:
            grid[c] = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]
    grid = grid[feature_cols]

    raw_scores = model.decision_function(grid)
    if raw_scores.ndim > 1:
        raw_scores = raw_scores[:, 0]
    Z = raw_scores.reshape(xxg.shape)

    fig = go.Figure()
    # margin bands: decision_function == -1, 0, +1
    fig.add_trace(go.Contour(
        x=xx, y=yy, z=Z,
        contours=dict(
            start=-1, end=1, size=1,
            coloring="none",
            showlabels=True,
            labelfont=dict(size=11, color="#47596a"),
        ),
        line=dict(width=2, color="rgba(9,23,34,0.55)", dash="dot"),
        showscale=False, hoverinfo="skip",
        name="Margin",
    ))
    # f(x)=0 boundary
    fig.add_trace(go.Contour(
        x=xx, y=yy, z=Z,
        contours=dict(start=0, end=0, size=0, coloring="none"),
        line=dict(width=3, color="#091722"),
        showscale=False, hoverinfo="skip",
        name="Boundary f(x)=0",
    ))
    # fill between margin lines (light band)
    fig.add_trace(go.Heatmap(
        x=xx, y=yy,
        z=np.where((Z >= -1) & (Z <= 1), 1, np.nan),
        colorscale=[[0, "rgba(24,197,216,0.08)"], [1, "rgba(24,197,216,0.08)"]],
        showscale=False, hoverinfo="skip",
    ))

    for i, lab in enumerate(labels):
        mask = y == lab
        fig.add_trace(go.Scatter(
            x=X.loc[mask, xcol], y=X.loc[mask, ycol],
            mode="markers", name=str(lab),
            marker=dict(color=_VIZ_PALETTE[i % len(_VIZ_PALETTE)], size=8, opacity=0.78,
                        line=dict(width=0.5, color="white")),
        ))

    # Support vectors
    sv_idx = svc.support_
    preprocessor = model.named_steps["preprocessor"]
    sv_transformed = svc.support_vectors_
    # Inverse-transform to get real coordinates
    try:
        num_pipe = preprocessor.named_transformers_["numeric"]
        scaler = num_pipe.named_steps["scaler"]
        num_cols = preprocessor.transformers_[0][2]
        sv_orig = pd.DataFrame(scaler.inverse_transform(sv_transformed[:, :len(num_cols)]),
                               columns=num_cols)
        sv_x = sv_orig[xcol] if xcol in sv_orig.columns else X.iloc[sv_idx][xcol]
        sv_y = sv_orig[ycol] if ycol in sv_orig.columns else X.iloc[sv_idx][ycol]
    except Exception:
        sv_x = X.iloc[sv_idx][xcol]
        sv_y = X.iloc[sv_idx][ycol]

    fig.add_trace(go.Scatter(
        x=sv_x, y=sv_y,
        mode="markers", name="Support vectors",
        marker=dict(symbol="diamond-open", color="#091722", size=14,
                    line=dict(width=2.5, color="#091722")),
        hovertemplate="Support vector<br>%{x:.3f}, %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Step 3 — The margin and support vectors",
        xaxis_title=xcol, yaxis_title=ycol,
        **_viz_plotly_layout(),
    )
    return fig


def _viz_step_4_decision_scores(X: pd.DataFrame, y: pd.Series, labels: list[str], model) -> go.Figure:
    """Step 4: Histogram of decision scores — where the model is confident vs uncertain."""
    raw = model.decision_function(X)
    if raw.ndim > 1:
        raw = np.abs(raw).min(axis=1)

    fig = go.Figure()
    for i, lab in enumerate(labels):
        mask = y == lab
        fig.add_trace(go.Histogram(
            x=raw[mask] if raw.ndim == 1 else raw,
            name=str(lab),
            marker_color=_VIZ_PALETTE[i % len(_VIZ_PALETTE)],
            opacity=0.72,
            nbinsx=40,
            hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="#091722", annotation_text="boundary",
                  annotation_position="top right")
    fig.add_vrect(x0=-1, x1=1, fillcolor="rgba(24,197,216,0.08)", line_width=0,
                  annotation_text="margin zone", annotation_position="top left")
    fig.update_layout(
        title="Step 4 — Decision scores (confidence)",
        xaxis_title="f(x) — distance from boundary",
        yaxis_title="Count",
        barmode="overlay",
        **_viz_plotly_layout(),
    )
    return fig


def _viz_step_5_pca(X: pd.DataFrame, y: pd.Series, labels: list[str], model) -> go.Figure:
    """Step 5: PCA projection for high-dimensional data — the big-picture view."""
    X_transformed = model.named_steps["preprocessor"].transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_transformed)
    ev1, ev2 = pca.explained_variance_ratio_[:2]
    svc = model.named_steps["svc"]
    sv_coords = pca.transform(svc.support_vectors_)

    fig = go.Figure()
    for i, lab in enumerate(labels):
        mask = y.values == lab
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=str(lab),
            marker=dict(color=_VIZ_PALETTE[i % len(_VIZ_PALETTE)], size=7, opacity=0.80),
            hovertemplate=f"Class: {lab}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=sv_coords[:, 0], y=sv_coords[:, 1],
        mode="markers", name="Support vectors",
        marker=dict(symbol="diamond-open", color="#091722", size=13, line=dict(width=2)),
        hovertemplate="SV<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Step 5 — PCA projection ({ev1:.0%} + {ev2:.0%} = {ev1+ev2:.0%} variance explained)",
        xaxis_title=f"PC1 ({ev1:.1%})", yaxis_title=f"PC2 ({ev2:.1%})",
        **_viz_plotly_layout(),
    )
    return fig


def render_visualizer_tab(frame: pd.DataFrame, source_name: str) -> None:
    """Full-page interactive SVM visual explainer — pure Plotly, zero matplotlib."""
    ui_shell.render_section_intro(
        "SVM Visualizer",
        "See the machine learn, step by step",
        "This page trains an SVM on your data and walks you through every stage "
        "of the algorithm visually — from raw points to decision boundary, margin, "
        "support vectors, and confidence scores. No math background required.",
    )
    ui_shell.render_step_strip(
        "What you'll see",
        [
            ("Data scatter", "Your two chosen columns plotted as coloured dots — one colour per class."),
            ("Boundary", "The dividing line (or curve) the SVM found between classes."),
            ("Margin + SVs", "The safety corridor and the critical points that pin it open."),
            ("Confidence", "How sure the model is about each prediction — and where it hesitates."),
            ("PCA overview", "All features compressed to 2D so you can see the full picture at once."),
        ],
    )

    columns = list(frame.columns)
    target_default = _default_target_column(columns)

    with st.form("viz_form"):
        viz_cols = st.columns([1, 1, 1, 1])
        with viz_cols[0]:
            viz_target = st.selectbox("Target column", columns,
                                      index=columns.index(target_default) if target_default in columns else 0,
                                      key="viz_target")
        feature_options = [c for c in columns if c != viz_target]
        numeric_options = [c for c in feature_options
                          if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
        with viz_cols[1]:
            viz_x = st.selectbox("X-axis feature", numeric_options,
                                 index=0 if numeric_options else 0, key="viz_x")
        y_opts = [c for c in numeric_options if c != viz_x]
        with viz_cols[2]:
            viz_y = st.selectbox("Y-axis feature", y_opts,
                                 index=0 if y_opts else 0, key="viz_y")
        with viz_cols[3]:
            viz_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="viz_kernel")
        viz_features = st.multiselect(
            "All features for the model (the two above are just for plotting)",
            feature_options,
            default=numeric_options[:min(6, len(numeric_options))],
            key="viz_features",
        )
        run_viz = st.form_submit_button("Train & Visualize", width='stretch')

    if run_viz:
        if len(viz_features) < 2 or viz_x not in viz_features or viz_y not in viz_features:
            st.warning("The two axis features must be included in 'All features'.")
        else:
            with st.status("Training SVM …", expanded=True) as status:
                try:
                    model, X, y_series = fit_custom_svm_estimator(
                        frame, viz_target, viz_features, kernel=viz_kernel,
                    )
                    labels = sorted(y_series.unique().tolist())
                    svc = model.named_steps["svc"]
                    status.update(label="SVM trained — building visuals", state="running")

                    fig1 = _viz_step_1_scatter(X, y_series, viz_x, viz_y, labels)
                    fig2 = _viz_step_2_boundary(X, y_series, viz_x, viz_y, labels, model, viz_features)
                    fig3 = _viz_step_3_margin(X, y_series, viz_x, viz_y, labels, model, viz_features)
                    fig4 = _viz_step_4_decision_scores(X, y_series, labels, model)
                    fig5 = _viz_step_5_pca(X, y_series, labels, model)

                    st.session_state["viz_result"] = {
                        "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
                        "kernel": viz_kernel, "target": viz_target,
                        "features": viz_features, "n_sv": svc.n_support_.sum(),
                        "n_classes": len(labels), "labels": labels,
                        "xcol": viz_x, "ycol": viz_y,
                    }
                    st.session_state.pop("viz_error", None)
                    status.update(label="Visualizer ready", state="complete", expanded=False)
                except Exception as exc:
                    st.session_state["viz_error"] = str(exc)
                    st.session_state.pop("viz_result", None)
                    status.update(label="Training failed", state="error")

    if "viz_error" in st.session_state:
        ui_shell.render_state_panel("error", "Visualizer failed", st.session_state["viz_error"])

    vr = st.session_state.get("viz_result")
    if vr is None:
        ui_shell.render_state_panel(
            "info", "Pick features and hit Train",
            "Choose a target, two numeric columns for the axes, and any extra features. "
            "The visualizer will walk you through how the SVM sees your data.")
        return

    ui_shell.render_stat_grid([
        ("Kernel", vr["kernel"].upper(), "The shape of the boundary."),
        ("Classes", str(vr["n_classes"]), f'{", ".join(str(l) for l in vr["labels"])}'),
        ("Support vectors", f'{vr["n_sv"]:,}', "The points that hold the margin open."),
        ("Plot axes", f'{vr["xcol"]}  vs  {vr["ycol"]}', f'{len(vr["features"])} total features in the model.'),
    ])

    # ── Step 1 ──
    st.plotly_chart(vr["fig1"], width='stretch')
    ui_shell.render_annotated_formula(
        "What you're seeing",
        [("Each dot", "one row in your data"), (" is coloured by ", None), ("class label", "target column")],
        note="Nothing has been learned yet — this is just your raw data on two axes.",
    )

    # ── Step 2 ──
    st.plotly_chart(vr["fig2"], width='stretch')
    ui_shell.render_annotated_formula(
        "The boundary line",
        [("f(x)", "decision function"), (" = 0", None),
         ("  ←  the SVM drew this line to separate the classes", None)],
        note="The coloured regions show which class the model would predict for any point in this space.",
    )

    # ── Step 3 ──
    st.plotly_chart(vr["fig3"], width='stretch')
    ui_shell.render_annotated_formula(
        "Margin and support vectors",
        [("f(x) = −1", "margin edge (class A)"),
         ("  ───  ", None),
         ("f(x) = 0", "boundary"),
         ("  ───  ", None),
         ("f(x) = +1", "margin edge (class B)")],
        note="The ◇ diamonds are the support vectors — the closest points to the boundary. "
             "Remove any of them and the boundary moves. All other points could vanish without changing the model.",
    )
    ui_shell.render_callout(
        "Why the margin matters",
        "A wider margin means the model is more confident. The SVM maximises this gap "
        "so it picks the boundary that is as far away from both classes as possible.",
    )

    # ── Step 4 ──
    st.plotly_chart(vr["fig4"], width='stretch')
    ui_shell.render_annotated_formula(
        "Reading the confidence histogram",
        [("f(x) > 0", "predicted class B"),
         ("     ", None),
         ("f(x) < 0", "predicted class A"),
         ("     ", None),
         ("|f(x)| near 0", "uncertain")],
        note="Points deep into positive or negative territory are easy calls. "
             "Points near zero are the ones the model struggles with — they live inside the margin.",
    )

    # ── Step 5 ──
    st.plotly_chart(vr["fig5"], width='stretch')
    ui_shell.render_annotated_formula(
        "PCA — the big picture",
        [("PC1", "most variance"), (" + ", None), ("PC2", "second most"),
         (" = a 2D summary of all ", None),
         (f"{len(vr['features'])} features", "compressed")],
        note="PCA squashes all your features into two axes so you can see whether the classes "
             "actually separate. Support vectors (◇) sit right on the boundary between clusters.",
    )

    _llm_explain_button(
        "viz_explain", "SVM Visualizer walkthrough",
        f"Kernel: {vr['kernel']}, Target: {vr['target']}, "
        f"Features: {', '.join(vr['features'])}, "
        f"Classes: {', '.join(str(l) for l in vr['labels'])}, "
        f"Support vectors: {vr['n_sv']}, "
        f"Plot axes: {vr['xcol']} vs {vr['ycol']}",
        frame,
    )


# ── Advanced SVM page (Active Learning + Universum) ────────────────────────

def _llm_explain_button(
    key: str,
    technique: str,
    result_summary: str,
    frame: pd.DataFrame | None,
) -> None:
    """Render an 'Explain with LLM' button that streams the reply token-by-token."""
    if st.button("Explain with LLM", key=key, width='stretch'):
        from svm_studio.llm_advisor import (
            explain_result_stream, _build_data_context,
            fetch_available_models, build_explain_messages,
        )

        base_url = (
            st.session_state.get("advisor_base_url_input")
            or st.session_state.get("chat_base_url")
            or os.environ.get("LLM_BASE_URL")
            or None
        )
        api_key = st.session_state.get("chat_api_key") or st.session_state.get("advisor_api_key_input") or None
        model = st.session_state.get("advisor_model_input") or st.session_state.get("chat_model") or None

        if not model and base_url:
            models = fetch_available_models(base_url)
            model = models[0] if models else None

        # Minimal context: shape + column names only, no sample rows or correlations
        if frame is not None:
            ctx_full = _build_data_context(
                frame, max_cat_samples=3, sample_rows=0, include_correlations=False
            )
            ctx = ctx_full[:600] + ("\n[...truncated...]" if len(ctx_full) > 600 else "")
        else:
            ctx = None

        # Show the prompt being sent
        msgs = build_explain_messages(technique, result_summary, ctx)
        total_chars = sum(len(m["content"]) for m in msgs)
        with st.expander(f"Prompt sent to LLM  ({total_chars:,} chars)", expanded=False):
            for m in msgs:
                st.markdown(f"**`{m['role']}`**")
                st.code(m["content"], language=None, wrap_lines=True)

        # Safe streaming wrapper — surfaces generator exceptions to the caller
        def _safe_stream():
            try:
                yield from explain_result_stream(
                    technique, result_summary, ctx,
                    api_key=api_key, model=model, base_url=base_url,
                )
            except Exception as _gen_exc:
                st.session_state[f"{key}_stream_error"] = str(_gen_exc)

        # Clear any previous error before a new attempt
        st.session_state.pop(f"{key}_error", None)
        st.session_state.pop(f"{key}_explanation", None)

        # Thinking indicator
        _think = st.empty()
        _think.info("Querying LLM — waiting for first token…")
        try:
            full_text = st.write_stream(_safe_stream())
            _think.empty()
            stream_err = st.session_state.pop(f"{key}_stream_error", None)
            if stream_err:
                st.session_state[f"{key}_error"] = stream_err
            elif full_text:
                st.session_state[f"{key}_explanation"] = full_text
            else:
                st.session_state[f"{key}_error"] = "Model returned an empty response."
        except Exception as exc:
            _think.empty()
            st.session_state[f"{key}_error"] = f"{type(exc).__name__}: {exc}"

    # Always render persisted error or explanation (survives reruns)
    error_msg = st.session_state.get(f"{key}_error")
    if error_msg:
        st.error(f"LLM explanation failed")
        st.code(error_msg)

    explanation = st.session_state.get(f"{key}_explanation")
    if explanation:
        with st.expander("LLM Explanation", expanded=True):
            st.markdown(explanation)


def render_advanced_tab(frame: pd.DataFrame, source_name: str) -> None:
    ui_shell.render_section_intro(
        "Advanced SVM",
        "Active learning and Universum-augmented boundaries",
        "Explore how SVMs behave when labels are scarce (active learning) or "
        "when synthetic midpoint examples are used to regularise the boundary (Universum SVM).",
    )
    ui_shell.render_step_strip(
        "Page flow",
        [
            ("Choose a technique", "Pick Active Learning or Universum SVM from the tabs below."),
            ("Configure and run", "Set the target, features, kernel, and technique-specific parameters."),
            ("Review results", "Inspect the learning curve or comparison table, then ask the LLM to explain."),
        ],
    )

    from svm_studio.advanced_svm import run_active_learning, run_universum_svm

    columns = list(frame.columns)
    target_default = _default_target_column(columns)

    al_tab, uni_tab = st.tabs(["Active Learning", "Universum SVM"])

    # ── Active Learning ────────────────────────────────────────────────
    with al_tab:
        ui_shell.render_method_box(
            "Uncertainty-based active learning",
            "Start with a tiny labelled seed, then iteratively ask for the labels of the most uncertain points — "
            "those closest to the decision boundary.",
            [
                ("Margin sampling", "|f(x)| → smallest", "Points with the smallest margin distance are selected first."),
                ("Learning curve", "accuracy vs. labels queried", "Shows how quickly the model improves as more labels are added."),
                ("Baseline", "train on all labels", "Accuracy when the SVM sees every label — the ceiling the active learner tries to approach."),
            ],
        )
        with st.form("active_learning_form"):
            al_target = st.selectbox("Target column", columns, index=columns.index(target_default) if target_default in columns else 0, key="al_target")
            al_feature_options = [c for c in columns if c != al_target]
            al_features = st.multiselect("Feature columns", al_feature_options, default=al_feature_options[:min(6, len(al_feature_options))], key="al_features")
            al_cols = st.columns(4)
            with al_cols[0]:
                al_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="al_kernel")
            with al_cols[1]:
                al_seed = st.number_input("Seed size", min_value=2, max_value=200, value=10, step=2, key="al_seed")
            with al_cols[2]:
                al_budget = st.number_input("Query budget", min_value=5, max_value=500, value=50, step=5, key="al_budget")
            with al_cols[3]:
                al_batch = st.number_input("Batch size", min_value=1, max_value=50, value=5, step=1, key="al_batch")
            run_al = st.form_submit_button("Run Active Learning", width='stretch')

        if run_al:
            if not al_features:
                st.warning("Select at least one feature column.")
            else:
                with st.status("Running active learning …", expanded=True) as status:
                    try:
                        status.write(f"Seed set: {al_seed} labels, budget: {al_budget}, batch: {al_batch}")
                        al_result = run_active_learning(
                            frame, al_target, al_features,
                            kernel=al_kernel, seed_size=al_seed,
                            budget=al_budget, batch_size=al_batch,
                        )
                        status.update(label="Active learning complete", state="complete", expanded=False)
                        st.session_state["al_result"] = al_result
                        st.session_state.pop("al_error", None)
                    except Exception as exc:
                        status.update(label="Active learning failed", state="error", expanded=True)
                        st.session_state["al_error"] = str(exc)
                        st.session_state.pop("al_result", None)

        if "al_error" in st.session_state:
            ui_shell.render_state_panel("error", "Active learning failed", st.session_state["al_error"])

        al_result = st.session_state.get("al_result")
        if al_result is not None:
            ui_shell.render_stat_grid([
                ("Final accuracy", f"{al_result.final_accuracy:.3f}", "Accuracy after the full query budget."),
                ("Final macro F1", f"{al_result.final_macro_f1:.3f}", "Macro-averaged F1 at the end of active learning."),
                ("Baseline accuracy", f"{al_result.baseline_accuracy:.3f}", "Accuracy with all pool labels (the ceiling)."),
                ("Labels used", f"{al_result.rounds[-1].labelled_count}", f"Out of a possible pool (seed {al_result.seed_size} + budget {al_result.budget})."),
            ])
            ui_shell.render_state_panel(
                "success", "Active learning complete",
                f"Reached {al_result.final_accuracy:.1%} accuracy using only "
                f"{al_result.rounds[-1].labelled_count} labels vs. {al_result.baseline_accuracy:.1%} baseline on the full pool.",
            )

            # Learning curve chart
            curve = al_result.learning_curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["labelled"], y=curve["accuracy"], mode="lines+markers", name="Accuracy"))
            fig.add_trace(go.Scatter(x=curve["labelled"], y=curve["macro_f1"], mode="lines+markers", name="Macro F1"))
            fig.add_hline(y=al_result.baseline_accuracy, line_dash="dash", annotation_text="Baseline (all labels)", line_color="grey")
            fig.update_layout(
                title="Active Learning Curve",
                xaxis_title="Labelled examples",
                yaxis_title="Score",
                yaxis_range=[0, 1.05],
                template="plotly_white",
            )
            st.plotly_chart(fig, width='stretch')

            st.dataframe(curve, width='stretch', hide_index=True)
            st.download_button(
                "Download learning curve CSV",
                dataframe_bytes(curve),
                file_name="active_learning_curve.csv",
                mime="text/csv",
                on_click="ignore",
                width='stretch',
            )

            # LLM explain
            al_summary = (
                f"Technique: Active Learning with uncertainty (margin) sampling\n"
                f"Target: {al_result.target_column}\n"
                f"Features: {', '.join(al_result.feature_columns)}\n"
                f"Kernel: {al_result.kernel}\n"
                f"Classes: {', '.join(str(c) for c in al_result.class_labels)}\n"
                f"Seed size: {al_result.seed_size}, Budget: {al_result.budget}, Batch: {al_result.batch_size}\n"
                f"Final accuracy: {al_result.final_accuracy:.4f}\n"
                f"Final macro F1: {al_result.final_macro_f1:.4f}\n"
                f"Baseline accuracy (all labels): {al_result.baseline_accuracy:.4f}\n"
                f"Labels used: {al_result.rounds[-1].labelled_count}\n"
                f"Rounds: {len(al_result.rounds)}\n"
                f"Learning curve:\n{curve.to_string(index=False)}"
            )
            _llm_explain_button("al_explain", "Active Learning (Uncertainty Sampling)", al_summary, frame)

        elif "al_error" not in st.session_state:
            ui_shell.render_state_panel(
                "info", "Active learning not started",
                "Configure the target, features, and budget, then press Run.",
            )

    # ── Universum SVM ──────────────────────────────────────────────────
    with uni_tab:
        ui_shell.render_method_box(
            "Universum-augmented SVM",
            "Synthetic examples that belong to neither class are injected between the real classes. "
            "Penalising the SVM for confidently classifying these 'nobody' points improves margin quality.",
            [
                ("Universum points", "midpoint / noise / convex", "Synthetic examples placed between class centroids."),
                ("Universum weight", "class_weight penalty", "Lower weight = softer regularisation; higher = stronger push away from midpoint."),
                ("Comparison", "Standard vs. Universum SVM", "Side-by-side accuracy and F1 to see if the extra regularisation helps."),
            ],
        )
        with st.form("universum_form"):
            uni_target = st.selectbox("Target column", columns, index=columns.index(target_default) if target_default in columns else 0, key="uni_target")
            uni_feature_options = [c for c in columns if c != uni_target]
            uni_features = st.multiselect("Feature columns", uni_feature_options, default=uni_feature_options[:min(6, len(uni_feature_options))], key="uni_features")
            uni_cols = st.columns(4)
            with uni_cols[0]:
                uni_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="uni_kernel")
            with uni_cols[1]:
                uni_size = st.number_input("Universum size", min_value=10, max_value=1000, value=100, step=10, key="uni_size")
            with uni_cols[2]:
                uni_strategy = st.selectbox("Strategy", ["midpoint", "gaussian_noise", "random_convex"], key="uni_strategy")
            with uni_cols[3]:
                uni_c = st.number_input("Universum weight", min_value=0.01, max_value=5.0, value=0.5, step=0.05, key="uni_c")
            run_uni = st.form_submit_button("Run Universum SVM", width='stretch')

        if run_uni:
            if not uni_features:
                st.warning("Select at least one feature column.")
            else:
                with st.status("Running Universum SVM …", expanded=True) as status:
                    try:
                        status.write(f"Generating {uni_size} universum points ({uni_strategy}) …")
                        uni_result = run_universum_svm(
                            frame, uni_target, uni_features,
                            kernel=uni_kernel, universum_size=uni_size,
                            universum_strategy=uni_strategy, universum_C=uni_c,
                        )
                        status.update(label="Universum SVM complete", state="complete", expanded=False)
                        st.session_state["uni_result"] = uni_result
                        st.session_state.pop("uni_error", None)
                    except Exception as exc:
                        status.update(label="Universum SVM failed", state="error", expanded=True)
                        st.session_state["uni_error"] = str(exc)
                        st.session_state.pop("uni_result", None)

        if "uni_error" in st.session_state:
            ui_shell.render_state_panel("error", "Universum SVM failed", st.session_state["uni_error"])

        uni_result = st.session_state.get("uni_result")
        if uni_result is not None:
            delta_color = "normal" if uni_result.accuracy_delta >= 0 else "inverse"
            ui_shell.render_stat_grid([
                ("Standard accuracy", f"{uni_result.standard_accuracy:.3f}", "Baseline SVM without universum augmentation."),
                ("Universum accuracy", f"{uni_result.universum_accuracy:.3f}", "SVM trained with synthetic midpoint examples."),
                ("Delta", f"{uni_result.accuracy_delta:+.4f}", "Universum minus standard. Positive = improvement."),
                ("Strategy", uni_result.universum_strategy, f"{uni_result.universum_size} synthetic points, weight={uni_c if 'uni_c' in dir() else uni_result.universum_strategy}."),
            ])
            sign = "improved" if uni_result.accuracy_delta > 0 else "did not improve" if uni_result.accuracy_delta == 0 else "decreased"
            ui_shell.render_state_panel(
                "success" if uni_result.accuracy_delta >= 0 else "warning",
                f"Universum SVM {sign} over standard",
                f"Standard: {uni_result.standard_accuracy:.3f} acc / {uni_result.standard_macro_f1:.3f} F1  →  "
                f"Universum: {uni_result.universum_accuracy:.3f} acc / {uni_result.universum_macro_f1:.3f} F1  (Δ = {uni_result.accuracy_delta:+.4f})",
            )

            st.dataframe(uni_result.comparison, width='stretch', hide_index=True)
            st.download_button(
                "Download comparison CSV",
                dataframe_bytes(uni_result.comparison),
                file_name="universum_comparison.csv",
                mime="text/csv",
                on_click="ignore",
                width='stretch',
            )

            # LLM explain
            uni_summary = (
                f"Technique: Universum SVM\n"
                f"Target: {uni_result.target_column}\n"
                f"Features: {', '.join(uni_result.feature_columns)}\n"
                f"Kernel: {uni_result.kernel}\n"
                f"Classes: {', '.join(str(c) for c in uni_result.class_labels)}\n"
                f"Universum strategy: {uni_result.universum_strategy}, size: {uni_result.universum_size}\n"
                f"Standard accuracy: {uni_result.standard_accuracy:.4f}, Macro F1: {uni_result.standard_macro_f1:.4f}\n"
                f"Universum accuracy: {uni_result.universum_accuracy:.4f}, Macro F1: {uni_result.universum_macro_f1:.4f}\n"
                f"Delta (Uni − Std): {uni_result.accuracy_delta:+.4f}\n"
                f"Comparison table:\n{uni_result.comparison.to_string(index=False)}"
            )
            _llm_explain_button("uni_explain", "Universum SVM", uni_summary, frame)

        elif "uni_error" not in st.session_state:
            ui_shell.render_state_panel(
                "info", "Universum SVM not started",
                "Configure the target, features, and universum parameters, then press Run.",
            )


def render_chat_tab(frame: pd.DataFrame | None, source_name: str | None) -> None:
    ui_shell.render_section_intro(
        "LLM Chat",
        "Ask the model anything about your data, features, or SVM results",
        "The model has access to the current dataset schema and any advisor recommendation "
        "already generated.  Use it to reason about feature choices, interpret grades, "
        "or explore what the data is telling you.",
    )

    # ── backend config (mirrors advisor form, persisted in session_state) ──
    with st.expander("LLM connection", expanded="chat_base_url" not in st.session_state):
        chat_backend = st.radio(
            "Backend",
            ["Local / Remote LLM", "OpenAI", "No LLM (disabled)"],
            horizontal=True,
            key="chat_backend",
        )
        if chat_backend == "OpenAI":
            st.text_input(
                "OpenAI API key",
                type="password",
                key="chat_api_key",
            )
            st.session_state["chat_base_url"] = ""
            st.session_state.setdefault("chat_model", "gpt-4o-mini")
        elif chat_backend == "Local / Remote LLM":
            _llm_url_and_model_widgets(
                url_key="chat_base_url",
                model_key="chat_model",
                default_url=st.session_state.get("advisor_base_url_input") or os.environ.get("LLM_BASE_URL", ""),
                inside_form=False,
            )
        if st.button("Clear chat history", use_container_width=False):
            st.session_state["chat_messages"] = []
            st.rerun()
        if frame is not None:
            st.selectbox(
                "Data context sent to LLM",
                ["Full (stats + sample)", "Stats only (no sample rows)", "Schema only (column names + ranges)"],
                key="chat_data_depth",
                help="Controls how much of the dataset the LLM sees. More context = better answers, larger prompt.",
            )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # ── starter prompts (shown only when history is empty) ─────────────────
    if not st.session_state["chat_messages"] and chat_backend != "No LLM (disabled)":
        st.markdown("**Starter prompts — click to send:**")
        starters = [
            "Suggest a good binary classification dataset for SVM — something about medical diagnosis.",
            "Where can I find a fraud detection dataset with at least 10k rows?",
            "I want to classify sensor readings into fault / no-fault. What dataset type and repo should I look at?",
            "What are the gotchas when using SVM on high-dimensional gene expression data?",
            "Explain what the grade score means and how I should interpret a 0.85 result.",
        ]
        cols = st.columns(len(starters))
        for col, s in zip(cols, starters):
            if col.button(s[:55] + "…", width='stretch', help=s):
                st.session_state["chat_messages"].append({"role": "user", "content": s})
                st.rerun()
        st.markdown("---")

    # ── chat history ───────────────────────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── input ──────────────────────────────────────────────────────────────
    if chat_backend == "No LLM (disabled)":
        st.info("Configure a backend above to start chatting.")
        return

    prompt = st.chat_input("Ask about your data, features, or SVM results …")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_ctx = _chat_system_context(frame, source_name)
        full_messages = [{"role": "system", "content": system_ctx}] + st.session_state["chat_messages"]

        base_url = st.session_state.get("chat_base_url") or None
        api_key = st.session_state.get("chat_api_key") or None
        model = st.session_state.get("chat_model") or None

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    from svm_studio.llm_advisor import chat_completion
                    reply = chat_completion(
                        full_messages,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                except Exception as exc:
                    reply = f"⚠️ LLM error: {exc}"
            st.markdown(reply)

            # ── URL import: scan reply for CSV/dataset links ───────────────
            _url_hits = re.findall(
                r"https?://[^\s\)\]>\"']+(?:\.csv|/download|datasets?/\d+|data_id=\d+)[^\s\)\]>\"']*",
                reply,
                re.IGNORECASE,
            )
            if _url_hits:
                st.caption("**Download detected dataset(s) from this reply:**")
                for _u in dict.fromkeys(_url_hits):  # deduplicate, preserve order
                    _short = _u if len(_u) <= 72 else _u[:69] + "…"
                    if st.button(f"⬇ {_short}", key=f"dl_{hash(_u)}", width='stretch'):
                        try:
                            import urllib.request as _ur
                            _raw, _ = _ur.urlretrieve(_u)
                            with open(_raw, "rb") as _fh:
                                _bytes = _fh.read()
                            _fname = _u.split("/")[-1].split("?")[0] or "dataset.csv"
                            st.download_button(
                                f"Save {_fname} to disk",
                                data=_bytes,
                                file_name=_fname,
                                mime="text/csv",
                                key=f"save_{hash(_u)}",
                            )
                        except Exception as _e:
                            st.warning(f"Could not fetch: {_e}")

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


# ── Batch test runner page ─────────────────────────────────────────────────

def render_batch_tab() -> None:
    from svm_studio.llm_advisor import advise_columns as _advise
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine
    import warnings as _w

    ui_shell.render_section_intro(
        "Batch Advisor Test",
        "Run the LLM advisor against all known datasets and rank results",
        "Tests every dataset in the registry: gets column advice, runs 50/50 hold-out + 10-fold CV "
        "on the best candidate, and ranks datasets by combined grade.",
    )

    with st.form("batch_form"):
        b_backend = st.radio(
            "LLM backend",
            ["Local / Remote LLM", "OpenAI", "Heuristic (no LLM)"],
            horizontal=True,
            key="batch_backend",
        )
        if b_backend == "OpenAI":
            b_api_key = st.text_input("OpenAI API key", type="password", key="batch_api_key")
            b_base_url: str | None = None
            b_model: str | None = None
        elif b_backend == "Local / Remote LLM":
            b_api_key = ""
            b_base_url, b_model = _llm_url_and_model_widgets(
                url_key="batch_base_url",
                model_key="batch_model",
                default_url=st.session_state.get("advisor_base_url_input") or os.environ.get("LLM_BASE_URL", ""),
            )
        else:
            b_api_key, b_base_url, b_model = "", None, None

        b_kernels = st.multiselect("Kernels", ["linear", "rbf", "poly"], default=["linear", "rbf"], key="batch_kernels")
        run_btn = st.form_submit_button("Run batch test", width='stretch')

    if not run_btn:
        ui_shell.render_state_panel("info", "Ready", "Configure the backend above and press Run.")
        return

    if not b_kernels:
        ui_shell.render_state_panel("warning", "No kernels selected", "Pick at least one kernel.")
        return

    # ── build dataset registry ─────────────────────────────────────────────
    def _make_registry() -> dict:
        _w.filterwarnings("ignore")
        reg = {}
        try:
            ds = load_iris(); df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df["species"] = [ds.target_names[t] for t in ds.target]; reg["Iris"] = df
        except Exception: pass
        try:
            ds = load_breast_cancer(); df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df["diagnosis"] = ["malignant" if t == 0 else "benign" for t in ds.target]; reg["Breast Cancer"] = df
        except Exception: pass
        try:
            ds = load_wine(); df = pd.DataFrame(ds.data, columns=ds.feature_names)
            df["wine_class"] = [ds.target_names[t] for t in ds.target]; reg["Wine UCI"] = df
        except Exception: pass
        for fname, key in [("cancer_uci.csv", "Cancer UCI"), ("fraud_openml.csv", "Fraud OpenML"),
                           ("wine_uci.csv", "Wine UCI CSV"), ("titanic_openml.csv", "Titanic")]:
            try:
                p = Path("data/external") / fname
                if p.exists():
                    df = pd.read_csv(p)
                    if fname == "fraud_openml.csv":
                        df = (df.groupby("Class", group_keys=False)
                                .apply(lambda g: g.sample(min(len(g), 2500), random_state=42))
                                .reset_index(drop=True))
                        df["Class"] = df["Class"].map({0: "legit", 1: "fraud"})
                    if fname == "cancer_uci.csv":
                        df = df.drop(columns=["target"], errors="ignore")
                    reg[key] = df
            except Exception: pass
        return reg

    registry = _make_registry()
    records, errors = [], []

    overall = st.status(f"Running {len(registry)} datasets …", expanded=True)
    for name, df in registry.items():
        overall.write(f"▶ {name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        try:
            advice = _advise(df, api_key=b_api_key or None, base_url=b_base_url or None, model=b_model or None)
            all_sets = advice.candidates if advice.candidates else [advice.feature_columns]
            seen_sets: list[list[str]] = []
            for c in all_sets:
                if c not in seen_sets:
                    seen_sets.append(c)
            best_result = None
            best_feats = advice.feature_columns
            for fs in seen_sets:
                try:
                    r = evaluate_column_set(df, target_column=advice.target_column,
                                            feature_columns=fs, kernels=b_kernels, n_cv_folds=10)
                    if best_result is None or r.grade > best_result.grade:
                        best_result = r; best_feats = fs
                except Exception: pass
            if best_result:
                records.append({
                    "Dataset": name, "Rows": df.shape[0], "Cols": df.shape[1],
                    "Target": advice.target_column, "Features": len(best_feats),
                    "Source": advice.source,
                    "Hold-out": round(best_result.holdout_accuracy, 4),
                    "CV mean": round(best_result.cv_mean_accuracy, 4),
                    "CV std": round(best_result.cv_std_accuracy, 4),
                    "Grade": round(best_result.grade, 4),
                })
        except Exception as exc:
            errors.append((name, str(exc)))

    overall.update(label="Batch run complete", state="complete", expanded=False)

    if errors:
        with st.expander(f"⚠ {len(errors)} error(s)", expanded=False):
            for nm, err in errors:
                st.error(f"**{nm}**: {err}")

    if not records:
        ui_shell.render_state_panel("error", "No results", "All datasets failed.")
        return

    results_df = pd.DataFrame(records).sort_values("Grade", ascending=False).reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    def _grade_color(val: float) -> str:
        if val >= 0.90: return "background-color: #1a3a1a; color: #6fcf6f"
        if val >= 0.70: return "background-color: #1a2a3a; color: #6fafcf"
        if val >= 0.55: return "background-color: #3a2a10; color: #cfaf6f"
        return "background-color: #3a1a1a; color: #cf6f6f"

    st.dataframe(
        results_df.style.applymap(_grade_color, subset=["Grade"]),
        width='stretch',
        hide_index=True,
    )

    best = results_df.iloc[0]
    ui_shell.render_state_panel(
        "success",
        f"Best: {best['Dataset']} — grade {best['Grade']:.4f}",
        f"Target: {best['Target']}  |  {best['Features']} features  |  "
        f"Hold-out {best['Hold-out']:.3f}  CV {best['CV mean']:.3f} ± {best['CV std']:.3f}",
    )

    st.download_button(
        "Download results CSV",
        data=results_df.to_csv(index=False),
        file_name="batch_advisor_results.csv",
        mime="text/csv",
        width='stretch',
    )


def main() -> None:
    st.set_page_config(page_title="SVM Data Studio", layout="wide", initial_sidebar_state="expanded")
    apply_style()
    ui_shell.inject_app_css()

    with st.sidebar:
        st.markdown("### Control Deck")
        st.markdown("Tune the input source here, then move through the workspace pages to model, mine, and visualize.")

    frame, source_name = load_data_source()

    # ── Row-count scalability warning ──────────────────────────────────────
    if frame is not None and len(frame) > 10_000:
        st.warning(
            f"**Large dataset detected: {len(frame):,} rows.** "
            "Streamlit rerenders the full page on every interaction. "
            "Interactive replotting may be slow past ~10k rows. "
            "SVM training with an RBF kernel also scales as O(n²) in memory — "
            "consider sampling before running the full benchmark.",
            icon="⚠️",
        )
    with st.sidebar:
        st.divider()
        with st.expander("Session settings", expanded=False):
            st.download_button(
                "Save current settings",
                data=session_settings_bytes(),
                file_name="svm_session.json",
                mime="application/json",
                on_click="ignore",
                width='stretch',
                help="Download a JSON snapshot of all current control settings.",
            )
            restore_file = st.file_uploader(
                "Restore settings from JSON",
                type=["json"],
                key="session_restore_uploader",
                label_visibility="collapsed",
            )
            if restore_file is not None:
                try:
                    config = json.loads(restore_file.getvalue())
                    apply_session_settings(config)
                    st.session_state.pop("session_restore_uploader", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not restore settings: {exc}")

        # ── Export workspace ───────────────────────────────────────────────
        if frame is not None:
            with st.expander("Export workspace", expanded=False):
                st.caption("Download all computed results, the session config, and a data preview in one ZIP.")
                st.download_button(
                    "Download workspace ZIP",
                    data=workspace_export_bytes(frame),
                    file_name="svm_workspace_export.zip",
                    mime="application/zip",
                    on_click="ignore",
                    width='stretch',
                )

    # Chat + Batch + Benchmark work without a dataset — register before the early-return guard
    def chat_page() -> None:
        render_chat_tab(frame, source_name)

    def batch_page() -> None:
        render_batch_tab()

    def benchmark_page() -> None:
        render_benchmark_tab()

    def history_page() -> None:
        render_history_tab()

    if frame is None or source_name is None:
        page = st.navigation(
            [
                st.Page(benchmark_page, title="Benchmark"),
                st.Page(history_page, title="Run History"),
                st.Page(batch_page, title="Batch Test"),
                st.Page(chat_page, title="Chat"),
            ]
        )
        page.run()
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
            ("Auto-Advisor", "Use the LLM Auto-Advisor to get a recommended column set and a mathematically graded score."),
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

    def advisor_page() -> None:
        render_advisor_tab(frame, source_name)

    def advanced_page() -> None:
        render_advanced_tab(frame, source_name)

    def visualizer_page() -> None:
        render_visualizer_tab(frame, source_name)

    page = st.navigation(
        [
            st.Page(data_page, title="Data Atlas"),
            st.Page(svm_page, title="SVM Lab"),
            st.Page(visualizer_page, title="SVM Visualizer"),
            st.Page(itemset_page, title="Itemsets"),
            st.Page(episode_page, title="Episodes"),
            st.Page(advisor_page, title="LLM Auto-Advisor"),
            st.Page(advanced_page, title="Advanced SVM"),
            st.Page(benchmark_page, title="Benchmark"),
            st.Page(history_page, title="Run History"),
            st.Page(batch_page, title="Batch Test"),
            st.Page(chat_page, title="Chat"),
        ]
    )
    page.run()


if __name__ == "__main__":
    main()
