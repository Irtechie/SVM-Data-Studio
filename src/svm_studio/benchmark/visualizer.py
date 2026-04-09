"""Benchmark-specific Plotly visualizations.

All charts return Plotly Figure objects so they render interactively in the
Streamlit app (``st.plotly_chart(fig, use_container_width=True)``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .svm_evaluator import EvalResult


# ── colour palette (matches the app's futurist-lab theme) ─────────────────
_CYAN = "#00d4ff"
_ORANGE = "#ff7c40"
_GREY = "#a0aabb"
_LIGHT = "#e8eaf0"
_TEMPLATE = "plotly_dark"


def plot_accuracy_comparison(result: EvalResult) -> go.Figure:
    """Side-by-side bar chart — LLM-trained vs control SVM accuracy with CV error bars."""
    models = ["LLM Labels", "True Labels (control)"]
    test_accs = [result.llm_metrics.test_accuracy, result.control_metrics.test_accuracy]
    cv_means = [result.llm_metrics.cv_mean_accuracy, result.control_metrics.cv_mean_accuracy]
    cv_stds = [result.llm_metrics.cv_std_accuracy, result.control_metrics.cv_std_accuracy]
    colours = [_CYAN, _ORANGE]

    fig = go.Figure()
    for i, (model, test_acc, cv_mean, cv_std, colour) in enumerate(
        zip(models, test_accs, cv_means, cv_stds, colours)
    ):
        fig.add_trace(go.Bar(
            name=model,
            x=[model],
            y=[test_acc],
            marker_color=colour,
            error_y=dict(type="data", array=[cv_std], visible=True, color=_LIGHT),
            text=[f"{test_acc:.3f}"],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            name=f"{model} CV mean",
            x=[model],
            y=[cv_mean],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=_LIGHT),
            showlegend=(i == 0),
        ))

    fig.update_layout(
        title=f"SVM Accuracy — LLM Labels vs Ground Truth (labeling cost: {result.labeling_cost:+.3f})",
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        barmode="group",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_confusion_matrices(result: EvalResult) -> go.Figure:
    """Side-by-side heatmaps for LLM-trained and control SVMs."""
    labels = result.class_names

    # Clip labels for display (long class names overflow)
    disp_labels = [str(l)[:15] for l in labels]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LLM-Trained SVM", "Control SVM (true labels)"],
    )

    for col_idx, (metrics, title) in enumerate([
        (result.llm_metrics, "LLM"),
        (result.control_metrics, "Control"),
    ], start=1):
        cm = metrics.confusion
        n = cm.shape[0]
        # Normalise rows
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        if n <= len(disp_labels):
            x_labs = disp_labels[:n]
            y_labs = disp_labels[:n]
        else:
            x_labs = [f"cls_{i}" for i in range(n)]
            y_labs = x_labs

        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=x_labs,
                y=y_labs,
                colorscale="Blues",
                showscale=(col_idx == 2),
                text=cm,
                texttemplate="%{text}",
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title="Confusion Matrices (normalised rows; counts in cells)",
        template=_TEMPLATE,
        height=max(400, 40 * len(labels)),
    )
    return fig


def plot_per_class_metrics(result: EvalResult) -> go.Figure:
    """Grouped bar chart — per-class precision and recall, LLM vs control."""
    llm_df = result.llm_metrics.class_report.copy()
    ctrl_df = result.control_metrics.class_report.copy()

    # Only keep real class rows (not macro/weighted avg)
    llm_df = llm_df[llm_df.index.isin(result.class_names)]
    ctrl_df = ctrl_df[ctrl_df.index.isin(result.class_names)]

    classes = llm_df.index.tolist()
    fig = go.Figure()

    for metric, colour, dash in [
        ("precision", _CYAN, "solid"),
        ("recall", _ORANGE, "dot"),
    ]:
        if metric not in llm_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=f"LLM {metric}",
            x=classes,
            y=llm_df[metric].values,
            marker_color=colour,
            opacity=0.85,
        ))
        if metric in ctrl_df.columns:
            fig.add_trace(go.Bar(
                name=f"Control {metric}",
                x=classes,
                y=ctrl_df[metric].values,
                marker_color=colour,
                opacity=0.45,
            ))

    fig.update_layout(
        title="Per-Class Precision & Recall — LLM vs Control",
        xaxis_title="Class",
        yaxis=dict(title="Score", range=[0, 1.05]),
        barmode="group",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_confidence_distribution(confidence_scores: list[float]) -> go.Figure:
    """Histogram of LLM label confidence scores."""
    fig = go.Figure(go.Histogram(
        x=confidence_scores,
        nbinsx=20,
        marker_color=_CYAN,
        opacity=0.8,
    ))
    mean_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    fig.add_vline(
        x=mean_conf, line_dash="dash", line_color=_ORANGE,
        annotation_text=f"mean = {mean_conf:.2f}", annotation_position="top right",
    )
    fig.update_layout(
        title="LLM Label Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        template=_TEMPLATE,
    )
    return fig


def plot_disagreement_table(disagreements: pd.DataFrame) -> go.Figure:
    """Table of examples where LLM and ground truth disagree."""
    if disagreements.empty:
        fig = go.Figure()
        fig.add_annotation(text="No disagreements — LLM labels matched ground truth perfectly.",
                           showarrow=False, font=dict(size=14))
        fig.update_layout(template=_TEMPLATE)
        return fig

    cols = list(disagreements.columns)
    fig = go.Figure(go.Table(
        header=dict(values=cols, fill_color="#1e2a3a", font=dict(color=_LIGHT, size=12), align="left"),
        cells=dict(
            values=[disagreements[c].astype(str).tolist() for c in cols],
            fill_color="#121a24",
            font=dict(color=_LIGHT, size=11),
            align="left",
        ),
    ))
    fig.update_layout(
        title=f"Disagreement Table ({len(disagreements)} examples where LLM ≠ ground truth)",
        template=_TEMPLATE,
    )
    return fig


def plot_cv_fold_comparison(result: EvalResult) -> go.Figure:
    """Line chart — per-fold accuracy for LLM-trained vs control SVM."""
    llm_folds = [f.accuracy for f in result.llm_metrics.cv_folds]
    ctrl_folds = [f.accuracy for f in result.control_metrics.cv_folds]
    folds = list(range(1, len(llm_folds) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=folds, y=llm_folds, mode="lines+markers",
                              name="LLM-trained SVM", line=dict(color=_CYAN)))
    fig.add_trace(go.Scatter(x=folds, y=ctrl_folds, mode="lines+markers",
                              name="Control SVM", line=dict(color=_ORANGE)))
    fig.update_layout(
        title="CV Fold Accuracy — LLM-Trained vs Control SVM",
        xaxis_title="Fold",
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        template=_TEMPLATE,
    )
    return fig
