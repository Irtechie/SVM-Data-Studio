from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA

from .svm_analysis import SvmStudyResult


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette=["#18c5d8", "#ff7a18", "#7ef2c8", "#295fe7", "#ffd166"],
        rc={
            "axes.facecolor": "#f8fbfd",
            "figure.facecolor": "#f8fbfd",
            "axes.edgecolor": "#bfd0da",
            "grid.color": "#d8e4eb",
            "axes.labelcolor": "#091722",
            "axes.titlecolor": "#091722",
            "xtick.color": "#425567",
            "ytick.color": "#425567",
            "text.color": "#091722",
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d5e0e7",
        },
    )
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 160


def plot_iris_kernel_boundaries(result: SvmStudyResult, output_path: Path) -> None:
    dataset = result.dataset
    palette = sns.color_palette("Set2", n_colors=len(dataset.target_names))
    figure, axes = plt.subplots(1, len(result.kernel_runs), figsize=(6 * len(result.kernel_runs), 5), constrained_layout=True)

    if len(result.kernel_runs) == 1:
        axes = [axes]

    x_min, x_max = dataset.X[:, 0].min() - 0.6, dataset.X[:, 0].max() + 0.6
    y_min, y_max = dataset.X[:, 1].min() - 0.6, dataset.X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for axis, run in zip(axes, result.kernel_runs):
        model = clone(run.estimator)
        model.fit(dataset.X, dataset.y)
        predictions = model.predict(grid).reshape(xx.shape)
        svc = model.named_steps["svc"]
        scaler = model.named_steps["scaler"]
        support_vectors = scaler.inverse_transform(svc.support_vectors_)

        axis.contourf(xx, yy, predictions, alpha=0.20, levels=np.arange(len(dataset.target_names) + 1) - 0.5, cmap="viridis")

        for class_index, class_name in enumerate(dataset.target_names):
            mask = dataset.y == class_index
            axis.scatter(
                dataset.X[mask, 0],
                dataset.X[mask, 1],
                color=palette[class_index],
                edgecolor="white",
                linewidth=0.5,
                s=60,
                label=class_name,
            )

        axis.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            s=150,
            label="support vectors",
        )
        axis.set_title(f"{run.kernel.upper()} | acc={run.test_accuracy:.3f}")
        axis.set_xlabel(dataset.feature_names[0])
        axis.set_ylabel(dataset.feature_names[1])

    axes[0].legend(loc="best", frameon=True)
    figure.suptitle("SVM Kernel Comparison on Iris", y=1.02)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(result: SvmStudyResult, output_path: Path) -> None:
    width = 8 if len(result.dataset.target_names) > 5 else 6
    figure, axis = plt.subplots(figsize=(width, 5.5), constrained_layout=True)
    sns.heatmap(
        result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=result.dataset.target_names,
        yticklabels=result.dataset.target_names,
        ax=axis,
    )
    axis.set_title(f"{result.dataset.title} Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_projection_with_support_vectors(result: SvmStudyResult, output_path: Path) -> None:
    dataset = result.dataset
    model = clone(result.selected_estimator)
    model.fit(dataset.X, dataset.y)

    scaled_X = model.named_steps["scaler"].transform(dataset.X)
    support_vectors = model.named_steps["svc"].support_vectors_
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled_X)
    support_coordinates = pca.transform(support_vectors)

    palette_name = "tab10" if len(dataset.target_names) > 5 else "Set2"
    palette = sns.color_palette(palette_name, n_colors=len(dataset.target_names))
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    for class_index, class_name in enumerate(dataset.target_names):
        mask = dataset.y == class_index
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=palette[class_index],
            alpha=0.68,
            s=36 if len(dataset.target_names) < 6 else 22,
            label=class_name,
        )

    axis.scatter(
        support_coordinates[:, 0],
        support_coordinates[:, 1],
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        s=130,
        label="support vectors",
    )
    axis.set_title(f"{dataset.title} PCA Projection with Support Vectors")
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy_overview(selected_results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    sns.barplot(
        data=selected_results,
        x="dataset_title",
        y="test_accuracy",
        hue="selected_kernel",
        palette="deep",
        ax=axis,
    )
    axis.set_title("Selected SVM Accuracy by Dataset")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.05)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_kernel_comparison(kernel_results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    sns.barplot(
        data=kernel_results,
        x="dataset_title",
        y="test_accuracy",
        hue="kernel",
        palette="muted",
        ax=axis,
    )
    axis.set_title("Kernel Comparison Across Datasets")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.05)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_support_vector_counts(selected_results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    sns.barplot(
        data=selected_results,
        x="dataset_title",
        y="support_vector_count",
        hue="dataset_title",
        dodge=False,
        palette="crest",
        legend=False,
        ax=axis,
    )
    axis.set_title("Support Vector Counts")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Number of support vectors")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_itemsets(itemset_results: pd.DataFrame, output_path: Path) -> None:
    filtered = itemset_results[itemset_results["length"] >= 2].copy()
    top = (
        filtered.sort_values(["dataset_name", "support", "length"], ascending=[True, False, False])
        .groupby("dataset_name", as_index=False)
        .head(5)
    )
    datasets = list(top["dataset_name"].drop_duplicates())
    figure, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), constrained_layout=True)

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        subset = top[top["dataset_name"] == dataset_name].sort_values("support", ascending=True)
        sns.barplot(data=subset, x="support", y="itemset", hue="itemset", dodge=False, palette="flare", legend=False, ax=axis)
        axis.set_title(dataset_name)
        axis.set_xlabel("Support")
        axis.set_ylabel("Itemset")

    figure.suptitle("Top Frequent Itemsets", y=1.02)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_episodes(episode_results: pd.DataFrame, output_path: Path) -> None:
    filtered = episode_results[episode_results["length"] >= 2].copy()
    top = (
        filtered.sort_values(["dataset_name", "support", "length"], ascending=[True, False, False])
        .groupby("dataset_name", as_index=False)
        .head(5)
    )
    datasets = list(top["dataset_name"].drop_duplicates())
    figure, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), constrained_layout=True)

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        subset = top[top["dataset_name"] == dataset_name].sort_values("support", ascending=True)
        sns.barplot(data=subset, x="support", y="episode", hue="episode", dodge=False, palette="mako", legend=False, ax=axis)
        axis.set_title(dataset_name)
        axis.set_xlabel("Support")
        axis.set_ylabel("Episode")

    figure.suptitle("Top Serial Episodes", y=1.02)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
