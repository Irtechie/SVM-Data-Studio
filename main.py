from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm_studio.datasets import load_svm_datasets
from svm_studio.episode_mining import episodes_to_frame, load_episode_datasets, mine_all_episode_datasets
from svm_studio.itemset_mining import itemsets_to_frame, mine_itemsets_for_datasets
from svm_studio.svm_analysis import kernel_runs_frame, run_all_svm_studies, selected_runs_frame
from svm_studio.visualization import (
    apply_style,
    plot_accuracy_overview,
    plot_confusion_matrix,
    plot_episodes,
    plot_iris_kernel_boundaries,
    plot_itemsets,
    plot_kernel_comparison,
    plot_projection_with_support_vectors,
    plot_support_vector_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SVM and mining study pipeline.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where CSV summaries and charts will be written.",
    )
    return parser.parse_args()


def _top_rows(frame, label_column: str, group_column: str) -> str:
    lines: list[str] = []
    top = (
        frame.sort_values([group_column, "support", "length"], ascending=[True, False, False])
        .groupby(group_column, as_index=False)
        .head(3)
    )

    for group_name in top[group_column].drop_duplicates():
        lines.append(f"### {group_name}")
        subset = top[top[group_column] == group_name]
        for _, row in subset.iterrows():
            lines.append(f"- {row[label_column]} (support={row['support']:.3f}, count={int(row['count'])})")
        lines.append("")

    return "\n".join(lines).strip()


def build_summary(selected_results, itemset_results, episode_results) -> str:
    lines = [
        "# Project Summary",
        "",
        "## Selected SVM models",
        "",
    ]

    for _, row in selected_results.iterrows():
        lines.append(
            f"- {row['dataset_title']}: kernel={row['selected_kernel']}, "
            f"accuracy={row['test_accuracy']:.3f}, macro_f1={row['macro_f1']:.3f}, "
            f"support_vectors={int(row['support_vector_count'])}"
        )

    lines.extend(
        [
            "",
            "## Strongest frequent itemsets",
            "",
            _top_rows(itemset_results[itemset_results["length"] >= 2], "itemset", "dataset_name"),
            "",
            "## Strongest serial episodes",
            "",
            _top_rows(episode_results[episode_results["length"] >= 2], "episode", "dataset_name"),
            "",
            "## Interpretation",
            "",
            "- Simple data makes the SVM geometry easy to inspect visually.",
            "- Medium and complex data show why scaling, tuning, and kernel choice matter.",
            "- Itemsets expose co-occurring attribute patterns after discretization.",
            "- Episodes expose ordered behavior that classification alone would miss.",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    svm_datasets = load_svm_datasets()
    svm_results = run_all_svm_studies(svm_datasets)
    kernel_results = kernel_runs_frame(svm_results)
    selected_results = selected_runs_frame(svm_results)

    itemset_results = itemsets_to_frame(mine_itemsets_for_datasets(svm_datasets))
    episode_results = episodes_to_frame(mine_all_episode_datasets(load_episode_datasets()))

    kernel_results.to_csv(output_dir / "svm_kernel_results.csv", index=False)
    selected_results.to_csv(output_dir / "svm_selected_results.csv", index=False)
    itemset_results.to_csv(output_dir / "itemset_results.csv", index=False)
    episode_results.to_csv(output_dir / "episode_results.csv", index=False)

    for result in svm_results:
        safe_name = result.dataset.key
        plot_confusion_matrix(result, output_dir / f"{safe_name}_confusion_matrix.png")
        if result.dataset.is_two_dimensional:
            plot_iris_kernel_boundaries(result, output_dir / f"{safe_name}_kernel_boundaries.png")
        else:
            plot_projection_with_support_vectors(result, output_dir / f"{safe_name}_support_vectors.png")

    plot_accuracy_overview(selected_results, output_dir / "svm_accuracy_overview.png")
    plot_kernel_comparison(kernel_results, output_dir / "svm_kernel_comparison.png")
    plot_support_vector_counts(selected_results, output_dir / "svm_support_vector_counts.png")
    plot_itemsets(itemset_results, output_dir / "itemset_overview.png")
    plot_episodes(episode_results, output_dir / "episode_overview.png")

    summary = build_summary(selected_results, itemset_results, episode_results)
    (output_dir / "project_summary.md").write_text(summary, encoding="utf-8")

    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
