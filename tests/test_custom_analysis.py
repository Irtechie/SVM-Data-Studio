from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm_studio.custom_analysis import fit_custom_svm_estimator, run_custom_svm_analysis
from svm_studio.episode_mining import (
    build_episode_dataset_from_event_columns,
    build_episode_dataset_from_sequence_column,
)
from svm_studio.itemset_mining import build_transactions_from_frame


class CustomAnalysisTests(unittest.TestCase):
    def test_custom_svm_analysis_returns_feature_ranking(self) -> None:
        iris = load_iris(as_frame=True)
        frame = iris.frame.copy()
        frame["target_name"] = frame["target"].map(lambda value: iris.target_names[int(value)])

        result = run_custom_svm_analysis(
            frame=frame,
            target_column="target_name",
            feature_columns=iris.feature_names,
            kernels=["linear", "rbf"],
            test_size=0.25,
        )

        self.assertIn(result.selected_kernel, {"linear", "rbf"})
        self.assertEqual(set(result.feature_importance["feature"]), set(iris.feature_names))
        self.assertFalse(result.kernel_results.empty)

    def test_build_transactions_from_frame_handles_numeric_and_categorical(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [22, 35, 35, 48],
                "plan": ["basic", "pro", "pro", "basic"],
                "target": ["stay", "buy", "buy", "stay"],
            }
        )

        transactions = build_transactions_from_frame(frame, ["age", "plan"], target_column="target")

        self.assertEqual(len(transactions), 4)
        self.assertTrue(any("plan=pro" in transaction for transaction in transactions))
        self.assertTrue(any("target=buy" in transaction for transaction in transactions))

    def test_build_episode_datasets_from_columns_and_sequences(self) -> None:
        frame = pd.DataFrame(
            {
                "sequence": ["A,B,C", "A,B,D", "A,C,D"],
                "step_1": ["A", "A", "A"],
                "step_2": ["B", "B", "C"],
                "step_3": ["C", "D", "D"],
            }
        )

        sequence_dataset = build_episode_dataset_from_sequence_column(
            frame=frame,
            sequence_column="sequence",
            name="demo",
            min_support=0.5,
            max_span=3,
            separator=",",
        )
        column_dataset = build_episode_dataset_from_event_columns(
            frame=frame,
            event_columns=["step_1", "step_2", "step_3"],
            name="demo",
            min_support=0.5,
            max_span=3,
        )

        self.assertEqual(len(sequence_dataset.sequences), 3)
        self.assertEqual(len(column_dataset.sequences), 3)
        self.assertEqual(sequence_dataset.sequences[0], ["A", "B", "C"])
        self.assertEqual(column_dataset.sequences[1], ["A", "B", "D"])

    def test_fit_custom_svm_estimator_returns_fitted_pipeline(self) -> None:
        iris = load_iris(as_frame=True)
        frame = iris.frame.copy()
        frame["target_name"] = frame["target"].map(lambda value: iris.target_names[int(value)])

        model, X, y = fit_custom_svm_estimator(
            frame=frame,
            target_column="target_name",
            feature_columns=iris.feature_names[:2],
            kernel="linear",
            best_params={"svc__C": 1.0},
        )

        predictions = model.predict(X.head(5))
        self.assertEqual(len(predictions), 5)
        self.assertEqual(len(X.columns), 2)
        self.assertEqual(y.nunique(), 3)


if __name__ == "__main__":
    unittest.main()
