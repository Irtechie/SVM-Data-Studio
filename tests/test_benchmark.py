"""Tests for the benchmark pipeline modules.

Uses sklearn's iris / wine datasets as fixtures so no external dependencies
(ucimlrepo, HuggingFace) are required.  LLM calls are mocked via a simple
monkeypatch so the tests are fully offline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svm_studio.benchmark.dataset_loader import (
    DatasetLoader,
    StandardDataset,
    _infer_label_column,
    _infer_text_column,
)
from svm_studio.benchmark.llm_labeler import (
    LLMLabeler,
    _parse_label_response,
    _build_prompt,
)
from svm_studio.benchmark.svm_evaluator import SVMEvaluator
from svm_studio.benchmark.prompts import (
    LABELING_PROMPT_TABULAR,
    LABELING_PROMPT_TEXT,
    LABELING_PROMPT_UNCERTAINTY_RETRY,
    REPORT_GENERATION_PROMPT,
    OPTIONAL_TECHNIQUE_EXPLANATION_PROMPT,
)
from svm_studio.benchmark.visualizer import (
    plot_accuracy_comparison,
    plot_confidence_distribution,
    plot_disagreement_table,
)
from svm_studio.benchmark.db import save_run, list_runs, compare_runs, load_run


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iris_dataset() -> StandardDataset:
    loader = DatasetLoader(max_rows=100)
    return loader.load("sklearn", "iris")


@pytest.fixture(scope="module")
def wine_dataset() -> StandardDataset:
    loader = DatasetLoader(max_rows=80)
    return loader.load("sklearn", "wine")


def _make_fake_llm(dataset: StandardDataset, correct_rate: float = 0.70):
    """Return a chat_completion mock that answers correctly ~correct_rate of the time."""
    import json
    import random
    rng = random.Random(42)

    def _fake_chat_completion(messages, **kwargs):
        # Extract class names from the prompt content
        prompt = messages[-1]["content"]
        classes = dataset.class_names
        # Decide correct or wrong
        true_class = classes[rng.randint(0, len(classes) - 1)]
        if rng.random() < correct_rate:
            label = true_class
        else:
            wrong = [c for c in classes if c != true_class]
            label = wrong[rng.randint(0, len(wrong) - 1)] if wrong else true_class
        return json.dumps({"label": label, "confidence": round(rng.uniform(0.5, 1.0), 2), "reasoning": "Test reasoning."})

    return _fake_chat_completion


# ── DatasetLoader tests ───────────────────────────────────────────────────

class TestDatasetLoader:

    def test_load_iris_shape(self, iris_dataset: StandardDataset):
        assert len(iris_dataset.X) == 100
        assert len(iris_dataset.y) == 100

    def test_iris_class_names(self, iris_dataset: StandardDataset):
        assert len(iris_dataset.class_names) == 3
        assert "setosa" in " ".join(iris_dataset.class_names).lower()

    def test_iris_feature_names(self, iris_dataset: StandardDataset):
        assert len(iris_dataset.feature_names) == 4

    def test_iris_standard_fields(self, iris_dataset: StandardDataset):
        assert iris_dataset.task_type == "classification"
        assert iris_dataset.data_type == "tabular"
        assert iris_dataset.n_examples == 100
        assert iris_dataset.n_classes == 3

    def test_load_wine(self, wine_dataset: StandardDataset):
        assert len(wine_dataset.X) == 80
        assert wine_dataset.n_classes == 3

    def test_load_digits(self):
        loader = DatasetLoader(max_rows=50)
        ds = loader.load("sklearn", "digits")
        assert ds.data_type == "image_features"
        assert len(ds.class_names) == 10
        assert len(ds.X) == 50

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            DatasetLoader().load("badSource", "iris")

    def test_unknown_sklearn_name_raises(self):
        with pytest.raises(ValueError, match="Unknown sklearn dataset"):
            DatasetLoader().load("sklearn", "nonexistent_dataset")

    def test_csv_load(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "label": ["x", "y", "x"]})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        ds = DatasetLoader(max_rows=None).load("csv", str(csv_path), target_column="label")
        assert list(ds.class_names) == ["x", "y"]
        assert len(ds.X.columns) == 2

    def test_infer_label_column_by_name(self):
        df = pd.DataFrame({"text": ["a", "b"], "label": ["x", "y"], "extra": [1, 2]})
        assert _infer_label_column(df) == "label"

    def test_infer_text_column_by_name(self):
        df = pd.DataFrame({"text": ["hello world", "foo bar"], "label": [0, 1]})
        assert _infer_text_column(df) == "text"


# ── LLMLabeler tests ──────────────────────────────────────────────────────

class TestLLMLabeler:

    def test_label_returns_correct_length(self, iris_dataset: StandardDataset):
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   side_effect=_make_fake_llm(iris_dataset)):
            labeler = LLMLabeler(model="test-model")
            labeled = labeler.label(iris_dataset, max_examples=10)
        assert len(labeled.y_llm) == 10
        assert len(labeled.records) == 10

    def test_label_agreement_rate_in_range(self, iris_dataset: StandardDataset):
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   side_effect=_make_fake_llm(iris_dataset, correct_rate=0.8)):
            labeler = LLMLabeler(model="test-model")
            labeled = labeler.label(iris_dataset, max_examples=30)
        assert 0.0 <= labeled.agreement_rate <= 1.0

    def test_fallback_on_bad_response(self, iris_dataset: StandardDataset):
        """If LLM always returns garbage, all labels fall back to 'unknown'."""
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   return_value="this is not JSON at all"):
            labeler = LLMLabeler(model="test-model", max_retries=0)
            labeled = labeler.label(iris_dataset, max_examples=5)
        assert all(r.fallback_used for r in labeled.records)
        assert all(lbl == "unknown" for lbl in labeled.y_llm)

    def test_parse_clean_json(self):
        raw = '{"label": "setosa", "confidence": 0.9, "reasoning": "petal width is small"}'
        result = _parse_label_response(raw, ["setosa", "versicolor", "virginica"])
        assert result is not None
        label, conf, reasoning = result
        assert label == "setosa"
        assert conf == 0.9

    def test_parse_strips_markdown_fences(self):
        raw = "```json\n{\"label\": \"versicolor\", \"confidence\": 0.7, \"reasoning\": \"mid range\"}\n```"
        result = _parse_label_response(raw, ["setosa", "versicolor", "virginica"])
        assert result is not None
        assert result[0] == "versicolor"

    def test_parse_case_insensitive(self):
        raw = '{"label": "SETOSA", "confidence": 0.85, "reasoning": "small petals"}'
        result = _parse_label_response(raw, ["setosa", "versicolor"])
        assert result is not None
        assert result[0] == "setosa"

    def test_parse_invalid_label_returns_none(self):
        raw = '{"label": "completely_wrong", "confidence": 0.9, "reasoning": "test"}'
        result = _parse_label_response(raw, ["setosa", "versicolor"])
        assert result is None

    def test_build_prompt_tabular(self, iris_dataset: StandardDataset):
        row = iris_dataset.X.iloc[0]
        prompt = _build_prompt(row, iris_dataset)
        assert "setosa" in prompt or "class" in prompt.lower()
        assert LABELING_PROMPT_TABULAR.split("{")[0] in prompt or "label" in prompt.lower()


# ── SVMEvaluator tests ────────────────────────────────────────────────────

class TestSVMEvaluator:

    @pytest.fixture(scope="class")
    def evaluation(self, iris_dataset: StandardDataset):
        """Run a full evaluation with mocked ~70% agreement labels."""
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   side_effect=_make_fake_llm(iris_dataset, correct_rate=0.70)):
            labeler = LLMLabeler(model="test-model")
            labeled = labeler.label(iris_dataset, max_examples=90)
        evaluator = SVMEvaluator(kernel="rbf", n_folds=3)
        return evaluator.evaluate(iris_dataset, labeled)

    def test_eval_result_fields(self, evaluation):
        assert 0.0 <= evaluation.llm_metrics.test_accuracy <= 1.0
        assert 0.0 <= evaluation.control_metrics.test_accuracy <= 1.0

    def test_confusion_matrix_shape(self, evaluation):
        n = evaluation.llm_metrics.confusion.shape[0]
        assert n >= 2  # at least 2 classes present

    def test_labeling_cost_type(self, evaluation):
        assert isinstance(evaluation.labeling_cost, float)

    def test_cv_folds_count(self, evaluation):
        assert len(evaluation.llm_metrics.cv_folds) == 3
        assert len(evaluation.control_metrics.cv_folds) == 3

    def test_class_report_has_classes(self, evaluation):
        classes_in_report = set(evaluation.llm_metrics.class_report.index)
        assert len(classes_in_report) >= 2

    def test_disagreements_frame_columns(self, evaluation):
        if not evaluation.disagreements.empty:
            assert "true_label" in evaluation.disagreements.columns
            assert "llm_label" in evaluation.disagreements.columns


# ── Prompts template tests ────────────────────────────────────────────────

class TestPrompts:

    def test_tabular_prompt_formats(self):
        rendered = LABELING_PROMPT_TABULAR.format(
            dataset_name="iris",
            dataset_description="flower classification",
            class_names="setosa, versicolor, virginica",
            feature_dict="  sepal_length: 5.1\n  petal_width: 0.2",
        )
        assert "iris" in rendered
        assert "setosa" in rendered
        assert "JSON" in rendered

    def test_text_prompt_formats(self):
        rendered = LABELING_PROMPT_TEXT.format(
            dataset_name="imdb",
            task_description="sentiment classification",
            class_names="positive, negative",
            text="This movie was great.",
        )
        assert "positive" in rendered
        assert "JSON" in rendered

    def test_uncertainty_retry_prompt_formats(self):
        rendered = LABELING_PROMPT_UNCERTAINTY_RETRY.format(
            dataset_name="iris",
            class_names="setosa, versicolor",
            example_data="sepal: 5.1",
            previous_label="setosa",
            previous_confidence=0.55,
        )
        assert "setosa" in rendered
        assert "changed" in rendered.lower()

    def test_report_prompt_formats(self):
        rendered = REPORT_GENERATION_PROMPT.format(
            dataset_name="iris", dataset_description="flowers", data_type="tabular",
            n_examples=100, n_classes=3, class_names="setosa, versicolor, virginica",
            llm_model="gemma3", optional_techniques="none",
            llm_agreement_pct=72.0, svm_llm_accuracy=81.0, svm_control_accuracy=95.0,
            gap=14.0, variance_llm=0.03, variance_control=0.02,
            most_common_error="versicolor → virginica", worst_class="versicolor",
            best_class="setosa", optional_results_block="No optional techniques.",
        )
        assert "iris" in rendered
        assert "81.0" in rendered

    def test_optional_technique_prompt_formats(self):
        rendered = OPTIONAL_TECHNIQUE_EXPLANATION_PROMPT.format(
            technique_name="Universum SVM",
            dataset_description="flowers",
            technique_results="accuracy improved by 0.02",
        )
        assert "Universum" in rendered


# ── Visualizer smoke tests ────────────────────────────────────────────────

class TestVisualizer:

    @pytest.fixture(scope="class")
    def eval_result(self, iris_dataset: StandardDataset):
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   side_effect=_make_fake_llm(iris_dataset, correct_rate=0.75)):
            labeler = LLMLabeler(model="test-model")
            labeled = labeler.label(iris_dataset, max_examples=60)
        evaluator = SVMEvaluator(kernel="rbf", n_folds=3)
        return evaluator.evaluate(iris_dataset, labeled)

    def test_accuracy_chart_returns_figure(self, eval_result):
        import plotly.graph_objects as go
        fig = plot_accuracy_comparison(eval_result)
        assert isinstance(fig, go.Figure)

    def test_confidence_chart_returns_figure(self):
        import plotly.graph_objects as go
        fig = plot_confidence_distribution([0.6, 0.7, 0.8, 0.9])
        assert isinstance(fig, go.Figure)

    def test_disagreement_table_empty(self):
        import plotly.graph_objects as go
        fig = plot_disagreement_table(pd.DataFrame())
        assert isinstance(fig, go.Figure)


# ── DB tests ─────────────────────────────────────────────────────────────

class TestDB:

    @pytest.fixture
    def tmp_db(self, tmp_path: Path) -> Path:
        return tmp_path / "test_runs.db"

    @pytest.fixture
    def eval_result(self, iris_dataset: StandardDataset):
        with patch("svm_studio.benchmark.llm_labeler.chat_completion",
                   side_effect=_make_fake_llm(iris_dataset, correct_rate=0.75)):
            labeler = LLMLabeler(model="test-model")
            labeled = labeler.label(iris_dataset, max_examples=40)
        evaluator = SVMEvaluator(kernel="rbf", n_folds=3)
        return evaluator.evaluate(iris_dataset, labeled)

    def test_save_and_list(self, eval_result, tmp_db: Path):
        run_id = save_run(eval_result, ["none"], "Test report", tmp_db)
        assert isinstance(run_id, int)
        df = list_runs(tmp_db)
        assert len(df) == 1
        assert df["id"].iloc[0] == run_id

    def test_load_run(self, eval_result, tmp_db: Path):
        run_id = save_run(eval_result, [], "Report text", tmp_db)
        data = load_run(run_id, tmp_db)
        assert data["dataset_name"] == eval_result.dataset_name
        assert data["report_text"] == "Report text"

    def test_compare_runs(self, eval_result, tmp_db: Path):
        id1 = save_run(eval_result, [], "report1", tmp_db)
        id2 = save_run(eval_result, ["universum_svm"], "report2", tmp_db)
        cmp = compare_runs([id1, id2], tmp_db)
        assert len(cmp) == 2
        assert "llm_accuracy" in cmp.columns

    def test_load_nonexistent_raises(self, tmp_db: Path):
        # Create the DB first
        from svm_studio.benchmark.db import _get_conn
        _get_conn(tmp_db).close()
        with pytest.raises(KeyError):
            load_run(9999, tmp_db)
