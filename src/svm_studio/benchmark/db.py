"""SQLite run history for the benchmark pipeline.

Stores every experiment run so users can compare across models, datasets, and
optional technique configurations.

Schema
------
runs
  id             INTEGER PRIMARY KEY
  timestamp      TEXT    ISO-8601
  dataset_name   TEXT
  llm_model      TEXT
  optional_techs TEXT    JSON list
  n_examples     INTEGER
  n_classes      INTEGER
  llm_accuracy   REAL
  control_accuracy REAL
  labeling_cost  REAL
  llm_agreement  REAL
  cv_variance_llm REAL
  most_common_error TEXT
  worst_class    TEXT
  best_class     TEXT
  report_text    TEXT    full Markdown report
  metrics_json   TEXT    full EvalResult-derived JSON for comparison view
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .svm_evaluator import EvalResult

_DEFAULT_DB = Path(__file__).resolve().parents[4] / "data" / "benchmark_runs.db"


def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            dataset_name    TEXT    NOT NULL,
            llm_model       TEXT    NOT NULL,
            optional_techs  TEXT    NOT NULL DEFAULT '[]',
            n_examples      INTEGER,
            n_classes       INTEGER,
            llm_accuracy    REAL,
            control_accuracy REAL,
            labeling_cost   REAL,
            llm_agreement   REAL,
            cv_variance_llm REAL,
            most_common_error TEXT,
            worst_class     TEXT,
            best_class      TEXT,
            report_text     TEXT,
            metrics_json    TEXT
        )
    """)
    conn.commit()


def save_run(
    result: EvalResult,
    optional_techniques: list[str] | None = None,
    report_text: str = "",
    db_path: Path | None = None,
) -> int:
    """Persist one experiment run and return its row ID."""
    conn = _get_conn(db_path)
    metrics = _result_to_dict(result)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    cursor = conn.execute(
        """
        INSERT INTO runs (
            timestamp, dataset_name, llm_model, optional_techs,
            n_examples, n_classes, llm_accuracy, control_accuracy,
            labeling_cost, llm_agreement, cv_variance_llm,
            most_common_error, worst_class, best_class,
            report_text, metrics_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ts,
            result.dataset_name,
            result.llm_model,
            json.dumps(optional_techniques or []),
            result.n_train + result.n_test,
            len(result.class_names),
            result.llm_metrics.test_accuracy,
            result.control_metrics.test_accuracy,
            result.labeling_cost,
            result.llm_agreement_rate,
            result.llm_metrics.cv_std_accuracy,
            result.most_common_error,
            result.worst_class,
            result.best_class,
            report_text,
            json.dumps(metrics),
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


def load_run(run_id: int, db_path: Path | None = None) -> dict[str, Any]:
    """Return all columns for a single run as a dict."""
    conn = _get_conn(db_path)
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        raise KeyError(f"No run with id={run_id}")
    cols = [d[0] for d in conn.execute("SELECT * FROM runs WHERE 0").description] if False else _run_columns()
    data = dict(zip(cols, row))
    data["optional_techs"] = json.loads(data.get("optional_techs", "[]"))
    data["metrics_json"] = json.loads(data.get("metrics_json", "{}"))
    return data


def list_runs(db_path: Path | None = None) -> pd.DataFrame:
    """Return a summary DataFrame of all stored runs, newest first."""
    conn = _get_conn(db_path)
    df = pd.read_sql_query(
        """
        SELECT id, timestamp, dataset_name, llm_model, optional_techs,
               n_examples, n_classes, llm_accuracy, control_accuracy,
               labeling_cost, llm_agreement, cv_variance_llm,
               most_common_error, worst_class, best_class
        FROM runs ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df


def compare_runs(run_ids: list[int], db_path: Path | None = None) -> pd.DataFrame:
    """Return a side-by-side comparison of the requested run IDs.

    Columns are the selected metrics; rows are the individual runs.
    """
    conn = _get_conn(db_path)
    placeholders = ",".join("?" * len(run_ids))
    df = pd.read_sql_query(
        f"""
        SELECT id, timestamp, dataset_name, llm_model,
               llm_accuracy, control_accuracy, labeling_cost,
               llm_agreement, cv_variance_llm,
               most_common_error, worst_class, best_class
        FROM runs WHERE id IN ({placeholders}) ORDER BY id
        """,
        conn,
        params=run_ids,
    )
    conn.close()
    return df


# ── helpers ────────────────────────────────────────────────────────────────

def _run_columns() -> list[str]:
    return [
        "id", "timestamp", "dataset_name", "llm_model", "optional_techs",
        "n_examples", "n_classes", "llm_accuracy", "control_accuracy",
        "labeling_cost", "llm_agreement", "cv_variance_llm",
        "most_common_error", "worst_class", "best_class",
        "report_text", "metrics_json",
    ]


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    return {
        "dataset_name": result.dataset_name,
        "llm_model": result.llm_model,
        "class_names": result.class_names,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "llm_test_accuracy": result.llm_metrics.test_accuracy,
        "llm_cv_mean": result.llm_metrics.cv_mean_accuracy,
        "llm_cv_std": result.llm_metrics.cv_std_accuracy,
        "llm_macro_f1": result.llm_metrics.test_macro_f1,
        "control_test_accuracy": result.control_metrics.test_accuracy,
        "control_cv_mean": result.control_metrics.cv_mean_accuracy,
        "control_cv_std": result.control_metrics.cv_std_accuracy,
        "control_macro_f1": result.control_metrics.test_macro_f1,
        "labeling_cost": result.labeling_cost,
        "llm_agreement_rate": result.llm_agreement_rate,
        "most_common_error": result.most_common_error,
        "worst_class": result.worst_class,
        "best_class": result.best_class,
    }
