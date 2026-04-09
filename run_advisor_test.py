"""
Advisor grading test.

Runs the LLM column advisor (heuristic path unless OPENAI_API_KEY is set)
against every dataset, evaluates each recommendation with:
  - 50 / 50 hold-out split
  - 10-fold stratified CV

Results are ranked by combined grade.  The best column set is printed last.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from svm_studio.custom_analysis import EvaluationResult, evaluate_column_set
from svm_studio.llm_advisor import ColumnAdvice, advise_columns


# ── dataset registry ───────────────────────────────────────────────────────

def _iris_df() -> pd.DataFrame:
    ds = load_iris()
    df = pd.DataFrame(ds.data, columns=ds.feature_names)
    df["species"] = [ds.target_names[t] for t in ds.target]
    return df


def _breast_cancer_df() -> pd.DataFrame:
    ds = load_breast_cancer()
    df = pd.DataFrame(ds.data, columns=ds.feature_names)
    df["diagnosis"] = ["malignant" if t == 0 else "benign" for t in ds.target]
    return df


def _cancer_uci_df() -> pd.DataFrame:
    df = pd.read_csv("data/external/cancer_uci.csv")
    # Drop the numeric target code so the advisor only sees the text label
    return df.drop(columns=["target"], errors="ignore")


def _fraud_df() -> pd.DataFrame:
    # Use a stratified 5 000-row sample so SVM stays fast
    df = pd.read_csv("data/external/fraud_openml.csv")
    # Class is heavily imbalanced; sample keeps the ratio
    df_sampled = (
        df.groupby("Class", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 2500), random_state=42))
        .reset_index(drop=True)
    )
    df_sampled["Class"] = df_sampled["Class"].map({0: "legit", 1: "fraud"})
    return df_sampled


def _wine_df() -> pd.DataFrame:
    ds = load_wine()
    df = pd.DataFrame(ds.data, columns=ds.feature_names)
    df["wine_class"] = [ds.target_names[t] for t in ds.target]
    return df


def _titanic_df() -> pd.DataFrame:
    df = pd.read_csv("data/external/titanic_openml.csv")
    return df


def _csv(fname: str) -> pd.DataFrame:
    """Load a CSV from data/external/ — generic loader for clean datasets."""
    return pd.read_csv(f"data/external/{fname}")


DATASETS: dict[str, pd.DataFrame] = {
    "Iris (3-class)":             _iris_df(),
    "Breast Cancer Wisconsin":    _breast_cancer_df(),
    "Cancer UCI CSV":             _cancer_uci_df(),
    "Credit Card Fraud (sample)": _fraud_df(),
    "Wine UCI (13 features)":     _wine_df(),
    "Titanic (OpenML)":           _titanic_df(),
    "Banknote Authentication":    _csv("banknote_auth.csv"),
    "Diabetes Pima Indians":      _csv("diabetes_pima.csv"),
    "German Credit":              _csv("credit_german.csv"),
    "Telecom Churn":              _csv("churn_telecom.csv"),
    "Steel Plates Fault":         _csv("steel_plates.csv"),
    "Handwritten Digits":         _csv("digits_sklearn.csv"),
    "HAR Smartphone (8k)": _csv("har_smartphone.csv"),
    "Sensorless Drive":           _csv("sensorless_drive.csv"),
}


# ── result container ───────────────────────────────────────────────────────

@dataclass
class RunRecord:
    dataset_name: str
    advice: ColumnAdvice
    grade: EvaluationResult


# ── helpers ────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _grade_label(g: float) -> str:
    if g >= 0.90:
        return "A  (Excellent)"
    if g >= 0.80:
        return "B  (Good)"
    if g >= 0.70:
        return "C  (Fair)"
    if g >= 0.55:
        return "D  (Marginal)"
    return "F  (Poor)"


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    records: list[RunRecord] = []
    errors: list[tuple[str, str]] = []

    base_url = os.environ.get("LLM_BASE_URL", "") or None
    model = os.environ.get("LLM_MODEL", "") or None
    backend_label = f"Ollama @ {base_url}" if base_url else ("OpenAI" if os.environ.get("OPENAI_API_KEY") else "heuristic")

    print("\n" + "=" * 72)
    print("  SVM ADVISOR GRADING TEST  —  10-fold CV + 50/50 hold-out")
    print(f"  Backend: {backend_label}" + (f"  model={model}" if model else ""))
    print("=" * 72)

    for name, df in DATASETS.items():
        print(f"\n[{name}]  {df.shape[0]:,} rows x {df.shape[1]} cols")
        try:
            advice = advise_columns(df, base_url=base_url, model=model)
            print(f"  Advisor      : {advice.source} ({advice.model_used})")
            print(f"  Target       : {advice.target_column}  "
                  f"({df[advice.target_column].nunique()} classes)")
            print(f"  Features     : {len(advice.feature_columns)} columns")
            print(f"  Rationale    : {advice.rationale[:110]}")

            print("  Running evaluations …", end="", flush=True)
            result = evaluate_column_set(
                df,
                target_column=advice.target_column,
                feature_columns=advice.feature_columns,
                kernels=["linear", "rbf"],
                n_cv_folds=10,
            )
            print(" done")

            print(f"  Hold-out acc : {result.holdout_accuracy:.4f}  "
                  f"macro-F1 {result.holdout_macro_f1:.4f}  "
                  f"{_bar(result.holdout_accuracy)}")
            print(f"  10-fold CV   : {result.cv_mean_accuracy:.4f} "
                  f"+/- {result.cv_std_accuracy:.4f}  "
                  f"{_bar(result.cv_mean_accuracy)}")
            print(f"  Grade        : {result.grade:.4f}  "
                  f"{_grade_label(result.grade)}")

            records.append(RunRecord(dataset_name=name, advice=advice, grade=result))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append((name, str(exc)))

    # ── ranking table ───────────────────────────────────────────────────────
    if not records:
        print("\nNo successful runs to rank.")
        return

    ranked = sorted(records, key=lambda r: r.grade.grade, reverse=True)

    print("\n" + "=" * 72)
    print("  RANKING  (highest combined grade first)")
    print("=" * 72)
    print(f"  {'Rank':<5} {'Dataset':<32} {'Target':<22} {'Grade':>7}  {'Label'}")
    print("  " + "-" * 68)
    for rank, rec in enumerate(ranked, start=1):
        print(
            f"  {rank:<5} {rec.dataset_name:<32} "
            f"{rec.advice.target_column:<22} "
            f"{rec.grade.grade:>7.4f}  "
            f"{_grade_label(rec.grade.grade)}"
        )

    # ── best set details ────────────────────────────────────────────────────
    best = ranked[0]
    print("\n" + "=" * 72)
    print("  BEST COLUMN SET")
    print("=" * 72)
    print(f"  Dataset   : {best.dataset_name}")
    print(f"  Target    : {best.advice.target_column}")
    print(f"  Features  : {', '.join(best.advice.feature_columns[:8])}"
          + (" …" if len(best.advice.feature_columns) > 8 else ""))
    print(f"  Grade     : {best.grade.grade:.4f}  {_grade_label(best.grade.grade)}")
    print(f"  Hold-out  : acc={best.grade.holdout_accuracy:.4f}  "
          f"F1={best.grade.holdout_macro_f1:.4f}")
    cv = best.grade
    print(f"  CV        : {cv.cv_mean_accuracy:.4f} +/- {cv.cv_std_accuracy:.4f}  "
          f"({cv.n_cv_folds} folds)")
    print(f"  Rationale : {best.advice.rationale}")
    print()

    if errors:
        print("Datasets that failed:")
        for ds_name, msg in errors:
            print(f"  {ds_name}: {msg}")


if __name__ == "__main__":
    main()
