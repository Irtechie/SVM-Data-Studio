"""Report generator for the benchmark pipeline.

Calls the LLM once with all experiment metrics to produce a plain-language
report.  Assembles the final Markdown document and optionally exports to PDF
via ``fpdf2`` (if available) or plain Markdown bytes as fallback.
"""
from __future__ import annotations

import datetime
from typing import Any

from ..llm_advisor import chat_completion
from .prompts import OPTIONAL_TECHNIQUE_EXPLANATION_PROMPT, REPORT_GENERATION_PROMPT
from .svm_evaluator import EvalResult

try:
    from fpdf import FPDF  # type: ignore[import-untyped]
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False


def generate_report(
    result: EvalResult,
    optional_results: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    dataset_description: str = "",
    class_names_str: str = "",
) -> str:
    """Generate a plain-language Markdown report via LLM.

    Parameters
    ----------
    result : EvalResult
        Full evaluation output from SVMEvaluator.
    optional_results : dict, optional
        Mapping of technique name → result summary string.
    Returns markdown string.
    """
    optional_results = optional_results or {}
    enabled = list(optional_results.keys()) if optional_results else ["none"]

    # Build optional results block
    optional_blocks: list[str] = []
    for tech_name, tech_summary in optional_results.items():
        optional_blocks.append(f"### {tech_name}\n{tech_summary}")
    optional_results_block = "\n\n".join(optional_blocks) or "No optional techniques enabled."

    llm_acc_pct = round(result.llm_metrics.test_accuracy * 100, 1)
    ctrl_acc_pct = round(result.control_metrics.test_accuracy * 100, 1)
    gap_pct = round(result.labeling_cost * 100, 1)
    agreement_pct = round(result.llm_agreement_rate * 100, 1)

    main_prompt = REPORT_GENERATION_PROMPT.format(
        dataset_name=result.dataset_name,
        dataset_description=dataset_description or result.dataset_name,
        data_type="tabular",
        n_examples=result.n_train + result.n_test,
        n_classes=len(result.class_names),
        class_names=class_names_str or ", ".join(result.class_names),
        llm_model=result.llm_model,
        optional_techniques=", ".join(enabled),
        llm_agreement_pct=agreement_pct,
        svm_llm_accuracy=llm_acc_pct,
        svm_control_accuracy=ctrl_acc_pct,
        gap=gap_pct,
        variance_llm=round(result.llm_metrics.cv_std_accuracy, 4),
        variance_control=round(result.control_metrics.cv_std_accuracy, 4),
        most_common_error=result.most_common_error,
        worst_class=result.worst_class,
        best_class=result.best_class,
        optional_results_block=optional_results_block,
    )

    messages = [{"role": "user", "content": main_prompt}]
    try:
        core_report = chat_completion(messages, api_key=api_key, model=model, base_url=base_url)
    except Exception as exc:
        core_report = f"*Report generation failed: {exc}*"

    # Append per-technique explanations
    tech_sections: list[str] = []
    for tech_name, tech_summary in optional_results.items():
        tech_prompt = OPTIONAL_TECHNIQUE_EXPLANATION_PROMPT.format(
            technique_name=tech_name,
            dataset_description=dataset_description or result.dataset_name,
            technique_results=tech_summary,
        )
        try:
            explanation = chat_completion(
                [{"role": "user", "content": tech_prompt}],
                api_key=api_key, model=model, base_url=base_url,
            )
        except Exception:
            explanation = "*Explanation unavailable.*"
        tech_sections.append(f"\n## Optional Technique: {tech_name}\n\n{explanation}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        f"# Benchmark Report: {result.dataset_name}\n\n"
        f"*Generated {timestamp} | LLM: {result.llm_model} | "
        f"Dataset: {result.n_train + result.n_test} examples*\n\n"
        f"---\n\n"
    )
    full_report = header + core_report + "".join(tech_sections)
    return full_report


def report_to_pdf_bytes(markdown_text: str) -> bytes:
    """Convert a Markdown report to PDF bytes.

    Uses fpdf2 if available, otherwise returns UTF-8 Markdown bytes so
    callers always receive a downloadable artifact.
    """
    if not _FPDF_AVAILABLE:
        return markdown_text.encode("utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, stripped[2:], ln=True)
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, stripped[3:], ln=True)
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("**") and stripped.endswith("**"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, stripped.strip("*"), ln=True)
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("---"):
            pdf.ln(2)
            pdf.set_draw_color(100, 100, 100)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
        elif stripped == "":
            pdf.ln(4)
        else:
            # Strip remaining inline markdown
            text = stripped.replace("**", "").replace("*", "").replace("`", "")
            pdf.multi_cell(0, 6, text)

    return bytes(pdf.output())
