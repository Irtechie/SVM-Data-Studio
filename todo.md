# Purpose

Maintain one live runner for the professional look-and-feel backlog of the Support Vector Machine workspace.

## Objective

Turn the current Streamlit application into a polished, review-ready product surface with consistent visual design, clearer interaction states, stronger readability, and disciplined UX quality gates.

## Current Focus

Queue is clear. See `todo_done.md` for all completed work.

## Current Truth

- The app has a custom hero, styled sidebar, card-based sections, geometry views, and boxed math explanations.
- The active UI shell routes through `src/svm_studio/ui_shell.py` (split into `_ui_css.py`, `_ui_components.py`, `_ui_hero.py`).
- The app includes a quick-start strip plus page-level flow strips for new users.
- The visual direction is a futurist lab system with metallic light surfaces, cyan/orange accenting, and matching chart styling.
- The workbench sanitizes session state on dataset changes and marks stale outputs when controls change.
- A release checklist exists in `ui_release_checklist.md` for final browser-side review.
- The SVM Lab uses interactive Plotly charts for kernel comparison, feature importance, and PCA projection.
- `src/svm_studio/llm_advisor.py` provides heuristic + OpenAI-based column advice; endpoint auto-detected (OpenAI-compat vs. Ollama-native).
- `src/svm_studio/custom_analysis.py` exports `EvaluationResult` and `evaluate_column_set` for 50/50 + 10-fold CV grading.
- LLM advisor uses a 4-step grounded reasoning protocol and returns up to 10 candidate feature sets; the app grades all and keeps the best.
- A Chat page is available at all times with dataset schema, advisor recommendation, and grade injected as system context.
- Wine UCI and Titanic OpenML datasets are in `data/external/`; `run_advisor_test.py` covers 6 datasets.
- Session settings can be saved/restored (JSON); all computed results exportable as a ZIP.
- `src/svm_studio/benchmark/` — 12-module LLM-vs-ground-truth benchmark pipeline (bench-01 through bench-15). 43/43 tests pass.
- Nav has 10 pages; `requirements.txt` includes `plotly`, `openai`, and `fpdf2`.
- `render_annotated_formula()` is available from `ui_shell`; CSS in `_ui_css.py`.  SVM Lab shows the kernel-specific decision formula with each term underlined + labelled below, plus actual parameter values in scientific notation. LLM Explain button is now on all 4 result pages (SVM Lab, Itemsets, Episodes, Advanced SVM).

## Success Criteria

- The UI feels visually consistent across `Data Atlas`, `SVM Lab`, `Itemsets`, and `Episodes`.
- Key user states are explicit: loading, empty, error, success, and blocked input.
- Tables, plots, cards, and math boxes look intentional rather than stitched together.
- The app remains legible and usable at common desktop and tablet breakpoints.
- Basic accessibility expectations are met for contrast, focus visibility, and keyboard flow.

## Active Tasks

All completed work has been moved to `todo_done.md`.

- (none — queue is clear)

## Parked / Cold Storage

- (empty)

## Blocked

- [!] No active blockers.

## Blocked

- [!] No active blockers.

## Work Log

- 2026-03-29: Created the live professional-UI runner and imported the shared `todo_rules.md` convention into the repo root.
- 2026-04-06: Promoted all six Cold Storage items to Active Tasks (ui-cold-01 through ui-cold-06).
- 2026-04-06: Added new LLM-Graded Feature Selection workstream (llm-feat-01 through llm-feat-05).
- 2026-04-06: Completed all actionable tasks in one burn-down pass. All LLM workstream tasks done; all ready UI-debt tasks done. 6 existing tests still pass.
- 2026-04-06: Completed ui-cold-03 and ui-cold-06. Split ui_shell.py into _ui_css.py / _ui_components.py / _ui_hero.py with a 40-line facade. Removed 499 lines of dead code from streamlit_app.py. All 6 tests still pass.
- 2026-04-07: Added LLM-vs-Ground-Truth Benchmark Pipeline workstream (bench-01 through bench-15).
- 2026-04-07: Completed all 15 bench tasks. 43/43 tests pass. Nav expanded to 10 pages.
- 2026-04-08: Added render_annotated_formula component (CSS + _ui_components.py + ui_shell.py re-export). SVM Lab now shows kernel-specific prediction formula with underlined annotated terms and scientific-notation parameter note. LLM Explain button added to SVM Lab (now present on all 4 result pages).
- 2026-04-07: Completed all 15 bench tasks. New package: src/svm_studio/benchmark/ (12 modules + optional/ subpackage). 37 new unit tests. Nav expanded to 10 pages. 43/43 tests pass.
- 2026-04-07: Completed LLM Backend & Advisor Hardening, Chat Page, Next/Open, Advanced SVM & LLM Explainer workstreams.
- 2026-04-07: All completed work moved to todo_done.md. Queue is clear.
