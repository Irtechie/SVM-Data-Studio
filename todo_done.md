# Completed Work Ledger

Historical completions moved out of `todo.md` so the live runner stays focused on active UI work.

## Completed Tasks

- [x] ~~Margin quality transparency: report F1 and calibration check alongside accuracy in benchmark leaderboard~~
  - Task ID: review-01
  - Completed: 2026-04-09
  - Validation: Benchmark stat grid now shows Macro-F1 for both LLM and control SVMs. State panel fires a calibration warning when accuracy − F1 > 0.10 (likely majority-class collapse).

- [x] ~~Row-count scalability warning: notify users when dataset exceeds 10k rows~~
  - Task ID: review-02
  - Completed: 2026-04-09
  - Validation: Warning banner shown after `load_data_source()` when `len(frame) > 10_000`, explaining Streamlit rerender cost and O(n²) SVM scaling.

- [x] ~~Guidance quality guardrails: canned core-concept explanations, LLM restricted to dataset-specific commentary~~
  - Task ID: review-03
  - Completed: 2026-04-09
  - Validation: `_EXPLAIN_SYSTEM_PROMPT` rewritten to ban technique re-explanation; LLM told to focus only on the specific numbers and dataset domain in front of it.

- [x] ~~"Why did this fail?" advisor mode: diagnose bad SVM runs with the LLM~~
  - Task ID: review-04
  - Completed: 2026-04-09
  - Validation: `diagnose_bad_result_stream()` added to `llm_advisor.py`. "Why did this fail?" button appears on SVM Lab when test accuracy < 75%; streams a diagnosis of the top SVM failure modes with concrete fixes.

- [x] ~~Adopt the shared todo rules and seed a single live runner for UI polish work~~
  - Task ID: ui-runner-01
  - Completed: 2026-03-29
  - Validation: copied `todo_rules.md` into the repo root and created `todo.md` as the single live execution queue.

- [x] ~~Consolidate the visual system into clearly named UI primitives and theme tokens.~~
  - Task ID: ui-polish-01
  - Completed: 2026-03-29
  - Validation: the active UI shell now routes through `src/svm_studio/ui_shell.py`, which owns the shared CSS theme, hero, stat grid, callout, method box, and state surfaces used by `streamlit_app.py`.

- [x] ~~Professionalize the app state model for empty, loading, error, and success views.~~
  - Task ID: ui-polish-02
  - Depends on: ui-polish-01
  - Completed: 2026-03-29
  - Validation: generic `st.info`, `st.warning`, and `st.error` fallbacks were replaced with shared product-style state panels across dataset loading, SVM analysis, geometry constraints, itemset mining, and episode mining.

- [x] ~~Tighten the content hierarchy so every tab has a clean narrative flow from controls to explanation to output.~~
  - Task ID: ui-polish-03
  - Depends on: ui-polish-01
  - Completed: 2026-03-29
  - Validation: each page now opens with an explicit step-strip that makes the workflow read from intent to controls to outputs instead of dropping users into dense control blocks.

- [x] ~~Evaluate a shift from tab-heavy flow to page-level navigation using official Streamlit multipage patterns.~~
  - Task ID: ui-polish-08
  - Depends on: ui-polish-03
  - Completed: 2026-03-29
  - Validation: the app now uses `st.navigation` page-level flow for `Data Atlas`, `SVM Lab`, `Itemsets`, and `Episodes` instead of a dense tab stack.

- [x] ~~Bring tables and charts up to presentation quality.~~
  - Task ID: ui-polish-04
  - Depends on: ui-polish-01
  - Completed: 2026-03-29
  - Validation: the app now uses a coordinated futurist lab theme across cards, table surfaces, and charts, with deliberate cyan/orange palette choices and consistent figure styling.

- [x] ~~Harden session-state behavior so data source, computed results, and page context persist cleanly across reruns.~~
  - Task ID: ui-polish-09
  - Depends on: ui-polish-02, ui-polish-08
  - Completed: 2026-03-29
  - Validation: the app now sanitizes control state on dataset changes, preserves valid selections across reruns, and explicitly labels stale result views when controls no longer match the last completed run.

- [x] ~~Run a responsive QA pass and tune the layout for narrower widths.~~
  - Task ID: ui-polish-05
  - Depends on: ui-polish-02, ui-polish-03, ui-polish-04
  - Completed: 2026-03-29
  - Validation: the shared CSS now adds breakpoint-specific layout tuning for the hero, chips, step strips, method grids, and content padding so smaller widths collapse cleanly.

- [x] ~~Run an accessibility and readability pass.~~
  - Task ID: ui-polish-06
  - Depends on: ui-polish-02, ui-polish-03
  - Completed: 2026-03-29
  - Validation: visible focus rings, reduced-motion handling, stronger muted-text contrast, and roomier reading line-height were added across the shared UI shell.

- [x] ~~Create a final visual QA checklist for release readiness.~~
  - Task ID: ui-polish-07
  - Depends on: ui-polish-04, ui-polish-05, ui-polish-06
  - Completed: 2026-03-29
  - Validation: added `ui_release_checklist.md` with review items for data loading, workflow state, responsiveness, accessibility, charts, exports, and copy consistency.

- [x] ~~Add explicit progress and status surfaces for long-running actions.~~
  - Task ID: ui-polish-10
  - Depends on: ui-polish-02
  - Completed: 2026-03-29
  - Validation: SVM analysis, itemset mining, and episode mining now use explicit `st.status` progress surfaces with visible complete and error states.

- [x] ~~Prevent avoidable reruns during downloads and stage export preparation more deliberately.~~
  - Task ID: ui-polish-11
  - Depends on: ui-polish-04
  - Completed: 2026-03-29
  - Validation: download actions now use cached CSV conversion and `on_click="ignore"` to avoid unnecessary reruns during export.

- [x] ~~Add a stronger quick-start layer inspired by polished gallery apps.~~
  - Task ID: ui-polish-12
  - Depends on: ui-polish-03
  - Completed: 2026-03-29
  - Validation: the app now includes a global quick-start strip beneath the hero plus page-level flow strips so first-time users can see where to start and what each page is for.

## Validation Snapshots

- 2026-03-29: `.\.venv\Scripts\python -m compileall streamlit_app.py src tests`
- 2026-03-29: `.\.venv\Scripts\python -c "import streamlit_app"`
- 2026-03-29: `.\.venv\Scripts\python -m unittest discover -s tests`

---

## UI Debt (promoted from Cold Storage 2026-04-06)

- [x] ~~Add a PCA or UMAP projection tab for high-dimensional boundary interpretation.~~
  - Task ID: ui-cold-01
  - Completed: PCA projection added as a third geometry mode in SVM Lab (interactive Plotly scatter).

- [x] ~~Add a branded export flow for report snapshots and presentation-ready summaries.~~
  - Task ID: ui-cold-02
  - Completed: "Export workspace" expander in sidebar downloads a ZIP containing kernel_results.csv, feature_importance.csv, itemset_results.csv, episode_results.csv, advisor_grade.json, session_config.json, and data_preview.csv.

- [x] ~~Borrow richer interaction patterns from Plotly science and engineering examples.~~
  - Task ID: ui-cold-05
  - Completed: Kernel comparison and feature importance charts replaced with interactive Plotly figures (hover, error bars, live filtering). PCA projection also uses Plotly.

- [x] ~~Extract the UI shell into smaller view modules.~~
  - Task ID: ui-cold-03
  - Completed: `ui_shell.py` is now a 40-line re-export facade; implementation split into `_ui_css.py` (CSS + inject), `_ui_components.py` (6 render helpers), `_ui_hero.py` (hero section).

- [x] ~~Remove the legacy inline UI helper definitions from `streamlit_app.py`.~~
  - Task ID: ui-cold-06
  - Completed: deleted ~500 lines of dead duplicate code (`inject_app_css`, `render_section_intro`, `render_stat_grid`, `render_callout`, `render_method_box`, `render_hero`) — none were called; all routing already went through `ui_shell.*`.

- [x] ~~Add session save and restore for long exploratory workflows.~~
  - Task ID: ui-cold-04
  - Completed: "Session settings" sidebar expander with download (JSON) and restore (file upload) controls.

---

## LLM-Graded Feature Selection (completed 2026-04-06)

- [x] ~~Add LLM column advisor backend — given a CSV schema, call an LLM to recommend the target column and feature columns with a brief rationale.~~
  - Task ID: llm-feat-01
  - Completed: `src/svm_studio/llm_advisor.py` — heuristic fallback + OpenAI gpt-4o-mini path.

- [x] ~~Add 50/50 train/test evaluation runner.~~
  - Task ID: llm-feat-02
  - Completed: `evaluate_column_set` in `custom_analysis.py` uses `run_custom_svm_analysis(test_size=0.50)` for the hold-out pass.

- [x] ~~Add 10-fold stratified CV evaluation runner.~~
  - Task ID: llm-feat-03
  - Completed: same `evaluate_column_set` runs `StratifiedKFold(n_splits=10)` + `cross_val_score` on the full dataset after the hold-out pass.

- [x] ~~Build grading surface — display LLM recommended columns with hold-out score, CV score, and a combined grade.~~
  - Task ID: llm-feat-04
  - Completed: grade rendered as `stat_grid` + `state_panel` on the LLM Auto-Advisor page, colour-coded by quality.

- [x] ~~Add "LLM Auto-Advisor" page to the Streamlit app.~~
  - Task ID: llm-feat-05
  - Completed: fifth page in `st.navigation`, with form, advice panel, grade button, and "Apply to SVM Lab" shortcut.

---

## LLM Backend & Advisor Hardening (completed 2026-04-07)

- [x] ~~LLM endpoint auto-detection — probe `/v1/models` (OpenAI-compat) and `/api/tags` (Ollama-native); dispatch to the right caller without manual toggle.~~
  - Task ID: llm-backend-01
  - Works with: OpenAI, vLLM, LM Studio, llama.cpp, TRT-LLM, Ollama native and /v1.

- [x] ~~Configurable LLM endpoint — base URL, model name, and API key all overridable via UI or env vars (`LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY`).~~
  - Task ID: llm-backend-02

- [x] ~~Upgrade advisor prompt to 4-step grounded reasoning protocol — target identification, column elimination with cited rules, evidence-based feature scoring, combinatorial candidate generation.~~
  - Task ID: llm-prompt-01

- [x] ~~Allow up to 10 candidate feature sets — combinations derived from schema evidence, not guessing.~~
  - Task ID: llm-prompt-02

- [x] ~~Candidate review UI — per-candidate expanders with column list and reasoning, individual checkboxes, Grade Selected and Yolo buttons.~~
  - Task ID: llm-ui-01

- [x] ~~Download and register Wine UCI (178 rows, 13 numeric features, 3-class) and Titanic OpenML (1309 rows, mixed types, binary) as real datasets in `data/external/`.~~
  - Task ID: llm-data-01

- [x] ~~Extend `run_advisor_test.py` to cover all 6 datasets including Wine and Titanic.~~
  - Task ID: llm-data-02

---

## Chat Page (completed 2026-04-07)

- [x] ~~Add `chat_completion()` to `llm_advisor.py` — multi-turn, same backend detection as advisor, separate chat system prompt.~~
  - Task ID: chat-01

- [x] ~~Add Chat nav page — available before any dataset is loaded, `st.chat_input` pinned to bottom, full message history.~~
  - Task ID: chat-02

- [x] ~~Inject workspace context into chat system message — current dataset schema, active advisor recommendation, and latest grade.~~
  - Task ID: chat-03

- [x] ~~Add starter prompt chips — 5 clickable one-liners shown when chat history is empty.~~
  - Task ID: chat-04

- [x] ~~Embed dataset discovery knowledge in chat system prompt — 10 real repos with direct URLs, SVM task-type table, per-suggestion format rules.~~
  - Task ID: chat-05

---

## Next / Open (completed 2026-04-07)

- [x] ~~Batch test runner UI page — run advisor against all known datasets from within the Streamlit app, show ranked results table.~~
  - Task ID: batch-ui-01
  - Completed: `render_batch_tab()` page in Streamlit nav.

- [x] ~~Download-and-import flow from chat — when LLM suggests a dataset URL, add a "Download this dataset" button that fetches and stages the CSV directly into the app.~~
  - Task ID: chat-import-01
  - Completed: URL scanner in `render_chat_tab()` detects dataset links in LLM replies with download buttons.

- [x] ~~Candidate reasoning display — store per-candidate `reasoning` text from LLM JSON and surface it in the expander.~~
  - Task ID: llm-ui-02
  - Completed: `candidate_reasoning` field populated from LLM JSON and shown in expanders.

---

## Advanced SVM & LLM Explainer (completed 2026-04-08)

- [x] ~~Active Learning (uncertainty sampling) backend — margin-based query loop, learning curve, baseline comparison.~~
  - Task ID: adv-svm-01
  - Completed: `run_active_learning()` in `src/svm_studio/advanced_svm.py`.

- [x] ~~Universum SVM backend — synthetic midpoint/noise/convex examples, standard vs universum comparison.~~
  - Task ID: adv-svm-02
  - Completed: `run_universum_svm()` in `src/svm_studio/advanced_svm.py`.

- [x] ~~LLM result explainer — generic `explain_result()` function that asks the LLM to interpret any technique's results in plain English with counter-examples.~~
  - Task ID: adv-svm-03
  - Completed: `explain_result()` + `_EXPLAIN_SYSTEM_PROMPT` in `llm_advisor.py`.

- [x] ~~Advanced SVM page — Active Learning and Universum SVM tabs with forms, results display, learning curve chart, comparison table, and LLM Explain buttons.~~
  - Task ID: adv-svm-04
  - Completed: `render_advanced_tab()` in `streamlit_app.py`, wired as 8th nav page.

- [x] ~~LLM Explain buttons on Itemset and Episode pages — post-result button that sends top patterns to `explain_result()`.~~
  - Task ID: adv-svm-05
  - Completed: `_llm_explain_button()` helper added after result sections in both tabs.

---

## LLM-vs-Ground-Truth Benchmark Pipeline (completed 2026-04-07)

Goal: extend the workbench into a full LLM labeling benchmark — LLM labels a dataset, SVM evaluates label quality against ground truth, optional advanced techniques diagnose failure modes, and a final LLM-generated report explains everything in plain English.

All 15 tasks completed. New package: `src/svm_studio/benchmark/` (12 modules + `optional/` subpackage). 37 new unit tests; 43 total passing.

- [x] ~~`prompts.py` — 5 LLM prompt templates optimised for 30B models.~~ Task ID: bench-01
- [x] ~~`dataset_loader.py` — `DatasetLoader.load(source, name)` → `StandardDataset`. Sources: sklearn (11 datasets), OpenML (any id), ucimlrepo (guarded), HuggingFace (guarded), local CSV.~~ Task ID: bench-02
- [x] ~~`llm_labeler.py` — `LLMLabeler` with retry/fallback, confidence tracking, batching, wraps `chat_completion`.~~ Task ID: bench-03
- [x] ~~`svm_evaluator.py` — `SVMEvaluator`: dual SVM (LLM vs control), 5-fold CV, 20% holdout, confusion matrices, disagreements, labeling cost.~~ Task ID: bench-04
- [x] ~~`optional/uncertainty_sampling.py` — margin-based UBS, LLM re-query, before/after metrics.~~ Task ID: bench-05
- [x] ~~`optional/universum_svm.py` — wraps `run_universum_svm`, applies to LLM-labeled data.~~ Task ID: bench-06
- [x] ~~`optional/itemset_mining.py` — error vs correct pattern comparison via Apriori.~~ Task ID: bench-07
- [x] ~~`optional/episode_mining.py` — sequence pattern comparison, auto-detects sequence columns.~~ Task ID: bench-08
- [x] ~~`visualizer.py` — 6 Plotly charts: accuracy bar, CV fold line, confusion heatmaps, per-class bars, confidence histogram, disagreement table.~~ Task ID: bench-09
- [x] ~~`report_generator.py` — LLM report via `REPORT_GENERATION_PROMPT`, MD + PDF (fpdf2).~~ Task ID: bench-10
- [x] ~~`db.py` — SQLite run history: `save_run`, `load_run`, `list_runs`, `compare_runs`.~~ Task ID: bench-11
- [x] ~~`experiment.py` — `run_experiment(source, name, llm_model, optional_techniques=[])` orchestrator with progress callback.~~ Task ID: bench-12
- [x] ~~Streamlit "Benchmark" page — dataset picker, LLM config, optional checkboxes, progress bar, full results display.~~ Task ID: bench-13
- [x] ~~Streamlit "Run History" page — list + multi-select compare + report viewer.~~ Task ID: bench-14
- [x] ~~`requirements.txt` updated — `fpdf2` added; optional deps commented with install instructions.~~ Task ID: bench-15
