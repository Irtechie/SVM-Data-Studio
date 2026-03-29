# Purpose

Maintain one live runner for the professional look-and-feel backlog of the Support Vector Machine workspace.

## Objective

Turn the current Streamlit application into a polished, review-ready product surface with consistent visual design, clearer interaction states, stronger readability, and disciplined UX quality gates.

## Current Focus

Professional UI and UX hardening for the interactive Streamlit experience in `streamlit_app.py`.

## Current Truth

- The app already has a custom hero, styled sidebar, card-based sections, geometry views, and boxed math explanations.
- The active UI shell now routes through `src/svm_studio/ui_shell.py`, which owns the shared CSS theme, hero, stat grid, callout, method box, and state surfaces used by `streamlit_app.py`.
- The app now includes a quick-start strip plus page-level flow strips so new users can see the intended workflow before interacting with controls.
- The visual direction is now a more deliberate futurist lab system with metallic light surfaces, cyan/orange accenting, and chart styling that matches the shell.
- The workbench now sanitizes session state on dataset changes and clearly marks stale outputs when controls differ from the last completed run.
- A release checklist now exists in `ui_release_checklist.md` for final browser-side review.
- Historical completed items were moved to `todo_done.md` so this runner stays focused on actionable work.
- Responsive and accessibility-oriented code-level passes are now recorded in the shared UI shell.
- The repo now includes `todo_rules.md` as the local convention file for this runner.
- Reference patterns for this runner are being drawn from official Streamlit docs on multipage apps, forms, session state, status elements, and download behavior, plus Plotly's science and engineering app examples.

## Success Criteria

- The UI feels visually consistent across `Data Atlas`, `SVM Lab`, `Itemsets`, and `Episodes`.
- Key user states are explicit: loading, empty, error, success, and blocked input.
- Tables, plots, cards, and math boxes look intentional rather than stitched together.
- The app remains legible and usable at common desktop and tablet breakpoints.
- Basic accessibility expectations are met for contrast, focus visibility, and keyboard flow.

## Active Tasks

Completed history is now tracked in `todo_done.md`.

- [x] The current professional-UI backlog is complete. New work should be promoted from `Parked / Cold Storage` or added as a new active task after review.

## Parked / Cold Storage

- [-] Add a PCA or UMAP projection tab for high-dimensional boundary interpretation.
  - Task ID: ui-cold-01
  - Discovered from: user request about hyperplane interpretation

- [-] Add a branded export flow for report snapshots and presentation-ready summaries.
  - Task ID: ui-cold-02
  - Discovered from: current output and presentation workflow

- [-] Borrow richer interaction patterns from Plotly science and engineering examples once the core Streamlit shell stabilizes.
  - Task ID: ui-cold-05
  - Discovered from: Plotly examples reference pass

- [-] Extract the UI shell into smaller view modules once the visual direction stabilizes.
  - Task ID: ui-cold-03
  - Discovered from: ui-polish-01

- [-] Remove the legacy inline UI helper definitions from `streamlit_app.py` once the remaining page logic is split out safely.
  - Task ID: ui-cold-06
  - Discovered from: ui-polish-01

- [-] Add session save and restore for long exploratory workflows.
  - Task ID: ui-cold-04
  - Discovered from: multi-step data exploration use case

## Blocked

- [!] No active blockers at the moment.

## Work Log

- 2026-03-29: Created the live professional-UI runner and imported the shared `todo_rules.md` convention into the repo root.
- 2026-03-29: Source rules file was found at `C:\Users\marowe\OneDrive - Microsoft\Microsoft Teams Chat Files\todo_rules.md` rather than the Downloads folder.
- 2026-03-29: Expanded the runner using ideas from official Streamlit docs and Plotly example galleries to make the professional-UI queue more concrete.
- 2026-03-29: Completed `ui-polish-10` and `ui-polish-11` by adding visible status surfaces for long-running actions and non-rerunning download behavior.
- 2026-03-29: Completed `ui-polish-08` by switching the app from tabs to official Streamlit page navigation for a clearer workflow.
- 2026-03-29: Completed `ui-polish-01` by moving the active UI theme and shared render primitives into `src/svm_studio/ui_shell.py` and routing `streamlit_app.py` through that shared layer.
- 2026-03-29: Completed `ui-polish-02` by replacing generic empty, warning, and error fallbacks with product-style state panels across the app.
- 2026-03-29: Completed `ui-polish-03` by adding shared page-flow strips so each page reads in a clearer top-to-bottom sequence.
- 2026-03-29: Completed `ui-polish-12` by adding a global quick-start strip beneath the hero and tying first-run guidance into the page flow.
- 2026-03-29: Completed `ui-polish-04` by pushing the app into a stronger futurist-lab theme and aligning the chart styling with the main UI shell.
- 2026-03-29: Completed `ui-polish-09` by sanitizing control state on dataset changes and explicitly marking stale result views when controls change after a run.
- 2026-03-29: Completed `ui-polish-05` by adding smaller-width layout tuning for the hero, chips, step strips, grids, and content padding.
- 2026-03-29: Completed `ui-polish-06` by improving focus visibility, reduced-motion handling, contrast, and readability in the shared shell.
- 2026-03-29: Completed `ui-polish-07` by adding `ui_release_checklist.md` as the final browser-side visual review guide.
- 2026-03-29: Moved completed runner history into `todo_done.md` to keep the live queue focused on actionable items.
