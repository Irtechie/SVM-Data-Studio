# UI Release Checklist

Use this before calling the Streamlit workbench visually complete.

## Data Load

- Verify built-in demo loading works from the sidebar and preserves a valid target, feature set, and spotlight column.
- Verify CSV upload works with the default separator and with a non-comma separator when changed manually.
- Verify switching data sources clears stale result tables while keeping the control model valid.

## Workflow State

- Verify `SVM Lab` keeps the last successful run visible until a new run completes.
- Verify `Itemsets` keeps the last successful run visible until a new run completes.
- Verify `Episodes` keeps the last successful run visible until a new run completes.
- Verify each workflow shows an explicit info panel when controls differ from the last completed run.
- Verify each workflow still renders a clear state for empty, warning, success, and error cases.

## Layout And Responsiveness

- Review the shell at roughly `1440px`, `1280px`, `1024px`, and `768px` widths.
- Confirm the hero, quick-start strip, and stat cards do not clip or overlap at narrower widths.
- Confirm step strips and method boxes collapse cleanly to single-column layouts on smaller screens.
- Confirm data tables remain readable and scroll without breaking surrounding cards.

## Accessibility And Readability

- Tab through the sidebar, page navigation, forms, and download buttons and confirm the focus ring is always visible.
- Confirm the cyan/orange accent system does not reduce legibility for body copy or labels.
- Confirm muted text remains readable against panel backgrounds.
- Confirm math boxes and state-detail panels stay readable on both dense and sparse pages.
- Confirm motion remains minimal and does not create distracting shifts during interaction.

## Charts And Tables

- Confirm chart colors match the main UI shell and remain distinguishable across classes.
- Confirm confusion matrices, boundary charts, and bar charts keep readable labels and titles.
- Confirm scientific notation appears only where it improves comprehension.
- Confirm exported CSV files match the currently displayed raw result data.

## Copy And Product Feel

- Confirm every page reads from intent to controls to outputs without awkward jumps.
- Confirm button labels, panel titles, and captions use consistent language.
- Confirm the app still feels like one product rather than four separate demos.
