"""Public facade for the SVM Studio UI shell.

All callers should continue to import from this module::

    from svm_studio import ui_shell
    ui_shell.inject_app_css()
    ui_shell.render_hero(source_name, frame)

The implementation is split across three private sub-modules:

* ``_ui_css``        — APP_CSS constant and ``inject_app_css``
* ``_ui_components`` — card/panel/strip render helpers
* ``_ui_hero``       — the full-width hero section
"""

from __future__ import annotations

from ._ui_components import (
    render_annotated_formula,
    render_callout,
    render_method_box,
    render_section_intro,
    render_stat_grid,
    render_state_panel,
    render_step_strip,
)
from ._ui_css import APP_CSS, inject_app_css
from ._ui_hero import render_hero

__all__ = [
    "APP_CSS",
    "inject_app_css",
    "render_annotated_formula",
    "render_callout",
    "render_hero",
    "render_method_box",
    "render_section_intro",
    "render_stat_grid",
    "render_state_panel",
    "render_step_strip",
]

