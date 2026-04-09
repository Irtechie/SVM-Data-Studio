"""Reusable Streamlit render components for the SVM Studio shell."""

from __future__ import annotations

import html

import streamlit as st


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <p class="section-kicker">{html.escape(kicker)}</p>
            <h2 class="section-title">{html.escape(title)}</h2>
            <p class="section-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_grid(cards: list[tuple[str, str, str]]) -> None:
    body = "".join(
        f"""
        <div class="stat-card">
            <p class="stat-label">{html.escape(label)}</p>
            <p class="stat-value">{html.escape(value)}</p>
            <p class="stat-note">{html.escape(note)}</p>
        </div>
        """
        for label, value, note in cards
    )
    st.markdown(f'<div class="stat-grid">{body}</div>', unsafe_allow_html=True)


def render_callout(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="callout-card">
            <p class="callout-title">{html.escape(title)}</p>
            <p class="callout-copy">{html.escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_state_panel(kind: str, title: str, copy: str, detail: str | None = None) -> None:
    allowed_kinds = {"info", "success", "warning", "error"}
    state_kind = kind if kind in allowed_kinds else "info"
    detail_html = f'<p class="state-detail">{html.escape(detail)}</p>' if detail else ""
    st.markdown(
        f"""
        <section class="state-panel" data-kind="{state_kind}">
            <p class="state-label">{html.escape(state_kind)}</p>
            <p class="state-title">{html.escape(title)}</p>
            <p class="state-copy">{html.escape(copy)}</p>
            {detail_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_step_strip(title: str, steps: list[tuple[str, str]]) -> None:
    body = "".join(
        f"""
        <div class="step-card">
            <p class="step-index">Step {index}</p>
            <p class="step-name">{html.escape(name)}</p>
            <p class="step-copy">{html.escape(copy)}</p>
        </div>
        """
        for index, (name, copy) in enumerate(steps, start=1)
    )
    st.markdown(
        f"""
        <section class="step-strip">
            <p class="step-strip-title">{html.escape(title)}</p>
            <div class="step-strip-grid">{body}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_annotated_formula(
    title: str,
    segments: list[tuple[str, str | None]],
    note: str = "",
) -> None:
    """Render a formula with underlined, labelled terms.

    Parameters
    ----------
    title:
        Kicker shown above the formula row (displayed in monospace caps).
    segments:
        List of ``(text, label)`` pairs.  When *label* is ``None`` the text is
        rendered as plain formula connector text (e.g. ``" + "``).  When
        *label* is a string the term is underlined and the label appears below.
    note:
        Optional line of context (e.g. active parameter values) shown beneath
        the formula in monospace.
    """
    parts: list[str] = []
    for text, label in segments:
        esc_text = html.escape(text)
        if label is None:
            parts.append(f'<span class="af-plain">{esc_text}</span>')
        else:
            esc_label = html.escape(label)
            parts.append(
                f'<span class="af-term">'
                f'<span class="af-text">{esc_text}</span>'
                f'<span class="af-label">{esc_label}</span>'
                f'</span>'
            )

    formula_html = "".join(parts)
    note_html = f'<p class="af-note">{html.escape(note)}</p>' if note else ""
    st.markdown(
        f"""
        <div class="annotated-formula-block">
            <p class="af-title">{html.escape(title)}</p>
            <div class="af-row">{formula_html}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_box(title: str, copy: str, items: list[tuple[str, str, str]]) -> None:
    item_html = "".join(
        f"""
        <div class="method-item">
            <p class="method-name">{html.escape(name)}</p>
            <p class="method-formula">{html.escape(formula)}</p>
            <p class="method-note">{html.escape(note)}</p>
        </div>
        """
        for name, formula, note in items
    )
    st.markdown(
        f"""
        <section class="method-box">
            <p class="method-box-title">{html.escape(title)}</p>
            <p class="method-box-copy">{html.escape(copy)}</p>
            <div class="method-box-grid">{item_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
