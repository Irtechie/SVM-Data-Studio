from __future__ import annotations

import html

import pandas as pd
import streamlit as st

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-shell: #eef4f7;
    --bg-metal: #f7fafc;
    --bg-grid: rgba(13, 32, 46, 0.055);
    --ink: #091722;
    --muted: #47596a;
    --panel: rgba(248, 251, 253, 0.74);
    --panel-strong: rgba(255, 255, 255, 0.9);
    --line: rgba(73, 101, 125, 0.18);
    --line-strong: rgba(73, 101, 125, 0.28);
    --accent: #18c5d8;
    --accent-hot: #ff7a18;
    --accent-warm: #ffd166;
    --accent-soft: #7ef2c8;
    --shadow: 0 28px 60px rgba(10, 22, 34, 0.14);
    --shadow-soft: 0 16px 32px rgba(10, 22, 34, 0.08);
    --radius-xl: 32px;
    --radius-lg: 26px;
    --radius-md: 18px;
}

html, body, [class*="css"] {
    font-family: "Sora", "Segoe UI", sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(var(--bg-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px),
        radial-gradient(circle at 12% 0%, rgba(255, 122, 24, 0.14), transparent 26%),
        radial-gradient(circle at 88% 2%, rgba(24, 197, 216, 0.18), transparent 24%),
        linear-gradient(180deg, #f3f7fa 0%, var(--bg-shell) 48%, #edf2f5 100%);
    background-size: 36px 36px, 36px 36px, auto, auto, auto;
}

.main .block-container {
    max-width: 1420px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 100% 0%, rgba(24, 197, 216, 0.18), transparent 22%),
        linear-gradient(180deg, rgba(7, 16, 27, 0.98) 0%, rgba(10, 25, 39, 0.97) 100%);
    border-right: 1px solid rgba(126, 242, 200, 0.12);
}

[data-testid="stSidebar"] * {
    color: #f8f4ec !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p {
    color: #f8f4ec !important;
}

a,
[role="button"],
button,
input,
select,
textarea {
    transition: box-shadow 120ms ease, transform 120ms ease, border-color 120ms ease, background 120ms ease;
}

button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
[role="tab"]:focus-visible,
a:focus-visible {
    outline: 3px solid rgba(24, 197, 216, 0.42) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(24, 197, 216, 0.16) !important;
}

[data-baseweb="base-input"] > div,
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--line) !important;
    background: rgba(255, 255, 255, 0.82) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
}

[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: var(--radius-lg);
    border: 1px dashed rgba(255, 255, 255, 0.24);
    background: rgba(255, 255, 255, 0.05);
    min-height: 8.5rem;
}

[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem 0.4rem 1.1rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--panel-strong);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 0.35rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(14px);
}

[data-testid="stDataFrame"] div[role="grid"],
[data-testid="stTable"] table {
    font-size: 0.94rem;
}

div.stButton > button,
div.stDownloadButton > button,
button[kind="primaryFormSubmit"] {
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 999px;
    background: linear-gradient(135deg, #071827 0%, #0f2b3d 54%, #13445d 100%);
    color: white;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.78rem 1.28rem;
    box-shadow:
        0 18px 30px rgba(9, 23, 34, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 0.18);
    transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease, border-color 120ms ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px);
    filter: saturate(1.08);
    border-color: rgba(24, 197, 216, 0.42);
    box-shadow:
        0 20px 34px rgba(9, 23, 34, 0.24),
        0 0 0 1px rgba(24, 197, 216, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.22);
}

[data-testid="stTabs"] button[role="tab"] {
    height: auto;
    padding: 0.8rem 1.05rem;
    border-radius: 999px;
    border: 1px solid rgba(73, 101, 125, 0.14);
    background: rgba(255, 255, 255, 0.62);
    color: var(--muted);
    font-weight: 700;
    margin-right: 0.55rem;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #071827 0%, #103a53 72%, #18c5d8 100%);
    color: white;
    border-color: transparent;
    box-shadow: 0 18px 34px rgba(16, 58, 83, 0.26);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    border-radius: var(--radius-xl);
    background:
        linear-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.06) 1px, transparent 1px),
        radial-gradient(circle at 85% 14%, rgba(255, 209, 102, 0.22), transparent 24%),
        linear-gradient(135deg, rgba(6, 16, 27, 0.98) 0%, rgba(10, 32, 46, 0.96) 48%, rgba(19, 68, 93, 0.94) 100%);
    background-size: 28px 28px, 28px 28px, auto, auto;
    box-shadow: 0 34px 72px rgba(8, 18, 30, 0.28);
    margin-bottom: 1.6rem;
}

.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -5% -28% 48%;
    height: 260px;
    background: radial-gradient(circle, rgba(255, 122, 24, 0.28) 0%, transparent 62%);
    filter: blur(22px);
    pointer-events: none;
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
    gap: 1.35rem;
    align-items: stretch;
}

.hero-kicker,
.section-kicker,
.stat-label,
.panel-kicker {
    font-family: "IBM Plex Mono", monospace;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}

.hero-kicker {
    margin: 0 0 0.65rem 0;
    font-size: 0.78rem;
    color: rgba(255, 248, 240, 0.72);
}

.hero-title {
    margin: 0;
    font-size: clamp(2.4rem, 4.3vw, 4.9rem);
    line-height: 0.92;
    letter-spacing: -0.04em;
    color: #f7fbff;
}

.hero-copy {
    max-width: 52rem;
    margin: 0.95rem 0 0 0;
    font-size: 1rem;
    line-height: 1.65;
    color: rgba(255, 248, 240, 0.82);
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 1rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    padding: 0.55rem 0.88rem;
    border-radius: 999px;
    border: 1px solid rgba(126, 242, 200, 0.18);
    background: rgba(255, 255, 255, 0.08);
    color: #f7fbff;
    font-size: 0.88rem;
    backdrop-filter: blur(12px);
}

.hero-panel {
    border-radius: 26px;
    border: 1px solid rgba(126, 242, 200, 0.16);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.07));
    padding: 1.2rem 1.25rem;
    backdrop-filter: blur(16px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.panel-kicker {
    margin: 0 0 0.55rem 0;
    font-size: 0.75rem;
    color: rgba(255, 248, 240, 0.72);
}

.panel-number {
    margin: 0;
    font-size: 2.6rem;
    font-weight: 700;
    color: #fffaf1;
}

.panel-copy {
    margin: 0.55rem 0 0 0;
    color: rgba(255, 248, 240, 0.78);
    line-height: 1.55;
}

.section-intro {
    margin: 0.15rem 0 1rem 0;
}

.section-kicker {
    margin: 0 0 0.3rem 0;
    font-size: 0.78rem;
    color: var(--accent);
}

.section-title {
    margin: 0;
    font-size: 1.95rem;
    line-height: 1.05;
    color: var(--ink);
    letter-spacing: -0.03em;
}

.section-copy {
    margin: 0.6rem 0 0 0;
    max-width: 58rem;
    color: var(--muted);
    line-height: 1.68;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.95rem;
    margin: 1rem 0 1.6rem 0;
}

.stat-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(244, 249, 252, 0.88));
    border: 1px solid rgba(73, 101, 125, 0.14);
    border-radius: 24px;
    padding: 1rem 1.05rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(14px);
}

.stat-card::before,
.callout-card::before,
.state-panel::before,
.step-strip::before,
.method-box::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(24, 197, 216, 0.55), transparent);
}

.stat-label {
    margin: 0;
    font-size: 0.76rem;
    color: var(--muted);
}

.stat-value {
    margin: 0.45rem 0 0 0;
    font-size: 1.95rem;
    line-height: 1;
    color: var(--ink);
    font-weight: 700;
}

.stat-note {
    margin: 0.5rem 0 0 0;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.52;
}

.callout-card {
    position: relative;
    margin: 0.55rem 0 1.1rem 0;
    padding: 1rem 1.1rem;
    border-radius: 22px;
    border: 1px solid rgba(24, 197, 216, 0.16);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(239, 248, 251, 0.94));
    box-shadow: var(--shadow-soft);
}

.callout-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
}

.callout-copy {
    margin: 0.35rem 0 0 0;
    color: var(--muted);
    line-height: 1.62;
}

.state-panel {
    position: relative;
    margin: 0.7rem 0 1.15rem 0;
    padding: 1rem 1.1rem;
    border-radius: 22px;
    border: 1px solid rgba(73, 101, 125, 0.14);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(242, 247, 250, 0.9));
    box-shadow: var(--shadow-soft);
}

.state-panel[data-kind="info"] {
    border-color: rgba(15, 139, 141, 0.18);
    background: linear-gradient(135deg, rgba(240, 250, 249, 0.98), rgba(255, 255, 255, 0.92));
}

.state-panel[data-kind="success"] {
    border-color: rgba(43, 125, 98, 0.18);
    background: linear-gradient(135deg, rgba(241, 251, 246, 0.98), rgba(255, 255, 255, 0.92));
}

.state-panel[data-kind="warning"] {
    border-color: rgba(244, 180, 0, 0.24);
    background: linear-gradient(135deg, rgba(255, 249, 235, 0.98), rgba(255, 255, 255, 0.92));
}

.state-panel[data-kind="error"] {
    border-color: rgba(211, 84, 0, 0.24);
    background: linear-gradient(135deg, rgba(255, 245, 239, 0.98), rgba(255, 255, 255, 0.92));
}

.state-label {
    margin: 0 0 0.45rem 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.76rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.state-panel[data-kind="info"] .state-label {
    color: var(--accent);
}

.state-panel[data-kind="success"] .state-label {
    color: #2b7d62;
}

.state-panel[data-kind="warning"] .state-label {
    color: #ac7b00;
}

.state-panel[data-kind="error"] .state-label {
    color: #c25716;
}

.state-title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--ink);
}

.state-copy {
    margin: 0.38rem 0 0 0;
    color: var(--muted);
    line-height: 1.66;
}

.state-detail {
    margin: 0.7rem 0 0 0;
    padding: 0.7rem 0.8rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.72);
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.84rem;
    line-height: 1.55;
    color: var(--ink);
    white-space: pre-wrap;
}

.step-strip {
    position: relative;
    margin: 0.7rem 0 1.2rem 0;
    padding: 1rem 1.05rem;
    border-radius: 24px;
    border: 1px solid rgba(73, 101, 125, 0.14);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(242, 248, 251, 0.94));
    box-shadow: var(--shadow-soft);
}

.step-strip-title {
    margin: 0 0 0.8rem 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
}

.step-strip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.85rem;
}

.step-card {
    padding: 0.9rem 0.92rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid rgba(24, 197, 216, 0.12);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

.step-index {
    margin: 0 0 0.35rem 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.74rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-hot);
}

.step-name {
    margin: 0;
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink);
}

.step-copy {
    margin: 0.38rem 0 0 0;
    font-size: 0.93rem;
    line-height: 1.58;
    color: var(--muted);
}

.method-box {
    position: relative;
    margin: 0.75rem 0 1.1rem 0;
    padding: 1.05rem 1.1rem;
    border-radius: 24px;
    border: 1px solid rgba(73, 101, 125, 0.14);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(242, 247, 250, 0.92));
    box-shadow: var(--shadow-soft);
}

.method-box-title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--ink);
}

.method-box-copy {
    margin: 0.38rem 0 0.9rem 0;
    color: var(--muted);
    line-height: 1.62;
}

.method-box-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.8rem;
}

.method-item {
    padding: 0.85rem 0.9rem;
    border-radius: 18px;
    border: 1px solid rgba(24, 197, 216, 0.12);
    background: rgba(255, 255, 255, 0.82);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

.method-name {
    margin: 0;
    font-size: 0.83rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.method-formula {
    margin: 0.55rem 0 0 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--ink);
    white-space: pre-wrap;
}

.method-note {
    margin: 0.55rem 0 0 0;
    color: var(--muted);
    line-height: 1.58;
    font-size: 0.93rem;
}

@media (max-width: 980px) {
    .hero-grid,
    .stat-grid {
        grid-template-columns: 1fr;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-shell {
        padding: 1.4rem;
    }

    .hero-panel {
        min-height: auto;
    }

    .method-box-grid,
    .step-strip-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .main .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 2.4rem;
    }

    .hero-shell {
        padding: 1.15rem;
        border-radius: 24px;
    }

    .hero-title {
        font-size: clamp(2rem, 11vw, 3rem);
    }

    .chip-row {
        gap: 0.5rem;
    }

    .chip {
        width: 100%;
        justify-content: center;
    }

    .section-title {
        font-size: 1.55rem;
    }

    .stat-card,
    .callout-card,
    .state-panel,
    .step-strip,
    .method-box {
        border-radius: 20px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}
</style>
"""


def inject_app_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


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


def render_hero(source_name: str, frame: pd.DataFrame) -> None:
    numeric_count = int(frame.select_dtypes(include="number").shape[1])
    categorical_count = int(frame.shape[1] - numeric_count)
    missing_count = int(frame.isna().sum().sum())
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-grid">
                <div>
                    <p class="hero-kicker">SVM / ITEMSET / EPISODE WORKBENCH</p>
                    <h1 class="hero-title">SVM Data Studio</h1>
                    <p class="hero-copy">
                        Load a dataset, isolate the columns that matter, inspect the separator geometry,
                        and generate mining outputs from the same polished workspace.
                    </p>
                    <div class="chip-row">
                        <span class="chip">{html.escape(source_name)}</span>
                        <span class="chip">{frame.shape[0]} rows</span>
                        <span class="chip">{frame.shape[1]} columns</span>
                        <span class="chip">{numeric_count} numeric</span>
                        <span class="chip">{categorical_count} categorical</span>
                    </div>
                </div>
                <aside class="hero-panel">
                    <div>
                        <p class="panel-kicker">Data Snapshot</p>
                        <p class="panel-number">{frame.shape[0]:,}</p>
                        <p class="panel-copy">
                            Rows currently staged for modeling. Missing cells: {missing_count:,}. Use the pages below
                            to move from schema review to SVM geometry and mining outputs.
                        </p>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
