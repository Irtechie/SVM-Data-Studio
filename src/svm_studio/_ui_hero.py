"""Hero section renderer for the SVM Studio shell."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st


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
