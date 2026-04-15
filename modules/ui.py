"""
UI helpers for Dorm-Net.

This module keeps Streamlit styling and reusable rendering functions separate
from the main application flow. That makes `main_app.py` easier to read.
"""

from __future__ import annotations

import streamlit as st


def apply_dark_theme() -> None:
    """
    Inject custom CSS so the app has a consistent dark-mode interface.
    """

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b1020 0%, #111827 45%, #0f172a 100%);
                color: #e5e7eb;
            }
            section[data-testid="stSidebar"] {
                background: #0f172a;
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero-card {
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
            }
            .source-card {
                background: rgba(30, 41, 59, 0.75);
                border-left: 4px solid #38bdf8;
                border-radius: 10px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.7rem;
            }
            div[data-testid="stChatMessage"] {
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 16px;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 10px;
                border: 1px solid rgba(56, 189, 248, 0.3);
                background: #0ea5e9;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_card(selected_course: str) -> None:
    """
    Show the main header card for the current course.
    """

    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin-bottom:0.3rem;">Dorm-Net</h1>
            <p style="margin:0; color:#cbd5e1;">
                Course: <strong>{selected_course}</strong><br/>
                Ask questions from your uploaded PDFs and handwritten notes,
                all without leaving your local machine.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources) -> None:
    """
    Display retrieved chunks so students can inspect what the answer used.
    """

    if not sources:
        st.info("No local study sources were retrieved for this question yet.")
        return

    st.markdown("### Retrieved Study Context")
    for doc in sources:
        source_name = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        snippet = doc.page_content[:320].strip()
        st.markdown(
            f"""
            <div class="source-card">
                <strong>{source_name}</strong> | page {page}<br/>
                {snippet}...
            </div>
            """,
            unsafe_allow_html=True,
        )
