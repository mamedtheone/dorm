"""
ui_components.py - Streamlit UI helpers for Dorm-Net.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from modules.persona_module import PERSONAS


DARK_THEME_CSS = """
<style>
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --accent: #2ea043;
    --accent-soft: #1f6f3f;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
}

html, body, [class*="css"] {
    background: var(--bg-primary);
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

.dorm-header {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.dorm-subtle {
    color: var(--text-muted);
    font-size: 0.92rem;
}

.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    font-size: 0.78rem;
}
</style>
"""


def inject_css():
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "messages": [],
        "selected_model": "mistral:latest",
        "selected_persona": "software",
        "subject_hint": "engineering",
        "step_by_step": True,
        "debug_mode": False,
        "user_level": "auto",
        "current_mode": "answer",
        "indexed_docs": [],
        "ocr_result": None,
        "use_ocr_in_next_query": False,
        "is_processing": False,
        "ollama_online": False,
        "db_chunk_count": 0,
        "last_rag_sources": [],
        "last_followups": [],
        "last_detected_level": "basic",
        "last_quiz": [],
        "last_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    st.markdown(
        """
        <div class="dorm-header">
            <h1 style="margin-bottom:0.15rem;">Dorm-Net</h1>
            <div class="dorm-subtle">
                Offline multi-professor tutor for engineering study, grounded in your own materials.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    available_models: list[str],
    on_pdf_upload: Callable,
    on_clear_chat: Callable,
):
    with st.sidebar:
        st.subheader("System")
        render_status_badges()
        st.divider()

        st.subheader("Tutor Persona")
        persona_keys = list(PERSONAS.keys())
        current_persona = st.session_state.selected_persona
        persona_index = persona_keys.index(current_persona) if current_persona in persona_keys else 0
        st.session_state.selected_persona = st.selectbox(
            "Persona",
            persona_keys,
            index=persona_index,
            format_func=lambda key: PERSONAS[key].title,
        )
        active_persona = PERSONAS[st.session_state.selected_persona]
        st.caption(
            f"Style: {active_persona.explanation_style}. Focus: {active_persona.focus}."
        )

        st.subheader("Model")
        model_options = available_models or ["mistral:latest", "phi3:latest"]
        model_index = (
            model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_options
            else 0
        )
        st.session_state.selected_model = st.selectbox(
            "Ollama model",
            model_options,
            index=model_index,
        )

        st.subheader("Learning Controls")
        st.session_state.step_by_step = st.toggle(
            "Step-by-step mode",
            value=st.session_state.step_by_step,
        )
        st.session_state.debug_mode = st.toggle(
            "Debug / transparency mode",
            value=st.session_state.debug_mode,
        )
        st.session_state.user_level = st.selectbox(
            "Student level",
            ["auto", "basic", "intermediate"],
            index=["auto", "basic", "intermediate"].index(st.session_state.user_level),
        )
        st.session_state.current_mode = st.selectbox(
            "Tutor mode",
            ["answer", "concept_breakdown", "diagnosis", "notes"],
            format_func=lambda item: {
                "answer": "Answer questions",
                "concept_breakdown": "Concept breakdown",
                "diagnosis": "Error diagnosis",
                "notes": "Generate notes",
            }[item],
        )

        st.subheader("Subject")
        subject_options = [
            "engineering",
            "electrical",
            "mechanical",
            "software",
            "mathematics",
            "physics",
        ]
        current_subject = (
            st.session_state.subject_hint
            if st.session_state.subject_hint in subject_options
            else "engineering"
        )
        st.session_state.subject_hint = st.selectbox(
            "Subject hint",
            subject_options,
            index=subject_options.index(current_subject),
        )

        st.subheader("Documents")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file and st.button("Index PDF", use_container_width=True):
            on_pdf_upload(pdf_file)

        if st.session_state.indexed_docs:
            st.caption(f"Indexed chunks: {st.session_state.db_chunk_count}")
            for document in st.session_state.indexed_docs:
                st.markdown(
                    f"<span class='badge'>{document.get('source', '?')}</span>",
                    unsafe_allow_html=True,
                )

        st.subheader("Session")
        if st.button("Clear chat", use_container_width=True):
            on_clear_chat()


def render_status_badges():
    online = st.session_state.get("ollama_online", False)
    db_chunks = st.session_state.get("db_chunk_count", 0)
    model = st.session_state.get("selected_model", "n/a")

    st.markdown(
        f"<span class='badge'>{'LLM ready' if online else 'LLM offline'}</span>"
        f"<span class='badge'>{'DB ready' if db_chunks else 'DB empty'}</span>"
        f"<span class='badge'>{model}</span>",
        unsafe_allow_html=True,
    )
    if not online:
        st.warning("Run `ollama serve` and make sure your selected model is pulled.")


def render_chat_message_native(role: str, content: str):
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)


def render_ocr_panel(on_image_upload: Callable):
    with st.expander("OCR Notes", expanded=False):
        image_file = st.file_uploader(
            "Upload notebook image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="ocr_upload",
        )
        if image_file is not None:
            st.image(image_file, use_container_width=True)
            if st.button("Extract text", use_container_width=True):
                on_image_upload(image_file.read())

        result = st.session_state.get("ocr_result")
        if result and result.success and result.raw_text:
            st.success(
                f"Extracted {result.word_count} words at {result.confidence:.0f}% confidence."
            )
            st.text_area("Extracted text", result.raw_text, height=180)
            if st.button("Use OCR in next question", use_container_width=True):
                st.session_state["use_ocr_in_next_query"] = True
                st.toast("OCR text will be included in the next tutor turn.")


def render_rag_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources):
            score_text = ""
            if source.get("score") is not None:
                score_text = f" | score: {source['score']:.3f}"
            st.markdown(
                f"**{index + 1}. {source.get('source', '?')}** | page {source.get('page', '?')}{score_text}"
            )
            st.caption(source.get("snippet", "")[:500])


def render_followups(followups: list[str]):
    if not followups:
        return
    st.markdown("**Suggested follow-up questions**")
    for question in followups:
        st.markdown(f"- {question}")


def render_quiz(quiz_items: list):
    if not quiz_items:
        return
    with st.expander("Quiz Mode", expanded=False):
        for index, item in enumerate(quiz_items, start=1):
            st.markdown(f"**Q{index}. {item.question}**")
            if item.question_type == "mcq":
                for option_index, option in enumerate(item.options):
                    label = chr(65 + option_index)
                    st.markdown(f"- {label}. {option}")
                if item.answer_index is not None:
                    st.caption(
                        f"Answer: {chr(65 + item.answer_index)} | {item.explanation}"
                    )
            else:
                st.caption(f"Reference answer: {item.answer}")
                st.caption(item.explanation)


def toast_success(message: str):
    st.toast(message, icon="✅")


def toast_error(message: str):
    st.toast(message, icon="❌")


def toast_info(message: str):
    st.toast(message, icon="ℹ️")
