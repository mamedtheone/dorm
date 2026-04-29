"""
main_app.py — Dorm-Net Edizione Italiana
=========================================
Layout: Streamlit sidebar + full-width main area
  · Page header with dramatic Cormorant Garamond title
  · Mode bar
  · st.columns([5, 2]) for chat + reference panel
  · st.chat_input() at PAGE SCOPE (not inside columns)
  · Reference panel: sources / quiz / ocr / health via session state
"""

from __future__ import annotations
import asyncio
import logging
import os
import tempfile

import streamlit as st

from modules.brain_module import RAGManager
from modules.persona_module import Message, PersonaManager
from modules.tutor_controller import TutorController
from modules.ui_components import (
    MODE_META, PERSONA_META, NAV_ITEMS,
    init_session_state, inject_css,
    render_chat_message_native, render_followups,
    render_page_header, render_mode_bar,
    render_professor_card_inline,
    render_ocr_panel, render_quiz,
    render_rag_sources, render_sidebar,
    render_system_health,
    toast_error, toast_info, toast_success,
)
from modules.vision_module import VisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("dorm_net.main")

DB_PATH       = os.getenv("DORM_NET_DB_PATH", "./dorm_net_db")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")


@st.cache_resource(show_spinner="Initialising Dorm-Net…")
def get_rag_manager() -> RAGManager:
    return RAGManager(db_path=DB_PATH)

@st.cache_resource(show_spinner=False)
def get_persona_manager() -> PersonaManager:
    return PersonaManager(ollama_url=OLLAMA_URL, default_model="mistral:latest")

@st.cache_resource(show_spinner=False)
def get_controller() -> TutorController:
    return TutorController(get_rag_manager(), get_persona_manager())

@st.cache_resource(show_spinner=False)
def get_vision_engine() -> VisionEngine:
    return VisionEngine(tesseract_cmd=TESSERACT_CMD)


def handle_pdf_upload(uploaded_file):
    rag = get_rag_manager()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read()); tmp_path = tmp.name
    with st.spinner(f"Indexing {uploaded_file.name}…"):
        report = asyncio.run(rag.ingest_pdf_async(tmp_path))
    os.unlink(tmp_path)
    if report.success:
        toast_info(f"{uploaded_file.name} already indexed.") if report.skipped else toast_success(f"Indexed {uploaded_file.name}: {report.total_pages}p / {report.total_chunks} chunks.")
    else:
        toast_error(f"Index failed: {report.error}")
    refresh_state()


def handle_image_upload(image_bytes: bytes):
    with st.spinner("Running OCR…"):
        result = asyncio.run(get_vision_engine().extract_text_async(image_bytes))
    st.session_state["ocr_result"] = result
    toast_success(f"OCR done — {result.word_count} words @ {result.confidence:.0f}%.") if result.success else toast_error(f"OCR failed: {result.error}")


def handle_clear_chat():
    for k in ("messages","last_rag_sources","last_followups","last_quiz"):
        st.session_state[k] = []
    st.session_state["last_notes"] = ""
    st.session_state["ocr_result"] = None
    st.session_state["use_ocr_in_next_query"] = False
    st.session_state["_pending_input"] = None
    toast_info("Session cleared.")


def refresh_state():
    rag = get_rag_manager(); pm = get_persona_manager()
    st.session_state["db_chunk_count"] = rag.get_chunk_count()
    st.session_state["indexed_docs"]   = rag.list_indexed_docs()
    st.session_state["ollama_online"]  = pm.is_ollama_running()


def build_history() -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in st.session_state.messages[-8:]]


def current_user_level() -> str | None:
    level = st.session_state.get("user_level","auto")
    return None if level == "auto" else level


def next_ocr_text() -> str | None:
    if not st.session_state.get("use_ocr_in_next_query"): return None
    result = st.session_state.get("ocr_result")
    st.session_state["use_ocr_in_next_query"] = False
    return result.raw_text if (result and result.success) else None


def generate_streaming_answer(user_question: str):
    controller = get_controller()
    request, rag_sources, raw_context = controller.build_request(
        question=user_question, history=build_history(),
        model=st.session_state.selected_model,
        persona_key=st.session_state.selected_persona,
        step_by_step=st.session_state.step_by_step,
        subject_hint=st.session_state.subject_hint,
        mode=st.session_state.current_mode,
        user_level=current_user_level(), ocr_text=next_ocr_text(),
        debug_mode=st.session_state.debug_mode,
        top_k=5 if st.session_state.debug_mode else 4,
    )
    full_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for token in get_persona_manager().stream(request):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    turn = controller.parse_streamed_turn(request, full_text, rag_sources, raw_context)
    st.session_state.messages.append({"role":"assistant","content":turn.response.answer})
    st.session_state["last_rag_sources"]    = turn.rag_sources
    st.session_state["last_followups"]      = turn.response.follow_up_questions
    st.session_state["last_detected_level"] = turn.response.detected_level


def generate_quiz(topic: str):
    with st.spinner("Composing quiz…"):
        items = get_controller().generate_quiz(topic)
    st.session_state["last_quiz"] = items
    if items:
        toast_success(f"Quiz ready — {len(items)} questions.")
        st.session_state["active_panel"] = "quiz"
    else:
        toast_info("Need indexed material on that topic first.")


# ─── REFERENCE PANEL ────────────────────────────────────────────────────────

def render_reference_panel():
    active = st.session_state.get("active_panel","chat")

    # Panel switcher — minimal tab row
    st.markdown(
        """
        <div style="display:flex;gap:.3rem;padding:.6rem 0 .8rem;border-bottom:1px solid var(--border-2);margin-bottom:.9rem;">
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    labels = [("chat","◈ Sources"),("quiz","⊟ Quiz"),("ocr","⊡ OCR"),("health","◎ System")]
    for col,(key,label) in zip(cols, labels):
        with col:
            is_active = (key == active)
            style = "color:var(--gold);background:var(--gold-dim);border-color:rgba(201,168,76,.28);" if is_active else ""
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:.55rem;letter-spacing:.1em;text-transform:uppercase;'
                f'padding:.3rem .5rem;border:1px solid transparent;border-radius:2px;text-align:center;{style}">{label}</div>',
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"rp_{key}", use_container_width=True):
                st.session_state["active_panel"] = key
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if active == "chat":
        render_rag_sources(st.session_state.get("last_rag_sources",[]))
        render_followups(st.session_state.get("last_followups",[]))

    elif active == "quiz":
        # Topic input
        st.markdown('<div class="sources-label" style="margin-bottom:.5rem;">Generate Quiz</div>', unsafe_allow_html=True)
        topic = st.text_input("Topic", placeholder="e.g. Kirchhoff's laws…", label_visibility="collapsed", key="quiz_topic_rp")
        if st.button("▸ Generate", use_container_width=True, key="quiz_gen_rp"):
            t = topic.strip() or (" ".join(m["content"] for m in st.session_state.messages[-3:]) or st.session_state.subject_hint)
            generate_quiz(t)
        st.markdown('<div class="sect-divider"><div class="sect-divider-line"></div></div>', unsafe_allow_html=True)
        render_quiz(st.session_state.get("last_quiz",[]))

    elif active == "ocr":
        render_ocr_panel(on_image_upload=handle_image_upload)

    elif active == "health":
        render_system_health()


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Dorm-Net",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session_state()
    refresh_state()

    available_models = get_persona_manager().list_available_models()
    render_sidebar(
        available_models=available_models,
        on_pdf_upload=handle_pdf_upload,
        on_clear_chat=handle_clear_chat,
    )

    # ── Page header ──
    render_page_header()

    # ── Mode bar ──
    render_mode_bar()

    # ── chat_input at PAGE SCOPE ──
    mode_key = st.session_state.get("current_mode","answer")
    placeholder_map = {
        "answer":            "Ask a study question…",
        "concept_breakdown": "Describe a concept to unpack…",
        "diagnosis":         "Paste a bug or error to diagnose…",
        "notes":             "Request study notes on a topic…",
    }
    user_input = st.chat_input(
        placeholder_map.get(mode_key,"Ask a question…"),
        disabled=st.session_state.get("is_processing",False),
    )

    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        st.session_state["_pending_input"] = user_input
        if st.session_state.get("active_panel") not in ("chat","quiz"):
            st.session_state["active_panel"] = "chat"

    # ── Two-column layout: chat (wide) + reference panel (narrow) ──
    col_chat, col_ref = st.columns([5, 2], gap="large")

    # ── Chat column ──
    with col_chat:
        # Padding strip to align with mode-bar bottom
        st.markdown('<div style="height:1px;background:var(--border-2);margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

        # Pinned professor card
        render_professor_card_inline()

        # Offline warning
        if not st.session_state.get("ollama_online"):
            st.warning("Kernel offline — run `ollama serve` to connect.", icon=None)

        # OCR attachment notice
        if st.session_state.get("use_ocr_in_next_query"):
            st.info("📎 OCR text will be attached to your next message.", icon=None)

        # Chat history
        for msg in st.session_state.messages:
            render_chat_message_native(msg["role"], msg["content"])

        # Stream pending answer
        pending = st.session_state.get("_pending_input")
        if pending:
            st.session_state["_pending_input"] = None
            st.session_state["is_processing"] = True
            try:
                generate_streaming_answer(pending)
            finally:
                st.session_state["is_processing"] = False
            st.rerun()

        # Ornamental divider + quiz shortcut
        if st.session_state.messages:
            st.markdown(
                '<div class="sect-divider"><div class="sect-divider-line"></div>'
                '<span class="sect-divider-text">continue the session</span>'
                '<div class="sect-divider-line"></div></div>',
                unsafe_allow_html=True,
            )
            if st.button("⊟ Generate Quiz from this conversation", key="quiz_from_chat"):
                topic = " ".join(m["content"] for m in st.session_state.messages[-3:]) or st.session_state.subject_hint
                generate_quiz(topic)
                st.rerun()

    # ── Reference panel column ──
    with col_ref:
        st.markdown(
            '<div style="height:1px;background:var(--border-2);margin-bottom:1.5rem;border-left:1px solid var(--border-2);padding-left:1rem;"></div>',
            unsafe_allow_html=True,
        )
        render_reference_panel()


if __name__ == "__main__":
    main()
