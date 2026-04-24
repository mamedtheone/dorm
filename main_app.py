"""
main_app.py - Streamlit interface for Dorm-Net.
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
    init_session_state,
    inject_css,
    render_chat_message_native,
    render_followups,
    render_header,
    render_ocr_panel,
    render_quiz,
    render_rag_sources,
    render_sidebar,
    toast_error,
    toast_info,
    toast_success,
)
from modules.vision_module import VisionEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("dorm_net.main")


DB_PATH = os.getenv("DORM_NET_DB_PATH", "./dorm_net_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")


@st.cache_resource(show_spinner="Loading Dorm-Net resources...")
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        report = asyncio.run(rag.ingest_pdf_async(temp_path))

    os.unlink(temp_path)

    if report.success:
        if report.skipped:
            toast_info(f"{uploaded_file.name} was already indexed.")
        else:
            toast_success(
                f"Indexed {uploaded_file.name}: {report.total_pages} pages, {report.total_chunks} chunks."
            )
    else:
        toast_error(f"Failed to index {uploaded_file.name}: {report.error}")

    refresh_state()


def handle_image_upload(image_bytes: bytes):
    with st.spinner("Running OCR..."):
        result = asyncio.run(get_vision_engine().extract_text_async(image_bytes))
    st.session_state["ocr_result"] = result
    if result.success:
        toast_success("OCR completed.")
    else:
        toast_error(f"OCR failed: {result.error}")


def handle_clear_chat():
    st.session_state["messages"] = []
    st.session_state["last_rag_sources"] = []
    st.session_state["last_followups"] = []
    st.session_state["last_quiz"] = []
    st.session_state["last_notes"] = ""
    st.session_state["ocr_result"] = None
    st.session_state["use_ocr_in_next_query"] = False
    toast_info("Session cleared.")


def refresh_state():
    rag = get_rag_manager()
    persona = get_persona_manager()
    st.session_state["db_chunk_count"] = rag.get_chunk_count()
    st.session_state["indexed_docs"] = rag.list_indexed_docs()
    st.session_state["ollama_online"] = persona.is_ollama_running()


def build_history() -> list[Message]:
    return [
        Message(role=item["role"], content=item["content"])
        for item in st.session_state.messages[-8:]
    ]


def current_user_level() -> str | None:
    level = st.session_state.get("user_level", "auto")
    return None if level == "auto" else level


def next_ocr_text() -> str | None:
    if not st.session_state.get("use_ocr_in_next_query"):
        return None
    result = st.session_state.get("ocr_result")
    st.session_state["use_ocr_in_next_query"] = False
    if result and result.success:
        return result.raw_text
    return None


def generate_streaming_answer(user_question: str):
    controller = get_controller()
    request, rag_sources, raw_context = controller.build_request(
        question=user_question,
        history=build_history(),
        model=st.session_state.selected_model,
        persona_key=st.session_state.selected_persona,
        step_by_step=st.session_state.step_by_step,
        subject_hint=st.session_state.subject_hint,
        mode=st.session_state.current_mode,
        user_level=current_user_level(),
        ocr_text=next_ocr_text(),
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
    st.session_state.messages.append({"role": "assistant", "content": turn.response.answer})
    st.session_state["last_rag_sources"] = turn.rag_sources
    st.session_state["last_followups"] = turn.response.follow_up_questions
    st.session_state["last_detected_level"] = turn.response.detected_level


def generate_quiz(topic: str):
    quiz_items = get_controller().generate_quiz(topic)
    st.session_state["last_quiz"] = quiz_items
    if quiz_items:
        toast_success("Quiz generated from retrieved materials.")
    else:
        toast_info("Quiz generation needs indexed material related to the topic.")


def main():
    st.set_page_config(
        page_title="Dorm-Net",
        page_icon="🎓",
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

    render_header()
    render_ocr_panel(on_image_upload=handle_image_upload)

    if not st.session_state.get("ollama_online"):
        st.info("Ollama is offline. Start it with `ollama serve`.")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(
            f"Mode: {st.session_state.current_mode} | Persona: {st.session_state.selected_persona} | "
            f"Detected level: {st.session_state.last_detected_level}"
        )
    with col2:
        if st.button("Generate quiz", use_container_width=True):
            generate_quiz(" ".join([msg["content"] for msg in st.session_state.messages[-3:]]) or st.session_state.subject_hint)

    for item in st.session_state.messages:
        render_chat_message_native(item["role"], item["content"])

    if st.session_state.debug_mode:
        render_rag_sources(st.session_state.get("last_rag_sources", []))

    render_followups(st.session_state.get("last_followups", []))
    render_quiz(st.session_state.get("last_quiz", []))

    prompt_placeholder = {
        "answer": "Ask a study question...",
        "concept_breakdown": "Ask for a concept breakdown...",
        "diagnosis": "Paste a bug, wrong solution, or problem to diagnose...",
        "notes": "Ask for lightweight study notes from the indexed material...",
    }[st.session_state.current_mode]

    if user_input := st.chat_input(
        prompt_placeholder,
        disabled=st.session_state.is_processing,
    ):
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message_native("user", user_input)
        st.session_state.is_processing = True
        try:
            generate_streaming_answer(user_input)
        finally:
            st.session_state.is_processing = False
        st.rerun()


if __name__ == "__main__":
    main()
