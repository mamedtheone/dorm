"""
main_app.py — Dorm-Net Central Controller
==========================================
Entry point: `streamlit run main_app.py`
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Module imports ────────────────────────────────────────────────────────
from modules.vision_module  import VisionEngine
from modules.brain_module   import RAGManager
from modules.persona_module import PersonaManager, TutorRequest, Message
from modules.ui_components  import (
    init_session_state,
    inject_css,
    render_header,
    render_sidebar,
    render_ocr_panel,
    render_rag_sources,
    render_chat_message_native,
    toast_success,
    toast_error,
    toast_info,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Logging                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dorm_net.main")

# ─────────────────────────────────────────────────────────────────────────── #
#  Configuration                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

DB_PATH = os.getenv("DORM_NET_DB_PATH", "./dorm_net_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

# ─────────────────────────────────────────────────────────────────────────── #
#  Module singletons                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@st.cache_resource(show_spinner="Loading AI engine… (first run only)")
def get_rag_manager() -> RAGManager:
    return RAGManager(db_path=DB_PATH)

@st.cache_resource(show_spinner=False)
def get_persona_manager() -> PersonaManager:
    return PersonaManager(ollama_url=OLLAMA_URL,
                          default_model="llama3.2:3b")

@st.cache_resource(show_spinner=False)
def get_vision_engine() -> VisionEngine:
    return VisionEngine(tesseract_cmd=TESSERACT_CMD)

# ─────────────────────────────────────────────────────────────────────────── #

def handle_pdf_upload(uploaded_file):
    rag = get_rag_manager()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner(f"Indexing '{uploaded_file.name}'…"):
        report = asyncio.run(rag.ingest_pdf_async(tmp_path))

    os.unlink(tmp_path)

    if report.success:
        if report.skipped:
            toast_info(f"'{uploaded_file.name}' already indexed — skipped.")
        else:
            toast_success(
                f"✅ Indexed '{uploaded_file.name}': "
                f"{report.total_pages} pages, {report.total_chunks} chunks."
            )
    else:
        toast_error(f"Failed to index: {report.error}")

    _refresh_db_state(rag)


def handle_image_upload(image_bytes: bytes):
    vision = get_vision_engine()
    with st.spinner("Scanning handwriting…"):
        result = asyncio.run(vision.extract_text_async(image_bytes))

    st.session_state["ocr_result"] = result
    if result.success:
        toast_success(f"OCR complete — {result.word_count} words extracted.")
    else:
        toast_error(f"OCR failed: {result.error}")


def handle_clear_chat():
    st.session_state["messages"] = []
    st.session_state["ocr_result"] = None
    st.session_state["use_ocr_in_next_query"] = False
    st.session_state["last_rag_sources"] = []
    toast_info("Chat cleared.")

# ─────────────────────────────────────────────────────────────────────────── #

def generate_answer(user_question: str):
    rag     = get_rag_manager()
    persona = get_persona_manager()

    rag_context = []
    rag_sources = []

    if rag.get_chunk_count() > 0:
        retrieval = asyncio.run(rag.query_async(user_question, top_k=4))
        if retrieval.success and retrieval.chunks:
            rag_context = [c.text for c in retrieval.chunks]
            rag_sources = [
                {"source": c.source, "page": c.page, "snippet": c.text}
                for c in retrieval.chunks
            ]

    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[-10:]
    ]

    request = TutorRequest(
        user_question=user_question,
        rag_context=rag_context,
        ocr_text=None,
        history=history,
        subject_hint="general",
        model="llama3.2:3b",
    )

    full_answer = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for token in persona.stream(request):
            full_answer += token
            placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state["last_rag_sources"] = rag_sources

    return rag_sources

# ─────────────────────────────────────────────────────────────────────────── #

def _refresh_db_state(rag: RAGManager):
    st.session_state["db_chunk_count"] = rag.get_chunk_count()
    st.session_state["indexed_docs"]   = rag.list_indexed_docs()

def _refresh_ollama_state(persona: PersonaManager):
    st.session_state["ollama_online"] = persona.is_ollama_running()

# ─────────────────────────────────────────────────────────────────────────── #

def main():
    st.set_page_config(
        page_title="Dorm-Net · AASTU AI Tutor",
        page_icon="🎓",
        layout="wide",
    )

    inject_css()
    init_session_state()

    rag     = get_rag_manager()
    persona = get_persona_manager()

    _refresh_db_state(rag)
    _refresh_ollama_state(persona)

    render_sidebar(
        available_models=persona.list_available_models(),
        on_pdf_upload=handle_pdf_upload,
        on_clear_chat=handle_clear_chat,
    )

    render_header()

    # ✅ ADDED TEXT (TOP)
    st.markdown("""
    ### 🎓 Offline AI Tutor

    A smart learning system built for students without internet access.  
    You can ask questions, upload notes, and scan handwritten work — all processed locally.

    💡 Learn anytime, anywhere.
    """)

    # ✅ ADDED GUIDANCE
    st.success("""
    🎯 Study Tip:
    Try solving the problem first, then use AI to check your understanding.
    """)

    render_ocr_panel(on_image_upload=handle_image_upload)

    for msg in st.session_state.messages:
        render_chat_message_native(msg["role"], msg["content"])

    if st.session_state.get("last_rag_sources"):
        render_rag_sources(st.session_state["last_rag_sources"])

    if user_input := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message_native("user", user_input)

        rag_sources = generate_answer(user_input)

        if rag_sources:
            render_rag_sources(rag_sources)

        st.rerun()

    # ✅ ADDED TEXT (BOTTOM)
    st.markdown("""
    ---
    📌 This system works fully offline using local AI.

    ⚠️ It may sometimes make mistakes.  
    Always check important information before using it in exams or assignments.

    💭 Use it as a learning tool — not a replacement for thinking.
    """)


if __name__ == "__main__":
    main()