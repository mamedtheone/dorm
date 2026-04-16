"""
main_app.py — Dorm-Net Central Controller
==========================================
Entry point: `streamlit run main_app.py`

Wires together:
  VisionEngine   (vision_module.py)
  RAGManager     (brain_module.py)
  PersonaManager (persona_module.py)
  UI components  (ui_components.py)

Async processing is handled via asyncio + ThreadPoolExecutor inside each
module so the Streamlit event loop is never blocked for more than a frame.
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
# On Windows, set this to your Tesseract install path:
# e.g.  r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

# ─────────────────────────────────────────────────────────────────────────── #
#  Module singletons (cached so Streamlit doesn't re-init on every rerun)     #
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
#  Async helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def run_async(coro):
    """
    Safely run an async coroutine from synchronous Streamlit context.
    Creates a new event loop if one isn't running.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.ensure_future(coro, loop=loop)
            return concurrent.futures.Future()  # fallback: run sync
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────── #
#  Event handlers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def handle_pdf_upload(uploaded_file):
    """Save uploaded PDF to a temp file, then ingest asynchronously."""
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

    # Refresh sidebar state
    _refresh_db_state(rag)


def handle_image_upload(image_bytes: bytes):
    """Run OCR pipeline on uploaded image bytes."""
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
#  Core: generate answer                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def generate_answer(user_question: str):
    """
    Full pipeline:
      1. Retrieve relevant RAG chunks (async)
      2. Build TutorRequest with history + context
      3. Stream LLM response token-by-token into the UI
      4. Save assistant message to session state
    """
    rag     = get_rag_manager()
    persona = get_persona_manager()

    # ── 1. RAG retrieval ────────────────────────────────────────────────
    rag_context: list[str] = []
    rag_sources: list[dict] = []

    if rag.get_chunk_count() > 0:
        retrieval = asyncio.run(rag.query_async(user_question, top_k=4))
        if retrieval.success and retrieval.chunks:
            rag_context = [c.text for c in retrieval.chunks]
            rag_sources = [
                {"source": c.source, "page": c.page, "snippet": c.text}
                for c in retrieval.chunks
            ]

    # ── 2. Build request ────────────────────────────────────────────────
    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[-10:]
    ]

    ocr_text = None
    if st.session_state.get("use_ocr_in_next_query"):
        ocr_result = st.session_state.get("ocr_result")
        if ocr_result and ocr_result.success:
            ocr_text = ocr_result.raw_text
        st.session_state["use_ocr_in_next_query"] = False

    request = TutorRequest(
        user_question=user_question,
        rag_context=rag_context,
        ocr_text=ocr_text,
        history=history,
        subject_hint=st.session_state.get("subject_hint", "general"),
        model=st.session_state.get("selected_model", "llama3.2:3b"),
    )

    # ── 3. Stream response ──────────────────────────────────────────────
    full_answer = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for token in persona.stream(request):
            full_answer += token
            placeholder.markdown(full_answer + "▌")   # blinking cursor effect
        placeholder.markdown(full_answer)             # final render (no cursor)

    # ── 4. Persist to history ───────────────────────────────────────────
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state["last_rag_sources"] = rag_sources

    return rag_sources


# ─────────────────────────────────────────────────────────────────────────── #
#  State helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _refresh_db_state(rag: RAGManager):
    st.session_state["db_chunk_count"] = rag.get_chunk_count()
    st.session_state["indexed_docs"]   = rag.list_indexed_docs()


def _refresh_ollama_state(persona: PersonaManager):
    st.session_state["ollama_online"] = persona.is_ollama_running()


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    # ── Page config (must be first Streamlit call) ──────────────────────
    st.set_page_config(
        page_title="Dorm-Net · AASTU AI Tutor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Inject CSS + init state ─────────────────────────────────────────
    inject_css()
    init_session_state()

    # ── Load modules ────────────────────────────────────────────────────
    rag     = get_rag_manager()
    persona = get_persona_manager()

    # ── Refresh live status (cheap; runs every rerun) ───────────────────
    _refresh_db_state(rag)
    _refresh_ollama_state(persona)

    available_models = persona.list_available_models()

    # ── Sidebar ─────────────────────────────────────────────────────────
    render_sidebar(
        available_models=available_models,
        on_pdf_upload=handle_pdf_upload,
        on_clear_chat=handle_clear_chat,
    )

    # ── Main area: header + OCR panel ────────────────────────────────────
    render_header()
    render_ocr_panel(on_image_upload=handle_image_upload)

    # ── Chat history ─────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        render_chat_message_native(msg["role"], msg["content"])

    # ── RAG source display (last answer) ─────────────────────────────────
    if st.session_state.get("last_rag_sources"):
        render_rag_sources(st.session_state["last_rag_sources"])

    # ── Offline notice ───────────────────────────────────────────────────
    if not st.session_state.get("ollama_online"):
        st.info(
            "**Ollama is not running.** Start it with `ollama serve` "
            "in a terminal, then refresh this page.",
            icon="🔌",
        )

    # ── Chat input ───────────────────────────────────────────────────────
    if user_input := st.chat_input(
        "Ask Kebede anything… (e.g. 'Explain Kirchhoff's Voltage Law step by step')",
        disabled=st.session_state.get("is_processing", False),
    ):
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message_native("user", user_input)

        # Generate answer
        st.session_state["is_processing"] = True
        try:
            rag_sources = generate_answer(user_input)
        finally:
            st.session_state["is_processing"] = False

        # Show sources inline after the answer
        if rag_sources:
            render_rag_sources(rag_sources)

        st.rerun()


if __name__ == "__main__":
    main()