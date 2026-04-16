"""
ui_components.py — Dorm-Net Streamlit UI
==========================================
Provides:
  - Dark-mode CSS theme (AASTU green + deep slate)
  - Chat history renderer using st.session_state
  - Sidebar: model picker, PDF uploader, status indicators
  - Image uploader panel for OCR
  - Streaming-aware chat input handler
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, Callable

# ─────────────────────────────────────────────────────────────────────────── #
#  Theme                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

DARK_THEME_CSS = """
<style>
/* ── Base ───────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --accent-green:  #238636;
    --accent-bright: #2ea043;
    --accent-gold:   #d29922;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border:        #30363d;
    --user-bubble:   #1f3a5f;
    --bot-bubble:    #1c2128;
    --code-bg:       #161b22;
    --danger:        #f85149;
    --radius:        12px;
    --font-mono:     'JetBrains Mono', 'Fira Code', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Chat messages ───────────────────────────────────────────── */
.chat-message {
    display: flex;
    gap: 12px;
    padding: 14px 16px;
    margin: 8px 0;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    animation: fadeIn 0.25s ease-in-out;
}
.chat-message.user   { background: var(--user-bubble); }
.chat-message.bot    { background: var(--bot-bubble);  }

.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.avatar.user { background: var(--accent-green); }
.avatar.bot  { background: #21262d; border: 1px solid var(--border); }

.message-body {
    flex: 1;
    line-height: 1.65;
    font-size: 0.95rem;
}
.message-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 4px;
}

/* ── Code blocks ─────────────────────────────────────────────── */
code {
    background: var(--code-bg) !important;
    color: #79c0ff !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.88em;
}
pre { background: var(--code-bg) !important; border-radius: var(--radius); padding: 12px !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background-color: var(--accent-green) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s;
}
.stButton > button:hover { background-color: var(--accent-bright) !important; }

/* ── Inputs ──────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Status badges ───────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-green  { background: #1a3a1a; color: var(--accent-bright); border: 1px solid var(--accent-green); }
.badge-yellow { background: #3a2d00; color: var(--accent-gold);   border: 1px solid var(--accent-gold);  }
.badge-red    { background: #3a1a1a; color: var(--danger);        border: 1px solid var(--danger);       }

/* ── Header logo bar ─────────────────────────────────────────── */
.dorm-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.dorm-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.5px; }
.dorm-sub   { font-size: 0.8rem;  color: var(--text-muted); }

/* ── Divider ─────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Spinner ─────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent-bright) !important; }

/* ── Fade-in animation ───────────────────────────────────────── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────── #
#  Session state init                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def init_session_state():
    """
    Must be called once at app startup.
    Initialises all keys used throughout the session.
    """
    defaults = {
        "messages": [],               # list of {"role": str, "content": str}
        "selected_model": "llama3.2:3b",
        "subject_hint": "general",
        "indexed_docs": [],           # list of doc metadata dicts
        "ocr_result": None,           # last OCRResult
        "is_processing": False,
        "ollama_online": False,
        "db_chunk_count": 0,
        "last_rag_sources": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────── #
#  Layout helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def inject_css():
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
        <div class="dorm-header">
            <span style="font-size:2.2rem;">🎓</span>
            <div>
                <div class="dorm-title">Dorm-Net</div>
                <div class="dorm-sub">Local-first AI Tutor · AASTU Edition</div>
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
    """
    Renders the left sidebar.
    Calls on_pdf_upload(uploaded_file) when a new PDF is submitted.
    """
    with st.sidebar:
        # ── Status indicators ──────────────────────────────────────────
        st.markdown("### ⚡ System Status")
        _render_status_badges()
        st.divider()

        # ── Model selector ─────────────────────────────────────────────
        st.markdown("### 🤖 Model")
        model_options = available_models if available_models else ["llama3.2:3b", "llama3.2:1b"]
        idx = model_options.index(st.session_state.selected_model) \
              if st.session_state.selected_model in model_options else 0
        st.session_state.selected_model = st.selectbox(
            "Ollama model", model_options, index=idx, label_visibility="collapsed"
        )
        st.caption("💡 Use `llama3.2:1b` on 8 GB RAM laptops")
        st.divider()

        # ── Subject hint ───────────────────────────────────────────────
        st.markdown("### 📚 Subject")
        subjects = {
            "general": "General / Mixed",
            "mathematics": "Mathematics",
            "circuits": "Circuits & Electronics",
            "programming": "Programming",
            "physics": "Physics",
            "engineering": "Engineering Mechanics",
        }
        st.session_state.subject_hint = st.selectbox(
            "Subject", list(subjects.keys()),
            format_func=lambda k: subjects[k],
            label_visibility="collapsed",
        )
        st.divider()

        # ── PDF uploader ───────────────────────────────────────────────
        st.markdown("### 📄 Upload Textbook")
        pdf_file = st.file_uploader(
            "Drop a PDF to add it to the knowledge base",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if pdf_file and st.button("📥 Index PDF", use_container_width=True):
            on_pdf_upload(pdf_file)

        if st.session_state.indexed_docs:
            st.markdown("**Indexed documents:**")
            for doc in st.session_state.indexed_docs:
                st.markdown(
                    f"<span class='badge badge-green'>✓ {doc.get('source','?')}</span>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Total chunks: {st.session_state.db_chunk_count:,}")
        st.divider()

        # ── Chat controls ──────────────────────────────────────────────
        if st.button("🗑️ Clear Chat", use_container_width=True):
            on_clear_chat()

        st.markdown(
            "<div style='position:fixed;bottom:20px;font-size:0.72rem;"
            "color:#8b949e;'>Dorm-Net v1.0 · Local LLM</div>",
            unsafe_allow_html=True,
        )


def _render_status_badges():
    online = st.session_state.get("ollama_online", False)
    db_ok = st.session_state.get("db_chunk_count", 0) > 0

    ollama_badge = (
        "<span class='badge badge-green'>● Ollama Online</span>"
        if online
        else "<span class='badge badge-red'>● Ollama Offline</span>"
    )
    db_badge = (
        f"<span class='badge badge-green'>● DB {st.session_state.db_chunk_count} chunks</span>"
        if db_ok
        else "<span class='badge badge-yellow'>● DB Empty</span>"
    )
    st.markdown(ollama_badge + " " + db_badge, unsafe_allow_html=True)
    if not online:
        st.warning("Run `ollama serve` in a terminal to start the LLM.", icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────── #
#  Chat renderer                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def render_chat_history():
    """
    Iterates st.session_state["messages"] and renders each bubble.
    """
    for msg in st.session_state.messages:
        _render_bubble(msg["role"], msg["content"])


def _render_bubble(role: str, content: str):
    is_user = role == "user"
    avatar = "👤" if is_user  else "🎓"
    name   = "You" if is_user else "Kebede (AI Tutor)"
    css_class = "user" if is_user else "bot"

    st.markdown(
        f"""
        <div class="chat-message {css_class}">
            <div class="avatar {css_class}">{avatar}</div>
            <div class="message-body">
                <div class="message-meta">{name}</div>
                {_markdown_safe(content)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _markdown_safe(text: str) -> str:
    """Convert newlines to <br> for HTML rendering inside our custom bubble."""
    import html as html_lib
    # Don't escape — let Streamlit render markdown properly via st.markdown
    # We just wrap with a div here; actual content is rendered below via columns trick
    return f"<div style='white-space:pre-wrap;'>{html_lib.escape(text)}</div>"


def render_chat_message_native(role: str, content: str):
    """
    Alternative renderer using Streamlit's native st.chat_message.
    Supports full markdown + LaTeX rendering. Use this for new messages.
    """
    icon = "human" if role == "user" else "assistant"
    with st.chat_message(icon):
        st.markdown(content)


# ─────────────────────────────────────────────────────────────────────────── #
#  OCR Panel                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def render_ocr_panel(on_image_upload: Callable):
    """
    Expandable panel for uploading a photo of handwritten notes.
    Calls on_image_upload(image_bytes) when submitted.
    """
    with st.expander("📷 Scan Handwritten Notes (OCR)", expanded=False):
        st.caption("Upload a photo of your notebook page — Dorm-Net will read it for you.")
        img_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
            key="ocr_uploader",
        )
        if img_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img_file, caption="Uploaded image", use_container_width=True)
            with col2:
                if st.button("🔍 Extract Text", use_container_width=True):
                    on_image_upload(img_file.read())

        # Show previous OCR result
        result = st.session_state.get("ocr_result")
        if result and result.success and result.raw_text:
            st.success(f"✅ Extracted {result.word_count} words (confidence: {result.confidence:.0f}%)")
            with st.expander("View extracted text"):
                st.text(result.raw_text)
            if st.button("💬 Ask about these notes", use_container_width=True):
                # Trigger chat with OCR context (handled in main_app.py)
                st.session_state["use_ocr_in_next_query"] = True
                st.toast("OCR text will be included in your next question!", icon="📝")


# ─────────────────────────────────────────────────────────────────────────── #
#  RAG source display                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def render_rag_sources(sources: list[dict]):
    """Show which textbook chunks were used to answer the last question."""
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} textbook source(s) used", expanded=False):
        for i, src in enumerate(sources):
            st.markdown(
                f"**Source {i+1}** — *{src.get('source', '?')}*, page {src.get('page', '?')}",
            )
            st.caption(src.get("snippet", "")[:300] + "...")
            st.divider()


# ─────────────────────────────────────────────────────────────────────────── #
#  Notification helpers                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def toast_success(msg: str): st.toast(msg, icon="✅")
def toast_error(msg: str):   st.toast(msg, icon="❌")
def toast_info(msg: str):    st.toast(msg, icon="ℹ️")
