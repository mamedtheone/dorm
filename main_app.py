"""
Streamlit frontend for Dorm-Net.

The app lets students:
1. Pick a course.
2. Upload PDFs for the local knowledge base.
3. Upload photos of handwritten notes for OCR.
4. Chat with a local Ollama model using retrieved course context.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from modules.brain import KnowledgeBase
from modules.ui import apply_dark_theme, render_hero_card, render_sources
from modules.vision import extract_text


# Page settings should be configured before other Streamlit UI calls.
st.set_page_config(page_title="Dorm-Net", page_icon="📚", layout="wide")


def get_kb(course_name: str) -> KnowledgeBase:
    """
    Build or retrieve a course-specific knowledge base from Streamlit cache.
    """

    return KnowledgeBase(course_name=course_name)


@st.cache_resource(show_spinner=False)
def cached_kb(course_name: str) -> KnowledgeBase:
    return get_kb(course_name)


def initialize_state() -> None:
    """
    Prepare Streamlit session keys used throughout the app.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_ocr_text" not in st.session_state:
        st.session_state.latest_ocr_text = ""
    if "last_course" not in st.session_state:
        st.session_state.last_course = "Calculus"


def reset_chat_if_course_changed(course_name: str) -> None:
    """
    Keep chat history separate per course so responses stay relevant.
    """

    if st.session_state.last_course != course_name:
        st.session_state.messages = []
        st.session_state.latest_ocr_text = ""
        st.session_state.last_course = course_name


def main() -> None:
    """
    Main Streamlit application flow.
    """

    apply_dark_theme()
    initialize_state()

    with st.sidebar:
        st.title("Dorm-Net")
        st.caption("Offline AI study assistant for AASTU students")

        selected_course = st.selectbox(
            "Choose a course",
            ["Calculus", "DSA", "Physics"],
            index=0,
        )

        reset_chat_if_course_changed(selected_course)
        kb = cached_kb(selected_course)

        st.markdown("---")
        st.subheader("Upload Course PDF")
        pdf_file = st.file_uploader(
            "Add lecture notes or handouts",
            type=["pdf"],
            key=f"pdf_{selected_course}",
        )

        if pdf_file is not None:
            if st.button("Ingest PDF into Knowledge Base", use_container_width=True):
                with st.spinner("Reading and indexing the PDF locally..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(pdf_file.getbuffer())
                        temp_pdf_path = temp_pdf.name

                    saved_chunks = kb.ingest_pdf(temp_pdf_path, source_label=pdf_file.name)
                    Path(temp_pdf_path).unlink(missing_ok=True)

                if saved_chunks > 0:
                    st.success(f"Stored {saved_chunks} chunks from {pdf_file.name}.")
                else:
                    st.warning("No readable text was found in that PDF.")

        st.markdown("---")
        st.subheader("Upload Handwritten Notes")
        note_image = st.file_uploader(
            "Add a photo of class notes",
            type=["png", "jpg", "jpeg"],
            key=f"notes_{selected_course}",
        )

        if note_image is not None:
            st.image(note_image, caption="Uploaded handwritten note", use_container_width=True)

            if st.button("Extract and Save Notes", use_container_width=True):
                with st.spinner("Running OCR locally with OpenCV + Tesseract..."):
                    image_bytes = note_image.read()
                    extracted_text = extract_text(image_bytes)
                    st.session_state.latest_ocr_text = extracted_text
                    saved_chunks = kb.ingest_text(extracted_text, source_label=note_image.name)

                if extracted_text:
                    st.success(f"OCR complete. Stored {saved_chunks} note chunks.")
                    st.text_area(
                        "Extracted note text",
                        extracted_text,
                        height=180,
                    )
                else:
                    st.warning("OCR finished, but no readable text was detected.")

        st.markdown("---")
        st.info(
            "Make sure Ollama is running locally and the `llama3.2` model is installed."
        )

    render_hero_card(selected_course)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"Ask a {selected_course} question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking locally with Ollama..."):
                result = kb.query(prompt)
                st.markdown(result.answer)

            render_sources(result.sources)

        st.session_state.messages.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
