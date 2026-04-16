# ui_components.py
import streamlit as st

def apply_dark_theme():
    st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #1c1f26;
        color: white;
    }
    .stTextArea textarea {
        background-color: #1c1f26;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def header():
    st.title("🧠 Dorm-Net")
    st.caption("Offline Peer Tutor for AASTU Students")


def upload_section():
    st.subheader("📸 Upload Notebook")
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    return file


def query_section():
    st.subheader("❓ Ask a Question")
    query = st.text_input("Type your question here...")
    return query


def display_answer(answer, sources):
    st.subheader("📘 Answer")
    st.write(answer)

    st.subheader("📚 Sources")
    for s in sources:
        st.write("- ", s)


def display_confidence(score):
    st.subheader("📊 Confidence Score")
    st.progress(score)
    st.write(f"{score*100:.2f}% confidence")