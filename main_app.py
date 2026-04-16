# app.py
import streamlit as st
from modules.vision import extract_text
from modules.brain import add_documents, query
from modules.ui import *

apply_dark_theme()
header()

# -------- Upload Section --------
file = upload_section()

if file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(file.read())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    if st.button("🧾 Extract Notes"):
        text = extract_text("temp.jpg")

        st.subheader("📝 Extracted Text")
        st.text_area("", text, height=200)

        # Add to knowledge base
        chunks = text.split("\n")
        add_documents(chunks)

        st.success("✅ Notes added to brain!")

# -------- Query Section --------
query_text = query_section()

if st.button("🔍 Ask Dorm-Net"):
    results = query(query_text)

    # Flatten results
    sources = [item for sublist in results for item in sublist]

    # Simple answer (join top results)
    answer = " ".join(sources[:2])

    # Fake ground truth (for demo)
    

    display_answer(answer, sources)
    