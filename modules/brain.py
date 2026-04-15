"""
RAG engine for Dorm-Net.

This module keeps all knowledge-base logic in one place so the Streamlit UI
can stay focused on user interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import uuid

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader


@dataclass
class QueryResult:
    """
    Small container that keeps the answer and the retrieved study snippets
    together. This is useful in the frontend when we want to show sources.
    """

    answer: str
    sources: List[Document]


class KnowledgeBase:
    """
    Local Retrieval-Augmented Generation (RAG) engine.

    Responsibilities:
    1. Read and chunk course materials.
    2. Save chunks into a local Chroma vector database.
    3. Retrieve the most relevant chunks for a student question.
    4. Ask the local Ollama model to answer using those chunks.
    """

    def __init__(
        self,
        course_name: str,
        persist_root: str = "data/chroma",
        model_name: str = "llama3.2",
    ) -> None:
        """
        Create a course-specific knowledge base.

        Each course gets its own Chroma collection so Calculus notes do not mix
        with Physics or DSA content.
        """

        self.course_name = course_name
        self.model_name = model_name
        self.persist_directory = Path(persist_root) / course_name.lower().replace(" ", "_")
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # OllamaEmbeddings runs locally through the Ollama server.
        # No external network calls are made here.
        self.embeddings = OllamaEmbeddings(model=model_name)

        # Chroma stores vectors on disk so the knowledge base survives restarts.
        self.vector_store = Chroma(
            collection_name=f"dorm_net_{course_name.lower().replace(' ', '_')}",
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )

        # The local language model that writes the final answer.
        self.llm = Ollama(model=model_name, temperature=0.2)

        # This splitter creates overlapping chunks to preserve context.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_pdf(self, pdf_path: str, source_label: str | None = None) -> int:
        """
        Read a PDF, split it into chunks, and store the chunks in Chroma.

        Returns:
            int: Number of chunks saved to the vector database.
        """

        reader = PdfReader(pdf_path)
        pages = []

        # Read page by page so we can preserve useful metadata.
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()

            # Skip fully empty pages because they add noise to retrieval.
            if not cleaned_text:
                continue

            pages.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "course": self.course_name,
                        "source": source_label or Path(pdf_path).name,
                        "page": page_number,
                        "type": "pdf",
                    },
                )
            )

        if not pages:
            return 0

        chunks = self.splitter.split_documents(pages)
        self._save_documents(chunks)
        return len(chunks)

    def ingest_text(self, text: str, source_label: str = "handwritten_notes") -> int:
        """
        Save raw text into the vector database.

        The frontend uses this for OCR output from handwritten notes.
        """

        cleaned_text = text.strip()
        if not cleaned_text:
            return 0

        chunks = self.splitter.split_documents(
            [
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "course": self.course_name,
                        "source": source_label,
                        "page": 1,
                        "type": "ocr_note",
                    },
                )
            ]
        )
        self._save_documents(chunks)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> QueryResult:
        """
        Retrieve relevant study material and generate an answer.

        The assistant persona is a supportive AASTU senior student who explains
        things clearly and encourages the learner without sounding robotic.
        """

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        source_documents = retriever.get_relevant_documents(question)

        if source_documents:
            context = "\n\n".join(
                [
                    (
                        f"Source: {doc.metadata.get('source', 'unknown')} "
                        f"(page {doc.metadata.get('page', 'n/a')})\n"
                        f"{doc.page_content}"
                    )
                    for doc in source_documents
                ]
            )
        else:
            context = "No course context was found in the local knowledge base."

        prompt = f"""
You are Dorm-Net, a supportive AASTU senior student helping a junior learn.

Rules:
- Be warm, practical, and encouraging.
- Explain step by step when helpful.
- If the context is weak, say so honestly.
- Prefer examples connected to university study habits.
- Do not claim to use the internet or external sources.

Course: {self.course_name}

Retrieved context:
{context}

Student question:
{question}

Answer as the supportive AASTU senior student:
""".strip()

        answer = self.llm.invoke(prompt)
        return QueryResult(answer=answer, sources=source_documents)

    def _save_documents(self, documents: List[Document]) -> None:
        """
        Internal helper that writes documents to Chroma and persists them to disk.
        """

        if not documents:
            return

        ids = [str(uuid.uuid4()) for _ in documents]
        self.vector_store.add_documents(documents=documents, ids=ids)
        self.vector_store.persist()
