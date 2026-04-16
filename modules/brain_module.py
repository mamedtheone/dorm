"""
brain_module.py — Dorm-Net RAG Engine
======================================
Handles PDF ingestion, embedding, vector storage (ChromaDB),
and semantic retrieval for the tutoring pipeline.

Key design decisions:
  - Checks for existing DB to avoid re-indexing on every startup
  - Uses sentence-transformers (all-MiniLM-L6-v2) — CPU-friendly, ~80 MB
  - Recursive Character Text Splitting: chunk=1000, overlap=200
  - Async query so the UI never freezes during retrieval
"""

import os
import hashlib
import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# ── pysqlite3 shim ── must happen BEFORE any chromadb import ──────────────
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # Linux with system sqlite3 >= 3.35 is fine

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)

# ─────────────────────────────────────────────────────────────────────────── #
#  Data structures                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Chunk:
    text: str
    source: str          # filename
    page: int
    chunk_index: int
    doc_hash: str


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    distances: list[float]
    success: bool
    error: Optional[str] = None


@dataclass
class IngestionReport:
    source: str
    total_pages: int
    total_chunks: int
    skipped: bool        # True if doc was already indexed
    success: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────── #
#  Recursive Character Text Splitter                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class RecursiveTextSplitter:
    """
    Splits text hierarchically:
      1. Double newline  (paragraph breaks)
      2. Single newline
      3. Period / sentence end
      4. Space (word boundary)
      5. Character (last resort)

    Mirrors LangChain's RecursiveCharacterTextSplitter logic — no dependency needed.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        chunks: list[str] = []
        self._split_recursive(text, self.SEPARATORS, chunks)
        return [c.strip() for c in chunks if c.strip()]

    def _split_recursive(self, text: str, separators: list[str], result: list[str]):
        if len(text) <= self.chunk_size:
            result.append(text)
            return

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if separators else []

        if sep and sep in text:
            parts = text.split(sep)
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    # Carry overlap into next chunk
                    overlap_text = self._get_overlap(current)
                    current = (overlap_text + sep + part) if overlap_text else part
                    if len(current) > self.chunk_size:
                        self._split_recursive(current, remaining_seps, result)
                        current = ""
            if current:
                result.append(current)
        else:
            # No separator found at this level; recurse deeper
            if remaining_seps:
                self._split_recursive(text, remaining_seps, result)
            else:
                # Hard split by character
                start = 0
                while start < len(text):
                    result.append(text[start: start + self.chunk_size])
                    start += self.chunk_size - self.chunk_overlap

    def _get_overlap(self, text: str) -> str:
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]


# ─────────────────────────────────────────────────────────────────────────── #
#  RAG Manager                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class RAGManager:
    """
    Manages the full RAG lifecycle:
      ingest_pdf()  →  build or update the vector store
      query()       →  retrieve top-k relevant chunks

    Persistence:
      - ChromaDB is stored in `db_path` on disk (survives restarts).
      - A sidecar manifest file (ingested_docs.json) tracks which PDFs have
        been indexed so we never re-embed the same file twice.
    """

    COLLECTION_NAME = "dorm_net_textbooks"
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: str = "./dorm_net_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.db_path / "ingested_docs.json"

        self.splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)

        logger.info(f"Loading embedding model: {self.MODEL_NAME}")
        self.embedder = SentenceTransformer(self.MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready at '{self.db_path}' — "
            f"{self.collection.count()} chunks loaded."
        )

    # ── Public async API ─────────────────────────────────────────────── #

    async def ingest_pdf_async(self, pdf_path: str) -> IngestionReport:
        """Non-blocking PDF ingestion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.ingest_pdf, pdf_path)

    async def query_async(
        self,
        query: str,
        top_k: int = 4,
    ) -> RetrievalResult:
        """Non-blocking semantic search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.query, query, top_k)

    # ── Synchronous workers ──────────────────────────────────────────── #

    def ingest_pdf(self, pdf_path: str) -> IngestionReport:
        """
        Extract text from PDF, chunk it, embed, and store in ChromaDB.
        Skips the file if it has already been indexed (hash-based check).
        """
        pdf_path = Path(pdf_path)
        source_name = pdf_path.name

        try:
            doc_hash = self._file_hash(pdf_path)

            # ── Already indexed? ────────────────────────────────────────
            manifest = self._load_manifest()
            if doc_hash in manifest:
                logger.info(f"Skipping '{source_name}' — already indexed.")
                return IngestionReport(
                    source=source_name,
                    total_pages=manifest[doc_hash]["pages"],
                    total_chunks=manifest[doc_hash]["chunks"],
                    skipped=True,
                    success=True,
                )

            # ── Extract text page by page ───────────────────────────────
            doc = fitz.open(str(pdf_path))
            all_chunks: list[Chunk] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if not text.strip():
                    continue

                page_chunks = self.splitter.split(text)
                for idx, chunk_text in enumerate(page_chunks):
                    all_chunks.append(
                        Chunk(
                            text=chunk_text,
                            source=source_name,
                            page=page_num + 1,
                            chunk_index=idx,
                            doc_hash=doc_hash,
                        )
                    )

            doc.close()
            logger.info(f"Extracted {len(all_chunks)} chunks from '{source_name}'")

            # ── Embed and store ─────────────────────────────────────────
            if all_chunks:
                self._store_chunks(all_chunks)

            # ── Update manifest ─────────────────────────────────────────
            manifest[doc_hash] = {
                "source": source_name,
                "pages": len(doc) if not doc.is_closed else 0,
                "chunks": len(all_chunks),
            }
            self._save_manifest(manifest)

            return IngestionReport(
                source=source_name,
                total_pages=manifest[doc_hash]["pages"],
                total_chunks=len(all_chunks),
                skipped=False,
                success=True,
            )

        except Exception as exc:
            logger.exception(f"Ingestion failed for '{pdf_path}'")
            return IngestionReport(
                source=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                skipped=False,
                success=False,
                error=str(exc),
            )

    def query(self, query: str, top_k: int = 4) -> RetrievalResult:
        """
        Embed the query and retrieve the top_k most relevant chunks.
        """
        try:
            if self.collection.count() == 0:
                return RetrievalResult(
                    chunks=[], query=query, distances=[], success=True
                )

            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            chunks: list[Chunk] = []
            distances: list[float] = []

            for doc_text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(
                    Chunk(
                        text=doc_text,
                        source=meta.get("source", "unknown"),
                        page=int(meta.get("page", 0)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        doc_hash=meta.get("doc_hash", ""),
                    )
                )
                distances.append(float(dist))

            return RetrievalResult(
                chunks=chunks, query=query, distances=distances, success=True
            )

        except Exception as exc:
            logger.exception("Query failed")
            return RetrievalResult(
                chunks=[], query=query, distances=[], success=False, error=str(exc)
            )

    def list_indexed_docs(self) -> list[dict]:
        """Returns metadata for all currently indexed documents."""
        manifest = self._load_manifest()
        return list(manifest.values())

    def get_chunk_count(self) -> int:
        return self.collection.count()

    # ── Internal helpers ─────────────────────────────────────────────── #

    def _store_chunks(self, chunks: list[Chunk]):
        """Batch embed and upsert chunks into ChromaDB (batches of 64)."""
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
            ids = [
                f"{c.doc_hash}_{c.page}_{c.chunk_index}" for c in batch
            ]
            metadatas = [
                {
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "doc_hash": c.doc_hash,
                }
                for c in batch
            ]
            self.collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")

    @staticmethod
    def _file_hash(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()[:16]

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest: dict):
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
