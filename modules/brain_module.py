"""
brain_module.py - Dorm-Net RAG Engine (Upgraded)
================================================
Changes from v1:
  - Lazy PDF loading: pages streamed one at a time (no full doc in RAM)
  - BM25 index built in-memory alongside ChromaDB vectors
  - Hybrid retrieval: vector + BM25 fused with Reciprocal Rank Fusion (RRF)
  - Quiz generation: 3 MCQs + 2 short-answer questions from retrieved context
"""
import re
import random  # <--- ADD THIS LINE

import asyncio
import shutil
import gc
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys

try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
import fitz
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int
    doc_hash: str


@dataclass
class RetrievalResult:
    chunks: list
    query: str
    distances: list
    success: bool
    error: Optional[str] = None


@dataclass
class IngestionReport:
    source: str
    total_pages: int
    total_chunks: int
    skipped: bool
    success: bool
    error: Optional[str] = None


@dataclass
class QuizQuestion:
    question: str
    question_type: str = "mcq"
    options: list = field(default_factory=list)
    answer_index: Optional[int] = None
    answer: str = ""
    explanation: str = ""


class RecursiveTextSplitter:
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list:
        chunks = []
        self._split_recursive(text, self.SEPARATORS, chunks)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_recursive(self, text: str, separators: list, result: list):
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
                    overlap_text = self._get_overlap(current)
                    current = (overlap_text + sep + part) if overlap_text else part
                    if len(current) > self.chunk_size:
                        self._split_recursive(current, remaining_seps, result)
                        current = ""
            if current:
                result.append(current)
            return

        if remaining_seps:
            self._split_recursive(text, remaining_seps, result)
            return

        start = 0
        while start < len(text):
            result.append(text[start : start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap

    def _get_overlap(self, text: str) -> str:
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap :]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: dict[str, int] = defaultdict(int)
        self.avgdl: float = 0.0
        self.N: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def add_documents(self, texts: list[str], ids: list[str]):
        for text, doc_id in zip(texts, ids):
            tokens = self._tokenize(text)
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)
            for token in set(tokens):
                self.df[token] += 1
        self.N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / self.N if self.N else 1.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = []
        for tokens in self.corpus:
            tf_map: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            score = 0.0
            doc_length = len(tokens)
            for token in query_tokens:
                if token not in self.df:
                    continue
                tf = tf_map.get(token, 0)
                idf = math.log(
                    (self.N - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1
                )
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avgdl
                )
                score += idf * numerator / denominator
            scores.append(score)

        ranked = sorted(
            [(self.doc_ids[index], scores[index]) for index in range(len(scores))],
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


class RAGManager:
    COLLECTION_NAME = "dorm_net_textbooks"
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: str = "./dorm_net_db",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.db_path / "ingested_docs.json"
        self.source_docs_path = self.db_path / "source_docs"
        self.source_docs_path.mkdir(parents=True, exist_ok=True)

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

        self.bm25 = BM25Index()
        self._rebuild_bm25_index()

        logger.info(
            f"ChromaDB ready at '{self.db_path}' - "
            f"{self.collection.count()} chunks | BM25 docs={self.bm25.N}"
        )

    async def ingest_pdf_async(self, pdf_path: str) -> IngestionReport:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self.ingest_pdf, pdf_path)

    async def query_async(self, query: str, top_k: int = 5) -> RetrievalResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self.query, query, top_k)

    async def generate_quiz_async(
        self, topic: str, n_questions: int = 5
    ) -> list:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self.generate_quiz, topic, n_questions
        )

    def ingest_pdf(self, pdf_path: str) -> IngestionReport:
        pdf_path = Path(pdf_path)
        source_name = pdf_path.name

        try:
            doc_hash = self._file_hash(pdf_path)
            manifest = self._load_manifest()
            if doc_hash in manifest:
                logger.info(f"Skipping '{source_name}' - already indexed.")
                return IngestionReport(
                    source=source_name,
                    total_pages=manifest[doc_hash]["pages"],
                    total_chunks=manifest[doc_hash]["chunks"],
                    skipped=True,
                    success=True,
                )

            all_chunks: list[Chunk] = []
            doc = fitz.open(str(pdf_path))
            try:
                total_pages = len(doc)
                for page_num in range(total_pages):
                    page = doc[page_num]
                    text = page.get_text("text")
                    page = None
                    if not text or not text.strip():
                        continue

                    page_chunks = self.splitter.split(text)
                    for chunk_index, chunk_text in enumerate(page_chunks):
                        all_chunks.append(
                            Chunk(
                                text=chunk_text,
                                source=source_name,
                                page=page_num + 1,
                                chunk_index=chunk_index,
                                doc_hash=doc_hash,
                            )
                        )
            finally:
                doc.close()

            if all_chunks:
                self._store_chunks(all_chunks)
                self.bm25.add_documents(
                    [chunk.text for chunk in all_chunks],
                    [
                        f"{chunk.doc_hash}_{chunk.page}_{chunk.chunk_index}"
                        for chunk in all_chunks
                    ],
                )

            stored_pdf_path = self._persist_source_pdf(pdf_path, doc_hash)
            manifest[doc_hash] = {
                "source": source_name,
                "pages": total_pages,
                "chunks": len(all_chunks),
                "stored_path": str(stored_pdf_path),
            }
            self._save_manifest(manifest)
            gc.collect()

            return IngestionReport(
                source=source_name,
                total_pages=total_pages,
                total_chunks=len(all_chunks),
                skipped=False,
                success=True,
            )
        except Exception as exc:
            logger.exception(f"Ingestion failed for '{pdf_path}'")
            gc.collect()
            return IngestionReport(
                source=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                skipped=False,
                success=False,
                error=str(exc),
            )

    def query(self, query: str, top_k: int = 5) -> RetrievalResult:
        try:
            count = self.collection.count()
            if count == 0:
                return RetrievalResult(
                    chunks=[],
                    query=query,
                    distances=[],
                    success=True,
                )

            fetch_k = min(max(top_k * 2, top_k), count)
            query_embedding = self.embedder.encode(query).tolist()
            vec_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
            vec_ids: list[str] = vec_results["ids"][0]

            bm25_hits = self.bm25.search(query, top_k=fetch_k)
            bm25_ids = [item[0] for item in bm25_hits]
            fused_ids = reciprocal_rank_fusion([vec_ids, bm25_ids])[:top_k]

            id_to_text: dict[str, str] = {}
            id_to_meta: dict[str, dict] = {}
            id_to_dist: dict[str, float] = {}
            for doc_text, meta, dist, doc_id in zip(
                vec_results["documents"][0],
                vec_results["metadatas"][0],
                vec_results["distances"][0],
                vec_ids,
            ):
                id_to_text[doc_id] = doc_text
                id_to_meta[doc_id] = meta
                id_to_dist[doc_id] = float(dist)

            missing_ids = [doc_id for doc_id in fused_ids if doc_id not in id_to_text]
            if missing_ids:
                extra = self.collection.get(
                    ids=missing_ids,
                    include=["documents", "metadatas"],
                )
                for doc_text, meta, doc_id in zip(
                    extra["documents"], extra["metadatas"], extra["ids"]
                ):
                    id_to_text[doc_id] = doc_text
                    id_to_meta[doc_id] = meta
                    id_to_dist[doc_id] = 0.5

            chunks: list[Chunk] = []
            distances: list[float] = []
            for doc_id in fused_ids:
                if doc_id not in id_to_text:
                    continue
                meta = id_to_meta[doc_id]
                chunks.append(
                    Chunk(
                        text=id_to_text[doc_id],
                        source=meta.get("source", "unknown"),
                        page=int(meta.get("page", 0)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        doc_hash=meta.get("doc_hash", ""),
                    )
                )
                distances.append(id_to_dist.get(doc_id, 0.5))

            return RetrievalResult(
                chunks=chunks,
                query=query,
                distances=distances,
                success=True,
            )
        except Exception as exc:
            logger.exception("Hybrid query failed")
            return RetrievalResult(
                chunks=[],
                query=query,
                distances=[],
                success=False,
                error=str(exc),
            )

    def generate_quiz(self, topic: str, n_questions: int = 5) -> list:
        retrieval = self.query(topic, top_k=max(n_questions*3, 15))
        if not retrieval.success or not retrieval.chunks:
            return []

        sentence_bank = self._build_sentence_bank(retrieval.chunks)
        if not sentence_bank:
            return []
        random.shuffle(sentence_bank)

        questions: list[QuizQuestion] = []

        for item in sentence_bank[:3]:
            distractors = self._build_statement_distractors(
                item["sentence"],
                sentence_bank,
                answer_term=item["keyword"],
            )
            if len(distractors) < 3:
                continue

            options = [item["sentence"]] + distractors[:3]
            random.shuffle(options)
            answer_index = options.index(item["sentence"])
            questions.append(
                QuizQuestion(
                    question=(
                        f"Which statement is supported by the retrieved material "
                        f"about {item['keyword']}?"
                    ),
                    question_type="mcq",
                    options=options,
                    answer_index=answer_index,
                    answer=item["sentence"],
                    explanation=(
                        f"Supported by {item['chunk'].source}, page {item['chunk'].page}."
                    ),
                )
            )

        for item in sentence_bank[3:]:
            short_count = len(
                [question for question in questions if question.question_type == "short_answer"]
            )
            if short_count >= 2:
                break
            questions.append(
                QuizQuestion(
                    question=(
                        "Using only the retrieved material, explain this idea in your "
                        f"own words: {item['sentence']}"
                    ),
                    question_type="short_answer",
                    answer=item["sentence"],
                    explanation=(
                        f"Reference answer from {item['chunk'].source}, page {item['chunk'].page}."
                    ),
                )
            )

        return questions[: max(n_questions, 5)]

    def list_indexed_docs(self) -> list:
        return list(self._load_manifest().values())

    def get_chunk_count(self) -> int:
        return self.collection.count()

    def _rebuild_bm25_index(self):
        total = self.collection.count()
        if total == 0:
            return

        batch_size = 500
        offset = 0
        while offset < total:
            results = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents"],
            )
            if not results["ids"]:
                break
            self.bm25.add_documents(results["documents"], results["ids"])
            offset += batch_size

        logger.info(f"BM25 index rebuilt: {self.bm25.N} docs")

    def _build_sentence_bank(self, chunks: list[Chunk]) -> list[dict]:
        bank: list[dict] = []
        seen: set[str] = set()
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                cleaned = " ".join(sentence.split()).strip()
                if len(cleaned) < 50 or cleaned in seen:
                    continue
                keyword = self._extract_keyword(cleaned)
                if not keyword:
                    continue
                seen.add(cleaned)
                bank.append(
                    {
                        "sentence": cleaned,
                        "keyword": keyword,
                        "chunk": chunk,
                    }
                )
        return bank

    def _extract_keyword(self, sentence: str) -> str:
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "of",
            "to",
            "and",
            "or",
            "for",
            "with",
            "at",
            "by",
            "that",
            "this",
            "from",
        }
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", sentence)
        for word in candidates:
            if word.lower() not in stop:
                return word
        return ""

    def _build_statement_distractors(
        self, sentence: str, sentence_bank: list[dict], answer_term: str
    ) -> list[str]:
        distractors: list[str] = []
        replacements = [
            item["keyword"]
            for item in sentence_bank
            if item["keyword"].lower() != answer_term.lower()
        ]

        for replacement in replacements:
            mutated = re.sub(
                rf"\b{re.escape(answer_term)}\b",
                replacement,
                sentence,
                count=1,
            )
            if mutated != sentence and mutated not in distractors:
                distractors.append(mutated)
            if len(distractors) >= 3:
                break

        if len(distractors) < 3:
            for item in sentence_bank:
                alt_sentence = item["sentence"]
                if alt_sentence != sentence and alt_sentence not in distractors:
                    distractors.append(alt_sentence)
                if len(distractors) >= 3:
                    break

        return distractors

    def _store_chunks(self, chunks: list[Chunk]):
        batch_size = 64
        for index in range(0, len(chunks), batch_size):
            batch = chunks[index : index + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=False,
            ).tolist()
            ids = [
                f"{chunk.doc_hash}_{chunk.page}_{chunk.chunk_index}" for chunk in batch
            ]
            metadatas = [
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "doc_hash": chunk.doc_hash,
                }
                for chunk in batch
            ]
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")

    def _persist_source_pdf(self, pdf_path: Path, doc_hash: str) -> Path:
        suffix = pdf_path.suffix or ".pdf"
        target_path = self.source_docs_path / f"{doc_hash}{suffix.lower()}"
        if not target_path.exists():
            shutil.copy2(pdf_path, target_path)
        return target_path

    @staticmethod
    def _file_hash(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()[:16]

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    def _save_manifest(self, manifest: dict):
        with open(self.manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
