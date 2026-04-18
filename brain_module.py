"""
================================================================================
  DORM-NET: brain_module.py  —  The Knowledge Architect
================================================================================
  Role    : Option B — "The Memory" of the Dorm-Net system
  Author  : Knowledge Architect Team
  Stack   : ChromaDB (Vector DB) + SentenceTransformers (Embeddings) + Ollama (LLM)

  Responsibilities:
    1. Ingest "Golden Summaries" (AASTU textbook content) into ChromaDB
    2. Smart text chunking for large documents
    3. Semantic search to retrieve the most relevant context
    4. RAG pipeline: inject retrieved context into Ollama's prompt
    5. Conversation memory for multi-turn tutoring sessions
    6. Full document management (add, list, delete, update)
================================================================================
"""

import os
import sys
import textwrap
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ─────────────────────────────────────────────────────────────

# Where ChromaDB will persist its data (creates a local folder)
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Embedding model — lightweight (22MB), runs offline after first download
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama model — must be pulled first: `ollama pull llama3.2`
OLLAMA_MODEL = "llama3.2"

# How many top-matching chunks to retrieve per query
TOP_K_RESULTS = 3

# Maximum characters per chunk when splitting large documents
CHUNK_SIZE = 800

# Overlap between chunks so context isn't lost at boundaries
CHUNK_OVERLAP = 100

# ── Database Initialization ────────────────────────────────────────────────────

def _init_db():
    """
    Initializes ChromaDB and returns the collection.
    Called once at module load. Uses a persistent local client so data
    survives between sessions (stored in ./chroma_db/).
    """
    client = chromadb.PersistentClient(path=DB_DIR)

    # SentenceTransformer embedding function — converts text to semantic vectors
    # This is what enables MEANING-BASED search, not just keyword matching
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # One collection holds all AASTU knowledge. Think of it as a smart table
    # where each row is a chunk of text with its vector embedding.
    collection = client.get_or_create_collection(
        name="aastu_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"description": "AASTU course materials for Dorm-Net offline tutor"}
    )

    return collection

# Module-level collection — initialized once when the module is imported
try:
    _collection = _init_db()
except Exception as e:
    print(f"[BRAIN] CRITICAL: Could not initialize ChromaDB: {e}")
    sys.exit(1)

# ── Utility: Text Chunker ─────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a large document into overlapping chunks so:
    - Each chunk fits within the embedding model's context window
    - Overlap ensures key ideas at chunk boundaries are not lost

    Example:
        "ABCDE" with chunk_size=3, overlap=1 → ["ABC", "CDE"]
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to cut at a sentence boundary (period/newline) within the last 20%
        boundary_zone = text[start + int(chunk_size * 0.8): end]
        last_period = boundary_zone.rfind('. ')
        last_newline = boundary_zone.rfind('\n')
        boundary = max(last_period, last_newline)

        if boundary != -1:
            end = start + int(chunk_size * 0.8) + boundary + 1

        chunks.append(text[start:end].strip())
        start = end - overlap  # step back by overlap amount

    return [c for c in chunks if c]  # remove any empty strings


# ── Core API: Add Knowledge ───────────────────────────────────────────────────

def add_golden_summary(text: str, document_id: str, metadata: dict = None):
    """
    Ingests a 'Golden Summary' into the local Vector Database.

    Handles large documents automatically by chunking them.
    Each chunk is stored as a separate searchable entry with its own embedding.

    Parameters:
        text        : The full text content (lecture notes, textbook summary, etc.)
        document_id : A unique identifier (e.g., "calc2_integration_ch3")
        metadata    : Optional dict — e.g., {"course": "Calculus II", "chapter": 3}

    Example:
        add_golden_summary(
            text="Integration by parts formula: ...",
            document_id="calc2_ch3_integration",
            metadata={"course": "Calculus II", "chapter": 3}
        )
    """
    if not text or not text.strip():
        print("[BRAIN] WARNING: Empty text provided. Skipping.")
        return

    if metadata is None:
        metadata = {}

    # Split into chunks
    chunks = _chunk_text(text)
    total  = len(chunks)

    print(f"\n[BRAIN] Indexing '{document_id}' ({total} chunk(s))...")

    # Check if document already exists — delete old version before re-adding
    existing = _collection.get(where={"source_doc": document_id})
    if existing["ids"]:
        _collection.delete(ids=existing["ids"])
        print(f"[BRAIN]  -> Replaced {len(existing['ids'])} existing chunk(s).")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i:03d}" if total > 1 else document_id

        # Enrich metadata with chunk info for traceability
        chunk_meta = {
            **metadata,
            "source_doc" : document_id,
            "chunk_index": i,
            "total_chunks": total
        }

        _collection.add(
            documents=[chunk],
            ids=[chunk_id],
            metadatas=[chunk_meta]
        )
        print(f"[BRAIN]  -> Chunk [{i+1}/{total}] indexed (ID: {chunk_id})")

    print(f"[OK] '{document_id}' successfully stored in Local Memory.\n")


# ── Core API: Semantic Search ─────────────────────────────────────────────────

def retrieve_context(query: str, n_results: int = TOP_K_RESULTS,
                     course_filter: str = None) -> tuple[str, list[dict]]:
    """
    Semantic Search: Finds the most relevant chunks from the database.

    Unlike keyword search, this understands MEANING. A query about
    "how to find the area under a curve" will correctly match content
    about "definite integrals" even if those exact words aren't used.

    Parameters:
        query         : The student's question
        n_results     : How many top chunks to retrieve
        course_filter : Optionally restrict search to one course (e.g. "Physics II")

    Returns:
        - context_str : Combined text of all retrieved chunks, ready for the prompt
        - sources     : List of metadata dicts for each retrieved chunk (for transparency)
    """
    total_docs = _collection.count()
    if total_docs == 0:
        return "", []

    # Clamp n_results so we don't request more than what exists
    n_results = min(n_results, total_docs)

    # Optional filter to only search within a specific course
    where_clause = {"course": course_filter} if course_filter else None

    query_kwargs = {
        "query_texts": [query],
        "n_results"  : n_results,
        "include"    : ["documents", "distances", "metadatas"]
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    results = _collection.query(**query_kwargs)

    docs      = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]

    if not docs:
        return "", []

    # Build sources list with relevance score (convert distance to 0-100% score)
    sources = []
    for doc, dist, meta in zip(docs, distances, metas):
        similarity_pct = round((1 - dist) * 100, 1)
        sources.append({
            "text"      : doc,
            "similarity": similarity_pct,
            "course"    : meta.get("course", "Unknown"),
            "source_doc": meta.get("source_doc", "Unknown"),
        })

    # Combine retrieved chunks into a single context block
    context_str = "\n\n---\n\n".join(
        f"[Source: {s['source_doc']} | Course: {s['course']} | Relevance: {s['similarity']}%]\n{s['text']}"
        for s in sources
    )

    return context_str, sources


# ── Core API: RAG Pipeline (Ask the Tutor) ────────────────────────────────────

def ask_aastu_senior(query: str, conversation_history: list = None,
                     course_filter: str = None, verbose: bool = True) -> str:
    """
    The Complete RAG Pipeline — the heart of Dorm-Net.

    RETRIEVAL-AUGMENTED GENERATION in 3 steps:
      1. RETRIEVE: Search ChromaDB for relevant course content
      2. AUGMENT:  Inject that content into a carefully engineered prompt
      3. GENERATE: Send to Ollama (llama3.2) to produce a tutored response

    Parameters:
        query                : The student's question
        conversation_history : List of {"role": role, "content": content} dicts
                               Enables multi-turn tutoring sessions
        course_filter        : Restrict knowledge search to one course
        verbose              : Print status messages if True

    Returns:
        The AI tutor's response as a string
    """
    if verbose:
        print(f"\n[SEARCH] Looking up: '{query}'")
        if course_filter:
            print(f"[SEARCH] Filtering by course: {course_filter}")

    # ── Step 1: RETRIEVE ──────────────────────────────────────────────────────
    context, sources = retrieve_context(query, course_filter=course_filter)

    if verbose:
        if sources:
            print(f"[SEARCH] Found {len(sources)} relevant chunk(s):")
            for s in sources:
                print(f"         - [{s['similarity']}%] {s['source_doc']} ({s['course']})")
        else:
            print("[WARN] No matching content found in DB. Using general knowledge.")

    # ── Step 2: AUGMENT (Prompt Engineering) ─────────────────────────────────
    #
    # This is where Prompt Engineering happens. The system prompt defines:
    #   - The AI's persona (AASTU senior student)
    #   - Its behavior rules (prioritize course notes, be encouraging)
    #   - The retrieved context (actual textbook/note content)
    #
    context_section = (
        f"RELEVANT COURSE MATERIALS:\n{'='*40}\n{context}\n{'='*40}\n"
        if context else
        "NOTE: No specific course material found. Answering from general knowledge.\n"
    )

    system_prompt = textwrap.dedent(f"""
        You are "Senior", a brilliant and supportive 4th-year student at AASTU
        (Addis Ababa Science and Technology University) who tutors juniors offline.

        YOUR PERSONA:
        - You speak like a knowledgeable, encouraging friend — not a textbook.
        - You break down hard concepts into simple, clear steps.
        - You use analogies and real examples when helpful.
        - You always end by checking if the student understands or needs more help.
        - If the student seems stressed, be extra reassuring.

        YOUR RULES:
        1. ALWAYS prioritize the provided course materials below.
        2. If the answer is in the materials, cite it: "According to your notes..."
        3. If not in the materials, use general knowledge but say: "This isn't in your notes, but..."
        4. Never make up facts. If unsure, say so honestly.
        5. Keep your answers focused and exam-relevant.

        {context_section}
    """).strip()

    # ── Step 3: GENERATE (Call Ollama) ────────────────────────────────────────
    if verbose:
        print(f"[AI] Sending to {OLLAMA_MODEL}...")

    # Build the message list (supports multi-turn conversation)
    messages = [{"role": "system", "content": system_prompt}]

    # Inject conversation history if provided (multi-turn memory)
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current question
    messages.append({"role": "user", "content": query})

    try:
        import ollama
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer   = response["message"]["content"]

        if verbose:
            print("[AI] Response received.\n")

        return answer

    except ImportError:
        return "[ERROR] 'ollama' Python package not installed. Run: pip install ollama"
    except Exception as e:
        err_msg = str(e)
        if "connection refused" in err_msg.lower() or "connect" in err_msg.lower():
            return (
                "[ERROR] Cannot connect to Ollama. Please:\n"
                "  1. Open the Ollama app\n"
                "  2. Make sure it is running in the background\n"
                f"  Details: {e}"
            )
        return f"[ERROR] Ollama error: {e}"


# ── Document Management API ───────────────────────────────────────────────────

def list_documents() -> list[dict]:
    """
    Lists all documents currently stored in the knowledge base.

    Returns a list of dicts with 'source_doc', 'course', and 'chunks' count.
    """
    if _collection.count() == 0:
        print("[BRAIN] The knowledge base is empty.")
        return []

    all_items = _collection.get(include=["metadatas"])
    metas     = all_items.get("metadatas", [])

    # Group by source document
    doc_map = {}
    for meta in metas:
        doc_id = meta.get("source_doc", "unknown")
        course = meta.get("course", "Unknown")
        if doc_id not in doc_map:
            doc_map[doc_id] = {"source_doc": doc_id, "course": course, "chunks": 0}
        doc_map[doc_id]["chunks"] += 1

    docs = sorted(doc_map.values(), key=lambda x: x["course"])
    return docs


def delete_document(document_id: str) -> bool:
    """
    Removes all chunks belonging to a given document from the knowledge base.

    Returns True if deleted, False if document was not found.
    """
    existing = _collection.get(where={"source_doc": document_id})
    if not existing["ids"]:
        print(f"[BRAIN] Document '{document_id}' not found.")
        return False

    _collection.delete(ids=existing["ids"])
    print(f"[OK] Deleted '{document_id}' ({len(existing['ids'])} chunk(s) removed).")
    return True


def get_stats() -> dict:
    """
    Returns statistics about the current knowledge base.
    """
    count = _collection.count()
    docs  = list_documents()
    courses = list(set(d["course"] for d in docs))

    return {
        "total_chunks"   : count,
        "total_documents": len(docs),
        "total_courses"  : len(courses),
        "courses"        : courses,
        "db_path"        : DB_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_model"   : OLLAMA_MODEL,
    }


# ── Demo / Self-Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DORM-NET: brain_module.py — Self-Test")
    print("="*60)

    # ── 1. Show current knowledge base stats ──────────────────────────────────
    stats = get_stats()
    print(f"\n[STATS] Knowledge Base:")
    print(f"  Total chunks    : {stats['total_chunks']}")
    print(f"  Total documents : {stats['total_documents']}")
    print(f"  Courses indexed : {', '.join(stats['courses']) if stats['courses'] else 'None yet'}")
    print(f"  DB location     : {stats['db_path']}")

    # ── 2. Add a sample document ──────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 1] Adding a sample Golden Summary...")

    sample_text = """
    Calculus II at AASTU focuses on advanced integration techniques:
    
    1. Integration by Parts: The formula is ∫u dv = uv - ∫v du.
       Choose u using the LIATE rule: Logarithmic, Inverse trig, Algebraic,
       Trigonometric, Exponential. The first type in this list should be u.
    
    2. Partial Fractions: Decompose a complex rational function into simpler ones.
       Works when the degree of the numerator < degree of denominator.
       Factor the denominator and set up partial fraction form, then solve.
    
    3. Trigonometric Substitution:
       - If you see sqrt(a²-x²), let x = a sin(theta)
       - If you see sqrt(a²+x²), let x = a tan(theta)
       - If you see sqrt(x²-a²), let x = a sec(theta)
    
    Exam tip: The AASTU Calculus II final usually has 3 theory questions worth
    20 marks each and 4 computational problems worth 10 marks each. Focus on
    mastering all three integration techniques above.
    """

    add_golden_summary(
        text=sample_text,
        document_id="calc2_integration_demo",
        metadata={"course": "Calculus II", "chapter": "Integration Techniques"}
    )

    # ── 3. Test Semantic Retrieval ─────────────────────────────────────────────
    print("-"*60)
    print("[TEST 2] Testing Semantic Search...")
    test_query = "I have an integral with a square root in it, what do I do?"
    context, sources = retrieve_context(test_query)
    print(f"  Query   : {test_query}")
    print(f"  Results : {len(sources)} chunk(s) retrieved")
    for s in sources:
        print(f"    - [{s['similarity']}%] {s['source_doc']} ({s['course']})")

    # ── 4. Full RAG Pipeline Test ──────────────────────────────────────────────
    print("\n" + "-"*60)
    print("[TEST 3] Full RAG Pipeline (Retrieve + Generate)...")
    query = "How do I use integration by parts? Give me the rule and an example."
    print(f"\n  Student Question: {query}\n")

    answer = ask_aastu_senior(query)

    print("\n" + "="*60)
    print("  AASTU Senior Tutor Says:")
    print("="*60)
    # Word-wrap the answer for clean terminal display
    wrapped = textwrap.fill(answer, width=60)
    print(wrapped)
    print("="*60)

    # ── 5. List all documents ──────────────────────────────────────────────────
    print("\n[TEST 4] Listing all documents in Knowledge Base:")
    all_docs = list_documents()
    if all_docs:
        print(f"  {'Source Document':<35} {'Course':<25} {'Chunks':>6}")
        print(f"  {'-'*35} {'-'*25} {'-'*6}")
        for doc in all_docs:
            print(f"  {doc['source_doc']:<35} {doc['course']:<25} {doc['chunks']:>6}")
    else:
        print("  (no documents yet)")

    print("\n[DONE] brain_module.py self-test complete.")
    print("="*60 + "\n")
