"""
Retrieval layer. Reuses the EXACT ChromaDB collection + query shape as AI
Second Brain's backend/app/services/vector_service.py — do NOT rebuild a new
knowledge base for this project.

Second Brain's search_vectors() requires a user_id in the `where` filter
(chunks are stored with user_id in metadata, scoped per user), so this
module needs SECOND_BRAIN_USER_ID set to your account's numeric user id.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH")
if not CHROMA_DB_PATH:
    raise RuntimeError(
        "CHROMA_DB_PATH is not set. Create a .env file in the project root "
        "(next to requirements.txt, not inside app/) with CHROMA_DB_PATH=... "
        "pointing at Second_brain/backend/chroma_db. See .env.example."
    )
COLLECTION_NAME = "document_chunks"  # matches vector_service.py exactly, not configurable
_user_id_raw = os.environ.get("SECOND_BRAIN_USER_ID")
if not _user_id_raw:
    raise RuntimeError(
        "SECOND_BRAIN_USER_ID is not set in .env. See .env.example."
    )
USER_ID = int(_user_id_raw)
TOP_K = 4

_embedder: SentenceTransformer | None = None
_client = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # Exact model string from Second Brain's embedding_service.py
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client.get_or_create_collection(COLLECTION_NAME)


def retrieve_from_chroma(question: str, top_k: int = TOP_K) -> str:
    """
    Mirrors Second Brain's search_vectors() exactly: same collection, same
    user_id where-filter, same try/except-returns-None-on-error shape.
    Returns concatenated top-k chunks, or '' if nothing came back.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return ""

    query_embedding = _get_embedder().encode(question).tolist()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": USER_ID},
        )
    except Exception:
        return ""

    if not results:
        return ""
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)


def web_search_fallback(question: str) -> str:
    """
    Stub — plug in whatever search API you have (Tavily, Serper, Bing, etc).
    Only called when Chroma returns nothing relevant, so the graph doesn't
    dead-end on questions outside your document set.
    """
    # TODO: wire up a real web search API here.
    return f"[web search fallback not implemented — no external results for: {question}]"


def retrieve(question: str) -> str:
    evidence = retrieve_from_chroma(question)
    if not evidence.strip():
        evidence = web_search_fallback(question)
    return evidence