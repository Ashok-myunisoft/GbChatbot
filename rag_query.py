"""
rag_query.py

Utility for querying the FAISS general knowledge vector store.
Used by general_bot.

The vector store is built by knowledge_loader.load_all() at startup and saved
to disk. This module loads it lazily on the first search() call.
"""

import os
import hashlib
import logging

logger = logging.getLogger(__name__)

DATA_DIR   = "/app/data"
FAISS_PATH = os.path.join(DATA_DIR, "general_faiss")

_vectorstore = None


def _save_index_hash(index_dir: str) -> None:
    """Compute SHA-256 of the FAISS index file and write it alongside."""
    index_file = os.path.join(index_dir, "index.faiss")
    if not os.path.exists(index_file):
        return
    sha = hashlib.sha256(open(index_file, "rb").read()).hexdigest()
    with open(os.path.join(index_dir, "index.sha256"), "w") as f:
        f.write(sha)


def _verify_index_hash(index_dir: str) -> bool:
    """Return True only if the stored hash matches the current index file."""
    index_file = os.path.join(index_dir, "index.faiss")
    sig_file   = os.path.join(index_dir, "index.sha256")
    if not os.path.exists(index_file) or not os.path.exists(sig_file):
        return False
    current = hashlib.sha256(open(index_file, "rb").read()).hexdigest()
    stored  = open(sig_file).read().strip()
    return current == stored


def _get_vectorstore():
    """Lazy-load the FAISS store from disk (built by knowledge_loader)."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.path.exists(FAISS_PATH):
        logger.warning(
            f"FAISS store not found at '{FAISS_PATH}'. "
            "Ensure knowledge_loader.load_all() ran at startup."
        )
        return None

    try:
        from langchain_community.vectorstores import FAISS
        from shared_resources import ai_resources
        if not _verify_index_hash(FAISS_PATH):
            logger.error("RAG: FAISS index integrity check failed — refusing to load potentially tampered file.")
            return None
        _vectorstore = FAISS.load_local(
            FAISS_PATH,
            ai_resources.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("RAG: FAISS knowledge store loaded from disk.")
    except Exception as exc:
        logger.error(f"Failed to load FAISS store: {exc}")
        return None

    return _vectorstore


def search(query: str, k: int = 10) -> str:
    """
    Similarity-search the FAISS store and return a formatted context string
    suitable for injection into an LLM prompt.
    """
    vs = _get_vectorstore()
    if vs is None:
        return "Knowledge base not available. Documents may not have been loaded at startup."

    try:
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            parts.append(f"--- Document {i} (Source: {source}) ---")
            parts.append(doc.page_content)
            parts.append("")
        return "\n".join(parts)

    except Exception as exc:
        logger.error(f"RAG search error: {exc}")
        return "Error while searching the knowledge base."
