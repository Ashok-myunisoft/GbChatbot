"""
Schema RAG — Semantic table detection for SQL generation.

Builds a FAISS index from enriched table descriptions (name + columns + synonyms).
Used as Pass 0 inside get_schema_tool() to find relevant tables semantically.

Safe by design:
  - is_index_ready() returns False until index is fully built
  - search_schema() returns [] on any failure
  - get_schema_tool() falls back to existing keyword logic when [] is returned
"""
import json
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

# ── Index state ───────────────────────────────────────────────────────────────
_index        = None   # FAISS vectorstore (None until build_or_load_index finishes)
_index_lock   = threading.Lock()
_CACHE_DIR    = "schema_faiss_index"
_META_FILE    = os.path.join(_CACHE_DIR, "meta.json")

# ── ERP synonym map ────────────────────────────────────────────────────────────
_ERP_SYNONYMS: dict = {
    "purchase":  "buy procurement vendor items ordered po",
    "formula":   "calculation expression field compute",
    "menu":      "navigation screen module access page",
    "report":    "print output summary document",
    "file":      "project document attachment upload",
    "employee":  "staff worker person hr human resources",
    "vendor":    "supplier company provider",
    "customer":  "client buyer consumer",
    "item":      "product goods material inventory stock",
    "payment":   "pay invoice billing amount",
    "account":   "ledger finance general",
    "user":      "login access permission",
    "setting":   "config configuration system",
    "order":     "transaction request",
    "detail":    "line item breakdown sub",
    "master":    "main reference lookup",
    "type":      "category kind classification",
    "code":      "reference lookup value",
    "log":       "audit trail history record",
    "role":      "permission access group",
    "branch":    "location site office",
    "tax":       "gst vat rate",
    "currency":  "exchange rate money",
    "project":   "work task job",
    "approval":  "approve workflow authorize sign",
    "status":    "state flag condition",
    "price":     "cost rate amount value",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_table_name(name: str) -> str:
    """MPURCHASEORDER → 'purchase order buy procurement vendor items ordered po'"""
    stripped = re.sub(r'^[MTV]', '', name, flags=re.IGNORECASE)
    words    = re.findall(r'[A-Z][a-z]*|[a-z]+|[0-9]+', stripped)
    base     = ' '.join(w.lower() for w in words) if words else stripped.lower()
    extras   = [syn for kw, syn in _ERP_SYNONYMS.items() if kw in base]
    return f"{base} {' '.join(extras)}".strip()


def _make_table_doc(table: str, cols: list) -> str:
    """Build enriched text doc for a table — used for embedding."""
    parsed   = _parse_table_name(table)
    col_text = ' '.join(cols[:30])
    return f"{table} {parsed}: {col_text}"


# ── Public API ─────────────────────────────────────────────────────────────────

def is_index_ready() -> bool:
    """True only after build_or_load_index() has successfully completed."""
    with _index_lock:
        return _index is not None


def build_or_load_index() -> None:
    """
    Build FAISS index from all DB tables, or load from disk if schema unchanged.
    Called once at startup in a background thread — never blocks requests.
    """
    global _index

    try:
        from db_query import get_columns, get_tables
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from shared_resources import ai_resources

        tables        = get_tables()
        current_count = len(tables)

        # ── Try loading from disk if table count matches ───────────────────────
        if os.path.exists(_META_FILE):
            try:
                with open(_META_FILE) as f:
                    meta = json.load(f)
                if meta.get("table_count") == current_count:
                    loaded = FAISS.load_local(
                        _CACHE_DIR,
                        ai_resources.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    with _index_lock:
                        _index = loaded
                    logger.info(f"✅ Schema RAG loaded from disk ({current_count} tables)")
                    return
            except Exception as e:
                logger.warning(f"Schema RAG disk load failed — rebuilding: {e}")

        # ── Build fresh index ──────────────────────────────────────────────────
        logger.info(f"Building Schema RAG index for {current_count} tables...")
        documents = []
        for table in tables:
            cols = get_columns(table)
            doc  = _make_table_doc(table, cols)
            documents.append(Document(page_content=doc, metadata={"table": table}))

        new_index = FAISS.from_documents(documents, ai_resources.embeddings)

        # ── Save to disk ───────────────────────────────────────────────────────
        os.makedirs(_CACHE_DIR, exist_ok=True)
        new_index.save_local(_CACHE_DIR)
        with open(_META_FILE, "w") as f:
            json.dump({"table_count": current_count}, f)

        with _index_lock:
            _index = new_index

        logger.info(f"✅ Schema RAG index built and saved ({current_count} tables)")

    except Exception as e:
        logger.warning(f"Schema RAG build failed (non-fatal, keyword matching still works): {e}")


def search_schema(question: str, top_k: int = 5) -> list:
    """
    Return up to top_k table names most relevant to the question.
    Returns [] on any failure — caller falls back to keyword matching.
    """
    with _index_lock:
        idx = _index

    if idx is None:
        return []

    try:
        results = idx.similarity_search_with_score(question, k=top_k)
        # all-MiniLM-L6-v2 uses L2 distance; < 1.2 = confidently relevant
        # If even the best match is weak, return [] so keyword logic handles it
        if not results or results[0][1] > 1.2:
            return []
        return [
            doc.metadata["table"]
            for doc, score in results
            if score <= 1.2 and doc.metadata.get("table")
        ]
    except Exception as e:
        logger.warning(f"Schema RAG search failed (non-fatal): {e}")
        return []
