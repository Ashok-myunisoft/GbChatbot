"""
kms_qdrant.py

Semantic search against the Qdrant kms_extracted_knowledge collection.
Used by report_bot, formula_bot, schema_bot, and menu_bot as a drop-in
replacement for db_query.query_table().

Returns a formatted context string (same contract as query_table):
  - Non-empty string when results found
  - "" (empty string) when no results — bots' existing empty-check handles this
"""

import os
import re
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
QDRANT_HOST         = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION          = os.getenv("QDRANT_COLLECTION", "kms_extracted_knowledge")
EMBED_MODEL         = "text-embedding-3-small"
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
KMS_LIMIT           = int(os.getenv("KMS_LIMIT", "10"))
KMS_LIMIT_BROAD     = int(os.getenv("KMS_LIMIT_BROAD", "50"))
KMS_SCORE_THRESHOLD = float(os.getenv("KMS_SCORE_THRESHOLD", "0.25"))

# ── Named constants ────────────────────────────────────────────────────────────
MIN_BROAD_QUERY_WORDS = 8   # queries this long or longer fetch KMS_LIMIT_BROAD chunks
MIN_KEYWORD_LENGTH    = 4   # minimum chars for a word to count as "meaningful"
MAX_CONTEXT_TERMS     = 3   # max prior-turn terms injected into an enriched query

# ── Startup validation ─────────────────────────────────────────────────────────
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "[kms_qdrant] OPENAI_API_KEY is not set. "
        "Add it to .env or the environment before starting the application."
    )

logger.info("[kms_qdrant] Connecting to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)

# ── Single OpenAI client (created once at import time) ────────────────────────
from openai import OpenAI as _OpenAI
_oai_client = _OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> list:
    """Embed text using the shared OpenAI client. Returns a float list."""
    return _oai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL,
    ).data[0].embedding


# ── Qdrant client singleton ────────────────────────────────────────────────────
_qdrant_client = None


def _get_client(reset: bool = False):
    global _qdrant_client
    if reset or _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False
        )
    return _qdrant_client


# ── Dynamic retrieval limit ────────────────────────────────────────────────────
def get_dynamic_limit(question: str) -> int:
    """
    Determine how many chunks to fetch based on query structure — no topic keywords.

    Rule 1 — standalone "all" (word boundary): enumeration/listing intent → broad.
              Catches: "list all modules", "show all menus", "what all features"
    Rule 2 — query length >= MIN_BROAD_QUERY_WORDS: complex multi-aspect question → broad.
              Catches: "explain the inventory module and give its use cases" (9 words)
    Everything else → specific fact lookup → default k.
    """
    q = question.lower().strip()
    if re.search(r'\ball\b', q):
        logger.debug("[kms_qdrant] Broad query (contains 'all') → k=%d", KMS_LIMIT_BROAD)
        return KMS_LIMIT_BROAD
    if len(q.split()) >= MIN_BROAD_QUERY_WORDS:
        logger.debug(
            "[kms_qdrant] Broad query (length≥%d) → k=%d",
            MIN_BROAD_QUERY_WORDS, KMS_LIMIT_BROAD,
        )
        return KMS_LIMIT_BROAD
    return KMS_LIMIT


# ── Shared result formatter ────────────────────────────────────────────────────
def _format_results(results: list) -> tuple:
    """
    Convert raw Qdrant hits into a formatted context string and deduplicated source list.
    Returns: (context_str: str, sources: list[dict])
    """
    parts        = []
    sources      = []
    seen_titles  = set()

    for i, hit in enumerate(results, 1):
        payload    = hit.payload or {}
        title      = payload.get("title", f"Result {i}")
        content    = payload.get("content", "")
        chunk_text = payload.get("source_chunk_text", "")

        if content or chunk_text:
            parts.append(f"--- {i}: {title} ---")
            if content:
                parts.append(content)
            if chunk_text and chunk_text != content:
                parts.append(f"Source: {chunk_text}")
            parts.append("")

        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({"title": title, "source": chunk_text or title})

    return "\n".join(parts).strip(), sources


# ── Retry classifier ───────────────────────────────────────────────────────────
def _is_qdrant_connection_error(exc: Exception) -> bool:
    """
    Return True only for Qdrant transport/connection failures.
    OpenAI auth errors, invalid requests, and embedding failures are NOT retried
    because they will not self-heal on a second attempt.
    """
    msg = str(exc).lower()
    qdrant_signals = ("connection", "timeout", "grpc", "rpc", "unreachable", "refused", "reset")
    return any(s in msg for s in qdrant_signals) or "qdrant" in type(exc).__name__.lower()


# ── Core search (shared by search() and search_with_sources()) ────────────────
def _run_search(query: str, reset: bool = False) -> tuple:
    """
    Embed query, search Qdrant, format and return results.
    Returns: (context_str: str, sources: list[dict])
    Raises on any failure — retry logic lives in _run_with_retry().
    """
    t0        = time.monotonic()
    embedding = get_embedding(query)
    limit     = get_dynamic_limit(query)

    results = _get_client(reset=reset).query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=limit,
        score_threshold=KMS_SCORE_THRESHOLD,
        with_payload=True,
    ).points

    elapsed   = time.monotonic() - t0
    n         = len(results)
    top_score = f"{results[0].score:.3f}" if results else "n/a"

    logger.info(
        "[kms_qdrant] query=%r  chunks=%d  limit=%d  threshold=%.2f  top_score=%s  elapsed=%.3fs",
        query[:80], n, limit, KMS_SCORE_THRESHOLD, top_score, elapsed,
    )

    if not results:
        return "", []

    return _format_results(results)


def _run_with_retry(query: str) -> tuple:
    """
    Execute _run_search with one retry on Qdrant connection failures.
    OpenAI errors are surfaced immediately — retrying them wastes time.
    Returns: (context_str: str, sources: list[dict])
    """
    try:
        return _run_search(query)
    except Exception as exc:
        if not _is_qdrant_connection_error(exc):
            logger.error("[kms_qdrant] Non-retryable error for query %r: %s", query[:80], exc)
            return "", []
        logger.warning(
            "[kms_qdrant] Qdrant connection error — resetting client and retrying: %s", exc
        )
        try:
            return _run_search(query, reset=True)
        except Exception as exc2:
            logger.error("[kms_qdrant] search failed after retry: %s", exc2)
            return "", []


# ── Public API ─────────────────────────────────────────────────────────────────
def search(query: str) -> str:
    """
    Embed query with OpenAI text-embedding-3-small, search Qdrant,
    and return a formatted context string for LLM consumption.
    Retrieval count is dynamic — see get_dynamic_limit().
    Retries once on Qdrant connection errors; does not retry OpenAI failures.
    """
    context, _ = _run_with_retry(query)
    return context


def search_with_sources(query: str):
    """
    Same as search() but also returns a deduplicated list of source dicts.
    Returns: (context_str: str, sources: list[dict])
    Each source dict: {"title": str, "source": str}
    """
    return _run_with_retry(query)


def enrich_search_query(question: str, context: str) -> str:
    """
    Enrich vague follow-up questions with topic keywords from conversation context
    so Qdrant gets enough signal to return relevant docs.

    A query is considered vague when it has fewer than 2 meaningful terms
    (words >= MIN_KEYWORD_LENGTH chars that are not stop words) OR contains
    explicit vague references ("it", "that", "this", "more", etc.).

    Specific queries — those with 2+ meaningful terms — are returned unchanged
    to prevent prior-turn topics from corrupting the KMS search.
    """
    q = question.strip()

    _FOLLOWUP_MARKERS = {
        'it', 'that', 'this', 'those', 'these', 'them', 'more',
        'elaborate', 'again', 'same', 'above', 'mentioned',
    }
    _STOPS = {
        'what', 'where', 'when', 'how', 'which', 'who', 'does',
        'the', 'are', 'can', 'give', 'show', 'list', 'tell', 'about',
        'with', 'from', 'have', 'your', 'for', 'and', 'that', 'this',
    }

    words         = q.lower().split()
    has_vague_ref = any(w in _FOLLOWUP_MARKERS for w in words)
    meaningful    = [w for w in words if len(w) >= MIN_KEYWORD_LENGTH and w not in _STOPS]
    is_vague      = len(meaningful) < 2

    if not (is_vague or has_vague_ref) or not context:
        return q

    user_lines = re.findall(r'User:\s*(.+?)(?:\n|$)', context)
    if not user_lines:
        return q

    prev_question = user_lines[-1].strip()
    key_terms = [
        w for w in re.findall(r'\b\w{4,}\b', prev_question.lower())
        if w not in _STOPS
    ]
    if not key_terms:
        return q

    enriched = q + ' ' + ' '.join(key_terms[:MAX_CONTEXT_TERMS])
    logger.debug("[enrich_search_query] %r → %r", q, enriched)
    return enriched
