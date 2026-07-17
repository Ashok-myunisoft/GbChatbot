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

# Deep search uses a lower threshold and more chunks
KMS_DEEP_LIMIT      = int(os.getenv("KMS_DEEP_LIMIT", "80"))
KMS_DEEP_THRESHOLD  = float(os.getenv("KMS_DEEP_THRESHOLD", "0.15"))

# ── Named constants ────────────────────────────────────────────────────────────
MIN_BROAD_QUERY_WORDS = 8
MIN_KEYWORD_LENGTH    = 4
MAX_CONTEXT_TERMS     = 3

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
    msg = str(exc).lower()
    qdrant_signals = ("connection", "timeout", "grpc", "rpc", "unreachable", "refused", "reset")
    return any(s in msg for s in qdrant_signals) or "qdrant" in type(exc).__name__.lower()


# ── Core search (shared by search() and search_with_sources()) ────────────────
def _run_search(query: str, reset: bool = False, limit: int = None, threshold: float = None) -> tuple:
    t0        = time.monotonic()
    embedding = get_embedding(query)
    _limit    = limit if limit is not None else get_dynamic_limit(query)
    _thresh   = threshold if threshold is not None else KMS_SCORE_THRESHOLD

    results = _get_client(reset=reset).query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=_limit,
        score_threshold=_thresh,
        with_payload=True,
    ).points

    elapsed   = time.monotonic() - t0
    n         = len(results)
    top_score = f"{results[0].score:.3f}" if results else "n/a"

    logger.info(
        "[kms_qdrant] query=%r  chunks=%d  limit=%d  threshold=%.2f  top_score=%s  elapsed=%.3fs",
        query[:80], n, _limit, _thresh, top_score, elapsed,
    )

    if not results:
        return "", []

    return _format_results(results)


def _run_with_retry(query: str, limit: int = None, threshold: float = None) -> tuple:
    try:
        return _run_search(query, limit=limit, threshold=threshold)
    except Exception as exc:
        if not _is_qdrant_connection_error(exc):
            logger.error("[kms_qdrant] Non-retryable error for query %r: %s", query[:80], exc)
            return "", []
        logger.warning(
            "[kms_qdrant] Qdrant connection error — resetting client and retrying: %s", exc
        )
        try:
            return _run_search(query, reset=True, limit=limit, threshold=threshold)
        except Exception as exc2:
            logger.error("[kms_qdrant] search failed after retry: %s", exc2)
            return "", []


def _chunk_overlap_ratio(chunk_text: str, previous_answer: str) -> float:
    """Word-overlap ratio of a result chunk against the rejected answer text."""
    a = set(re.findall(r"\b\w{4,}\b", chunk_text.lower()))
    b = set(re.findall(r"\b\w{4,}\b", previous_answer.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


_DUPLICATE_OVERLAP_THRESHOLD = 0.6


def _drop_chunks_matching_answer(context: str, sources: list, previous_answer: str) -> tuple:
    """
    Remove "--- i: Title ---" blocks from a formatted context string whose
    content heavily overlaps the rejected previous answer, so a retry can't
    silently regenerate the same wrong answer from the same source chunk.
    Falls back to the original context if filtering would remove everything.
    """
    blocks = re.split(r"(?=--- \d+: )", context)
    kept_blocks, dropped_titles = [], set()
    for block in blocks:
        if not block.strip():
            continue
        if _chunk_overlap_ratio(block, previous_answer) >= _DUPLICATE_OVERLAP_THRESHOLD:
            m = re.match(r"--- \d+: (.+?) ---", block)
            if m:
                dropped_titles.add(m.group(1))
            continue
        kept_blocks.append(block)

    if not kept_blocks:
        # Filtering would remove every chunk — better to keep the original
        # results than to return nothing.
        return context, sources

    if dropped_titles:
        logger.info("[DeepSearch] Dropped %d duplicate chunk(s) matching rejected answer: %s",
                     len(dropped_titles), ", ".join(list(dropped_titles)[:5]))

    filtered_sources = [s for s in sources if s.get("title") not in dropped_titles]
    return "\n".join(kept_blocks).strip(), filtered_sources


# ── Retry-mode context marker (shared with orchestrator_main.py) ──────────────
DEEP_SEARCH_MARKER = "[DEEP SEARCH"


def extract_deep_search_context(orchestrator_context: str) -> str:
    """
    Retry-mode plumbing helper for the 6 bot files.

    orchestrator_main._execute_retry() injects deep-search KMS results into
    the context string it hands each bot, wrapped between the
    "[DEEP SEARCH ...]" marker and the "[RETRY MODE]" tag. Each bot's own
    chat() would otherwise ignore that and re-run its OWN normal (shallow)
    KMS search for `context_str` — the "Primary source" field in its prompt —
    which just reproduces the same wrong answer the deep search was meant to
    fix. Bots call this first; if it returns non-empty, that becomes the
    bot's primary KMS context for this turn instead of a fresh shallow search.

    Returns "" when there is no deep-search block to promote.
    """
    if not orchestrator_context or DEEP_SEARCH_MARKER not in orchestrator_context:
        return ""
    start = orchestrator_context.find(DEEP_SEARCH_MARKER)
    end = orchestrator_context.find("[RETRY MODE]", start)
    block = orchestrator_context[start:end] if end != -1 else orchestrator_context[start:]
    # Drop the marker line itself, keep just the KB content beneath it.
    block = block.split("\n\n", 1)[1] if "\n\n" in block else block
    return block.strip()


# ── Public API ─────────────────────────────────────────────────────────────────
def search(query: str) -> str:
    context, _ = _run_with_retry(query)
    return context


def search_with_sources(query: str):
    return _run_with_retry(query)


def deep_search_with_sources(
    question: str,
    correction_hint: str = "",
    feedback_category: str = "",
    previous_answer: str = "",
) -> tuple:
    """
    Deep search — the Qdrant equivalent of SQL deep search.

    Called when the user says 'wrong', 'try again', 'not correct', etc.
    Tries THREE strategies and returns the best (most chunks found):

    Strategy 1 — Original question with lower threshold + more chunks.
                 Catches results that normal search missed because they
                 scored just below the normal threshold.

    Strategy 2 — Enriched query: original question + correction hint.
                 If user said 'not asking about leave balance, give salary info',
                 the hint 'salary info' is added to the search query so Qdrant
                 finds the right topic.

    Strategy 3 — Rephrased question: strips question words (what/how/show/list)
                 and searches with just the core keywords.
                 Catches cases where Qdrant matched the question structure
                 rather than the actual topic.

    feedback_category / previous_answer (optional): when the caller already
    knows this is a WRONG_DATA retry and has the rejected answer text, chunks
    that are near-duplicates of that rejected answer are filtered out of the
    winning strategy's results — the whole point of a retry is to surface
    something DIFFERENT, not the same chunk that produced the wrong answer.

    Returns the result from whichever strategy found the most chunks.
    Falls back to normal search_with_sources() if all strategies fail.
    """
    logger.info(
        "[DeepSearch] Starting deep search for: '%s' (category=%s)",
        question[:80], feedback_category or "n/a",
    )

    best_context = ""
    best_sources = []
    best_count   = 0

    # ── Strategy 1: lower threshold, more chunks ───────────────────────────────
    try:
        ctx1, src1 = _run_with_retry(
            question,
            limit=KMS_DEEP_LIMIT,
            threshold=KMS_DEEP_THRESHOLD,
        )
        if len(ctx1) > best_count:
            best_context, best_sources, best_count = ctx1, src1, len(ctx1)
            logger.info("[DeepSearch] Strategy 1 found %d chars", len(ctx1))
    except Exception as e:
        logger.warning("[DeepSearch] Strategy 1 failed: %s", e)

    # ── Strategy 2: inject correction hint into query ──────────────────────────
    if correction_hint:
        enriched_query = f"{question} {correction_hint}"
        try:
            ctx2, src2 = _run_with_retry(
                enriched_query,
                limit=KMS_DEEP_LIMIT,
                threshold=KMS_DEEP_THRESHOLD,
            )
            if len(ctx2) > best_count:
                best_context, best_sources, best_count = ctx2, src2, len(ctx2)
                logger.info("[DeepSearch] Strategy 2 (with hint) found %d chars", len(ctx2))
        except Exception as e:
            logger.warning("[DeepSearch] Strategy 2 failed: %s", e)

    # ── Strategy 3: keyword-only rephrased query ───────────────────────────────
    _STRIP_WORDS = {
        'what', 'how', 'show', 'list', 'give', 'tell', 'get',
        'me', 'the', 'all', 'are', 'is', 'a', 'an', 'of',
        'for', 'in', 'about', 'can', 'you', 'please', 'do',
    }
    keywords = [
        w for w in re.findall(r'\b\w+\b', question.lower())
        if w not in _STRIP_WORDS and len(w) >= 3
    ]
    if keywords:
        keyword_query = " ".join(keywords)
        try:
            ctx3, src3 = _run_with_retry(
                keyword_query,
                limit=KMS_DEEP_LIMIT,
                threshold=KMS_DEEP_THRESHOLD,
            )
            if len(ctx3) > best_count:
                best_context, best_sources, best_count = ctx3, src3, len(ctx3)
                logger.info("[DeepSearch] Strategy 3 (keywords) found %d chars", len(ctx3))
        except Exception as e:
            logger.warning("[DeepSearch] Strategy 3 failed: %s", e)

    # ── Fallback if all strategies return nothing ──────────────────────────────
    if not best_context:
        logger.warning("[DeepSearch] All strategies returned empty — falling back to normal search")
        return search_with_sources(question)

    # ── Strip chunks that are near-duplicates of the rejected answer ───────────
    # If we know what the (wrong) previous answer said, drop any result block
    # whose text is largely the same — otherwise the "corrected" answer can
    # just be re-derived from the same chunk the user already rejected.
    if previous_answer and previous_answer.strip():
        best_context, best_sources = _drop_chunks_matching_answer(
            best_context, best_sources, previous_answer
        )

    logger.info("[DeepSearch] Best result: %d chars, %d sources", len(best_context), len(best_sources))
    return best_context, best_sources


def enrich_search_query(question: str, context: str) -> str:
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