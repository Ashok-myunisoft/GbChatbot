"""
learning_store.py

Persistent learning layer for db_query.py.
ALL operations are:
  - Wrapped in try/except — never raises exceptions to callers
  - Atomic writes (tmp file + os.replace) — safe on concurrent requests
  - Non-invasive — can be deleted/disabled without breaking anything
  - Read-only toward the database — only stores metadata, never DB rows

Stores (JSON files in LEARNING_DIR):
  query_patterns.json  — successful GPT SQL patterns        (7-day TTL)
  synonyms.json        — learned word → table mappings       (permanent)
  failures.json        — SQL error counts by table           (24-hour TTL)
"""

import os
import json
import time
import hashlib
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Storage location ──────────────────────────────────────────────────────────
# Try /app/learning first (Docker), fallback to ./learning_data (local dev)
_LEARNING_DIR = os.getenv("LEARNING_DIR", "/app/learning")
try:
    os.makedirs(_LEARNING_DIR, exist_ok=True)
    _probe = os.path.join(_LEARNING_DIR, ".probe")
    open(_probe, "w").close()
    os.remove(_probe)
except Exception:
    _LEARNING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning_data")
    os.makedirs(_LEARNING_DIR, exist_ok=True)

logger.info(f"[LearningStore] Storage: {_LEARNING_DIR}")

_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save(path: str, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)          # atomic — prevents corrupt JSON on crash
    except Exception as e:
        logger.warning(f"[LearningStore] Write failed {os.path.basename(path)}: {e}")


# =============================================================================
# 1. QUERY PATTERN CACHE  (7-day TTL)
#
#    Stores:  question_hash → { question, sql, table, timestamp, hits }
#    Purpose: Skip GPT entirely for questions already answered successfully.
#             Cached SQL is always re-executed against live DB — data is never stale.
#             If re-execution fails (schema changed), pattern is auto-invalidated.
# =============================================================================

_PATTERN_FILE = os.path.join(_LEARNING_DIR, "query_patterns.json")
_PATTERN_TTL  = 7 * 24 * 3600   # 7 days


def _qhash(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def save_pattern(question: str, sql: str, table: str):
    """
    Persist a successful GPT-generated SQL for this question.
    Only called after the SQL executed successfully and returned data.
    """
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return
    try:
        with _LOCK:
            store = _load(_PATTERN_FILE, {})
            key   = _qhash(question)
            prev  = store.get(key, {})
            store[key] = {
                "question":  question[:200],
                "sql":       sql,
                "table":     table,
                "timestamp": time.time(),
                "hits":      prev.get("hits", 0),
            }
            # Evict expired entries (keep file small)
            now   = time.time()
            store = {k: v for k, v in store.items()
                     if now - v.get("timestamp", 0) < _PATTERN_TTL}
            _save(_PATTERN_FILE, store)
            logger.info(f"[LearningStore] Pattern saved: {question[:60]}")
    except Exception as e:
        logger.debug(f"[LearningStore] save_pattern: {e}")


def get_pattern(question: str) -> Optional[str]:
    """Return cached SQL if found and within TTL, else None."""
    try:
        store = _load(_PATTERN_FILE, {})
        key   = _qhash(question)
        entry = store.get(key)
        if entry and time.time() - entry.get("timestamp", 0) < _PATTERN_TTL:
            # Increment hit counter asynchronously
            try:
                with _LOCK:
                    s2 = _load(_PATTERN_FILE, {})
                    if key in s2:
                        s2[key]["hits"] = s2[key].get("hits", 0) + 1
                        _save(_PATTERN_FILE, s2)
            except Exception:
                pass
            logger.info(f"[LearningStore] Pattern hit ({entry.get('hits',0)+1}x): {question[:60]}")
            return entry.get("sql")
    except Exception as e:
        logger.debug(f"[LearningStore] get_pattern: {e}")
    return None


def invalidate_pattern(question: str):
    """
    Remove a cached pattern.
    Called when re-execution of cached SQL returns error — likely schema changed.
    """
    try:
        with _LOCK:
            store = _load(_PATTERN_FILE, {})
            key   = _qhash(question)
            if key in store:
                del store[key]
                _save(_PATTERN_FILE, store)
                logger.info(f"[LearningStore] Pattern invalidated: {question[:60]}")
    except Exception as e:
        logger.debug(f"[LearningStore] invalidate_pattern: {e}")


# =============================================================================
# 2. SYNONYM STORE  (permanent — no TTL, grows over time)
#
#    Stores:  word.lower() → { table, confirmed_count, last_seen }
#    Purpose: Map user vocabulary to table names.
#             e.g. "payslip" → MSALARYPROCESSING, "voucher" → MJOURNALVOUCHER
#             Only saves words confirmed by actual data returned from DB.
# =============================================================================

_SYNONYM_FILE = os.path.join(_LEARNING_DIR, "synonyms.json")


def save_synonym(word: str, table: str):
    """
    Save/update word → table mapping.
    Only called when a query using this word returned real data.
    """
    try:
        wl = word.lower().strip()
        if len(wl) < 3:
            return
        with _LOCK:
            store = _load(_SYNONYM_FILE, {})
            prev  = store.get(wl, {"confirmed_count": 0})
            store[wl] = {
                "table":           table,
                "confirmed_count": prev.get("confirmed_count", 0) + 1,
                "last_seen":       time.time(),
            }
            _save(_SYNONYM_FILE, store)
    except Exception as e:
        logger.debug(f"[LearningStore] save_synonym: {e}")


def get_synonym(word: str) -> Optional[str]:
    """Return learned table name for this word, or None if not learned yet."""
    try:
        store = _load(_SYNONYM_FILE, {})
        entry = store.get(word.lower().strip())
        if entry:
            return entry.get("table")
    except Exception as e:
        logger.debug(f"[LearningStore] get_synonym: {e}")
    return None


# =============================================================================
# 3. FAILURE BLACKLIST  (24-hour TTL)
#
#    Stores:  "table:error_category" → { count, last_seen }
#    Purpose: Track tables that repeatedly cause GPT SQL errors.
#             After 3+ failures, flag the table so callers can log a warning.
#             Does NOT skip any engine — purely informational for now.
# =============================================================================

_FAILURE_FILE = os.path.join(_LEARNING_DIR, "failures.json")
_FAILURE_TTL  = 24 * 3600   # 24 hours


def _error_category(error: str) -> str:
    e = error.lower()
    if "operator does not exist" in e or "invalid input syntax" in e:
        return "type_mismatch"
    if "does not exist" in e:
        return "col_missing"
    if "syntax error" in e:
        return "syntax"
    return "other"


def save_failure(table: str, error: str):
    """Record a SQL error for this table (for monitoring and future skip logic)."""
    try:
        with _LOCK:
            store = _load(_FAILURE_FILE, {})
            key   = f"{table.lower()}:{_error_category(error)}"
            prev  = store.get(key, {"count": 0})
            store[key] = {
                "count":     prev.get("count", 0) + 1,
                "last_seen": time.time(),
            }
            # Evict expired entries
            now   = time.time()
            store = {k: v for k, v in store.items()
                     if now - v.get("last_seen", 0) < _FAILURE_TTL}
            _save(_FAILURE_FILE, store)
    except Exception as e:
        logger.debug(f"[LearningStore] save_failure: {e}")


def get_failure_count(table: str) -> int:
    """Return total recent failure count for this table across all error categories."""
    try:
        store = _load(_FAILURE_FILE, {})
        now   = time.time()
        return sum(
            v.get("count", 0)
            for k, v in store.items()
            if k.startswith(table.lower() + ":")
            and now - v.get("last_seen", 0) < _FAILURE_TTL
        )
    except Exception as e:
        logger.debug(f"[LearningStore] get_failure_count: {e}")
    return 0
