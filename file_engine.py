"""
file_engine.py

Per-user file intelligence engine with production-safe persistence.
- Accepts uploaded files (PDF, CSV, Excel, JSON, TXT) up to 10 MB
- Parses and chunks content into searchable pieces
- Persists FAISS indexes to disk so uploads survive restarts and multiple workers
- Persists metadata in PostgreSQL so any worker can resolve file ownership
- Searches the persisted index dynamically during question answering
- Supports thread isolation and inactivity expiry
"""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db_setup import get_pg_conn, release_pg_conn

logger = logging.getLogger(__name__)

FILE_TTL = 1800  # 30 minutes idle expiry
MAX_FILE_MB = 10
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
RELEVANCE_THRESHOLD = 0.3

FILE_STORAGE_ROOT = Path(os.getenv("FILE_ENGINE_STORAGE_DIR", "data/file_intelligence"))
FILE_METADATA_TABLE = "file_upload_sessions"

_lock = threading.Lock()
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
_metadata_ready = False
_EMBEDDINGS = None


def set_embeddings(embeddings) -> None:
    """Configure the embedding model used to load and query FAISS indexes."""
    global _EMBEDDINGS
    _EMBEDDINGS = embeddings


def _ensure_storage_root() -> Path:
    FILE_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return FILE_STORAGE_ROOT


def _username_key(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]


def _session_dir(username: str, upload_token: str) -> Path:
    return _ensure_storage_root() / _username_key(username) / upload_token


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_metadata_table() -> None:
    global _metadata_ready
    if _metadata_ready:
        return
    with _lock:
        if _metadata_ready:
            return
        conn = None
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {FILE_METADATA_TABLE} (
                        username       TEXT PRIMARY KEY,
                        thread_id      TEXT,
                        upload_token   TEXT,
                        filename       TEXT,
                        file_ext       TEXT,
                        file_size      INTEGER,
                        status         TEXT NOT NULL DEFAULT 'processing',
                        storage_dir    TEXT NOT NULL,
                        index_path     TEXT,
                        dataframe_path TEXT,
                        full_text_path TEXT,
                        chunks         INTEGER DEFAULT 0,
                        created_at     TIMESTAMPTZ DEFAULT NOW(),
                        updated_at     TIMESTAMPTZ DEFAULT NOW(),
                        last_access_at TIMESTAMPTZ DEFAULT NOW(),
                        last_error     TEXT
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{FILE_METADATA_TABLE}_thread_id ON {FILE_METADATA_TABLE}(thread_id);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{FILE_METADATA_TABLE}_status ON {FILE_METADATA_TABLE}(status);"
                )
            conn.commit()
            _metadata_ready = True
        except Exception as exc:
            if conn is not None:
                conn.rollback()
            logger.error(f"[FileEngine] Failed to ensure metadata table: {exc}", exc_info=True)
            raise
        finally:
            if conn is not None:
                release_pg_conn(conn)


def _row_to_dict(row: Any) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return dict(row)


def _get_record(username: str) -> Optional[Dict[str, Any]]:
    try:
        _ensure_metadata_table()
        conn = get_pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT username, thread_id, upload_token, filename, file_ext, file_size, status,
                           storage_dir, index_path, dataframe_path, full_text_path, chunks,
                           created_at, updated_at, last_access_at, last_error
                    FROM {FILE_METADATA_TABLE}
                    WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
                return _row_to_dict(row)
        finally:
            release_pg_conn(conn)
    except Exception as exc:
        logger.error(f"[FileEngine] Metadata lookup failed for {username}: {exc}", exc_info=True)
        return None


def _get_record_by_token(username: str, upload_token: str) -> Optional[Dict[str, Any]]:
    try:
        _ensure_metadata_table()
        conn = get_pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT username, thread_id, upload_token, filename, file_ext, file_size, status,
                           storage_dir, index_path, dataframe_path, full_text_path, chunks,
                           created_at, updated_at, last_access_at, last_error
                    FROM {FILE_METADATA_TABLE}
                    WHERE username = %s AND upload_token = %s
                    """,
                    (username, upload_token),
                )
                return _row_to_dict(cur.fetchone())
        finally:
            release_pg_conn(conn)
    except Exception as exc:
        logger.error(f"[FileEngine] Metadata lookup by token failed for {username}: {exc}", exc_info=True)
        return None


def _upsert_record(record: Dict[str, Any]) -> None:
    _ensure_metadata_table()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {FILE_METADATA_TABLE}
                    (username, thread_id, upload_token, filename, file_ext, file_size, status,
                     storage_dir, index_path, dataframe_path, full_text_path, chunks,
                     created_at, updated_at, last_access_at, last_error)
                VALUES
                    (%(username)s, %(thread_id)s, %(upload_token)s, %(filename)s, %(file_ext)s,
                     %(file_size)s, %(status)s, %(storage_dir)s, %(index_path)s, %(dataframe_path)s,
                     %(full_text_path)s, %(chunks)s, %(created_at)s, %(updated_at)s,
                     %(last_access_at)s, %(last_error)s)
                ON CONFLICT (username) DO UPDATE SET
                    thread_id = EXCLUDED.thread_id,
                    upload_token = EXCLUDED.upload_token,
                    filename = EXCLUDED.filename,
                    file_ext = EXCLUDED.file_ext,
                    file_size = EXCLUDED.file_size,
                    status = EXCLUDED.status,
                    storage_dir = EXCLUDED.storage_dir,
                    index_path = EXCLUDED.index_path,
                    dataframe_path = EXCLUDED.dataframe_path,
                    full_text_path = EXCLUDED.full_text_path,
                    chunks = EXCLUDED.chunks,
                    updated_at = EXCLUDED.updated_at,
                    last_access_at = EXCLUDED.last_access_at,
                    last_error = EXCLUDED.last_error
                """,
                record,
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"[FileEngine] Failed to persist metadata for {record.get('username')}: {exc}", exc_info=True)
        raise
    finally:
        release_pg_conn(conn)


def _touch_last_access(username: str) -> None:
    try:
        _ensure_metadata_table()
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {FILE_METADATA_TABLE} SET last_access_at = NOW(), updated_at = NOW() WHERE username = %s",
                    (username,),
                )
            conn.commit()
        finally:
            release_pg_conn(conn)
    except Exception as exc:
        logger.warning(f"[FileEngine] Could not update last access for {username}: {exc}")


def _delete_record(username: str) -> None:
    try:
        _ensure_metadata_table()
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {FILE_METADATA_TABLE} WHERE username = %s", (username,))
            conn.commit()
        finally:
            release_pg_conn(conn)
    except Exception as exc:
        logger.warning(f"[FileEngine] Failed to delete metadata for {username}: {exc}")


def _remove_storage_dir(storage_dir: Optional[str]) -> None:
    if not storage_dir:
        return
    try:
        shutil.rmtree(storage_dir, ignore_errors=True)
    except Exception as exc:
        logger.warning(f"[FileEngine] Failed to remove storage dir '{storage_dir}': {exc}")


def _is_expired(record: Dict[str, Any]) -> bool:
    ref = _parse_dt(record.get("last_access_at")) or _parse_dt(record.get("updated_at")) or _parse_dt(record.get("created_at"))
    if not ref:
        return False
    return (_utc_now() - ref).total_seconds() > FILE_TTL


def _load_dataframe(dataframe_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not dataframe_path or not os.path.exists(dataframe_path):
        return None
    try:
        return pd.read_csv(dataframe_path)
    except Exception as exc:
        logger.error(f"[FileEngine] Failed to load dataframe '{dataframe_path}': {exc}", exc_info=True)
        return None


def _load_full_text(full_text_path: Optional[str]) -> Optional[str]:
    if not full_text_path or not os.path.exists(full_text_path):
        return None
    try:
        return Path(full_text_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error(f"[FileEngine] Failed to load full text '{full_text_path}': {exc}", exc_info=True)
        return None


def _load_index_with_embeddings(index_path: Optional[str], embeddings) -> Optional[FAISS]:
    if not index_path or not os.path.exists(index_path):
        logger.warning(f"[FileEngine] Missing vector index at '{index_path}'")
        return None
    try:
        try:
            return FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except TypeError:
            return FAISS.load_local(index_path, embeddings)
    except Exception as exc:
        logger.error(f"[FileEngine] Failed to load FAISS index '{index_path}': {exc}", exc_info=True)
        return None


def has_file(username: str, thread_id: str = None) -> bool:
    """Return True if user has a valid (non-expired) uploaded file for this thread."""
    record = _get_record(username)
    if not record:
        return False
    if record.get("status") == "processing":
        return False
    if _is_expired(record):
        logger.info(f"[FileEngine] Session expired for {username}")
        _remove_storage_dir(record.get("storage_dir"))
        _delete_record(username)
        return False
    stored_tid = record.get("thread_id")
    if stored_tid and thread_id and stored_tid != thread_id:
        logger.info(
            f"[FileEngine] Thread mismatch for {username} - file belongs to thread '{stored_tid}', request from '{thread_id}'"
        )
        return False
    index_path = record.get("index_path")
    if not index_path or not os.path.exists(index_path):
        logger.warning(f"[FileEngine] No persistent index found for {username}")
        return False
    return True


def mark_processing(username: str, filename: str, thread_id: str = None) -> str:
    """Register a file as processing so the UI can avoid routing too early."""
    upload_token = uuid.uuid4().hex
    storage_dir = str(_session_dir(username, upload_token))
    record = {
        "username": username,
        "thread_id": thread_id,
        "upload_token": upload_token,
        "filename": filename,
        "file_ext": os.path.splitext(filename)[1].lower(),
        "file_size": 0,
        "status": "processing",
        "storage_dir": storage_dir,
        "index_path": None,
        "dataframe_path": None,
        "full_text_path": None,
        "chunks": 0,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "last_access_at": _utc_now(),
        "last_error": None,
    }
    _upsert_record(record)
    logger.info(f"[FileEngine] Marked processing for {username} -> '{filename}'")
    return upload_token


def is_processing(username: str, thread_id: str = None) -> bool:
    """Return True when the user's uploaded file is still being indexed."""
    record = _get_record(username)
    if not record or record.get("status") != "processing":
        return False
    stored_tid = record.get("thread_id")
    if stored_tid and thread_id and stored_tid != thread_id:
        return False
    return True


def process(username: str, filename: str, content: bytes, embeddings, thread_id: str = None) -> str:
    """
    Parse uploaded file, build per-user FAISS index, and persist it to disk.
    Returns a human-readable status message.
    """
    if len(content) > MAX_FILE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        return (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_MB} MB. "
            "Please upload a smaller file."
        )

    record = _get_record(username) or {}
    expected_token = record.get("upload_token")
    storage_dir = Path(record.get("storage_dir") or _session_dir(username, expected_token or uuid.uuid4().hex))
    storage_dir.mkdir(parents=True, exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    docs = []
    df: Optional[pd.DataFrame] = None

    try:
        if ext == ".pdf":
            docs = _parse_pdf(content, filename)
        elif ext == ".csv":
            df, docs = _parse_tabular(content, filename, "csv")
        elif ext in (".xlsx", ".xls"):
            df, docs = _parse_tabular(content, filename, "excel")
        elif ext == ".json":
            docs = _parse_json(content, filename)
        elif ext == ".txt":
            text = content.decode("utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={"source": filename})]
        else:
            return (
                f"Unsupported file type **{ext}**. "
                "Supported formats: PDF, CSV, Excel (.xlsx/.xls), JSON, TXT."
            )

        if not docs:
            return "File appears to be empty or could not be read. Please check the file and try again."

        chunks = _splitter.split_documents(docs)
        index = FAISS.from_documents(chunks, embeddings)
        full_text = "\n\n".join(d.page_content for d in docs)

        index_path = storage_dir / "faiss_index"
        dataframe_path = storage_dir / "dataframe.csv"
        full_text_path = storage_dir / "full_text.txt"
        meta_path = storage_dir / "metadata.json"

        index.save_local(str(index_path))

        if df is not None:
            df.to_csv(dataframe_path, index=False)

        full_text_path.write_text(full_text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "username": username,
                    "thread_id": thread_id,
                    "filename": filename,
                    "file_ext": ext,
                    "chunks": len(chunks),
                    "created_at": _utc_now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        current = _get_record(username)
        if expected_token and current and current.get("upload_token") != expected_token:
            logger.warning(f"[FileEngine] Stale upload ignored for {username}; a newer upload replaced it.")
            return "A newer file upload replaced this one. Please use the latest file."

        final_record = {
            "username": username,
            "thread_id": thread_id or (current or {}).get("thread_id"),
            "upload_token": expected_token or (current or {}).get("upload_token") or uuid.uuid4().hex,
            "filename": filename,
            "file_ext": ext,
            "file_size": len(content),
            "status": "ready",
            "storage_dir": str(storage_dir),
            "index_path": str(index_path),
            "dataframe_path": str(dataframe_path) if df is not None else None,
            "full_text_path": str(full_text_path),
            "chunks": len(chunks),
            "created_at": (current or {}).get("created_at") or _utc_now(),
            "updated_at": _utc_now(),
            "last_access_at": _utc_now(),
            "last_error": None,
        }
        _upsert_record(final_record)

        logger.info(f"[FileEngine] {username} -> '{filename}' ({len(chunks)} chunks indexed and persisted)")
        return (
            f"File **{filename}** uploaded successfully — "
            f"{len(chunks)} sections indexed. "
            f"You can now ask me anything about it."
        )

    except Exception as exc:
        logger.error(f"[FileEngine] process error for {username}: {exc}", exc_info=True)
        try:
            _upsert_record(
                {
                    "username": username,
                    "thread_id": thread_id or record.get("thread_id"),
                    "upload_token": expected_token or record.get("upload_token") or uuid.uuid4().hex,
                    "filename": filename,
                    "file_ext": ext,
                    "file_size": len(content),
                    "status": "error",
                    "storage_dir": str(storage_dir),
                    "index_path": None,
                    "dataframe_path": None,
                    "full_text_path": None,
                    "chunks": 0,
                    "created_at": record.get("created_at") or _utc_now(),
                    "updated_at": _utc_now(),
                    "last_access_at": _utc_now(),
                    "last_error": str(exc),
                }
            )
        except Exception:
            logger.exception(f"[FileEngine] Failed to persist upload error state for {username}")
        return f"Error processing file: {str(exc)}"


def search(username: str, question: str, thread_id: str = None, k: int = 5) -> Optional[str]:
    """
    Search user's file index for an answer.
    Returns formatted answer string, or None if nothing relevant found.
    """
    record = _get_record(username)
    if not record:
        return None
    if record.get("status") == "processing":
        return None
    if _is_expired(record):
        logger.info(f"[FileEngine] Session expired for {username}")
        _remove_storage_dir(record.get("storage_dir"))
        _delete_record(username)
        return None
    stored_tid = record.get("thread_id")
    if stored_tid and thread_id and stored_tid != thread_id:
        logger.info(
            f"[FileEngine] Thread mismatch for {username} - file belongs to thread '{stored_tid}', request from '{thread_id}'"
        )
        return None

    index_path = record.get("index_path")
    filename = record.get("filename") or "upload"
    if not index_path or not os.path.exists(index_path):
        logger.warning(f"[FileEngine] Missing vector index for {username} at '{index_path}'")
        return None

    try:
        embeddings = _EMBEDDINGS
        if embeddings is None:
            logger.error("[FileEngine] Embeddings are not configured. Call set_embeddings() at startup.")
            return None

        index = _load_index_with_embeddings(index_path, embeddings)
        if index is None:
            return None

        _touch_last_access(username)

        df = _load_dataframe(record.get("dataframe_path"))
        if df is not None:
            pandas_result = _query_dataframe(question, df)
            if pandas_result:
                logger.info(f"[FileEngine] Pandas fast-path: '{question[:60]}'")
                return f"From **{filename}**:\n\n{pandas_result}"

        best_score = 0.0
        try:
            scored_docs = index.similarity_search_with_relevance_scores(question, k=k)
            best_score = max((score for _, score in scored_docs), default=0.0)
            if best_score < RELEVANCE_THRESHOLD:
                logger.info(
                    f"[FileEngine] Low relevance ({best_score:.2f} < {RELEVANCE_THRESHOLD}) "
                    f"for '{question[:60]}' - skipping file, routing to existing bots"
                )
                return None
            docs = [doc for doc, _ in scored_docs]
        except Exception:
            docs = index.similarity_search(question, k=k)

        if not docs:
            return None

        context = "\n\n".join(d.page_content for d in docs)
        if len(context.strip()) < 20:
            return None

        logger.info(f"[FileEngine] FAISS: '{question[:60]}' ({len(docs)} chunks, score={best_score:.2f})")
        return f"From **{filename}**:\n\n{context}"

    except Exception as exc:
        logger.error(f"[FileEngine] search error for {username}: {exc}", exc_info=True)
        return None


def get_dataframe(username: str) -> Optional[pd.DataFrame]:
    """Return stored DataFrame (for export use). None if not tabular or expired."""
    record = _get_record(username)
    if not record or record.get("status") == "processing" or _is_expired(record):
        return None
    return _load_dataframe(record.get("dataframe_path"))


def get_filename(username: str) -> Optional[str]:
    record = _get_record(username)
    if not record or _is_expired(record):
        return None
    return record.get("filename")


def get_full_text(username: str) -> Optional[str]:
    """Return full parsed text of the uploaded file (for summarization)."""
    record = _get_record(username)
    if not record or record.get("status") == "processing" or _is_expired(record):
        return None
    return _load_full_text(record.get("full_text_path"))


def _compact_preview(text: str, max_chars: int = 700, max_sentences: int = 3) -> str:
    """Create a short, readable preview from extracted text."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    preview = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
    if len(preview) < 80:
        preview = cleaned
    return preview[:max_chars].rstrip()


def build_upload_summary(filename: str, content: bytes) -> str:
    """
    Build a lightweight summary for the uploaded file without changing the
    main indexing/search pipeline.

    This is intentionally extractive and fast so the upload response can show a
    helpful summary immediately while background indexing continues.
    """
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            docs = _parse_pdf(content, filename)
            text = "\n\n".join(d.page_content for d in docs)
            preview = _compact_preview(text)
            if not preview:
                return f"File **{filename}** is a PDF, but no readable text was extracted yet."
            return (
                f"File **{filename}** is a PDF with **{len(docs)}** readable page sections.\n\n"
                f"**Quick summary:** {preview}"
            )

        if ext == ".csv":
            df, _ = _parse_tabular(content, filename, "csv")
            if df is None or df.empty:
                return f"File **{filename}** is a CSV file, but no rows were extracted."
            cols = ", ".join(map(str, df.columns[:12]))
            if len(df.columns) > 12:
                cols += ", ..."
            sample = df.head(3).to_string(index=False)
            return (
                f"File **{filename}** is a CSV with **{len(df)}** rows and **{len(df.columns)}** columns.\n\n"
                f"**Columns:** {cols}\n\n"
                f"**Sample rows:**\n{sample}"
            )

        if ext in (".xlsx", ".xls"):
            df, _ = _parse_tabular(content, filename, "excel")
            if df is None or df.empty:
                return f"File **{filename}** is an Excel file, but no rows were extracted."
            cols = ", ".join(map(str, df.columns[:12]))
            if len(df.columns) > 12:
                cols += ", ..."
            sample = df.head(3).to_string(index=False)
            return (
                f"File **{filename}** is an Excel file with **{len(df)}** rows and **{len(df.columns)}** columns.\n\n"
                f"**Columns:** {cols}\n\n"
                f"**Sample rows:**\n{sample}"
            )

        if ext == ".json":
            try:
                data = json.loads(content.decode("utf-8", errors="ignore"))
                if isinstance(data, list):
                    preview = _compact_preview(json.dumps(data[:2], ensure_ascii=False, indent=2))
                    return (
                        f"File **{filename}** is a JSON array with **{len(data)}** item(s).\n\n"
                        f"**Quick preview:** {preview}"
                    )
                if isinstance(data, dict):
                    keys = ", ".join(list(data.keys())[:15])
                    if len(data.keys()) > 15:
                        keys += ", ..."
                    preview = _compact_preview(json.dumps(data, ensure_ascii=False, indent=2))
                    return (
                        f"File **{filename}** is a JSON object.\n\n"
                        f"**Top-level keys:** {keys or 'none'}\n\n"
                        f"**Quick preview:** {preview}"
                    )
                preview = _compact_preview(str(data))
                return (
                    f"File **{filename}** is a JSON file.\n\n"
                    f"**Quick preview:** {preview}"
                )
            except Exception:
                return f"File **{filename}** is a JSON file, but a quick preview could not be generated."

        if ext == ".txt":
            text = content.decode("utf-8", errors="ignore")
            preview = _compact_preview(text)
            if not preview:
                return f"File **{filename}** is a text file, but it appears empty."
            return (
                f"File **{filename}** is a text file.\n\n"
                f"**Quick summary:** {preview}"
            )

        return (
            f"File **{filename}** was uploaded successfully.\n\n"
            "A quick summary is not available for this file type yet, but the file is being indexed for question answering."
        )
    except Exception as exc:
        logger.warning(f"[FileEngine] Quick summary generation failed for {filename}: {exc}")
        return (
            f"File **{filename}** was uploaded successfully.\n\n"
            "A quick summary could not be generated right now, but the file is still being indexed for question answering."
        )


def clear(username: str):
    """Manually clear a user's file session."""
    record = _get_record(username)
    if record:
        _remove_storage_dir(record.get("storage_dir"))
    _delete_record(username)
    logger.info(f"[FileEngine] Cleared session for {username}")


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _parse_pdf(content: bytes, filename: str) -> list:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        docs = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
        logger.info(f"[FileEngine] PDF: {len(docs)} pages from '{filename}'")
        return docs
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                logger.warning("[FileEngine] No PDF reader installed. Run: pip install pymupdf pypdf")
                return []

        try:
            reader = PdfReader(io.BytesIO(content))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
            logger.info(f"[FileEngine] PDF (fallback): {len(docs)} pages from '{filename}'")
            return docs
        except Exception as exc:
            logger.error(f"[FileEngine] PDF fallback parse error: {exc}")
            return []
    except Exception as exc:
        logger.error(f"[FileEngine] PDF parse error: {exc}")
        return []


def _parse_tabular(content: bytes, filename: str, fmt: str) -> Tuple[Optional[pd.DataFrame], list]:
    try:
        if fmt == "csv":
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8", encoding_errors="replace")
        else:
            df = pd.read_excel(io.BytesIO(content))

        docs = []
        for i, row in df.iterrows():
            row_text = " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if val is not None and str(val).strip() not in ("", "nan", "NaT", "None")
            )
            if row_text.strip():
                docs.append(Document(page_content=row_text, metadata={"source": filename, "row": i}))
        logger.info(f"[FileEngine] Tabular: {len(df)} rows -> {len(docs)} docs from '{filename}'")
        return df, docs
    except Exception as exc:
        logger.error(f"[FileEngine] Tabular parse error: {exc}")
        return None, []


def _parse_json(content: bytes, filename: str) -> list:
    try:
        data = json.loads(content.decode("utf-8", errors="ignore"))
        text = "\n".join(json.dumps(item, indent=2) for item in data) if isinstance(data, list) else json.dumps(data, indent=2)
        return [Document(page_content=text, metadata={"source": filename})]
    except Exception as exc:
        logger.error(f"[FileEngine] JSON parse error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Tabular fast-path
# ---------------------------------------------------------------------------

_DF_STOP = {"the", "is", "are", "was", "for", "from", "with", "and", "all", "me", "my", "please", "show", "give", "get", "list", "find"}


def _query_dataframe(question: str, df: pd.DataFrame) -> Optional[str]:
    """
    Answer simple structured questions directly from DataFrame - no LLM.
    Returns formatted string or None (falls back to FAISS if None).
    """
    q = re.sub(r"[^\w\s]", " ", question).lower()
    q_words = [w for w in q.split() if len(w) > 2 and w not in _DF_STOP]
    cols_lower = {c.lower(): c for c in df.columns}

    if any(p in q for p in ["how many", "count", "total number", "number of"]):
        return f"Total records: **{len(df)}**"

    if any(p in q for p in ["show all", "list all", "give all", "all records", "show me all", "display all", "fetch all"]):
        return df.head(50).to_string(index=False)

    for word in q_words:
        if word in cols_lower:
            col = cols_lower[word]
            after = q[q.find(word) + len(word):]
            val_words = [w for w in after.split() if len(w) > 2 and w not in _DF_STOP]
            if val_words:
                val = val_words[0]
                filtered = df[df[col].astype(str).str.lower().str.contains(val, na=False)]
                if not filtered.empty:
                    return filtered.head(20).to_string(index=False)

    for word in q_words:
        if word in cols_lower:
            col = cols_lower[word]
            return df[[col]].head(20).to_string(index=False)

    return None
