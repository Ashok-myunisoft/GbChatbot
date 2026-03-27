"""
file_engine.py

Per-user file intelligence engine.
- Accepts uploaded files (PDF, CSV, Excel, JSON, TXT) up to 10 MB
- Parses and chunks content into searchable pieces
- Builds a FAISS index per user (isolated — no cross-user bleed)
- Searches index to answer user questions
- Auto-expires after FILE_TTL seconds of inactivity

Zero impact on existing system — only called when user has uploaded a file.
"""

import io
import os
import re
import json
import time
import logging
import threading
from typing import Optional, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

FILE_TTL     = 1800          # 30 minutes idle expiry
MAX_FILE_MB  = 10            # hard limit: 10 MB
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Per-user store: { username → { index, filename, df, ts, chunks } }
_store: dict = {}
_lock  = threading.Lock()

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


# ── Public API ────────────────────────────────────────────────────────────────

def has_file(username: str) -> bool:
    """Return True if user has a valid (non-expired) uploaded file."""
    with _lock:
        entry = _store.get(username)
        if not entry:
            return False
        if time.time() - entry["ts"] > FILE_TTL:
            del _store[username]
            logger.info(f"[FileEngine] Session expired for {username}")
            return False
        return True


def process(username: str, filename: str, content: bytes, embeddings) -> str:
    """
    Parse uploaded file, build per-user FAISS index.
    Returns a human-readable status message.
    """
    # ── Size guard ────────────────────────────────────────────────────────────
    if len(content) > MAX_FILE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        return (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_MB} MB. "
            "Please upload a smaller file."
        )

    ext  = os.path.splitext(filename)[1].lower()
    docs = []
    df   = None

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
        index  = FAISS.from_documents(chunks, embeddings)

        with _lock:
            _store[username] = {
                "index":    index,
                "filename": filename,
                "df":       df,
                "ts":       time.time(),
                "chunks":   len(chunks),
            }

        logger.info(f"[FileEngine] {username} → '{filename}' ({len(chunks)} chunks indexed)")
        return (
            f"File **{filename}** uploaded successfully — "
            f"{len(chunks)} sections indexed. "
            f"You can now ask me anything about it."
        )

    except Exception as e:
        logger.error(f"[FileEngine] process error for {username}: {e}")
        return f"Error processing file: {str(e)}"


def search(username: str, question: str, k: int = 5) -> Optional[str]:
    """
    Search user's file index for an answer.
    Returns formatted answer string, or None if nothing relevant found.
    """
    with _lock:
        entry = _store.get(username)
        if not entry:
            return None
        if time.time() - entry["ts"] > FILE_TTL:
            del _store[username]
            return None
        entry["ts"] = time.time()          # refresh TTL on each access
        index    = entry["index"]
        df       = entry["df"]
        filename = entry["filename"]

    try:
        # For tabular data (CSV/Excel): try direct pandas query first (faster, no FAISS)
        if df is not None:
            pandas_result = _query_dataframe(question, df)
            if pandas_result:
                logger.info(f"[FileEngine] Pandas fast-path: '{question[:60]}'")
                return f"From **{filename}**:\n\n{pandas_result}"

        # FAISS semantic search
        docs = index.similarity_search(question, k=k)
        if not docs:
            return None

        context = "\n\n".join(d.page_content for d in docs)
        if len(context.strip()) < 20:
            return None

        logger.info(f"[FileEngine] FAISS: '{question[:60]}' ({len(docs)} chunks)")
        return f"From **{filename}**:\n\n{context}"

    except Exception as e:
        logger.error(f"[FileEngine] search error for {username}: {e}")
        return None


def get_dataframe(username: str) -> Optional[pd.DataFrame]:
    """Return stored DataFrame (for export use). None if not tabular or expired."""
    with _lock:
        entry = _store.get(username)
        return entry["df"] if entry else None


def get_filename(username: str) -> Optional[str]:
    with _lock:
        entry = _store.get(username)
        return entry["filename"] if entry else None


def clear(username: str):
    """Manually clear a user's file session."""
    with _lock:
        _store.pop(username, None)
    logger.info(f"[FileEngine] Cleared session for {username}")


# ── File parsers ──────────────────────────────────────────────────────────────

def _parse_pdf(content: bytes, filename: str) -> list:
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=content, filetype="pdf")
        docs = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1}
                ))
        logger.info(f"[FileEngine] PDF: {len(docs)} pages from '{filename}'")
        return docs
    except ImportError:
        logger.warning("[FileEngine] PyMuPDF not installed. Run: pip install pymupdf")
        return []
    except Exception as e:
        logger.error(f"[FileEngine] PDF parse error: {e}")
        return []


def _parse_tabular(content: bytes, filename: str,
                   fmt: str) -> Tuple[Optional[pd.DataFrame], list]:
    try:
        if fmt == "csv":
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8", errors="replace")
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
                docs.append(Document(
                    page_content=row_text,
                    metadata={"source": filename, "row": i}
                ))
        logger.info(f"[FileEngine] Tabular: {len(df)} rows → {len(docs)} docs from '{filename}'")
        return df, docs
    except Exception as e:
        logger.error(f"[FileEngine] Tabular parse error: {e}")
        return None, []


def _parse_json(content: bytes, filename: str) -> list:
    try:
        data = json.loads(content.decode("utf-8", errors="ignore"))
        text = (
            "\n".join(json.dumps(item, indent=2) for item in data)
            if isinstance(data, list)
            else json.dumps(data, indent=2)
        )
        return [Document(page_content=text, metadata={"source": filename})]
    except Exception as e:
        logger.error(f"[FileEngine] JSON parse error: {e}")
        return []


# ── Tabular fast-path ─────────────────────────────────────────────────────────

_DF_STOP = {"the", "is", "are", "was", "for", "from", "with", "and", "all",
            "me", "my", "please", "show", "give", "get", "list", "find"}

def _query_dataframe(question: str, df: pd.DataFrame) -> Optional[str]:
    """
    Answer simple structured questions directly from DataFrame — no LLM.
    Returns formatted string or None (falls back to FAISS if None).
    """
    q          = re.sub(r"[^\w\s]", " ", question).lower()
    q_words    = [w for w in q.split() if len(w) > 2 and w not in _DF_STOP]
    cols_lower = {c.lower(): c for c in df.columns}

    # Count / how many
    if any(p in q for p in ["how many", "count", "total number", "number of"]):
        return f"Total records: **{len(df)}**"

    # Show all / list all
    if any(p in q for p in ["show all", "list all", "give all", "all records",
                              "show me all", "display all", "fetch all"]):
        return df.head(50).to_string(index=False)

    # Column + value filter: "employees where department is Finance"
    for word in q_words:
        if word in cols_lower:
            col       = cols_lower[word]
            after     = q[q.find(word) + len(word):]
            val_words = [w for w in after.split() if len(w) > 2 and w not in _DF_STOP]
            if val_words:
                val      = val_words[0]
                filtered = df[df[col].astype(str).str.lower().str.contains(val, na=False)]
                if not filtered.empty:
                    return filtered.head(20).to_string(index=False)

    # Specific column values: "what is the salary?"
    for word in q_words:
        if word in cols_lower:
            col = cols_lower[word]
            return df[[col]].head(20).to_string(index=False)

    return None
