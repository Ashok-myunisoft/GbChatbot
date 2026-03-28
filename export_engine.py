"""
export_engine.py

On-demand file export engine.
- Converts a DataFrame or text answer to PDF / CSV / Excel / JSON
- Stores the output in a temp dict with a 10-minute TTL
- Caller gets back a file_id; the download endpoint serves the bytes

Zero impact on existing system — only called when user asks for a file export.
"""

import io
import json
import time
import uuid
import logging
import threading
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

EXPORT_TTL = 600          # 10-minute expiry for generated files
_store: dict = {}         # { file_id → { bytes, filename, mime, ts } }
_lock  = threading.Lock()

# ── Keyword detection ──────────────────────────────────────────────────────────

_EXPORT_KEYWORDS = {
    "as pdf", "as csv", "as excel", "as xlsx", "as json",
    "as a pdf", "as a csv", "as a excel", "as an excel", "as a json",
    "to pdf", "to csv", "to excel", "to xlsx", "to json",
    "in pdf", "in csv", "in excel", "in json",
    "give pdf", "give csv", "give excel", "give xlsx", "give json",
    "export pdf", "export csv", "export excel", "export json",
    "download pdf", "download csv", "download excel", "download json",
    "pdf download", "csv download", "excel download", "json download",
    "pdf to download", "csv to download", "excel to download",
    "save as pdf", "save as csv", "save as excel", "save as json",
    "convert to pdf", "convert to csv", "convert to excel", "convert to json",
    "give as pdf", "give as csv", "give as excel", "give as json",
    "get pdf", "get csv", "get excel", "get json",
    "generate pdf", "generate csv", "generate excel", "generate json",
}


def detect_export_format(text: str) -> Optional[str]:
    """
    Scan user message for an export request.
    Returns one of 'pdf' | 'csv' | 'excel' | 'json', or None.
    """
    t = text.lower()
    for kw in _EXPORT_KEYWORDS:
        if kw in t:
            # determine format from keyword
            for fmt in ("pdf", "csv", "excel", "xlsx", "json"):
                if fmt in kw:
                    return "excel" if fmt == "xlsx" else fmt
    return None


# ── Export builders ────────────────────────────────────────────────────────────

def build_export(
    fmt: str,
    answer: str,
    df: Optional[pd.DataFrame] = None,
    filename_hint: str = "export",
) -> Optional[str]:
    """
    Build the export file and return a file_id for download.
    Returns None on failure.

    Priority: if a DataFrame is available use it (structured data);
    otherwise fall back to rendering the answer text.
    """
    try:
        if fmt == "csv":
            return _build_csv(df, answer, filename_hint)
        elif fmt == "excel":
            return _build_excel(df, answer, filename_hint)
        elif fmt == "json":
            return _build_json(df, answer, filename_hint)
        elif fmt == "pdf":
            return _build_pdf(df, answer, filename_hint)
        else:
            logger.warning(f"[ExportEngine] Unknown format: {fmt}")
            return None
    except Exception as e:
        logger.error(f"[ExportEngine] build_export({fmt}) failed: {e}")
        return None


def get_file(file_id: str) -> Optional[dict]:
    """
    Retrieve export by file_id.  Returns dict with keys:
      bytes, filename, mime
    or None if not found / expired.
    """
    with _lock:
        entry = _store.get(file_id)
        if not entry:
            return None
        if time.time() - entry["ts"] > EXPORT_TTL:
            del _store[file_id]
            logger.info(f"[ExportEngine] Expired: {file_id}")
            return None
        return {
            "bytes":    entry["bytes"],
            "filename": entry["filename"],
            "mime":     entry["mime"],
        }


def evict_expired():
    """Remove all expired entries (call periodically or on demand)."""
    now = time.time()
    with _lock:
        expired = [fid for fid, e in _store.items() if now - e["ts"] > EXPORT_TTL]
        for fid in expired:
            del _store[fid]
    if expired:
        logger.info(f"[ExportEngine] Evicted {len(expired)} expired exports")


# ── Internal builders ──────────────────────────────────────────────────────────

def _store_file(data: bytes, filename: str, mime: str) -> str:
    file_id = str(uuid.uuid4())
    with _lock:
        _store[file_id] = {
            "bytes":    data,
            "filename": filename,
            "mime":     mime,
            "ts":       time.time(),
        }
    logger.info(f"[ExportEngine] Stored {filename} ({len(data)} bytes) → {file_id}")
    return file_id


def _build_csv(df, answer: str, hint: str) -> str:
    if df is not None and not df.empty:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
    else:
        # Wrap plain-text answer as a single-column CSV
        lines = [l.strip() for l in answer.splitlines() if l.strip()]
        buf   = io.StringIO()
        buf.write("response\n")
        for line in lines:
            buf.write(f'"{line.replace(chr(34), chr(34)+chr(34))}"\n')
        data = buf.getvalue().encode("utf-8")
    return _store_file(data, f"{hint}.csv", "text/csv")


def _build_excel(df, answer: str, hint: str) -> str:
    buf = io.BytesIO()
    if df is not None and not df.empty:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
    else:
        lines  = [l.strip() for l in answer.splitlines() if l.strip()]
        tmp_df = pd.DataFrame({"Response": lines})
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            tmp_df.to_excel(writer, index=False, sheet_name="Results")
    data = buf.getvalue()
    return _store_file(
        data,
        f"{hint}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _build_json(df, answer: str, hint: str) -> str:
    if df is not None and not df.empty:
        payload = df.to_dict(orient="records")
    else:
        payload = {"response": answer}
    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    return _store_file(data, f"{hint}.json", "application/json")


def _build_pdf(df, answer: str, hint: str) -> str:
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("[ExportEngine] fpdf2 not installed. Run: pip install fpdf2")
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    if df is not None and not df.empty:
        # Table header
        pdf.set_font("Helvetica", style="B", size=10)
        cols    = list(df.columns)
        col_w   = min(40, int(190 / max(len(cols), 1)))
        for col in cols:
            pdf.cell(col_w, 8, str(col)[:20], border=1)
        pdf.ln()
        pdf.set_font("Helvetica", size=9)
        for _, row in df.head(200).iterrows():
            for col in cols:
                val = str(row[col]) if row[col] is not None else ""
                pdf.cell(col_w, 7, val[:20], border=1)
            pdf.ln()
    else:
        # Plain text answer
        for line in answer.splitlines():
            line = line.strip()
            if not line:
                pdf.ln(4)
                continue
            # Bold for markdown-style headers (lines starting with **)
            if line.startswith("**") and line.endswith("**"):
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.multi_cell(0, 8, line.strip("*"))
                pdf.set_font("Helvetica", size=11)
            else:
                pdf.multi_cell(0, 7, line)

    data = pdf.output()
    if isinstance(data, str):
        data = data.encode("latin-1")
    return _store_file(data, f"{hint}.pdf", "application/pdf")
