"""
knowledge_loader.py

Runs once at FastAPI startup.
  - .txt / .json files  → FAISS vector store  (general_bot knowledge base)
  - .csv / .xlsx files  → DuckDB tables        (structured bot knowledge base)
"""

import os
import logging
import duckdb
import pandas as pd
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/app/data"
DUCKDB_PATH = os.path.join(DATA_DIR, "knowledge.duckdb")
FAISS_PATH  = os.path.join(DATA_DIR, "general_faiss")

# ── File → DuckDB table name mapping ──────────────────────────────────────────
CSV_TABLE_MAP = {
    "menu.csv":          "menu",
    "MFILE.csv":         "MFILE",
    "MFORMULAFIELD.csv": "MFORMULAFIELD",
    "MREPORT.csv":       "MREPORT",
}

XLSX_TABLE_MAP = {
    "unisoft_all_tables_export.xlsx": "Unisoft",
}


def load_all(embeddings) -> None:
    """Entry point — called once from FastAPI startup event."""
    _load_structured()
    _load_rag(embeddings)


# ── Structured data (DuckDB) ───────────────────────────────────────────────────
def _load_structured() -> None:
    """Load CSV and xlsx files into DuckDB tables."""
    if not os.path.exists(DATA_DIR):
        logger.warning(f"DATA_DIR '{DATA_DIR}' not found — skipping DuckDB load.")
        return

    conn = duckdb.connect(DUCKDB_PATH)
    try:
        # CSV files
        for filename, table in CSV_TABLE_MAP.items():
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                logger.warning(f"{path} not found — skipping table '{table}'.")
                continue
            try:
                df = pd.read_csv(path, encoding="cp1252")
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
                n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                logger.info(f"DuckDB: {n} rows from '{filename}' -> table '{table}'")
            except Exception as exc:
                logger.error(f"Error loading '{filename}' into DuckDB: {exc}")

        # Excel files
        for filename, table in XLSX_TABLE_MAP.items():
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                logger.warning(f"{path} not found — skipping table '{table}'.")
                continue
            try:
                df = pd.read_excel(path)
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df')
                n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                logger.info(f"DuckDB: {n} rows from '{filename}' -> table '{table}'")
            except Exception as exc:
                logger.error(f"Error loading '{filename}' into DuckDB: {exc}")
    finally:
        conn.close()


# ── Unstructured data (FAISS) ──────────────────────────────────────────────────
def _load_rag(embeddings) -> None:
    """Load .txt and .json files into a FAISS vector store for general_bot."""
    if not os.path.exists(DATA_DIR):
        logger.warning(f"DATA_DIR '{DATA_DIR}' not found — skipping FAISS build.")
        return

    all_docs = []
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.endswith(".txt"):
                docs = TextLoader(fpath, encoding="utf-8").load()
                all_docs.extend(docs)
            elif fname.endswith(".json"):
                docs = JSONLoader(fpath, jq_schema=".", text_content=False).load()
                all_docs.extend(docs)
        except Exception as exc:
            logger.error(f"Error loading '{fpath}' for RAG: {exc}")

    if not all_docs:
        logger.warning("No .txt/.json documents found — FAISS store not built.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(all_docs)
    vs       = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_PATH)
    logger.info(
        f"FAISS: built from {len(all_docs)} docs "
        f"({len(chunks)} chunks), saved to '{FAISS_PATH}'"
    )
