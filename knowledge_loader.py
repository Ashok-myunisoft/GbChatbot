"""
knowledge_loader.py

Runs once at FastAPI startup.
  - data.zip or .txt / .json files -> FAISS vector store (general_bot knowledge base)
"""

import os
import hashlib
import logging
import zipfile

from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "/app/data"
DATA_ZIP = "/app/data.zip"
FAISS_PATH = os.path.join(DATA_DIR, "general_faiss")


def load_all(embeddings) -> None:
    """Entry point called once from FastAPI startup."""
    _ensure_data_dir_from_zip()
    _load_rag(embeddings)


def _ensure_data_dir_from_zip() -> None:
    """
    Ensure /app/data contains source knowledge files.
    If the folder is empty, extract supported files from data.zip.
    """
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            logger.info(f"Created DATA_DIR '{DATA_DIR}'")
        except Exception as exc:
            logger.error(f"Cannot create DATA_DIR '{DATA_DIR}': {exc}")
            return

    existing_docs = [
        name
        for name in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, name))
        and (name.endswith(".txt") or name.endswith(".json"))
    ]
    if existing_docs:
        logger.info(f"DATA_DIR already populated with {len(existing_docs)} knowledge files.")
        return

    if not os.path.exists(DATA_ZIP):
        logger.warning(f"{DATA_ZIP} not found - cannot populate DATA_DIR from archive.")
        return

    try:
        extracted = 0
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                if not (member.endswith(".txt") or member.endswith(".json")):
                    continue
                target_name = os.path.basename(member)
                target_path = os.path.join(DATA_DIR, target_name)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
        if extracted:
            logger.info(
                f"Extracted {extracted} knowledge files from '{DATA_ZIP}' into '{DATA_DIR}'"
            )
        else:
            logger.warning(f"{DATA_ZIP} contains no supported knowledge files.")
    except Exception as exc:
        logger.error(f"Failed to extract '{DATA_ZIP}' into '{DATA_DIR}': {exc}")


def _load_rag(embeddings) -> None:
    """Load .txt and .json files into a FAISS vector store for general_bot."""
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            logger.info(f"Created DATA_DIR '{DATA_DIR}'")
        except Exception as exc:
            logger.error(f"Cannot create DATA_DIR '{DATA_DIR}': {exc} - skipping FAISS build.")
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
        logger.warning("No .txt/.json documents found - FAISS store not built.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_PATH)

    index_file = os.path.join(FAISS_PATH, "index.faiss")
    if os.path.exists(index_file):
        sha = hashlib.sha256(open(index_file, "rb").read()).hexdigest()
        with open(os.path.join(FAISS_PATH, "index.sha256"), "w") as f:
            f.write(sha)

    logger.info(
        f"FAISS: built from {len(all_docs)} docs "
        f"({len(chunks)} chunks), saved to '{FAISS_PATH}'"
    )
