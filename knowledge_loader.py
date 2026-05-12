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
ZIP_HASH_PATH = os.path.join(FAISS_PATH, "source.zip.sha256")


def load_all(embeddings) -> None:
    """Entry point called once from FastAPI startup."""
    current_zip_hash = _current_zip_hash()
    if _is_cached_index_valid(current_zip_hash):
        logger.info("FAISS cache is up to date — skipping extraction and rebuild.")
        return

    _ensure_data_dir_from_zip(current_zip_hash)
    _load_rag(embeddings, current_zip_hash)


def _hash_file(path: str) -> str:
    """Return SHA-256 hex digest for a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _current_zip_hash() -> str | None:
    if not os.path.exists(DATA_ZIP):
        return None
    return _hash_file(DATA_ZIP)


def _stored_zip_hash() -> str | None:
    if not os.path.exists(ZIP_HASH_PATH):
        return None
    try:
        with open(ZIP_HASH_PATH, "r") as f:
            return f.read().strip() or None
    except Exception:
        return None


def _is_cached_index_valid(current_zip_hash: str | None = None) -> bool:
    index_file = os.path.join(FAISS_PATH, "index.faiss")
    current_hash = current_zip_hash or _current_zip_hash()
    stored_hash = _stored_zip_hash()
    return bool(
        current_hash
        and stored_hash
        and current_hash == stored_hash
        and os.path.exists(index_file)
    )


def _clear_supported_docs() -> None:
    """Remove extracted source files before refreshing from the archive."""
    if not os.path.exists(DATA_DIR):
        return

    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".txt") or name.endswith(".json"):
            try:
                os.remove(path)
            except Exception as exc:
                logger.warning(f"Could not remove stale file '{path}': {exc}")


def _ensure_data_dir_from_zip(current_zip_hash: str | None = None) -> None:
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
    stored_hash = _stored_zip_hash()
    current_hash = current_zip_hash or _current_zip_hash()

    if existing_docs and stored_hash and current_hash and stored_hash == current_hash:
        logger.info(f"DATA_DIR already populated with {len(existing_docs)} knowledge files.")
        return

    if not os.path.exists(DATA_ZIP):
        logger.warning(f"{DATA_ZIP} not found - cannot populate DATA_DIR from archive.")
        return

    try:
        _clear_supported_docs()
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


def _load_rag(embeddings, current_zip_hash: str | None = None) -> None:
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
        sha = _hash_file(index_file)
        with open(os.path.join(FAISS_PATH, "index.sha256"), "w") as f:
            f.write(sha)
        zip_hash = current_zip_hash or _current_zip_hash()
        if zip_hash:
            with open(ZIP_HASH_PATH, "w") as f:
                f.write(zip_hash)

    logger.info(
        f"FAISS: built from {len(all_docs)} docs "
        f"({len(chunks)} chunks), saved to '{FAISS_PATH}'"
    )
