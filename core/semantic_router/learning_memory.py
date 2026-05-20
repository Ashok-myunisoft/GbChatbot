from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from shared_resources import ai_resources

logger = logging.getLogger(__name__)

_LEARNING_DIR = os.getenv("LEARNING_DIR", "/app/learning")
try:
    Path(_LEARNING_DIR).mkdir(parents=True, exist_ok=True)
except Exception:
    _LEARNING_DIR = str(Path(__file__).resolve().parents[2] / "learning_data")
    Path(_LEARNING_DIR).mkdir(parents=True, exist_ok=True)

_FEEDBACK_FILE = Path(_LEARNING_DIR) / "semantic_routing_feedback.json"
_LOCK = threading.Lock()
_STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "for", "to", "my", "me", "show",
    "list", "give", "get", "tell", "please", "about", "with", "in", "on",
    "this", "that", "these", "those", "can", "you", "do", "does", "did",
    "how", "many", "which", "where", "who", "when", "why",
}


def _atomic_write(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
    tmp_path.replace(path)


def _load_records() -> List[Dict]:
    if not _FEEDBACK_FILE.exists():
        return []
    try:
        with _FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning("[SemanticRouter] Could not load feedback file: %s", exc)
    return []


def _normalize_query(query: str) -> str:
    text = re.sub(r"[^\w\s]", " ", query.lower().strip())
    tokens = [token for token in text.split() if token and token not in _STOP_WORDS]
    tokens.sort()
    return " ".join(tokens)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    embeddings = ai_resources.embeddings
    try:
        return embeddings.embed_documents(texts)
    except Exception:
        return [embeddings.embed_query(text) for text in texts]


def _cosine(a: List[float], b: List[float]) -> float:
    denom_a = sum(x * x for x in a) ** 0.5
    denom_b = sum(x * x for x in b) ** 0.5
    if not denom_a or not denom_b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (denom_a * denom_b)


@dataclass
class FeedbackRecord:
    query: str
    wrong_route: str
    correct_route: str
    reason: str
    timestamp: float
    username: str = ""
    thread_id: str = ""

    @property
    def normalized_query(self) -> str:
        return _normalize_query(self.query)


class SemanticLearningMemory:
    def __init__(self) -> None:
        self._cache: List[FeedbackRecord] = []
        self._cache_loaded = False
        self._cache_lock = threading.Lock()
        self._query_embeddings: Dict[str, List[float]] = {}

    def _ensure_cache(self) -> None:
        if self._cache_loaded:
            return
        with self._cache_lock:
            if self._cache_loaded:
                return
            raw = _load_records()
            self._cache = [FeedbackRecord(**record) for record in raw if isinstance(record, dict)]
            self._cache_loaded = True

    def add_feedback(
        self,
        query: str,
        wrong_route: str,
        correct_route: str,
        reason: str,
        username: str = "",
        thread_id: str = "",
    ) -> FeedbackRecord:
        self._ensure_cache()
        record = FeedbackRecord(
            query=query.strip(),
            wrong_route=wrong_route.strip(),
            correct_route=correct_route.strip(),
            reason=reason.strip(),
            timestamp=time.time(),
            username=username.strip(),
            thread_id=thread_id.strip(),
        )
        with self._cache_lock:
            self._cache.append(record)
            _atomic_write(_FEEDBACK_FILE, [asdict(item) for item in self._cache])
            self._query_embeddings.pop(record.normalized_query, None)
        logger.info(
            "[SemanticRouter] Feedback stored wrong=%s correct=%s query=%s",
            record.wrong_route,
            record.correct_route,
            record.query[:80],
        )
        return record

    def _embed_query(self, query: str) -> List[float]:
        key = _normalize_query(query)
        if key in self._query_embeddings:
            return self._query_embeddings[key]
        vec = ai_resources.embeddings.embed_query(query)
        self._query_embeddings[key] = vec
        return vec

    def find_matching_correction(
        self,
        query: str,
        threshold: float = 0.86,
    ) -> Optional[Dict]:
        self._ensure_cache()
        if not self._cache:
            return None

        query_vec = self._embed_query(query)
        query_norm = _normalize_query(query)

        best_record: Optional[FeedbackRecord] = None
        best_score = 0.0

        for record in self._cache:
            if record.normalized_query and record.normalized_query == query_norm:
                return {
                    "query": record.query,
                    "wrong_route": record.wrong_route,
                    "correct_route": record.correct_route,
                    "reason": record.reason,
                    "timestamp": record.timestamp,
                    "score": 1.0,
                }

            record_vec = self._query_embeddings.get(record.normalized_query)
            if record_vec is None:
                record_vec = self._embed_query(record.query)
                self._query_embeddings[record.normalized_query] = record_vec

            score = _cosine(query_vec, record_vec)
            if score > best_score:
                best_score = score
                best_record = record

        if best_record and best_score >= threshold:
            return {
                "query": best_record.query,
                "wrong_route": best_record.wrong_route,
                "correct_route": best_record.correct_route,
                "reason": best_record.reason,
                "timestamp": best_record.timestamp,
                "score": best_score,
            }
        return None

