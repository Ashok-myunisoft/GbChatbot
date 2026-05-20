from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
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


def get_feedback_records() -> List[Dict]:
    """Public accessor for stored routing feedback records."""
    return _load_records()


def get_feedback_file_path() -> str:
    """Public accessor for the routing feedback file path."""
    return str(_FEEDBACK_FILE)


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
    bot_type: str = ""
    thread_title: str = ""
    last_user_message: str = ""
    last_bot_response: str = ""
    context_text: str = ""
    domain_hint: str = ""
    intent_hint: str = ""

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
            allowed_fields = {field.name for field in dataclass_fields(FeedbackRecord)}
            cleaned: List[FeedbackRecord] = []
            for record in raw:
                if not isinstance(record, dict):
                    continue
                payload = {key: value for key, value in record.items() if key in allowed_fields}
                try:
                    cleaned.append(FeedbackRecord(**payload))
                except TypeError as exc:
                    logger.warning("[SemanticRouter] Skipping malformed feedback record: %s", exc)
            self._cache = cleaned
            self._cache_loaded = True

    @staticmethod
    def _context_to_text(context: object) -> str:
        if context is None:
            return ""
        if isinstance(context, str):
            return context.strip()
        if isinstance(context, dict):
            parts: List[str] = []
            for key in (
                "thread_title",
                "last_user_message",
                "last_bot_response",
                "bot_type",
                "domain_hint",
                "intent_hint",
                "context_text",
            ):
                value = context.get(key)
                if value:
                    parts.append(f"{key}: {value}")
            for key in ("recent_turns", "recent_messages"):
                value = context.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            parts.append(item.strip())
                        elif isinstance(item, dict):
                            user_message = (item.get("user_message") or "").strip()
                            bot_response = (item.get("bot_response") or "").strip()
                            bot_type = (item.get("bot_type") or "").strip()
                            if user_message or bot_response or bot_type:
                                parts.append(
                                    " | ".join(
                                        part for part in (
                                            f"user: {user_message}" if user_message else "",
                                            f"bot: {bot_response}" if bot_response else "",
                                            f"type: {bot_type}" if bot_type else "",
                                        )
                                        if part
                                    )
                                )
            return "\n".join(parts).strip()
        if isinstance(context, (list, tuple)):
            return "\n".join(str(item).strip() for item in context if str(item).strip())
        return str(context).strip()

    def add_feedback(
        self,
        query: str,
        wrong_route: str,
        correct_route: str,
        reason: str,
        username: str = "",
        thread_id: str = "",
        context: object = None,
    ) -> FeedbackRecord:
        self._ensure_cache()
        context_text = self._context_to_text(context)
        context_payload = context if isinstance(context, dict) else {}
        record = FeedbackRecord(
            query=query.strip(),
            wrong_route=wrong_route.strip(),
            correct_route=correct_route.strip(),
            reason=reason.strip(),
            timestamp=time.time(),
            username=username.strip(),
            thread_id=thread_id.strip(),
            bot_type=str(context_payload.get("bot_type", "")).strip(),
            thread_title=str(context_payload.get("thread_title", "")).strip(),
            last_user_message=str(context_payload.get("last_user_message", "")).strip(),
            last_bot_response=str(context_payload.get("last_bot_response", "")).strip(),
            context_text=context_text,
            domain_hint=str(context_payload.get("domain_hint", "")).strip(),
            intent_hint=str(context_payload.get("intent_hint", "")).strip(),
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
        context_text: str = "",
        username: str = "",
        thread_id: str = "",
        threshold: float = 0.86,
    ) -> Optional[Dict]:
        self._ensure_cache()
        if not self._cache:
            return None

        query_vec = self._embed_query(query)
        context_norm = _normalize_query(context_text) if context_text else ""
        context_vec = self._embed_query(context_text) if context_text else []
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
            if context_norm and record.context_text:
                record_context_norm = _normalize_query(record.context_text)
                if record_context_norm == context_norm:
                    score = min(1.0, score + 0.12)
                else:
                    record_context_vec = self._query_embeddings.get(record_context_norm)
                    if record_context_vec is None:
                        record_context_vec = self._embed_query(record.context_text)
                        self._query_embeddings[record_context_norm] = record_context_vec
                    context_score = _cosine(context_vec, record_context_vec)
                    score = (score * 0.78) + (context_score * 0.22)
            if username and record.username and record.username == username:
                score = min(1.0, score + 0.02)
            if thread_id and record.thread_id and record.thread_id == thread_id:
                score = min(1.0, score + 0.03)
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
