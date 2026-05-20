from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from db_setup import get_pg_conn, release_pg_conn
from shared_resources import ai_resources

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (query or "").lower().strip())
    tokens = [token for token in text.split() if token]
    tokens.sort()
    return " ".join(tokens)


def _atomic_write(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
    tmp_path.replace(path)


@dataclass
class AnswerFeedbackRecord:
    feedback_id: str
    query: str
    original_answer: str
    corrected_answer: str
    feedback_type: str
    rating: int
    reason: str
    timestamp: float
    username: str = ""
    thread_id: str = ""
    bot_type: str = ""
    user_role: str = ""
    context_text: str = ""

    @property
    def normalized_query(self) -> str:
        return _normalize_query(self.query)


class AnswerLearningMemory:
    def __init__(self, vectorstore_path: str = "answer_learning_memory_store", metadata_file: str = "answer_learning_memory_meta.json") -> None:
        self.vectorstore_path = vectorstore_path
        self.metadata_file = metadata_file
        self.embeddings = ai_resources.embeddings
        self.memory_vectorstore = None
        self.memory_counter = 0
        self._cache: List[AnswerFeedbackRecord] = []
        self._cache_loaded = False
        self._cache_lock = threading.Lock()
        self._query_embeddings: Dict[str, List[float]] = {}
        self.load_memory_vectorstore()

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
            recent_turns = context.get("recent_turns")
            if isinstance(recent_turns, list):
                for turn in recent_turns:
                    if isinstance(turn, str) and turn.strip():
                        parts.append(turn.strip())
                    elif isinstance(turn, dict):
                        user_message = (turn.get("user_message") or "").strip()
                        bot_response = (turn.get("bot_response") or "").strip()
                        bot_type = (turn.get("bot_type") or "").strip()
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

    def _ensure_cache(self) -> None:
        if self._cache_loaded:
            return
        with self._cache_lock:
            if self._cache_loaded:
                return
            try:
                conn = get_pg_conn()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT feedback_id, query, original_answer, corrected_answer, feedback_type,
                               rating, reason, username, thread_id, bot_type, user_role,
                               context_text, created_at
                        FROM answer_feedback_vectors
                        ORDER BY created_at DESC
                        LIMIT 5000
                        """
                    )
                    rows = cur.fetchall()
                release_pg_conn(conn)
                allowed_fields = {field.name for field in dataclass_fields(AnswerFeedbackRecord)}
                records: List[AnswerFeedbackRecord] = []
                for row in rows:
                    payload = {key: value for key, value in dict(row).items() if key in allowed_fields}
                    if "feedback_id" not in payload:
                        continue
                    payload.setdefault("original_answer", "")
                    payload.setdefault("corrected_answer", "")
                    payload.setdefault("feedback_type", "answer_correction")
                    payload.setdefault("rating", 0)
                    payload.setdefault("reason", "")
                    payload.setdefault("timestamp", time.time())
                    payload.setdefault("username", "")
                    payload.setdefault("thread_id", "")
                    payload.setdefault("bot_type", "")
                    payload.setdefault("user_role", "")
                    payload.setdefault("context_text", "")
                    try:
                        records.append(AnswerFeedbackRecord(**payload))
                    except TypeError as exc:
                        logger.warning("[AnswerLearning] Skipping malformed feedback row: %s", exc)
                self._cache = records
                self._cache_loaded = True
                self._rebuild_vectorstore_from_cache()
            except Exception as exc:
                logger.error("[AnswerLearning] Failed to load feedback records: %s", exc, exc_info=True)
                self._cache = []
                self._cache_loaded = True
                dummy = Document(page_content="Answer learning memory initialized", metadata={"feedback_id": "init"})
                self.memory_vectorstore = FAISS.from_documents([dummy], self.embeddings)

    def refresh(self) -> None:
        """Reload feedback records and rebuild the vectorstore from PostgreSQL."""
        with self._cache_lock:
            self._cache_loaded = False
            self._cache = []
            self._query_embeddings.clear()
        self.load_memory_vectorstore()

    def get_statistics(self) -> Dict[str, int]:
        """Return lightweight counts for monitoring and admin endpoints."""
        self._ensure_cache()
        stats = {
            "total_feedback": len(self._cache),
            "answer_corrections": 0,
            "helpful": 0,
            "negative": 0,
            "rated_positive": 0,
            "rated_negative": 0,
        }
        for record in self._cache:
            if record.feedback_type == "answer_correction":
                stats["answer_corrections"] += 1
            elif record.feedback_type in {"helpful", "positive", "good_answer"}:
                stats["helpful"] += 1
            else:
                stats["negative"] += 1
            if record.rating >= 4:
                stats["rated_positive"] += 1
            if record.rating <= 2:
                stats["rated_negative"] += 1
        return stats

    def _rebuild_vectorstore_from_cache(self) -> None:
        docs: List[Document] = []
        for record in self._cache:
            content = self._build_content(record.query, record.original_answer, record.corrected_answer, record.reason, record.feedback_type, record.bot_type, record.user_role, record.context_text)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "feedback_id": record.feedback_id,
                        "username": record.username,
                        "thread_id": record.thread_id,
                        "bot_type": record.bot_type,
                        "user_role": record.user_role,
                        "feedback_type": record.feedback_type,
                        "rating": record.rating,
                        "query": record.query,
                        "original_answer": record.original_answer[:500],
                        "corrected_answer": record.corrected_answer[:500],
                        "reason": record.reason,
                        "context_text": record.context_text[:1500],
                        "timestamp": str(record.timestamp),
                    },
                )
            )
        if docs:
            self.memory_vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.memory_counter = len(docs)
        else:
            dummy = Document(page_content="Answer learning memory initialized", metadata={"feedback_id": "init"})
            self.memory_vectorstore = FAISS.from_documents([dummy], self.embeddings)

    @staticmethod
    def _build_content(
        query: str,
        original_answer: str,
        corrected_answer: str,
        reason: str,
        feedback_type: str,
        bot_type: str,
        user_role: str,
        context_text: str,
    ) -> str:
        preferred_answer = corrected_answer.strip() if corrected_answer.strip() else original_answer.strip()
        return (
            f"Question: {query}\n"
            f"Preferred Answer: {preferred_answer[:1200]}\n"
            f"Original Answer: {original_answer[:500]}\n"
            f"Feedback Type: {feedback_type}\n"
            f"Bot Type: {bot_type}\n"
            f"User Role: {user_role}\n"
            f"Reason: {reason}\n"
            f"Context: {context_text[:1500]}"
        )

    def store_feedback(
        self,
        username: str,
        query: str,
        original_answer: str,
        corrected_answer: str = "",
        feedback_type: str = "answer_correction",
        rating: int = 0,
        reason: str = "",
        bot_type: str = "",
        user_role: str = "",
        thread_id: str = "",
        context: object = None,
    ) -> AnswerFeedbackRecord:
        self._ensure_cache()
        context_text = self._context_to_text(context)
        feedback_id = f"{username}_{int(time.time() * 1000)}_{len(self._cache)}"
        record = AnswerFeedbackRecord(
            feedback_id=feedback_id,
            query=(query or "").strip(),
            original_answer=(original_answer or "").strip(),
            corrected_answer=(corrected_answer or "").strip(),
            feedback_type=(feedback_type or "answer_correction").strip(),
            rating=int(rating or 0),
            reason=(reason or "").strip(),
            timestamp=time.time(),
            username=(username or "").strip(),
            thread_id=(thread_id or "").strip(),
            bot_type=(bot_type or "").strip(),
            user_role=(user_role or "").strip(),
            context_text=context_text,
        )
        content = self._build_content(
            record.query,
            record.original_answer,
            record.corrected_answer,
            record.reason,
            record.feedback_type,
            record.bot_type,
            record.user_role,
            record.context_text,
        )
        with self._cache_lock:
            self._cache.append(record)
            self._query_embeddings.pop(record.normalized_query, None)
            try:
                if self.memory_vectorstore:
                    self.memory_vectorstore.add_documents(
                        [
                            Document(
                                page_content=content,
                                metadata={
                                    "feedback_id": record.feedback_id,
                                    "username": record.username,
                                    "thread_id": record.thread_id,
                                    "bot_type": record.bot_type,
                                    "user_role": record.user_role,
                                    "feedback_type": record.feedback_type,
                                    "rating": record.rating,
                                    "query": record.query,
                                    "original_answer": record.original_answer[:500],
                                    "corrected_answer": record.corrected_answer[:500],
                                    "reason": record.reason,
                                    "context_text": record.context_text[:1500],
                                    "timestamp": str(record.timestamp),
                                },
                            )
                        ]
                    )
            except Exception as exc:
                logger.warning("[AnswerLearning] FAISS add failed, will rely on DB rebuild: %s", exc)
            try:
                conn = get_pg_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO answer_feedback_vectors
                            (feedback_id, username, thread_id, bot_type, user_role,
                             feedback_type, rating, query, original_answer,
                             corrected_answer, reason, context_text)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (feedback_id) DO NOTHING
                        """,
                        (
                            record.feedback_id,
                            record.username,
                            record.thread_id,
                            record.bot_type,
                            record.user_role,
                            record.feedback_type,
                            record.rating,
                            record.query,
                            record.original_answer,
                            record.corrected_answer,
                            record.reason,
                            record.context_text,
                        ),
                    )
                conn.commit()
                release_pg_conn(conn)
            except Exception as exc:
                logger.error("[AnswerLearning] Failed to persist feedback: %s", exc, exc_info=True)
        self.memory_counter += 1
        logger.info(
            "[AnswerLearning] Stored feedback type=%s rating=%s query=%s",
            record.feedback_type,
            record.rating,
            record.query[:80],
        )
        return record

    def _should_use_record(self, record: AnswerFeedbackRecord) -> bool:
        if record.feedback_type in {"answer_correction", "helpful", "positive", "good_answer"}:
            return True
        if record.rating >= 4:
            return True
        if record.corrected_answer.strip():
            return True
        return False

    def retrieve_relevant_examples(
        self,
        username: str,
        query: str,
        k: int = 2,
        thread_id: str = "",
        thread_isolation: bool = False,
    ) -> List[Dict]:
        self._ensure_cache()
        if not self.memory_vectorstore:
            return []

        try:
            docs = self.memory_vectorstore.similarity_search(query, k=max(1, k * 3))
        except Exception as exc:
            logger.warning("[AnswerLearning] Similarity search failed: %s", exc)
            return []

        results: List[Dict] = []
        seen = set()
        for doc in docs:
            metadata = doc.metadata or {}
            feedback_id = metadata.get("feedback_id")
            if not feedback_id or feedback_id in seen or feedback_id == "init":
                continue
            seen.add(feedback_id)
            if metadata.get("username") not in ("", username):
                continue
            if thread_isolation and thread_id and metadata.get("thread_id") not in ("", thread_id):
                continue

            record = next((item for item in self._cache if item.feedback_id == feedback_id), None)
            if not record or not self._should_use_record(record):
                continue

            results.append(
                {
                    "feedback_id": record.feedback_id,
                    "query": record.query,
                    "original_answer": record.original_answer,
                    "corrected_answer": record.corrected_answer,
                    "preferred_answer": record.corrected_answer or record.original_answer,
                    "feedback_type": record.feedback_type,
                    "rating": record.rating,
                    "reason": record.reason,
                    "bot_type": record.bot_type,
                    "user_role": record.user_role,
                    "context_text": record.context_text,
                    "timestamp": record.timestamp,
                }
            )
            if len(results) >= k:
                break
        return results
