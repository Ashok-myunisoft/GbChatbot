from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2.extras

from db_setup import get_pg_conn, release_pg_conn
from core.semantic_router.learning_memory import get_feedback_file_path, get_feedback_records
import retry_handler

logger = logging.getLogger(__name__)

# Cap on how many examples per unique (normalized) question are allowed
# into the final dataset -- without this, a handful of over-tested
# questions (one seen ~20 times in manual testing) dominate the LoRA
# training signal while rarer real corrections get drowned out (#5).
_MAX_PER_QUESTION = 3


def _normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower().strip())
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_str(value: object, limit: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:limit]


def _write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _split_records(records: Sequence[Dict], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    data = list(records)
    rng = random.Random(seed)
    rng.shuffle(data)

    if not data:
        return [], [], []

    train_end = max(1, int(len(data) * train_ratio))
    val_end = max(train_end + 1, int(len(data) * (train_ratio + val_ratio)))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    if not val and len(data) > 1:
        val = [train.pop()]
    if not test and len(data) > 2:
        test = [val.pop()]

    return train, val, test


@dataclass
class ExportResult:
    output_dir: str
    all_file: str
    train_file: str
    val_file: str
    test_file: str
    records_exported: int
    answer_records: int
    routing_records: int


class LearningDatasetExporter:
    def __init__(self) -> None:
        self.feedback_file_path = Path(get_feedback_file_path())

    def _fetch_thread_context(self, thread_id: str) -> Dict[str, object]:
        if not thread_id:
            return {}
        try:
            conn = get_pg_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT thread_id, username, title, messages, user_role, user_name
                    FROM conversation_threads
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )
                row = cur.fetchone()
            release_pg_conn(conn)
            if not row:
                return {}

            messages = row.get("messages") or []
            recent = messages[-3:] if isinstance(messages, list) else []
            formatted_turns: List[str] = []
            for msg in recent:
                if not isinstance(msg, dict):
                    continue
                formatted_turns.append(
                    " | ".join(
                        part for part in (
                            f"user: {_safe_str(msg.get('user_message'), 250)}" if msg.get("user_message") else "",
                            f"bot: {_safe_str(msg.get('bot_response'), 250)}" if msg.get("bot_response") else "",
                            f"type: {_safe_str(msg.get('bot_type'), 80)}" if msg.get("bot_type") else "",
                        )
                        if part
                    )
                )
            return {
                "thread_id": thread_id,
                "thread_title": _safe_str(row.get("title"), 120),
                "username": _safe_str(row.get("username"), 80),
                "user_role": _safe_str(row.get("user_role"), 50),
                "user_name": _safe_str(row.get("user_name"), 80),
                "recent_turns": formatted_turns,
            }
        except Exception as exc:
            logger.warning("[LearningExport] Could not fetch thread context for %s: %s", thread_id, exc)
            return {}

    def _build_answer_example(self, row: Dict[str, object]) -> Dict[str, object]:
        thread_id = _safe_str(row.get("thread_id"))
        thread_context = self._fetch_thread_context(thread_id)
        corrected_answer = _safe_str(row.get("corrected_answer"))
        original_answer = _safe_str(row.get("original_answer"))
        preferred_answer = corrected_answer or original_answer
        context_lines = []
        if thread_context.get("thread_title"):
            context_lines.append(f"Thread: {thread_context['thread_title']}")
        if thread_context.get("recent_turns"):
            context_lines.append("Recent turns:")
            context_lines.extend(thread_context["recent_turns"])
        if row.get("context_text"):
            context_lines.append(f"Stored context: {_safe_str(row.get('context_text'), 1200)}")

        input_text = "\n".join(
            [
                f"Role: {_safe_str(row.get('user_role'))}",
                f"Bot Type: {_safe_str(row.get('bot_type'))}",
                f"Question: {_safe_str(row.get('query'))}",
                *context_lines,
            ]
        ).strip()

        return {
            "task_type": "answer",
            "source": "answer_feedback_vectors",
            "feedback_id": _safe_str(row.get("feedback_id")),
            "username": _safe_str(row.get("username")),
            "thread_id": thread_id,
            "bot_type": _safe_str(row.get("bot_type")),
            "user_role": _safe_str(row.get("user_role")),
            "feedback_type": _safe_str(row.get("feedback_type")),
            "rating": int(row.get("rating") or 0),
            "reason": _safe_str(row.get("reason"), 1000),
            "input": input_text,
            "output": preferred_answer,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "context": thread_context,
            "normalized_query": _normalize_text(_safe_str(row.get("query"))),
        }

    def _build_route_example(self, record: Dict[str, object]) -> Dict[str, object]:
        context_lines: List[str] = []
        context_text = _safe_str(record.get("context_text"))
        if context_text:
            context_lines.append(context_text)
        recent = record.get("recent_turns")
        if isinstance(recent, list):
            context_lines.extend(_safe_str(item, 600) for item in recent if _safe_str(item))

        input_text = "\n".join(
            [
                f"Question: {_safe_str(record.get('query'))}",
                f"Thread: {_safe_str(record.get('thread_id'))}",
                *context_lines,
            ]
        ).strip()

        return {
            "task_type": "route",
            "source": "semantic_routing_feedback",
            "username": _safe_str(record.get("username")),
            "thread_id": _safe_str(record.get("thread_id")),
            "wrong_route": _safe_str(record.get("wrong_route")),
            "correct_route": _safe_str(record.get("correct_route")),
            "reason": _safe_str(record.get("reason"), 1000),
            "input": input_text,
            "output": _safe_str(record.get("correct_route")),
            "context": _safe_str(record.get("context_text"), 1500),
            "normalized_query": _normalize_text(_safe_str(record.get("query"))),
        }

    def _load_retry_jsonl_rows(self) -> List[Dict[str, object]]:
        """
        Reads learning_exports/retry_feedback_dataset.jsonl (written by
        retry_handler.save_learning_example) and turns each confirmed
        negative/positive pair into one training example: "don't answer
        like the rejected one, answer like the corrected one." Records
        still labeled "still_wrong" (a positive that was itself rejected
        again -- see retry_handler.mark_last_positive_as_still_wrong) are
        excluded, since they were never actually confirmed good (#8).
        """
        path = Path(retry_handler.LEARNING_FILE)
        if not path.exists():
            return []

        rows: List[Dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("[LearningExport] Could not read retry JSONL: %s", exc)
            return []

        # Group by pair_id where available so a negative/positive pair is
        # matched explicitly (#6) rather than reconstructed by proximity.
        by_pair: Dict[str, Dict[str, Dict]] = {}
        legacy_pending_negative: Optional[Dict] = None
        examples: List[Dict[str, object]] = []

        for row in rows:
            label = row.get("label")
            pair_id = row.get("pair_id") or ""

            if pair_id:
                bucket = by_pair.setdefault(pair_id, {})
                bucket[label] = row
                continue

            # Legacy rows written before pair_id existed -- fall back to
            # adjacent negative->positive matching, best effort only.
            if label == "negative":
                legacy_pending_negative = row
            elif label == "positive" and legacy_pending_negative:
                examples.append(self._build_retry_example(legacy_pending_negative, row))
                legacy_pending_negative = None

        for bucket in by_pair.values():
            neg = bucket.get("negative")
            pos = bucket.get("positive")
            if neg and pos:
                examples.append(self._build_retry_example(neg, pos))
            # still_wrong / negative-only pairs are intentionally dropped --
            # there's no confirmed-good answer to teach from.

        return examples

    def _build_retry_example(self, negative_row: Dict, positive_row: Dict) -> Dict[str, object]:
        question = _safe_str(positive_row.get("instruction") or negative_row.get("instruction"))
        rejected_answer = _safe_str(negative_row.get("response"))
        corrected_answer = _safe_str(positive_row.get("response"))
        input_text = "\n".join(
            [
                f"Bot Type: {_safe_str(positive_row.get('bot_type'))}",
                f"Question: {question}",
                f"Rejected answer (do not repeat): {rejected_answer[:800]}",
            ]
        ).strip()
        return {
            "task_type": "answer",
            "source": "retry_feedback_dataset",
            "feedback_id": _safe_str(positive_row.get("pair_id")),
            "username": _safe_str(positive_row.get("username")),
            "thread_id": _safe_str(positive_row.get("thread_id")),
            "bot_type": _safe_str(positive_row.get("bot_type")),
            "user_role": "",
            "feedback_type": "retry_correction",
            "rating": 5,
            "reason": _safe_str(positive_row.get("category")),
            "input": input_text,
            "output": corrected_answer,
            "original_answer": rejected_answer,
            "corrected_answer": corrected_answer,
            "context": {},
            "normalized_query": _normalize_text(question),
        }

    def load_records(self) -> Tuple[List[Dict[str, object]], int, int]:
        answer_rows: List[Dict[str, object]] = []
        route_rows: List[Dict[str, object]] = []

        try:
            conn = get_pg_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT feedback_id, username, thread_id, bot_type, user_role,
                           feedback_type, rating, query, original_answer,
                           corrected_answer, reason, context_text, created_at
                    FROM answer_feedback_vectors
                    ORDER BY created_at DESC
                    """
                )
                answer_rows = list(cur.fetchall())
            release_pg_conn(conn)
        except Exception as exc:
            logger.error("[LearningExport] Failed to load answer feedback rows: %s", exc, exc_info=True)

        try:
            route_rows = list(get_feedback_records())
        except Exception as exc:
            logger.error("[LearningExport] Failed to load routing feedback rows: %s", exc, exc_info=True)

        return answer_rows, len(answer_rows), len(route_rows)

    def build_dataset(self) -> List[Dict[str, object]]:
        answer_rows, _, _ = self.load_records()
        records: List[Dict[str, object]] = []
        seen: set[str] = set()

        for row in answer_rows:
            record = self._build_answer_example(dict(row))
            key = f"answer::{record['normalized_query']}::{_normalize_text(record['output'])}"
            if key not in seen:
                seen.add(key)
                records.append(record)

        try:
            route_rows = get_feedback_records()
        except Exception:
            route_rows = []

        for row in route_rows:
            record = self._build_route_example(dict(row))
            key = f"route::{record['normalized_query']}::{_normalize_text(record['output'])}"
            if key not in seen:
                seen.add(key)
                records.append(record)

        for record in self._load_retry_jsonl_rows():
            key = f"retry::{record['normalized_query']}::{_normalize_text(record['output'])}"
            if key not in seen:
                seen.add(key)
                records.append(record)

        return self._cap_per_question(records)

    @staticmethod
    def _cap_per_question(records: List[Dict[str, object]], max_per_question: int = _MAX_PER_QUESTION) -> List[Dict[str, object]]:
        """
        Keeps at most `max_per_question` examples per unique normalized
        question so a handful of over-tested questions don't dominate the
        fine-tune (#5). Keeps the most recent-looking / longest-answer
        examples first as a simple quality proxy when trimming.
        """
        by_query: Dict[str, List[Dict[str, object]]] = {}
        for record in records:
            by_query.setdefault(record.get("normalized_query", ""), []).append(record)

        capped: List[Dict[str, object]] = []
        for query, group in by_query.items():
            if not query or len(group) <= max_per_question:
                capped.extend(group)
                continue
            group.sort(key=lambda r: len(r.get("output", "")), reverse=True)
            capped.extend(group[:max_per_question])
        return capped

    def export(
        self,
        output_dir: str = "learning_exports",
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> ExportResult:
        records = self.build_dataset()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_file = output_path / "learning_dataset_all.jsonl"
        train_file = output_path / "learning_dataset_train.jsonl"
        val_file = output_path / "learning_dataset_val.jsonl"
        test_file = output_path / "learning_dataset_test.jsonl"

        _write_jsonl(all_file, records)
        train, val, test = _split_records(records, train_ratio, val_ratio, seed)
        _write_jsonl(train_file, train)
        _write_jsonl(val_file, val)
        _write_jsonl(test_file, test)

        logger.info(
            "[LearningExport] Exported %s records (%s answer, %s route) to %s",
            len(records),
            len([r for r in records if r.get("task_type") == "answer"]),
            len([r for r in records if r.get("task_type") == "route"]),
            output_path,
        )

        return ExportResult(
            output_dir=str(output_path),
            all_file=str(all_file),
            train_file=str(train_file),
            val_file=str(val_file),
            test_file=str(test_file),
            records_exported=len(records),
            answer_records=len([r for r in records if r.get("task_type") == "answer"]),
            routing_records=len([r for r in records if r.get("task_type") == "route"]),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exporter = LearningDatasetExporter()
    result = exporter.export()
    print(json.dumps(result.__dict__, indent=2))
