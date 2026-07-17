import json
import os
import re
import time
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_RETRY_PHRASES = [
    r"^\s*wrong\s*[.!]?\s*$",
    r"^\s*(that'?s|this is|its|it'?s)\s+wrong\s*[.!]?\s*$",
    r"^\s*(not\s+correct|incorrect)\s*[.!]?\s*$",
    r"^\s*try\s+again\s*[.!]?\s*$",
    r"^\s*retry\s*[.!]?\s*$",
    r"^\s*(try\s+)?once\s+more\s*[.!]?\s*$",
    r"^\s*no+p*e?\s*[.!]?\s*$",
    r"^\s*that'?s\s+not\s+right\s*[.!]?\s*$",
    r"^\s*(that|this)\s+is\s+not\s+right\s*[.!]?\s*$",
    r"^\s*not\s+right\s*[.!]?\s*$",
    r"^\s*wrong\s+answer\s*[.!]?\s*$",
    r"^\s*bad\s+answer\s*[.!]?\s*$",
    r"^\s*this\s+is\s+wrong\s*[.!]?\s*$",
    r"^\s*deep\s*search\s*[.!]?\s*$",
    r"^\s*search\s+again\s*[.!]?\s*$",
    r"^\s*no\s*[.!]?\s*$",
    r"^\s*nope\s*[.!]?\s*$",
    r"^\s*nah\s*[.!]?\s*$",
    r"^\s*na\s*[.!]?\s*$",
    r"^\s*not\s+this\s*[.!]?\s*$",
    r"^\s*that'?s\s+not\s+it\s*[.!]?\s*$",
    r"^\s*still\s+wrong\s*[.!]?\s*$",
    r"^\s*still\s+not\s+right\s*[.!]?\s*$",
    r"^\s*(this|that)\s+isn'?t\s+it\s*[.!]?\s*$",
    r"^\s*redo\s*[.!]?\s*$",
    r"^\s*re-?search\s*[.!]?\s*$",
]
_RETRY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _RETRY_PHRASES]

_NON_RETRYABLE_BOT_TYPES = {
    "greeting", "role_setup", "role_prompt", "out_of_scope",
    "clarification", "error", "", "feedback_override", "retry_deep_search",
}


def is_retry_message(text: str) -> bool:
    """True only for short, generic 'this was wrong, try again' signals."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped.split()) > 6:
        return False
    return any(p.match(stripped) for p in _RETRY_PATTERNS)


def can_retry(last_bot_type: str) -> bool:
    return bool(last_bot_type) and last_bot_type not in _NON_RETRYABLE_BOT_TYPES


LEARNING_DIR = "learning_exports"
LEARNING_FILE = os.path.join(LEARNING_DIR, "retry_feedback_dataset.jsonl")

_MIN_LENGTH_RATIO = 0.5
_PROPER_ENDINGS = (".", "!", "?", '"', "'", "`", ")", "]", "*")


def _normalize_for_compare(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"```[a-zA-Z]*\n?", "", t)   # code fences (with/without lang tag)
    t = re.sub(r"[*_`#>-]+", "", t)          # markdown emphasis/heading/bullet chars
    t = re.sub(r"\s+", " ", t)               # collapse all whitespace
    return t.strip().lower()


def _word_overlap_ratio(a: str, b: str) -> float:
    wa = set(re.findall(r"\b\w{3,}\b", (a or "").lower()))
    wb = set(re.findall(r"\b\w{3,}\b", (b or "").lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _looks_truncated(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return True
    return not t.endswith(_PROPER_ENDINGS)


def is_meaningful_correction(old_answer: str, new_answer: str) -> tuple:
    if not new_answer or not new_answer.strip():
        return False, "empty_answer"

    old_norm = _normalize_for_compare(old_answer)
    new_norm = _normalize_for_compare(new_answer)

    if new_norm == old_norm:
        return False, "identical_after_normalization"

    overlap = _word_overlap_ratio(old_answer, new_answer)
    if overlap >= 0.92 and len(new_norm) <= len(old_norm) * 1.1:
        # Near-duplicate content, no meaningful new information added.
        return False, f"near_duplicate(overlap={overlap:.2f})"

    if _looks_truncated(new_answer):
        return False, "new_answer_truncated"

    if old_answer and len(new_answer.strip()) < _MIN_LENGTH_RATIO * len(old_answer.strip()):
        return False, "new_answer_shorter_than_old"

    return True, "ok"


def save_learning_example(
    question: str,
    answer: str,
    bot_type: str,
    label: str,          # "negative" (rejected) | "positive" (accepted) | "still_wrong"
    username: str = "",
    extra: Optional[dict] = None,
    thread_id: str = "",
    pair_id: str = "",
) -> None:
    try:
        os.makedirs(LEARNING_DIR, exist_ok=True)
        record = {
            "instruction": question,
            "response": answer,
            "bot_type": bot_type,
            "label": label,
            "username": username,
            "thread_id": thread_id,
            "pair_id": pair_id,
            "timestamp": time.time(),
        }
        if extra:
            record.update(extra)
        with open(LEARNING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[RetryHandler] Logged {label} example for training: {question[:60]}")
    except Exception as exc:
        logger.warning(f"[RetryHandler] Failed to write learning example: {exc}")


def new_pair_id() -> str:
    return uuid.uuid4().hex[:12]


# Hard cap: at most this many CONFIRMED pairs are kept for the same
# normalized question, no matter how many times it gets tested/retried.
_MAX_PAIRS_PER_QUESTION = 2


def _existing_pairs_for_question(question_norm: str) -> list:
    if not os.path.exists(LEARNING_FILE):
        return []
    out = []
    try:
        with open(LEARNING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("label") != "positive":
                    continue
                if _normalize_for_compare(rec.get("instruction", "")) == question_norm:
                    out.append(rec)
    except Exception as exc:
        logger.warning(f"[RetryHandler] Failed to scan existing pairs: {exc}")
    return out


def save_correction_pair(
    question: str,
    wrong_answer: str,
    corrected_answer: str,
    bot_type: str,
    username: str = "",
    thread_id: str = "",
    extra: Optional[dict] = None,
) -> tuple:
    ok, reason = is_meaningful_correction(wrong_answer, corrected_answer)
    if not ok:
        logger.info(f"[RetryHandler] Correction pair discarded ({reason}): '{question[:60]}'")
        return False, reason

    q_norm = _normalize_for_compare(question)
    existing = _existing_pairs_for_question(q_norm)

    if len(existing) >= _MAX_PAIRS_PER_QUESTION:
        return False, f"question_cap_reached({len(existing)})"

    new_norm = _normalize_for_compare(corrected_answer)
    for rec in existing:
        if _word_overlap_ratio(rec.get("response", ""), corrected_answer) >= 0.9:
            return False, "duplicate_of_existing_pair"

    # In case this is a re-ask of a question that was already confirmed
    # good once in this same thread and is now being rejected again (#8).
    mark_last_positive_as_still_wrong(thread_id, question)

    pair_id = new_pair_id()
    save_learning_example(question, wrong_answer, bot_type, "negative", username, extra, thread_id, pair_id)
    save_learning_example(question, corrected_answer, bot_type, "positive", username, extra, thread_id, pair_id)
    return True, "ok"


def mark_last_positive_as_still_wrong(thread_id: str, question: str) -> bool:
    if not thread_id or not question or not os.path.exists(LEARNING_FILE):
        return False
    try:
        q_norm = _normalize_for_compare(question)
        with open(LEARNING_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        target_idx = None
        for i in range(len(lines) - 1, -1, -1):
            try:
                rec = json.loads(lines[i])
            except Exception:
                continue
            if rec.get("thread_id") != thread_id:
                continue
            if _normalize_for_compare(rec.get("instruction", "")) != q_norm:
                continue
            if rec.get("label") == "positive":
                target_idx = i
            break  # stop at the most recent record for this (thread, question)

        if target_idx is None:
            return False

        rec = json.loads(lines[target_idx])
        rec["label"] = "still_wrong"
        rec["relabeled_at"] = time.time()
        lines[target_idx] = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(LEARNING_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"[RetryHandler] Downgraded prior positive to still_wrong: {question[:60]}")
        return True
    except Exception as exc:
        logger.warning(f"[RetryHandler] Failed to relabel still_wrong: {exc}")
        return False


async def reformat_answer(llm, cached_answer: str, requested_columns: list, format_preference: str = "") -> Optional[str]:
    if not cached_answer or not cached_answer.strip():
        return None

    ask = []
    if requested_columns:
        ask.append(f"Show only: {', '.join(requested_columns)}")
    if format_preference:
        ask.append(f"Format: {format_preference}")
    if not ask:
        ask.append("Reformat this more clearly (e.g. as a table if it's a list)")

    prompt = (
        "The user was happy with the DATA in this answer but wants it "
        "presented differently. Do not add, remove, or invent any facts — "
        "only reorganize/reformat what is already here.\n\n"
        f"Original answer:\n{cached_answer[:2000]}\n\n"
        f"Requested change: {'; '.join(ask)}\n\n"
        "Reformatted answer:"
    )
    try:
        response = await llm.ainvoke(prompt)
        result = (response.content if hasattr(response, "content") else str(response)).strip()
        return result if result else None
    except Exception as exc:
        logger.warning(f"[RetryHandler] reformat_answer LLM call failed: {exc}")
        return None