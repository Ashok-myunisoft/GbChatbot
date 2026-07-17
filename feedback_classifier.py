import json
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "and", "to",
    "list", "show", "give", "me", "please", "what", "how", "many", "there",
    "in", "on", "at", "with", "all", "get", "find",
}

_REJECTION_PHRASES = [
    "wrong", "incorrect", "not right", "not correct", "doesn't look right",
    "doesnt look right", "not what i", "bad answer", "that's not",
    "thats not", "no that", "not accurate", "are you sure", "seems off",
    "looks off", "that isn't right", "that isnt right", "this is wrong",
    "inaccurate", "mistaken", "that's incorrect", "thats incorrect",
    "still wrong", "still not right", "still incorrect", "redo", "re-search",
    "research this", "not it", "that's not it", "thats not it",
]


_BARE_REJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^\s*no\.?!?\s*$", r"^\s*nope\.?!?\s*$", r"^\s*nah\.?!?\s*$",
        r"^\s*na\.?!?\s*$", r"^\s*wrong\.?!?\s*$",
    )
]

_REFORMAT_PHRASES = [
    "just give me", "only give me", "give me only", "only show", "just show",
    "show as a table", "as a table", "in a table", "add the", "also give me",
    "along with", "just the", "only the", "with their", "include the",
    "show only", "give only", "without the", "don't need the", "dont need the",
]


def _has_any(text: str, phrases) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def _is_rejection(text: str) -> bool:
    if _has_any(text, _REJECTION_PHRASES):
        return True
    return any(p.match(text.strip()) for p in _BARE_REJECTION_PATTERNS)


def _content_words(text: str) -> set:
    words = {w for w in re.findall(r"\b\w{3,}\b", text.lower()) if w not in _STOP_WORDS}
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words}


def _word_overlap_ratio(new_message: str, prev_question: str) -> float:
    wa = _content_words(new_message)
    wb = _content_words(prev_question)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


_TOPIC_FILLER_WORDS = {
    "wrong", "incorrect", "asking", "meant", "mean", "not", "right",
    "correct", "accurate", "answer", "that's", "thats", "this", "isn't",
    "isnt", "about", "instead", "actually", "im", "want", "need",
}


def _extract_hint_from_text(text: str) -> Dict:
    hint = {"table": None, "column": None, "value": None, "condition": None, "topic": None}
    value_match = re.search(r"-?\d{3,}", text)
    if value_match:
        hint["value"] = value_match.group(0)
        hint["condition"] = "equals"
    col_match = re.search(r"\b([a-zA-Z_]+id|[a-zA-Z_]+code)\b", text, re.IGNORECASE)
    if col_match:
        hint["column"] = col_match.group(1)
    if not hint["value"] and not hint["column"]:
        topic_words = [w for w in _content_words(text) if w not in _TOPIC_FILLER_WORDS][:5]
        if topic_words:
            hint["topic"] = " ".join(topic_words)
    return hint


_KNOWN_COLUMN_CONCEPTS = [
    "code", "codes", "name", "names", "email", "emails", "id", "ids",
    "phone", "status", "date", "amount", "type", "description", "title",
]


def _extract_requested_columns(text: str) -> list:
    t = text.lower()
    for phrase in _REFORMAT_PHRASES:
        t = t.replace(phrase, " ")
    found = []
    for concept in _KNOWN_COLUMN_CONCEPTS:
        if re.search(rf"\b{concept}\b", t) and concept.rstrip("s") not in [f.rstrip("s") for f in found]:
            found.append(concept.rstrip("s"))
    if found:
        return found
    words = [w for w in re.findall(r"\b\w{3,}\b", t) if w not in _STOP_WORDS]
    return list(dict.fromkeys(words))[:8]


def _deterministic_classify(new_message: str, prev_question: str) -> Optional[Dict]:
    if _is_rejection(new_message):
        overlap = _word_overlap_ratio(new_message, prev_question)
        if not (len(_content_words(new_message)) >= 6 and overlap < 0.15):
            hint = _extract_hint_from_text(new_message)
            category = "WRONG_DATA_WITH_HINT" if (hint["value"] or hint["column"] or hint["topic"]) else "WRONG_DATA_GENERIC"
            return {"category": category, "raw_category": category, "hint": hint,
                    "requested_columns": [], "format_preference": "", "confidence": 1.0,
                    "decided_by": "deterministic_fallback:rejection"}
    if _has_any(new_message, _REFORMAT_PHRASES):
        return {"category": "REFORMAT", "raw_category": "REFORMAT",
                "hint": {"table": None, "column": None, "value": None, "condition": None, "topic": None},
                "requested_columns": _extract_requested_columns(new_message),
                "format_preference": "", "confidence": 1.0,
                "decided_by": "deterministic_fallback:reformat"}
    overlap = _word_overlap_ratio(new_message, prev_question)
    if len(_content_words(new_message)) >= 1 and overlap < 0.2:
        return {"category": "NEW_QUESTION", "raw_category": "NEW_QUESTION",
                "hint": {"table": None, "column": None, "value": None, "condition": None, "topic": None},
                "requested_columns": [], "format_preference": "", "confidence": 1.0,
                "decided_by": "deterministic_fallback:new_question"}
    return {"category": "WRONG_DATA_GENERIC", "raw_category": "AMBIGUOUS",
            "hint": {"table": None, "column": None, "value": None, "condition": None, "topic": None},
            "requested_columns": [], "format_preference": "", "confidence": 0.3,
            "decided_by": "deterministic_fallback:ambiguous_default"}


CLASSIFY_PROMPT = """You are reading a follow-up message from a user talking to a database assistant.

PREVIOUS QUESTION (what the user originally asked):
{prev_question}

PREVIOUS ANSWER (what the assistant replied, first 400 characters):
{prev_answer}

NEW MESSAGE (what the user just said):
{new_message}

Classify the NEW MESSAGE into EXACTLY ONE of these six categories:

1. WRONG_DATA_GENERIC - user says the answer was wrong, no specific detail given.
2. WRONG_DATA_WITH_HINT - user says the answer was wrong AND gives a specific clue
   (table name, column name, exact value/ID).
3. REFORMAT - the data was correct, user just wants it displayed differently
   (different/fewer/additional columns, a table instead of a list, etc). List
   the SPECIFIC column concepts they asked for in requested_columns (e.g. if
   they said "just the codes", requested_columns should be ["code"], not
   ["codes", "just", "the"] — one clean word per concept).
4. FOLLOWUP - user wants to extend/filter/drill into the existing result,
   without saying anything was wrong.
5. NEW_QUESTION - unrelated to the previous question and answer entirely.
   IMPORTANT: if the new message shares almost no topic/words with the
   previous question and contains no rejection language, it is almost
   certainly NEW_QUESTION, not a rejection of the old answer.
6. AMBIGUOUS - genuinely cannot be confidently placed in any category above.

PRECEDENCE RULE: any clear rejection signal (wrong, no, incorrect, not right,
doesn't look right, etc.) forces WRONG_DATA_GENERIC or WRONG_DATA_WITH_HINT,
even alongside formatting/follow-up language in the same message.

Respond with ONLY this JSON object, no other text:
{{
  "category": "WRONG_DATA_GENERIC" | "WRONG_DATA_WITH_HINT" | "REFORMAT" |
              "FOLLOWUP" | "NEW_QUESTION" | "AMBIGUOUS",
  "hint": {{"table": "<string or null>", "column": "<string or null>",
            "value": "<string or null>", "condition": "<string or null>",
            "topic": "<short phrase for what the user actually wants,
                       e.g. 'sales module' or 'GST formula', or null —
                       this is a knowledge-base search, not a SQL table,
                       so prefer topic over table/column when unsure>"}},
  "requested_columns": ["<single-concept column terms, only if REFORMAT>"],
  "format_preference": "<only if mentioned>",
  "confidence": <float 0.0 to 1.0>
}}
"""

_VALID_CATEGORIES = {
    "WRONG_DATA_GENERIC", "WRONG_DATA_WITH_HINT", "REFORMAT",
    "FOLLOWUP", "NEW_QUESTION", "AMBIGUOUS",
}
_VALID_CONDITIONS = {"equals", "contains", "greater_than", "less_than", "not_equals"}


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


async def classify_feedback(llm, new_message: str, prev_question: str, prev_answer: str) -> Dict:
    if not prev_question or not prev_answer:
        return {"category": "NEW_QUESTION", "raw_category": "NEW_QUESTION",
                "hint": {"table": None, "column": None, "value": None, "condition": None, "topic": None},
                "requested_columns": [], "format_preference": "", "confidence": 0.0,
                "decided_by": "no_prior_turn"}

    try:
        prompt = CLASSIFY_PROMPT.format(
            prev_question=prev_question[:300],
            prev_answer=(prev_answer or "")[:400],
            new_message=new_message[:300],
        )
        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _extract_json(raw)
        if not parsed:
            logger.warning("[FeedbackClassifier] LLM returned unparseable output, using deterministic fallback")
            result = _deterministic_classify(new_message, prev_question)
            logger.info(f"[FeedbackClassifier] '{new_message[:60]}' -> {result}")
            return result

        raw_category = str(parsed.get("category", "")).strip().upper()
        if raw_category not in _VALID_CATEGORIES:
            raw_category = "AMBIGUOUS"
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        category = "WRONG_DATA_GENERIC" if raw_category == "AMBIGUOUS" else raw_category

        hint_raw = parsed.get("hint") or {}
        hint = {
            "table": hint_raw.get("table") or None,
            "column": hint_raw.get("column") or None,
            "value": hint_raw.get("value") or None,
            "condition": hint_raw.get("condition") if hint_raw.get("condition") in _VALID_CONDITIONS else None,
            "topic": (str(hint_raw.get("topic")).strip() if hint_raw.get("topic") else None),
        }
        cols = parsed.get("requested_columns") or []
        if not isinstance(cols, list):
            cols = []
        norm_cols = []
        for c in [str(c).lower().strip() for c in cols]:
            c2 = c.rstrip("s") if c.rstrip("s") else c
            if c2 and c2 not in norm_cols:
                norm_cols.append(c2)

        has_rejection = _is_rejection(new_message)
        overlap = _word_overlap_ratio(new_message, prev_question)
        _msg_word_count = len(_content_words(new_message))
        _looks_like_new_question = _msg_word_count >= 6 and overlap < 0.15

        if has_rejection and not _looks_like_new_question and category not in ("WRONG_DATA_GENERIC", "WRONG_DATA_WITH_HINT"):
            logger.info(f"[FeedbackClassifier] Precedence override: LLM said {category}, but rejection word present -> forcing WRONG_DATA")
            extra_hint = _extract_hint_from_text(new_message)
            hint = {k: (hint.get(k) or extra_hint.get(k)) for k in hint}
            category = "WRONG_DATA_WITH_HINT" if (hint["value"] or hint["column"] or hint["topic"]) else "WRONG_DATA_GENERIC"

        if category in ("WRONG_DATA_GENERIC", "WRONG_DATA_WITH_HINT"):
            if _looks_like_new_question or (not has_rejection and _msg_word_count >= 1 and overlap < 0.15):
                logger.info(f"[FeedbackClassifier] Stuck-loop guard: LLM said {category} but overlap={overlap:.2f}, words={_msg_word_count} -> forcing NEW_QUESTION")
                category = "NEW_QUESTION"

        result = {
            "category": category, "raw_category": raw_category, "hint": hint,
            "requested_columns": _extract_requested_columns(new_message) if category == "REFORMAT" else [],
            "format_preference": str(parsed.get("format_preference", "") or "").strip(),
            "confidence": confidence, "decided_by": "llm",
        }
        logger.info(f"[FeedbackClassifier] '{new_message[:60]}' -> {result}")
        return result

    except Exception as exc:
        logger.warning(f"[FeedbackClassifier] LLM call failed ({exc}), using deterministic fallback")
        result = _deterministic_classify(new_message, prev_question)
        logger.info(f"[FeedbackClassifier] '{new_message[:60]}' -> {result}")
        return result