"""
formatter_agent.py  (v2 — Question-Aware)

Response Formatter Agent — intent-first, then structure-based formatting.
Converts a raw string response into structured, UI-friendly JSON.

KEY UPGRADE over v1:
  - Reads the USER'S QUESTION first to decide WHAT to show
  - Only surfaces fields/data the user actually asked for
  - Detects intent (single value, list, details, count, etc.)
  - Falls back to structure-based detection if intent is ambiguous

Zero impact on existing system:
  - Does NOT modify the "response" string field
  - Output is added as a new "formatted" field alongside existing response
  - If formatting fails for any reason, returns a safe fallback — never crashes

Output schema:
  {
    "type":         one of: text | bullet_list | numbered_steps | table | code_block | summary | comparison | single_value
    "title":        short title derived from the user's question
    "content":      structured content (format depends on type)
    "ui_component": recommended frontend component
    "priority":     low | medium | high
    "confidence":   0.0 to 1.0
    "intent":       detected user intent label (for debugging/logging)
  }
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── UI component mapping ──────────────────────────────────────────────────────
_UI_MAP = {
    "single_value":   "inline_text",      # "The module name is X."
    "text":           "paragraph",
    "bullet_list":    "list",
    "numbered_steps": "stepper",
    "table":          "table",
    "code_block":     "code_view",
    "summary":        "card",
    "comparison":     "comparison_table",
}


# ── Intent detection (question-aware) ─────────────────────────────────────────

# Maps intent label → (pattern list, format type)
_INTENT_PATTERNS = [
    # List requests — user wants multiple items
    # These are checked first so queries like "what menu names are in sales module"
    # are treated as lists instead of a single-value lookup.
    ("list_names",     [
        r"\bwhat are\b.*\bnames\b",
        r"\bwhat names\b",
        r"\bnames of\b",
        r"\bmenu names\b",
        r"\bmodule names\b",
        r"\bmenu(s)?\s+you\s+have\b",
        r"\bgive a name of that\b",
        r"\blist\b.*\bname",
        r"\ball.*names",
        r"\bshow.*names",
        r"\bshow.*modules",
        r"\bshow.*courses",
        r"\bshow.*products",
        r"\bwhat.*menus\b",
        r"\bwhat.*menu\b",
        r"\bmenus?\s+in\b",
        r"\bnames?\s+in\b",
        r"\bmodules?\s+have\b",
    ],  "bullet_list"),
    ("list_all",       [r"\blist all\b", r"\bshow all\b", r"\bget all\b", r"\bfetch all\b"],                                    "bullet_list"),

    # Single-value lookups — user wants ONE specific field
    ("single_name",    [r"\bwhat is the name\b", r"\bgive (me )?(the )?(module|course|product|menu|customer|employee)?\s*name\b", r"\bmodule name\b", r"\bcourse name\b", r"\bproduct name\b", r"\bcustomer name\b", r"\bemployee name\b"],  "single_value"),
    ("single_id",      [r"\bid\b", r"\bwhat is the id\b", r"\bmodule id\b", r"\bgive.*id\b", r"\bfetch.*id\b"],                                  "single_value"),
    ("single_status",  [r"\bstatus\b", r"\bis it active\b", r"\bcurrent status\b"],                                                      "single_value"),
    ("single_count",   [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bnumber of\b"],                                          "single_value"),
    ("single_price",   [r"\bprice\b", r"\bcost\b", r"\brate\b", r"\bhow much\b"],                                                "single_value"),
    ("single_date",    [r"\bdate\b", r"\bwhen\b", r"\bcreated at\b", r"\bdeadline\b"],                                           "single_value"),
    ("single_email",   [r"\bemail\b", r"\bcontact\b"],                                                                           "single_value"),

    # Detail requests — user wants full info
    ("full_details",   [r"\bdetails?\b", r"\bfull info\b", r"\btell me (about|more)", r"\bshow (me )?everything", r"\ball (fields|info|data)\b"], "table"),

    # Steps / how-to
    ("steps",          [r"\bhow to\b", r"\bsteps?\b", r"\bprocess\b", r"\bsetup\b", r"\bconfigure\b", r"\binstall\b"],          "numbered_steps"),

    # Comparison
    ("compare",        [r"\bvs\b", r"\bversus\b", r"\bcompare\b", r"\bdifference\b", r"\bpros.*cons\b"],                        "comparison"),

    # Summary / explain
    ("explain",        [r"\bexplain\b", r"\bsummarise\b", r"\bsummarize\b", r"\boverview\b", r"\bbrief\b", r"\babout\b"],       "summary"),
]


def _detect_intent(question: str) -> tuple[str, str]:
    """
    Analyze the user's question and return (intent_label, recommended_format_type).
    Returns ("unknown", None) if no clear intent is detected.
    """
    q = question.lower().strip()
    for intent_label, patterns, fmt_type in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q):
                return intent_label, fmt_type
    return "unknown", None


# ── Priority based on USER question, not raw data ─────────────────────────────
def _detect_priority(question: str, raw: str) -> str:
    q = question.lower()
    r = raw.lower()
    # High: errors or urgent actions
    if any(w in r for w in ["error", "failed", "critical", "urgent", "alert"]):
        return "high"
    # Medium: user is doing something (not just reading)
    if any(w in q for w in ["create", "update", "delete", "setup", "configure", "how to", "steps"]):
        return "medium"
    return "low"


# ── Title from user question (not raw data) ───────────────────────────────────
def _extract_title(question: str, fmt_type: str) -> str:
    """Generate a clean title from the user's question."""
    q = question.strip()
    # Remove question words at start
    q = re.sub(r"^(what is|give me|show me|list|get|fetch|find|tell me)\s+", "", q, flags=re.IGNORECASE)
    q = q.strip().rstrip("?").strip()
    # Capitalize first letter
    if q:
        q = q[0].upper() + q[1:]
    if len(q) > 60:
        q = q[:57] + "..."
    return q if q else fmt_type.replace("_", " ").title()


# ── Structure-based classifiers (fallback when intent is unknown) ─────────────
def _detect_code(text: str) -> bool:
    return bool(re.search(r"```[\s\S]*?```|^\s{4}\S", text, re.MULTILINE))

def _detect_table(text: str) -> bool:
    return bool(re.search(r"\|.+\|.+\|", text))

def _detect_numbered_steps(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    numbered = sum(1 for l in lines if re.match(r"^\d+[\.\)]\s+\S", l))
    return numbered >= 2

def _detect_bullet_list(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullets = sum(1 for l in lines if re.match(r"^[-•*🔹]\s+\S", l))
    return bullets >= 2

def _detect_comparison(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["vs ", "versus", "compared to", "difference between", "pros and cons"])

def _detect_summary(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    word_count = len(text.split())
    return word_count > 80 and len(lines) <= 5

def _classify_by_structure(text: str) -> str:
    """Fallback: classify purely by raw text structure."""
    if _detect_code(text):           return "code_block"
    if _detect_table(text):          return "table"
    if _detect_comparison(text):     return "comparison"
    if _detect_numbered_steps(text): return "numbered_steps"
    if _detect_bullet_list(text):    return "bullet_list"
    if _detect_summary(text):        return "summary"
    return "text"


# ── Content parsers ───────────────────────────────────────────────────────────

def _parse_single_value(text: str, intent: str) -> dict:
    """
    Extract just the one value the user asked for.
    Returns {"label": ..., "value": ...}
    """
    clean = re.sub(r"\*+", "", text).strip()
    clean = re.sub(r"#{1,6}\s*", "", clean)

    # Try to find key:value pairs
    kv_patterns = {
        "single_name":   [r"name\s*[:\-]\s*(.+)", r"^(.+)$"],
        "single_id":     [r"id\s*[:\-]\s*(.+)", r"\b([A-Z0-9\-]{3,})\b"],
        "single_status": [r"status\s*[:\-]\s*(.+)", r"\b(active|inactive|pending|completed|draft)\b"],
        "single_count":  [r"(\d[\d,]*)\s*(users?|modules?|courses?|products?|items?)?"],
        "single_price":  [r"[\$₹€£]?\s*(\d[\d,.]*)\s*(USD|INR|EUR)?"],
        "single_date":   [r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})"],
        "single_email":  [r"([\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,})"],
    }

    patterns = kv_patterns.get(intent, [r"^(.+)$"])
    for pattern in patterns:
        m = re.search(pattern, clean, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0).strip()
            label = intent.replace("single_", "").replace("_", " ").title()
            return {"label": label, "value": value}

    # Fallback: first non-empty line
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    return {"label": "Result", "value": lines[0] if lines else clean}


def _parse_code(text: str) -> dict:
    m = re.search(r"```(\w*)\n?([\s\S]*?)```", text)
    if m:
        return {"language": m.group(1) or "text", "code": m.group(2).strip()}
    lines = [l[4:] if l.startswith("    ") else l for l in text.split("\n")]
    return {"language": "text", "code": "\n".join(lines).strip()}

def _parse_table(text: str) -> dict:
    rows = []
    headers = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.match(r"^\|[-| ]+\|$", line):
            continue
        if "|" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [re.sub(r"\*+", "", c).strip() for c in cells]
            if not headers:
                headers = cells
            else:
                rows.append(cells)
    return {"headers": headers, "rows": rows}

def _parse_numbered_steps(text: str) -> list:
    steps = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s+(.*)", line)
        if m:
            steps.append(re.sub(r"\*+", "", m.group(1)).strip())
    return steps

def _parse_bullet_list(text: str) -> list:
    items = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^[-•*🔹]\s+(.*)", line)
        if m:
            items.append(re.sub(r"\*+", "", m.group(1)).strip())
        elif re.match(r"^\*\*(.+)\*\*", line):
            items.append(re.sub(r"\*+", "", line).strip())
    return items if items else [l.strip() for l in text.split("\n") if l.strip()]

def _parse_summary(text: str) -> list:
    """Extract key sentences — skip fluff."""
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#{1,6}\s*", "", clean)
    sentences = re.split(r"(?<=[.!?])\s+", clean.strip())
    # Filter out very short or filler sentences
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 15]
    return meaningful[:6]

def _parse_comparison(text: str) -> list:
    items = []
    blocks = re.split(r"\n{2,}", text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        name = re.sub(r"[#*\-•]+", "", lines[0]).strip()
        pros, cons, neutral = [], [], []
        for l in lines[1:]:
            l_clean = re.sub(r"^[-•*]\s*", "", l).strip()
            if any(w in l.lower() for w in ["advantage", "pro", "benefit", "positive", "good"]):
                pros.append(l_clean)
            elif any(w in l.lower() for w in ["disadvantage", "con", "limitation", "negative", "bad"]):
                cons.append(l_clean)
            else:
                neutral.append(l_clean)
        items.append({"name": name, "pros": pros, "cons": cons, "notes": neutral})
    return items if items else [{"name": "Comparison", "pros": [], "cons": [], "notes": []}]

def _parse_text(text: str) -> str:
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#{1,6}\s*", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence(fmt_type: str, text: str, intent: str) -> float:
    # If intent was clearly detected, higher base confidence
    if intent != "unknown":
        if fmt_type == "single_value": return 0.95
        return 0.88

    # Fallback structure-based confidence
    if fmt_type == "code_block":
        return 0.97 if "```" in text else 0.75
    if fmt_type == "table":
        pipe_lines = sum(1 for l in text.split("\n") if "|" in l)
        return min(0.95, 0.6 + pipe_lines * 0.05)
    if fmt_type == "numbered_steps":
        count = sum(1 for l in text.split("\n") if re.match(r"^\d+[\.\)]", l.strip()))
        return min(0.95, 0.6 + count * 0.05)
    if fmt_type == "bullet_list":
        count = sum(1 for l in text.split("\n") if re.match(r"^[-•*🔹]", l.strip()))
        return min(0.93, 0.6 + count * 0.04)
    if fmt_type == "comparison": return 0.80
    if fmt_type == "summary":    return 0.75
    return 0.70


# ── Main entry point ──────────────────────────────────────────────────────────

def format(question: str, raw: str) -> dict:
    """
    Convert raw response string into structured JSON for UI rendering.

    Args:
        question : original user question — used to decide WHAT to show
        raw      : raw string response from the backend

    Returns:
        Structured dict — always valid, never raises.
    """
    try:
        if not raw or not raw.strip():
            return _fallback("No response available.")

        text = raw.strip()
        question = (question or "").strip()

        # ── Step 1: Detect intent from user's question ────────────────────────
        intent, intent_fmt = _detect_intent(question)

        # ── Step 2: Decide final format type ─────────────────────────────────
        if intent_fmt:
            # Question was clear → trust the intent
            fmt_type = intent_fmt
        else:
            # Question was vague → fall back to structure detection
            fmt_type = _classify_by_structure(text)

        # ── Step 3: Parse content ─────────────────────────────────────────────
        if fmt_type == "single_value":
            content: Any = _parse_single_value(text, intent)
        elif fmt_type == "code_block":
            content = _parse_code(text)
        elif fmt_type == "table":
            content = _parse_table(text)
        elif fmt_type == "numbered_steps":
            content = _parse_numbered_steps(text)
        elif fmt_type == "bullet_list":
            content = _parse_bullet_list(text)
        elif fmt_type == "summary":
            content = _parse_summary(text)
        elif fmt_type == "comparison":
            content = _parse_comparison(text)
        else:
            content = _parse_text(text)

        return {
            "type":         fmt_type,
            "title":        _extract_title(question, fmt_type),
            "content":      content,
            "ui_component": _UI_MAP.get(fmt_type, "paragraph"),
            "priority":     _detect_priority(question, text),
            "confidence":   _confidence(fmt_type, text, intent),
            "intent":       intent,   # useful for debugging / analytics
        }

    except Exception as e:
        logger.warning(f"[FormatterAgent] format() failed: {e} — returning fallback")
        return _fallback(raw)


def _fallback(raw: str) -> dict:
    """Safe fallback — always returns valid structured output."""
    return {
        "type":         "text",
        "title":        "Response",
        "content":      raw[:500] if raw else "",
        "ui_component": "paragraph",
        "priority":     "low",
        "confidence":   0.5,
        "intent":       "unknown",
    }
