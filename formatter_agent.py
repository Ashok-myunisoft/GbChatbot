"""
formatter_agent.py

Response Formatter Agent — rule-based + LLM-hybrid.
Converts a raw string response into structured, UI-friendly JSON.

Zero impact on existing system:
  - Does NOT modify the "response" string field
  - Output is added as a new "formatted" field alongside existing response
  - If formatting fails for any reason, returns a safe fallback — never crashes

Output schema:
  {
    "type":         one of: text | bullet_list | numbered_steps | table | code_block | summary | comparison
    "title":        short title derived from content
    "content":      structured content (format depends on type)
    "ui_component": recommended frontend component
    "priority":     low | medium | high
    "confidence":   0.0 to 1.0
  }
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── UI component mapping ──────────────────────────────────────────────────────
_UI_MAP = {
    "text":           "paragraph",
    "bullet_list":    "list",
    "numbered_steps": "stepper",
    "table":          "table",
    "code_block":     "code_view",
    "summary":        "list",
    "comparison":     "comparison_table",
}

# ── Priority rules ────────────────────────────────────────────────────────────
def _detect_priority(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["error", "critical", "urgent", "failed", "warning", "alert"]):
        return "high"
    if any(w in t for w in ["step", "process", "configure", "setup", "install", "how to"]):
        return "medium"
    return "low"


# ── Title extractor ───────────────────────────────────────────────────────────
def _extract_title(text: str, fmt_type: str) -> str:
    """Extract a short title from the first meaningful line."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return fmt_type.replace("_", " ").title()
    first = re.sub(r"^[#*•\-\d.\s]+", "", lines[0]).strip()
    # Trim to max 60 chars
    if len(first) > 60:
        first = first[:57] + "..."
    return first if first else fmt_type.replace("_", " ").title()


# ── Format detectors ──────────────────────────────────────────────────────────

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
    return any(p in t for p in [
        "vs ", "versus", "compared to", "difference between",
        "pros and cons", "advantages", "disadvantages", "on the other hand"
    ])

def _detect_summary(text: str) -> bool:
    """Long prose with no clear list structure → summarize."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    word_count = len(text.split())
    return word_count > 80 and len(lines) <= 5

def _classify(text: str) -> str:
    """Classify the response into one of the supported types."""
    if _detect_code(text):        return "code_block"
    if _detect_table(text):       return "table"
    if _detect_comparison(text):  return "comparison"
    if _detect_numbered_steps(text): return "numbered_steps"
    if _detect_bullet_list(text): return "bullet_list"
    if _detect_summary(text):     return "summary"
    return "text"


# ── Content parsers ───────────────────────────────────────────────────────────

def _parse_code(text: str) -> dict:
    m = re.search(r"```(\w*)\n?([\s\S]*?)```", text)
    if m:
        return {"language": m.group(1) or "text", "code": m.group(2).strip()}
    # Indented code block
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
            # Bold lines treated as bullet items
            items.append(re.sub(r"\*+", "", line).strip())
    return items if items else [l.strip() for l in text.split("\n") if l.strip()]

def _parse_summary(text: str) -> list:
    """Break long prose into sentence-level bullet points."""
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#{1,6}\s*", "", clean)
    sentences = re.split(r"(?<=[.!?])\s+", clean.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10][:8]

def _parse_comparison(text: str) -> list:
    """Best-effort: extract named items with simple pros/cons."""
    items = []
    blocks = re.split(r"\n{2,}", text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        name = re.sub(r"[#*\-•]+", "", lines[0]).strip()
        pros, cons = [], []
        for l in lines[1:]:
            l_clean = re.sub(r"^[-•*]\s*", "", l).strip()
            if any(w in l.lower() for w in ["advantage", "pro", "benefit", "positive", "good"]):
                pros.append(l_clean)
            elif any(w in l.lower() for w in ["disadvantage", "con", "limitation", "negative", "bad"]):
                cons.append(l_clean)
        items.append({"name": name, "pros": pros, "cons": cons})
    return items if items else [{"name": "Comparison", "pros": [], "cons": []}]

def _parse_text(text: str) -> str:
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#{1,6}\s*", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence(fmt_type: str, text: str) -> float:
    """Simple confidence based on how strongly the type signals are present."""
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
    if fmt_type == "comparison":
        return 0.80
    if fmt_type == "summary":
        return 0.75
    return 0.70


# ── Main entry point ──────────────────────────────────────────────────────────

def format(question: str, raw: str) -> dict:
    """
    Convert raw response string into structured JSON for UI rendering.

    Args:
        question : original user question (used for title + priority hints)
        raw      : formatted string response (output of _fmt_response)

    Returns:
        Structured dict — always valid, never raises.
    """
    try:
        if not raw or not raw.strip():
            return _fallback("No response available.")

        text = raw.strip()
        fmt_type = _classify(text)

        # ── Parse content based on type ───────────────────────────────────────
        if fmt_type == "code_block":
            content: Any = _parse_code(text)
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
            "title":        _extract_title(text, fmt_type),
            "content":      content,
            "ui_component": _UI_MAP.get(fmt_type, "paragraph"),
            "priority":     _detect_priority(text),
            "confidence":   _confidence(fmt_type, text),
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
    }
