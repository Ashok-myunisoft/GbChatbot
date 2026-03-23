"""
response_formatter.py

Pure-Python Fast Response Formatter.
Converts raw bot output into clean, ChatGPT-style markdown — zero latency, no LLM call.

Format detection (automatic):
  1. Record blocks  (--- Record N ---)  → markdown table or bullet cards
  2. Numbered list  (1. value, 2. ...)  → clean bullet list
  3. Prose / LLM answer                 → whitespace cleanup only

"table" in question → always forces markdown table output.
"""

import re
from typing import Dict, List, Tuple


# ── Field emoji map — auto-decorates common field names ──────────────────────
_FIELD_EMOJI = {
    # identity
    "name": "👤", "username": "👤", "fullname": "👤", "employeename": "👤",
    "title": "📌", "description": "📝", "remarks": "📝", "notes": "📝",
    # date / time
    "date": "📅", "createdat": "📅", "updatedat": "📅", "modifiedat": "📅",
    "startdate": "📅", "enddate": "📅", "duedate": "📅", "posteddate": "📅",
    # money
    "amount": "💰", "price": "💰", "cost": "💰", "total": "💰",
    "value": "💰", "rate": "💰", "tax": "💰", "discount": "💰",
    # status / flag
    "status": "🔹", "state": "🔹", "active": "🔹", "isactive": "🔹",
    "flag": "🔹", "type": "🏷️", "category": "🏷️", "mode": "🏷️",
    # id / code
    "id": "🔑", "code": "🔑", "refno": "🔑", "referenceno": "🔑",
    "invoiceno": "🔑", "orderno": "🔑", "voucherno": "🔑",
    # contact
    "email": "📧", "phone": "📞", "mobile": "📞", "contact": "📞",
    "address": "📍", "city": "📍", "country": "📍",
    # org
    "department": "🏢", "branch": "🏢", "company": "🏢", "division": "🏢",
    "role": "🎭", "designation": "🎭", "position": "🎭",
    # quantity
    "quantity": "📦", "qty": "📦", "stock": "📦", "units": "📦",
}

def _field_emoji(field_name: str) -> str:
    """Return emoji for a field name, or empty string if not mapped."""
    key = field_name.lower().replace(" ", "").replace("_", "")
    # exact match
    if key in _FIELD_EMOJI:
        return _FIELD_EMOJI[key]
    # partial match — check if any keyword appears in the field name
    for kw, emoji in _FIELD_EMOJI.items():
        if kw in key:
            return emoji
    return ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip and collapse excess blank lines."""
    text = text.strip()
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text


def _split_results_header(text: str) -> Tuple[str, str]:
    """
    Separate the 'Results for ... (N rows):' prefix from the body.
    Returns (header_line, body).  header is "" when not present.
    """
    m = re.match(r"^(Results for '([^']*)' \((\d+) rows\):)\s*\n+(.*)", text, re.DOTALL)
    if m:
        query_text = m.group(2)
        row_count  = m.group(3)
        header = f"📊 **Results for '{query_text}'** — {row_count} record(s) found"
        return header, m.group(4).strip()
    return "", text


# ── Format detectors ──────────────────────────────────────────────────────────

def _has_record_blocks(text: str) -> bool:
    return bool(re.search(r'--- Record \d+ ---', text))


def _is_numbered_list(text: str) -> bool:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return False
    numbered = sum(1 for l in lines if re.match(r'^\d+\.\s+\S', l))
    return numbered >= max(2, len(lines) * 0.7)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_record_blocks(text: str) -> List[Dict[str, str]]:
    """Parse '--- Record N ---\\n  col: val' blocks into list of dicts."""
    records = []
    blocks = re.split(r'--- Record \d+ ---', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        record = {}
        for line in block.split('\n'):
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                if key:
                    record[key] = val
        if record:
            records.append(record)
    return records


def _is_tabular(records: List[Dict]) -> bool:
    """True if records have consistent keys and ≤ 8 columns — suitable for a table."""
    if not records:
        return False
    keys = set(records[0].keys())
    all_same = all(set(r.keys()) == keys for r in records)
    return all_same and 1 < len(keys) <= 8


# ── Converters ────────────────────────────────────────────────────────────────

def _records_to_table(records: List[Dict], header: str = "") -> str:
    """Convert record dicts → markdown table with bold column headers."""
    if not records:
        return "📭 No relevant data found."
    keys = list(records[0].keys())
    # Bold column headers
    header_row = "| " + " | ".join(f"**{k}**" for k in keys) + " |"
    separator  = "| " + " | ".join(["---"] * len(keys)) + " |"
    rows = [
        "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"
        for r in records
    ]
    table = "\n".join([header_row, separator] + rows)
    return f"{header}\n\n{table}" if header else table


def _records_to_bullets(records: List[Dict], header: str = "") -> str:
    """Convert record dicts → emoji + bold bullet cards."""
    lines = []
    for i, r in enumerate(records, 1):
        items = list(r.items())
        if not items:
            continue
        # First field = bold title with record number
        first_key, title_val = items[0]
        title_emoji = _field_emoji(first_key) or "🔷"
        lines.append(f"{title_emoji} **{title_val}**")
        for k, v in items[1:]:
            if v:
                emoji = _field_emoji(k)
                prefix = f"{emoji} " if emoji else ""
                lines.append(f"  - {prefix}**{k}:** {v}")
        lines.append("")  # blank line between records
    body = "\n".join(lines).strip()
    return f"{header}\n\n{body}" if header else body


def _numbered_to_bullets(text: str) -> str:
    """Convert '1. X\\n2. Y' → '🔹 **X**\\n🔹 **Y**'."""
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.*)', line)
        if m:
            lines.append(f"🔹 **{m.group(1)}**")
        elif line:
            lines.append(line)
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def format_response(question: str, raw: str) -> str:
    """
    Convert raw bot output to clean, user-friendly markdown with emojis and bold.

    Args:
        question : original user question (detects 'table' intent)
        raw      : raw bot response string

    Returns:
        Formatted string ready for the user.
    """
    if not raw or not raw.strip():
        return "📭 No relevant data found."

    text = raw.strip()

    # Pass through: errors, apologies, no-data signals — don't reformat these
    lower = text.lower()
    if (lower.startswith("error:")
            or lower.startswith("i apologize")
            or lower.startswith("no data found")
            or lower.startswith("no relevant data")
            or text in ("(no rows)",)):
        return text

    # Does the user explicitly ask for a table?
    wants_table = bool(re.search(r'\btable\b', question, re.IGNORECASE))

    # Split off the "Results for '...' (N rows):" header from DB output
    header, body = _split_results_header(text)
    target = body if body else text

    # ── Record block format (multi-column DB result from _format_df) ─────────
    if _has_record_blocks(target):
        records = _parse_record_blocks(target)
        if not records:
            return _clean(text)
        if wants_table or _is_tabular(records):
            return _records_to_table(records, header)
        return _records_to_bullets(records, header)

    # ── Numbered list (single-column DB result from _format_df) ──────────────
    if _is_numbered_list(target):
        bullets = _numbered_to_bullets(target)
        return f"{header}\n\n{bullets}" if header else bullets

    # ── Prose / LLM answer — just clean whitespace ────────────────────────────
    return _clean(text)
