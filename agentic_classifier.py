"""
agentic_classifier.py

Intent classifier + session tracker + API wrapper for the external agentic chatbot.

Responsibilities:
  1. classify(question)            → "personal" | "action" | "general"
  2. call_chat_interface(q, user)  → response string or None
  3. Session tracker               → keeps user in API conversation until slot filling completes

Slot-filling session logic:
  - When a user triggers a personal/action intent → session starts
  - ALL follow-up messages in the same thread go directly to the API (bypass classifier)
  - Session ends when API signals completion (no more questions)
  - Session ends on API failure → falls back to existing bots
  - Greetings are never sent to the API

Zero impact on existing system:
  - Only called from process_request() in orchestrator_main.py
  - On any API failure or session end → caller falls back to existing bots
"""

import re
import json
import time
import logging
import threading
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# ── External agentic chatbot API ──────────────────────────────────────────────
_DEFAULT_CHAT_INTERFACE_URL = "http://217.217.249.121:8000/gbaiapi/chat_Interface"
_API_TIMEOUT        = 30.0    # seconds
_SESSION_TTL        = 300.0   # 5 minutes — auto-expire idle sessions


# =============================================================================
# 1. SESSION TRACKER
# =============================================================================

_sessions: dict = {}        # { username: { "active": bool, "ts": float } }
_session_lock   = threading.Lock()

# Keywords that signal the API has finished collecting all slots
_COMPLETION_SIGNALS = {
    "has been applied", "has been submitted",
    "has been created", "has been approved", "has been rejected",
    "has been cancelled", "has been withdrawn", "leave applied",
    "leave submitted", "request submitted", "time slip submitted",
    "pack created", "successfully submitted", "successfully applied",
    "successfully created", "successfully recorded",
    "task completed", "process completed",
}

# Greetings — never sent to the API, handled by existing greeting flow
_GREETING_WORDS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "sup", "what's up",
}

# ── Slot-filling response detection ──────────────────────────────────────────
_SLOT_FILLING_KEYWORDS = {
    "leave", "permission", "half day", "full day", "first half", "second half",
    "sick", "casual", "earned", "loss of pay", "comp off", "maternity",
    "absent", "personal", "emergency", "health", "medical", "work from",
    "time slip", "pack", "overtime", "from date", "to date",
}

_MONTH_PATTERN = r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b"

_QUESTION_STARTERS = (
    "what is", "what are", "what was", "what were",
    "how do", "how does", "how can", "how to", "how many", "how much",
    "why is", "why does", "why can",
    "can you explain", "could you explain", "explain",
    "tell me about", "tell me how", "describe",
    "what exactly", "who is", "where is", "can you tell",
)

# ── Action message normalization ──────────────────────────────────────────────
_ACTION_TO_API_COMMAND = [
    (r"\b(apply|request|submit|book|take)\s+(for\s+|a\s+)?leave\b",  "apply leave"),
    (r"\b(apply|request|submit)\s+(for\s+|a\s+)?permission\b",       "apply permission"),
    (r"\bcancel\s+(my\s+)?leave\b",                                    "cancel leave"),
    (r"\bcancel\s+(my\s+)?permission\b",                               "cancel permission"),
    (r"\bwithdraw\s+(my\s+)?leave\b",                                  "withdraw leave"),
    (r"\bwithdraw\s+(my\s+)?permission\b",                             "withdraw permission"),
    (r"\bapprove\s+leave\b",                                           "approve leave"),
    (r"\breject\s+leave\b",                                            "reject leave"),
    (r"\bsubmit\s+(a\s+)?time\s*slip\b",                               "submit time slip"),
    (r"\bsubmit\s+.{0,10}time\s*slip\b",                              "submit time slip"),
    (r"\bmark\s+attendance\b",                                         "mark attendance"),
]


def is_greeting(question: str) -> bool:
    """Return True if the entire message is a greeting — should never hit the API."""
    q = question.lower().strip().rstrip("!.,?")
    return q in _GREETING_WORDS


def is_slot_filling_response(question: str) -> bool:
    """
    Return True  → message is a valid slot-filling response (keep session open).
    Return False → message is a general question (end session, route to existing bots).
    Conservative: only returns False when the message is clearly not slot-filling.
    """
    q = question.lower().strip()

    # Pure number → leave type / day type selection
    if re.match(r'^\d+$', q):
        return True

    # Date pattern → from/to date input
    if re.search(r'\b\d{1,2}[/\-\.]\d{1,2}([/\-\.]\d{2,4})?\b', q):
        return True

    # Month name → date input
    # Use a non-capturing group so "may" does not get matched inside unrelated words.
    if re.search(_MONTH_PATTERN, q):
        return True

    # Short control words
    if q.rstrip('.,!') in {'yes', 'no', 'ok', 'okay', 'y', 'n', 'sure',
                            'cancel', 'stop', 'quit', 'exit', 'done', 'confirm'}:
        return True

    # Contains a slot-filling keyword
    if any(kw in q for kw in _SLOT_FILLING_KEYWORDS):
        return True

    # Long question-style message → clearly not a slot-filling response
    if len(q.split()) > 5 and any(q.startswith(s) for s in _QUESTION_STARTERS):
        return False

    # Default: conservative — don't break session unless clearly non-slot
    return True


def normalize_action_message(question: str) -> str:
    """
    Map a verbose action intent sentence to a direct API command.
    This causes the API to skip the balance/offer display and go straight
    to slot-filling.
    Returns the normalized command, or the original question if no match.
    """
    q = question.lower().strip()
    for pattern, command in _ACTION_TO_API_COMMAND:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] Normalized: '{question[:60]}' → '{command}'")
            return command
    return question


def start_session(username: str) -> None:
    """Mark user as being in an active slot-filling session."""
    with _session_lock:
        _sessions[username] = {"active": True, "ts": time.time()}
    logger.info(f"[AgenticSession] Session STARTED for {username}")


def end_session(username: str) -> None:
    """Clear the user's slot-filling session."""
    with _session_lock:
        _sessions.pop(username, None)
    logger.info(f"[AgenticSession] Session ENDED for {username}")


def _start_session_if_needed(username: str) -> None:
    """Atomically start a session only if one is not already active."""
    with _session_lock:
        entry = _sessions.get(username)
        if not entry or time.time() - entry["ts"] > _SESSION_TTL:
            _sessions[username] = {"active": True, "ts": time.time()}
            logger.info(f"[AgenticSession] Session STARTED for {username}")


def is_in_session(username: str) -> bool:
    """
    Return True if user is currently in an active slot-filling session.
    Auto-expires sessions idle for more than SESSION_TTL seconds.
    """
    with _session_lock:
        entry = _sessions.get(username)
        if not entry:
            return False
        if time.time() - entry["ts"] > _SESSION_TTL:
            del _sessions[username]
            logger.info(f"[AgenticSession] Session EXPIRED for {username}")
            return False
        entry["ts"] = time.time()   # refresh TTL on each access
        return True


def _is_completion(response_text: str) -> bool:
    """
    Detect if the API response signals task completion (no more slots needed).
    Completion signals take priority — a trailing "?" (e.g. "Submitted! Anything else?")
    must not suppress detection of a genuine completion.
    Only if there is no completion signal do we treat a "?" as "still collecting".
    """
    lower = response_text.lower()
    if any(signal in lower for signal in _COMPLETION_SIGNALS):
        return True
    return False


def _resolve_chat_interface_url(login_dto: dict) -> str:
    """
    Resolve the agentic chatbot endpoint.

    Prefer a runtime-provided BaseURL when available, but keep the existing
    hardcoded endpoint as a safe fallback so the current integration continues
    to work even if the caller does not supply BaseURL.
    """
    base_url = ""
    if isinstance(login_dto, dict):
        base_url = str(login_dto.get("BaseURL") or login_dto.get("base_url") or "").strip()

    if base_url:
        return base_url.rstrip("/") + "/gbaiapi/chat_Interface"

    return _DEFAULT_CHAT_INTERFACE_URL


# =============================================================================
# 2. INTENT CLASSIFIER
# =============================================================================

# Personal intent: user asking about their own data
_PERSONAL_PATTERNS = [
    r"\bmy\s+leave\b",
    r"\bmy\s+leaves\b",
    r"\bmy\s+leave\s+balance\b",
    r"\bmy\s+permission\b",
    r"\bmy\s+permission\s+balance\b",
    r"\bmy\s+attendance\b",
    r"\bmy\s+salary\b",
    r"\bmy\s+payslip\b",
    r"\bmy\s+profile\b",
    r"\bmy\s+data\b",
    r"\bmy\s+details\b",
    r"\bmy\s+record\b",
    r"\bhow\s+many\s+leaves?\s+(do\s+)?i\s+have\b",
    r"\bhow\s+many\s+permissions?\s+(do\s+)?i\s+have\b",
    r"\bleave\s+balance\b",
    r"\bpermission\s+balance\b",
    r"\bremaining\s+leaves?\b",
    r"\bavailable\s+leaves?\b",
    r"\bleft\s+leaves?\b",
]

# Action intent: user wants to trigger a workflow
_ACTION_PATTERNS = [
    r"\bapply\s+(for\s+)?leave\b",
    r"\bapply\s+(a\s+)?leave\b",
    r"\bsubmit\s+(a\s+)?leave\b",
    r"\brequest\s+(a\s+)?leave\b",
    r"\bbook\s+(a\s+)?leave\b",
    r"\btake\s+(a\s+)?leave\b",
    r"\bapply\s+(for\s+)?permission\b",
    r"\bapply\s+(a\s+)?permission\b",
    r"\brequest\s+(a\s+)?permission\b",
    r"\bsubmit\s+(a\s+)?permission\b",
    r"\bcancel\s+(my\s+)?leave\b",
    r"\bcancel\s+(my\s+)?permission\b",
    r"\bwithdraw\s+(my\s+)?leave\b",
    r"\bwithdraw\s+(my\s+)?permission\b",
    r"\bapprove\s+leave\b",
    r"\breject\s+leave\b",
    r"\bmark\s+attendance\b",
]


def classify(question: str) -> str:
    """
    Classify question intent.
    Greetings always return "general" — never routed to the API.

    Returns:
        "personal"  → route to Chat Interface API
        "action"    → route to Chat Interface API
        "general"   → fall through to existing bot routing
    """
    # Greetings must never go to the API
    if is_greeting(question):
        logger.info(f"[AgenticClassifier] GREETING — skipping API: '{question[:60]}'")
        return "general"

    q = question.lower().strip()

    # Time slip requests should route to the agentic workflow even if phrased directly.
    if re.search(r"\bsubmit\s+(a\s+)?time\s*slip\b", q) or re.search(r"\btime\s*slip\b", q):
        logger.info(f"[AgenticClassifier] ACTION intent: '{question[:60]}'")
        return "action"

    for pattern in _PERSONAL_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] PERSONAL intent: '{question[:60]}'")
            return "personal"

    for pattern in _ACTION_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] ACTION intent: '{question[:60]}'")
            return "action"

    logger.info(f"[AgenticClassifier] GENERAL intent — passing to existing bots: '{question[:60]}'")
    return "general"


# =============================================================================
# 3. API HANDLER
# =============================================================================

def call_chat_interface(
    question: str,
    username: str,
    login_dto: dict
) -> Optional[str]:
    """
    Call the external agentic chatbot API with dynamic login DTO.

    - Uses BaseURL when supplied, otherwise falls back to the default endpoint
    - Starts session on first call
    - Ends session on completion or failure
    """

    # 🚫 Block greetings — preserve any active session so slot-filling is not aborted
    if is_greeting(question):
        logger.info(f"[AgenticClassifier] Greeting blocked from API for {username} — session preserved")
        return None

    payload = {
        "message": question
    }

    headers = {
        "Content-Type": "application/json",
        "Login": json.dumps(login_dto)   # ✅ FULL DTO PASSED HERE
    }

    chat_interface_url = _resolve_chat_interface_url(login_dto)

    try:
        logger.info(f"[AgenticClassifier] → API call for {username}: '{question[:60]}'")
        logger.debug(f"[AgenticClassifier] Login DTO sent: {login_dto}")

        resp = requests.post(
            chat_interface_url,
            json=payload,
            headers=headers,
            timeout=_API_TIMEOUT,
        )

        resp.raise_for_status()

        data = resp.json()

        status = data.get("status", "")
        answer = (
            data.get("response")
            or data.get("message")
            or data.get("answer")
            or data.get("text")
            or data.get("output")
        )

        if not answer or not str(answer).strip():
            logger.warning(f"[AgenticClassifier] Empty API response for {username}: {data}")
            end_session(username)
            return None

        answer = str(answer).strip()

        logger.info(f"[AgenticClassifier] API response ({len(answer)} chars): '{answer[:80]}'")

        # ✅ Start or maintain session (atomic check+set)
        _start_session_if_needed(username)

        # End session on completion
        if status == "error":
            logger.warning(f"[AgenticClassifier] API reported error status for {username} — falling back")
            end_session(username)
            return None

        if _is_completion(answer):
            end_session(username)
            logger.info(f"[AgenticClassifier] Task complete — session closed for {username}")

        return answer

    except requests.Timeout:
        logger.error(f"[AgenticClassifier] API timeout ({_API_TIMEOUT}s) for {username}")
        end_session(username)
        return None

    except requests.ConnectionError:
        logger.error(f"[AgenticClassifier] API connection error — {chat_interface_url}")
        end_session(username)
        return None

    except requests.HTTPError as e:
        logger.error(f"[AgenticClassifier] API HTTP error for {username}: {e}")
        end_session(username)
        return None

    except Exception as e:
        logger.error(f"[AgenticClassifier] Unexpected error for {username}: {e}")
        end_session(username)
        return None
        
        
