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
    
    Always use the hardcoded endpoint to ensure consistent
    routing to the chat interface, ignoring frontend BaseURL.
    """
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
    r"\bget\s+(my\s+|the\s+)?attendance\b",
    r"\bshow\s+(my\s+|the\s+)?attendance\b",
    r"\bview\s+(my\s+|the\s+)?attendance\b",
    r"\bcheck\s+(my\s+|the\s+)?attendance\b",
    r"\bwhat\s+is\s+my\s+attendance\b",
    r"\battendance\s+(details|record|status)\b",
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

# Navigational intent: user is asking WHERE a screen/menu/section lives, not
# requesting an action or their own data. These often mention "leave"/"attendance"
# incidentally (e.g. "how to access the leave request screen") and must be caught
# before the weak-signal LLM stage misreads them as ACTION/PERSONAL.
_NAVIGATION_PATTERNS = [
    r"\bhow\s+(do|can|to)\s+i?\s*(access|open|navigate|reach|get\s+to|go\s+to|find)\b",
    r"\bwhere\s+(is|are|can\s+i\s+find|do\s+i\s+find)\b",
    r"\b(which|what)\s+(section|screen|module|menu|tab|page)\b",
    r"\bin\s+which\s+(section|screen|module|menu|tab|page)\b",
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
    Two-stage intent classifier.

    Stage 1 — regex (existing patterns): strong, unambiguous matches resolved instantly.
    Stage 2 — LLM confirmation: only fires when weak signals are present but no strong
              pattern matched. Prevents false positives (e.g. "leave approval workflow"
              → GENERAL) and false negatives (e.g. "I need a day off" → ACTION).

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

    # ── Stage 1: Strong regex matches (existing logic, unchanged) ─────────────
    # Time slip — explicit fast path
    if re.search(r"\bsubmit\s+(a\s+)?time\s*slip\b", q) or re.search(r"\btime\s*slip\b", q):
        logger.info(f"[AgenticClassifier] ACTION intent (strong): '{question[:60]}'")
        return "action"

    for pattern in _PERSONAL_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] PERSONAL intent (strong): '{question[:60]}'")
            return "personal"

    for pattern in _ACTION_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] ACTION intent (strong): '{question[:60]}'")
            return "action"

    # ── Stage 1b: Navigational questions — never ACTION/PERSONAL ──────────────
    # Checked after the strong patterns (so "my leave balance" etc. still win)
    # but before the LLM stage, since these are deterministic UI-location asks.
    for pattern in _NAVIGATION_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] GENERAL intent (navigational): '{question[:60]}'")
            return "general"

    # ── Stage 2: Weak signal → LLM confirmation ───────────────────────────────
    # Only runs when no strong pattern matched but the message contains words that
    # COULD relate to personal/action intent (catches novel phrasings like
    # "I need a day off" or prevents "leave approval workflow" being misrouted).
    _WEAK_SIGNALS = {
        "leave", "permission", "attendance", "absent", "day off",
        "time off", "half day", "comp off", "holiday", "off day",
    }
    if any(w in q for w in _WEAK_SIGNALS):
        try:
            from shared_resources import ai_resources
            _prompt = (
                "Classify this ERP chatbot message into one category:\n"
                "PERSONAL — user asking about their OWN leave/permission/attendance/salary data\n"
                "ACTION   — user wants to apply/cancel/submit/approve leave, permission, or time slip\n"
                "GENERAL  — asking about ERP concepts, policies, workflows, reports, navigation, or general info\n\n"
                "IMPORTANT: a question about WHERE/HOW TO FIND, ACCESS, or NAVIGATE TO a screen, "
                "menu, or section is always GENERAL — even if it mentions leave/permission/attendance — "
                "because the user wants directions to a UI location, not to submit a request or view their data.\n\n"
                "Examples:\n"
                "  'I need a day off next Monday'          → ACTION\n"
                "  'show leave approval workflow'          → GENERAL\n"
                "  'how many leaves do I have'             → PERSONAL\n"
                "  'what is the leave approval process'    → GENERAL\n"
                "  'leave policy in HR module'             → GENERAL\n"
                "  'book me off on Friday'                 → ACTION\n"
                "  'how do I get to the leave screen'      → GENERAL\n"
                "  'where can I submit a leave request'    → GENERAL\n\n"
                "Reply with PERSONAL, ACTION, or GENERAL only.\n"
                f"Message: {question}"
            )
            raw = ai_resources.routing_llm.invoke(_prompt)
            answer = (raw.content if hasattr(raw, "content") else str(raw)).strip().upper()
            if answer.startswith("PERSONAL"):
                logger.info(f"[AgenticClassifier] PERSONAL intent (LLM confirmed): '{question[:60]}'")
                return "personal"
            if answer.startswith("ACTION"):
                logger.info(f"[AgenticClassifier] ACTION intent (LLM confirmed): '{question[:60]}'")
                return "action"
            logger.info(f"[AgenticClassifier] GENERAL intent (LLM confirmed): '{question[:60]}'")
        except Exception as _e:
            logger.warning(f"[AgenticClassifier] LLM confirmation failed, defaulting to general: {_e}")

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
        
