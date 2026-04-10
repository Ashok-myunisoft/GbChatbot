"""
agentic_classifier.py

Intent classifier + API wrapper for the external agentic chatbot.

Responsibilities:
  1. classify(question) → "personal" | "action" | "general"
  2. call_chat_interface(question, username) → response string

Routing rules:
  - personal  : user-specific queries ("my leave balance", "my permission")
  - action    : action-based queries  ("apply leave", "request permission")
  - general   : informational queries ("leave policy", "how to apply leave")
                → falls through to existing bots, this module does nothing

Zero impact on existing system:
  - Only called from one place in process_request()
  - Never modifies any existing routing logic
  - On any API failure → returns None so caller falls back to existing bots
"""

import re
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# ── External agentic chatbot API ──────────────────────────────────────────────
_CHAT_INTERFACE_URL = "http://217.217.249.121:8000/gbaiapi/chat_Interface"
_API_TIMEOUT        = 30.0   # seconds — prevents blocking the main thread


# =============================================================================
# 1. INTENT CLASSIFIER
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

# Action intent: user wants to do something (trigger a workflow)
_ACTION_PATTERNS = [
    r"\bapply\s+(for\s+)?leave\b",
    r"\bapply\s+(a\s+)?leave\b",
    r"\bsubmit\s+(a\s+)?leave\b",
    r"\brequest\s+(a\s+)?leave\b",
    r"\bbook\s+(a\s+)?leave\b",
    r"\btake\s+(a\s+)?leave\b",
    r"\bapply\s+(for\s+)?permission\b",
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

# General informational — should stay with existing bots
# (listed here only for logging clarity — not used in logic)
_GENERAL_SIGNALS = [
    "policy", "how to apply", "rules for", "what is leave",
    "explain leave", "types of leave", "leave types",
    "permission rules", "what is permission",
]


def classify(question: str) -> str:
    """
    Classify the question into one of three intents.

    Returns:
        "personal"  → route to Chat Interface API
        "action"    → route to Chat Interface API
        "general"   → fall through to existing bot routing
    """
    q = question.lower().strip()

    # Check personal intent first (highest priority)
    for pattern in _PERSONAL_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] PERSONAL intent detected: '{question[:60]}'")
            return "personal"

    # Check action intent second
    for pattern in _ACTION_PATTERNS:
        if re.search(pattern, q):
            logger.info(f"[AgenticClassifier] ACTION intent detected: '{question[:60]}'")
            return "action"

    # Default — let existing routing handle it
    logger.info(f"[AgenticClassifier] GENERAL intent — passing to existing bots: '{question[:60]}'")
    return "general"


# =============================================================================
# 2. API HANDLER
# =============================================================================

def call_chat_interface(question: str, username: str) -> Optional[str]:
    """
    Call the external agentic chatbot API.

    Args:
        question : the user's original question
        username : for logging purposes

    Returns:
        Response string on success.
        None on any failure — caller should fall back to existing bots.
    """
    payload = {"message": question}

    try:
        logger.info(f"[AgenticClassifier] Calling Chat Interface API for {username}: '{question[:60]}'")
        resp = requests.post(
            _CHAT_INTERFACE_URL,
            json=payload,
            timeout=_API_TIMEOUT,
        )
        resp.raise_for_status()

        data = resp.json()

        # Accept various response key names
        answer = (
            data.get("response")
            or data.get("message")
            or data.get("answer")
            or data.get("text")
            or data.get("output")
        )

        if answer and str(answer).strip():
            logger.info(f"[AgenticClassifier] API success for {username} ({len(str(answer))} chars)")
            return str(answer).strip()

        logger.warning(f"[AgenticClassifier] API returned empty response for {username}: {data}")
        return None

    except requests.Timeout:
        logger.error(f"[AgenticClassifier] API timeout ({_API_TIMEOUT}s) for {username}")
        return None
    except requests.ConnectionError:
        logger.error(f"[AgenticClassifier] API connection error — is {_CHAT_INTERFACE_URL} reachable?")
        return None
    except requests.HTTPError as e:
        logger.error(f"[AgenticClassifier] API HTTP error for {username}: {e}")
        return None
    except Exception as e:
        logger.error(f"[AgenticClassifier] Unexpected API error for {username}: {e}")
        return None
