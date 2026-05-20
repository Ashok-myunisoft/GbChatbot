from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .confidence_handler import build_clarification, classify_confidence
from .embedding_router import RoutingDecision, SemanticEmbeddingRouter
from .intent_registry import get_exact_command_patterns
from .learning_memory import SemanticLearningMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticRouteResult:
    domain: str
    intent: str
    action: str
    target: str
    confidence: float
    route_now: bool
    clarification: str = ""
    source: str = ""
    metadata: Dict[str, str] | None = None


class SemanticRoutingOrchestrator:
    def __init__(self) -> None:
        self.learning_memory = SemanticLearningMemory()
        self.router = SemanticEmbeddingRouter(learning_memory=self.learning_memory)
        self._lock = threading.Lock()
        self._exact_patterns = get_exact_command_patterns()

    @staticmethod
    def _context_to_text(context: object) -> str:
        if context is None:
            return ""
        if isinstance(context, str):
            return context.strip()
        if isinstance(context, dict):
            parts = []
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

    def _match_exact_command(self, query: str) -> Optional[SemanticRouteResult]:
        import re

        normalized = query.strip().lower()
        for pattern, target in self._exact_patterns.items():
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return SemanticRouteResult(
                    domain="General Chat",
                    intent="general_chat",
                    action="summary",
                    target=target,
                    confidence=1.0,
                    route_now=True,
                    clarification="",
                    source="critical_regex",
                    metadata={"pattern": pattern},
                )
        return None

    def route(
        self,
        query: str,
        username: str = "",
        thread_id: str = "",
        context: object = None,
    ) -> SemanticRouteResult:
        exact = self._match_exact_command(query)
        if exact:
            logger.info("[SemanticRouter] Critical regex route target=%s query=%s", exact.target, query[:80])
            return exact

        context_text = self._context_to_text(context)
        learned = self.learning_memory.find_matching_correction(
            query,
            context_text=context_text,
            username=username,
            thread_id=thread_id,
        )
        if learned:
            target = learned.get("correct_route", "general")
            logger.info(
                "[SemanticRouter] Learned route reused target=%s score=%.3f query=%s",
                target,
                learned.get("score", 0.0),
                query[:80],
            )
            return SemanticRouteResult(
                domain="General Chat",
                intent="general_chat",
                action="summary",
                target=target,
                confidence=float(learned.get("score", 0.0)),
                route_now=True,
                clarification="",
                source="learning_memory",
                metadata={
                    "wrong_route": learned.get("wrong_route", ""),
                    "correct_route": target,
                    "reason": learned.get("reason", ""),
                },
            )

        decision: RoutingDecision = self.router.route(query, context_text=context_text)
        confidence = decision.confidence
        confidence_state = classify_confidence(confidence, decision.score_gap)

        clarification = ""
        if confidence_state.ask_clarification or decision.should_clarify():
            clarification = build_clarification(
                decision.domain,
                decision.intent,
                decision.action,
                fallback=decision.clarification,
            )

        result = SemanticRouteResult(
            domain=decision.domain,
            intent=decision.intent,
            action=decision.action,
            target=decision.target,
            confidence=confidence,
            route_now=confidence_state.route_now and bool(decision.target),
            clarification=clarification,
            source=decision.source,
            metadata=decision.metadata or {},
        )

        logger.info(
            "[SemanticRouter] domain=%s intent=%s action=%s target=%s confidence=%.3f gap=%.3f source=%s",
            result.domain,
            result.intent,
            result.action,
            result.target,
            result.confidence,
            decision.score_gap,
            result.source,
        )
        return result

    def record_feedback(
        self,
        query: str,
        wrong_route: str,
        correct_route: str,
        reason: str,
        username: str = "",
        thread_id: str = "",
        context: object = None,
    ):
        return self.learning_memory.add_feedback(
            query=query,
            wrong_route=wrong_route,
            correct_route=correct_route,
            reason=reason,
            username=username,
            thread_id=thread_id,
            context=context,
        )


_SEMANTIC_ROUTER: Optional[SemanticRoutingOrchestrator] = None
_LOCK = threading.Lock()


def get_semantic_router() -> SemanticRoutingOrchestrator:
    global _SEMANTIC_ROUTER
    if _SEMANTIC_ROUTER is None:
        with _LOCK:
            if _SEMANTIC_ROUTER is None:
                _SEMANTIC_ROUTER = SemanticRoutingOrchestrator()
    return _SEMANTIC_ROUTER
