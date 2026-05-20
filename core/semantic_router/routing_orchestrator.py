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

    def route(self, query: str, username: str = "", thread_id: str = "") -> SemanticRouteResult:
        exact = self._match_exact_command(query)
        if exact:
            logger.info("[SemanticRouter] Critical regex route target=%s query=%s", exact.target, query[:80])
            return exact

        learned = self.learning_memory.find_matching_correction(query)
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

        decision: RoutingDecision = self.router.route(query)
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
    ):
        return self.learning_memory.add_feedback(
            query=query,
            wrong_route=wrong_route,
            correct_route=correct_route,
            reason=reason,
            username=username,
            thread_id=thread_id,
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

