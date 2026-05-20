from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .intent_registry import DOMAIN_REGISTRY, DomainDefinition
from .learning_memory import SemanticLearningMemory
from shared_resources import ai_resources

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower()).strip())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom_a = sum(x * x for x in a) ** 0.5
    denom_b = sum(x * x for x in b) ** 0.5
    if not denom_a or not denom_b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (denom_a * denom_b)


def _embed_documents(texts: List[str]) -> List[List[float]]:
    embeddings = ai_resources.embeddings
    try:
        return embeddings.embed_documents(texts)
    except Exception:
        return [embeddings.embed_query(text) for text in texts]


@dataclass(frozen=True)
class CandidateScore:
    name: str
    score: float
    metadata: Dict[str, str]


@dataclass
class RoutingDecision:
    domain: str
    intent: str
    action: str
    target: str
    confidence: float
    score_gap: float
    source: str
    clarification: Optional[str] = None
    matched_text: str = ""
    fallback_target: str = "general"
    metadata: Dict[str, str] = None

    def should_route_now(self) -> bool:
        return bool(self.target) and self.confidence >= 0.74 and not self.clarification

    def should_clarify(self) -> bool:
        return bool(self.clarification)


class SemanticEmbeddingRouter:
    def __init__(
        self,
        registry: Sequence[DomainDefinition] = DOMAIN_REGISTRY,
        learning_memory: Optional[SemanticLearningMemory] = None,
    ) -> None:
        self.registry = tuple(registry)
        self.learning_memory = learning_memory or SemanticLearningMemory()
        self._lock = threading.Lock()
        self._query_cache: Dict[str, RoutingDecision] = {}
        self._query_embedding_cache: Dict[str, List[float]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        self._domain_candidates: List[CandidateScore] = []
        self._intent_candidates: Dict[str, List[CandidateScore]] = {}
        self._action_candidates: Dict[Tuple[str, str], List[CandidateScore]] = {}
        self._route_lookup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        self._exact_targets: Dict[str, str] = {}

        for domain in self.registry:
            domain_text = " ".join([domain.name, domain.description, " ".join(domain.examples)])
            self._domain_candidates.append(
                CandidateScore(
                    name=domain.name,
                    score=0.0,
                    metadata={"text": domain_text, "clarification": domain.clarification},
                )
            )

            intent_candidates: List[CandidateScore] = []
            for intent in domain.intents:
                intent_text = " ".join([intent.name, intent.description, " ".join(intent.examples)])
                intent_candidates.append(
                    CandidateScore(
                        name=intent.name,
                        score=0.0,
                        metadata={
                            "domain": domain.name,
                            "text": intent_text,
                            "clarification": intent.clarification,
                            "default_target": intent.default_target,
                        },
                    )
                )

                action_candidates: List[CandidateScore] = []
                for action in intent.actions:
                    action_text = " ".join([action.name, action.description, " ".join(action.examples)])
                    action_candidates.append(
                        CandidateScore(
                            name=action.name,
                            score=0.0,
                            metadata={
                                "domain": domain.name,
                                "intent": intent.name,
                                "text": action_text,
                                "clarification": action.clarification,
                                "target": action.target,
                            },
                        )
                    )
                    self._route_lookup[(domain.name.lower(), intent.name.lower(), action.name.lower())] = {
                        "domain": domain.name,
                        "intent": intent.name,
                        "action": action.name,
                        "target": action.target,
                        "clarification": action.clarification or intent.clarification or domain.clarification,
                    }
                    for example in action.examples:
                        self._exact_targets[_normalize(example)] = action.target

                self._action_candidates[(domain.name.lower(), intent.name.lower())] = action_candidates

                self._route_lookup[(domain.name.lower(), intent.name.lower(), "__default__")] = {
                    "domain": domain.name,
                    "intent": intent.name,
                    "action": "summary",
                    "target": intent.default_target,
                    "clarification": intent.clarification or domain.clarification,
                }

                for example in intent.examples:
                    self._exact_targets[_normalize(example)] = intent.default_target

            # store intent candidates after they are built
            self._intent_candidates[domain.name.lower()] = intent_candidates
            for example in domain.examples:
                self._exact_targets[_normalize(example)] = domain.intents[0].default_target if domain.intents else "general"

        self._domain_embeddings = _embed_documents([candidate.metadata["text"] for candidate in self._domain_candidates])
        self._intent_embeddings: Dict[str, List[List[float]]] = {}
        self._action_embeddings: Dict[Tuple[str, str], List[List[float]]] = {}

        for key, candidates in self._intent_candidates.items():
            self._intent_embeddings[key] = _embed_documents([candidate.metadata["text"] for candidate in candidates])
        for key, candidates in self._action_candidates.items():
            self._action_embeddings[key] = _embed_documents([candidate.metadata["text"] for candidate in candidates])

    @lru_cache(maxsize=256)
    def _embed_query_cached(self, normalized_query: str) -> Tuple[float, ...]:
        vec = ai_resources.embeddings.embed_query(normalized_query)
        return tuple(vec)

    def _embed_query(self, query: str) -> List[float]:
        normalized = _normalize(query)
        if normalized in self._query_embedding_cache:
            return self._query_embedding_cache[normalized]
        vec = ai_resources.embeddings.embed_query(query)
        self._query_embedding_cache[normalized] = vec
        return vec

    def _score_candidates(
        self,
        query_vec: Sequence[float],
        candidates: Sequence[CandidateScore],
        embeddings: Sequence[Sequence[float]],
    ) -> List[CandidateScore]:
        scored: List[CandidateScore] = []
        for candidate, vector in zip(candidates, embeddings):
            scored.append(
                CandidateScore(
                    name=candidate.name,
                    score=_cosine(query_vec, vector),
                    metadata=candidate.metadata,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _best_match(self, query: str) -> RoutingDecision:
        normalized = _normalize(query)
        if normalized in self._exact_targets:
            target = self._exact_targets[normalized]
            return RoutingDecision(
                domain="General Chat",
                intent="general_chat",
                action="summary",
                target=target,
                confidence=1.0,
                score_gap=1.0,
                source="exact_match",
                matched_text=query,
                fallback_target=target,
                metadata={"normalized_query": normalized},
            )

        query_vec = self._embed_query(query)

        domain_scores = self._score_candidates(query_vec, self._domain_candidates, self._domain_embeddings)
        if not domain_scores:
            return RoutingDecision(
                domain="General Chat",
                intent="general_chat",
                action="summary",
                target="general",
                confidence=0.0,
                score_gap=0.0,
                source="empty_registry",
                clarification="Could you rephrase the request?",
                fallback_target="general",
                metadata={},
            )

        best_domain = domain_scores[0]
        second_domain = domain_scores[1] if len(domain_scores) > 1 else None
        domain_confidence = best_domain.score
        domain_gap = domain_confidence - (second_domain.score if second_domain else 0.0)

        intent_key = best_domain.name.lower()
        intent_candidates = self._intent_candidates.get(intent_key, [])
        intent_embeddings = self._intent_embeddings.get(intent_key, [])
        intent_scores = self._score_candidates(query_vec, intent_candidates, intent_embeddings) if intent_candidates else []

        if not intent_scores:
            clarification = f"Are you asking about {best_domain.name.lower()} or something else?"
            return RoutingDecision(
                domain=best_domain.name,
                intent="general_chat",
                action="summary",
                target="general",
                confidence=domain_confidence,
                score_gap=domain_gap,
                source="domain_only",
                clarification=clarification,
                matched_text=best_domain.metadata.get("text", ""),
                fallback_target="general",
                metadata={"domain_score": f"{domain_confidence:.4f}"},
            )

        best_intent = intent_scores[0]
        second_intent = intent_scores[1] if len(intent_scores) > 1 else None
        intent_gap = best_intent.score - (second_intent.score if second_intent else 0.0)

        action_key = (best_domain.name.lower(), best_intent.name.lower())
        action_candidates = self._action_candidates.get(action_key, [])
        action_embeddings = self._action_embeddings.get(action_key, [])
        action_scores = self._score_candidates(query_vec, action_candidates, action_embeddings) if action_candidates else []

        if action_scores:
            best_action = action_scores[0]
            second_action = action_scores[1] if len(action_scores) > 1 else None
            action_gap = best_action.score - (second_action.score if second_action else 0.0)
            route = self._route_lookup.get(
                (best_domain.name.lower(), best_intent.name.lower(), best_action.name.lower())
            ) or self._route_lookup.get(
                (best_domain.name.lower(), best_intent.name.lower(), "__default__")
            )
            confidence = (best_domain.score * 0.30) + (best_intent.score * 0.35) + (best_action.score * 0.35)
            score_gap = min(domain_gap, intent_gap, action_gap)
            clarification = None
            if confidence < 0.62 or score_gap < 0.04:
                clarification = route.get("clarification") or best_domain.metadata.get("clarification", "")
            return RoutingDecision(
                domain=route["domain"],
                intent=route["intent"],
                action=route["action"],
                target=route["target"],
                confidence=confidence,
                score_gap=score_gap,
                source="semantic",
                clarification=clarification,
                matched_text=best_action.metadata.get("text", ""),
                fallback_target=route["target"],
                metadata={
                    "domain_score": f"{best_domain.score:.4f}",
                    "intent_score": f"{best_intent.score:.4f}",
                    "action_score": f"{best_action.score:.4f}",
                },
            )

        route = self._route_lookup.get(
            (best_domain.name.lower(), best_intent.name.lower(), "__default__")
        )
        confidence = (best_domain.score * 0.45) + (best_intent.score * 0.55)
        clarification = None
        if confidence < 0.62 or intent_gap < 0.04:
            clarification = route.get("clarification") or best_intent.metadata.get("clarification", "")
        return RoutingDecision(
            domain=route["domain"],
            intent=route["intent"],
            action=route["action"],
            target=route["target"],
            confidence=confidence,
            score_gap=min(domain_gap, intent_gap),
            source="semantic",
            clarification=clarification,
            matched_text=best_intent.metadata.get("text", ""),
            fallback_target=route["target"],
            metadata={
                "domain_score": f"{best_domain.score:.4f}",
                "intent_score": f"{best_intent.score:.4f}",
            },
        )

    def route(self, query: str) -> RoutingDecision:
        normalized = _normalize(query)
        with self._lock:
            cached = self._query_cache.get(normalized)
            if cached:
                return cached

        decision = self._best_match(query)
        with self._lock:
            if len(self._query_cache) >= 500:
                self._query_cache = dict(list(self._query_cache.items())[250:])
            self._query_cache[normalized] = decision
        return decision
