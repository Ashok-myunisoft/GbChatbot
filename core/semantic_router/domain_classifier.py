from __future__ import annotations

from dataclasses import dataclass

from .embedding_router import SemanticEmbeddingRouter


@dataclass(frozen=True)
class DomainPrediction:
    domain: str
    confidence: float
    clarification: str = ""


class DomainClassifier:
    def __init__(self, router: SemanticEmbeddingRouter) -> None:
        self.router = router

    def classify(self, query: str) -> DomainPrediction:
        decision = self.router.route(query)
        return DomainPrediction(
            domain=decision.domain,
            confidence=decision.confidence,
            clarification=decision.clarification or "",
        )

