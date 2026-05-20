from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConfidenceDecision:
    route_now: bool
    ask_clarification: bool
    min_confidence: float


def classify_confidence(score: float, gap: float = 0.0) -> ConfidenceDecision:
    if score >= 0.82 and gap >= 0.04:
        return ConfidenceDecision(route_now=True, ask_clarification=False, min_confidence=0.82)
    if score >= 0.74 and gap >= 0.08:
        return ConfidenceDecision(route_now=True, ask_clarification=False, min_confidence=0.74)
    if score >= 0.62:
        return ConfidenceDecision(route_now=False, ask_clarification=True, min_confidence=0.62)
    return ConfidenceDecision(route_now=False, ask_clarification=True, min_confidence=0.0)


def build_clarification(domain: str, intent: str, action: str, fallback: Optional[str] = None) -> str:
    if domain == "HRMS":
        if intent == "attendance":
            return "Are you asking about attendance, leave, or payroll?"
        if intent == "leave":
            return "Do you want leave balance, leave application, or leave cancellation?"
        if intent == "payroll":
            return "Are you asking about salary, payslip, deductions, or payroll calculation?"
        return fallback or "Are you asking about HRMS attendance, leave, payroll, or employee details?"

    if domain == "GST":
        return fallback or "Are you asking about GST calculation, GST summary, or invoice-related tax data?"

    if domain == "Finance":
        return fallback or "Are you asking for a financial summary, comparison, or calculation?"

    if domain == "Document AI":
        return fallback or "Are you asking about document extraction, invoice parsing, or a file summary?"

    if domain == "Voice AI":
        return fallback or "Do you want transcription, speech output, or a text answer?"

    return fallback or "Could you clarify whether this is about a specific module or a general question?"

