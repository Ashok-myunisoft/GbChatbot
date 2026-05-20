from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    description: str
    examples: Tuple[str, ...]
    target: str
    clarification: str = ""


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    description: str
    examples: Tuple[str, ...]
    default_target: str
    clarification: str
    actions: Tuple[ActionDefinition, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    description: str
    examples: Tuple[str, ...]
    clarification: str
    intents: Tuple[IntentDefinition, ...] = field(default_factory=tuple)


def _t(*items: str) -> Tuple[str, ...]:
    return tuple(items)


DOMAIN_REGISTRY: Tuple[DomainDefinition, ...] = (
    DomainDefinition(
        name="HRMS",
        description="Employee attendance, leave, payroll, profiles, approvals, and HR operations.",
        examples=_t(
            "attendance summary",
            "how many days did I work",
            "leave balance",
            "apply leave",
            "cancel my leave",
            "my payslip",
            "employee details",
            "monthly payroll summary",
        ),
        clarification="Are you asking about attendance, leave, payroll, or employee details?",
        intents=(
            IntentDefinition(
                name="attendance",
                description="Attendance status, present days, worked days, late marks, and attendance summaries.",
                examples=_t(
                    "how many days did I work",
                    "attendance summary",
                    "attendance report",
                    "show my attendance",
                    "present days this month",
                ),
                default_target="general",
                clarification="Are you asking about attendance, leave, or payroll?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Return attendance summary or count.",
                        examples=_t("attendance summary", "worked days", "present days"),
                        target="general",
                    ),
                    ActionDefinition(
                        name="compare",
                        description="Compare attendance across periods or employees.",
                        examples=_t("compare attendance", "attendance comparison"),
                        target="report",
                    ),
                    ActionDefinition(
                        name="download",
                        description="Download attendance data or report.",
                        examples=_t("download attendance report", "export attendance"),
                        target="report",
                    ),
                ),
            ),
            IntentDefinition(
                name="leave",
                description="Leave balance, leave application, cancellation, and leave status.",
                examples=_t(
                    "leave balance",
                    "apply leave",
                    "cancel leave",
                    "remaining leaves",
                    "my leave status",
                ),
                default_target="general",
                clarification="Do you want leave balance, leave application, or leave status?",
                actions=(
                    ActionDefinition(
                        name="create",
                        description="Apply or create a leave request.",
                        examples=_t("apply leave", "request leave", "submit leave"),
                        target="general",
                    ),
                    ActionDefinition(
                        name="cancel",
                        description="Cancel or withdraw a leave request.",
                        examples=_t("cancel leave", "withdraw leave"),
                        target="general",
                    ),
                    ActionDefinition(
                        name="summary",
                        description="Summarize leave balance or leave usage.",
                        examples=_t("leave balance", "leave summary"),
                        target="general",
                    ),
                ),
            ),
            IntentDefinition(
                name="payroll",
                description="Salary, payslips, deductions, and payroll breakdowns.",
                examples=_t(
                    "my payslip",
                    "salary breakdown",
                    "payroll summary",
                    "deductions this month",
                    "how is salary calculated",
                ),
                default_target="formula",
                clarification="Are you asking about salary, payslip, deductions, or payroll calculations?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize payroll information.",
                        examples=_t("payroll summary", "salary summary"),
                        target="formula",
                    ),
                    ActionDefinition(
                        name="compare",
                        description="Compare payroll values across periods.",
                        examples=_t("compare salary", "payroll comparison"),
                        target="report",
                    ),
                    ActionDefinition(
                        name="download",
                        description="Download payslip or payroll report.",
                        examples=_t("download payslip", "export payroll"),
                        target="report",
                    ),
                ),
            ),
            IntentDefinition(
                name="employee_search",
                description="Find employee records, staff details, and profiles.",
                examples=_t(
                    "employee search",
                    "find employee details",
                    "show employee profile",
                    "lookup staff record",
                ),
                default_target="schema",
                clarification="Do you want to search employee details or another HRMS topic?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize an employee record.",
                        examples=_t("employee summary", "profile summary"),
                        target="schema",
                    ),
                    ActionDefinition(
                        name="download",
                        description="Download employee details or profile.",
                        examples=_t("download employee details", "export employee profile"),
                        target="schema",
                    ),
                ),
            ),
        ),
    ),
    DomainDefinition(
        name="GST",
        description="GST calculations, tax summaries, returns, invoices, and tax-related queries.",
        examples=_t(
            "gst calculation",
            "gst query",
            "tax invoice",
            "output tax",
            "input tax credit",
            "gst summary",
        ),
        clarification="Are you asking about GST calculation, GST summary, or a tax-related report?",
        intents=(
            IntentDefinition(
                name="gst_query",
                description="GST tax information, tax rules, and GST calculations.",
                examples=_t(
                    "gst calculation",
                    "how much gst",
                    "gst on this invoice",
                    "tax summary",
                    "input tax credit",
                ),
                default_target="formula",
                clarification="Is this a GST calculation, summary, or invoice-related query?",
                actions=(
                    ActionDefinition(
                        name="calculate",
                        description="Calculate GST or tax amounts.",
                        examples=_t("calculate gst", "gst amount", "tax calculation"),
                        target="formula",
                    ),
                    ActionDefinition(
                        name="summary",
                        description="Summarize GST data.",
                        examples=_t("gst summary", "tax summary"),
                        target="report",
                    ),
                ),
            ),
        ),
    ),
    DomainDefinition(
        name="Finance",
        description="Accounting, reports, balances, budgets, comparisons, and financial summaries.",
        examples=_t(
            "financial report",
            "compare accounts",
            "balance sheet",
            "p and l",
            "ledger summary",
            "expense breakdown",
        ),
        clarification="Are you asking for a financial summary, comparison, or calculation?",
        intents=(
            IntentDefinition(
                name="financial_summary",
                description="Summaries, balances, and financial overviews.",
                examples=_t("balance sheet", "ledger summary", "financial summary"),
                default_target="report",
                clarification="Do you need a summary, comparison, or calculation?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize financial data.",
                        examples=_t("financial summary", "balance summary"),
                        target="report",
                    ),
                    ActionDefinition(
                        name="compare",
                        description="Compare financial data across periods.",
                        examples=_t("compare accounts", "compare expense"),
                        target="report",
                    ),
                ),
            ),
            IntentDefinition(
                name="financial_calculation",
                description="Calculations, formulas, percentages, and derived values.",
                examples=_t("calculate tax", "profit calculation", "commission formula"),
                default_target="formula",
                clarification="Do you want a calculation, summary, or comparison?",
                actions=(
                    ActionDefinition(
                        name="calculate",
                        description="Perform a financial calculation.",
                        examples=_t("calculate tax", "calculate profit"),
                        target="formula",
                    ),
                ),
            ),
        ),
    ),
    DomainDefinition(
        name="Document AI",
        description="Invoice extraction, document understanding, and uploaded file intelligence.",
        examples=_t(
            "invoice extraction",
            "extract fields from document",
            "analyze uploaded file",
            "document summary",
            "parse invoice",
        ),
        clarification="Is this about an uploaded file, invoice extraction, or document summary?",
        intents=(
            IntentDefinition(
                name="invoice_extraction",
                description="Extract invoice fields and structured data from documents.",
                examples=_t("invoice extraction", "extract invoice fields", "parse invoice"),
                default_target="general",
                clarification="Do you want extraction, summary, or a document comparison?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize extracted document content.",
                        examples=_t("document summary", "invoice summary"),
                        target="general",
                    ),
                    ActionDefinition(
                        name="download",
                        description="Download extracted document output.",
                        examples=_t("download extracted invoice", "export document data"),
                        target="general",
                    ),
                ),
            ),
        ),
    ),
    DomainDefinition(
        name="General Chat",
        description="Open-ended help, product guidance, explanations, and general ERP support.",
        examples=_t(
            "what can you do",
            "help me with goodbooks",
            "explain this feature",
            "how does the system work",
            "tell me about goodbooks",
        ),
        clarification="Is this a general question, or do you want a specific module?",
        intents=(
            IntentDefinition(
                name="general_chat",
                description="General ERP help and open-ended questions.",
                examples=_t("what can you do", "help me with goodbooks", "tell me about goodbooks"),
                default_target="general",
                clarification="Do you want help with a specific module or general guidance?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize a general question or topic.",
                        examples=_t("summary of features", "quick overview"),
                        target="general",
                    ),
                ),
            ),
        ),
    ),
    DomainDefinition(
        name="Voice AI",
        description="Voice-based interactions, transcription, and spoken responses.",
        examples=_t(
            "transcribe this audio",
            "voice query",
            "speak the answer",
            "audio question",
        ),
        clarification="Is this a voice interaction or a text question?",
        intents=(
            IntentDefinition(
                name="voice_query",
                description="Voice input or voice response related requests.",
                examples=_t("voice query", "transcribe audio", "speak the answer"),
                default_target="general",
                clarification="Do you want transcription, speech output, or a text answer?",
                actions=(
                    ActionDefinition(
                        name="summary",
                        description="Summarize a voice query.",
                        examples=_t("voice summary", "audio summary"),
                        target="general",
                    ),
                ),
            ),
        ),
    ),
)


def iter_domains() -> Sequence[DomainDefinition]:
    return DOMAIN_REGISTRY


def domain_names() -> Tuple[str, ...]:
    return tuple(domain.name for domain in DOMAIN_REGISTRY)


def get_domain(name: str) -> DomainDefinition | None:
    key = name.strip().lower()
    for domain in DOMAIN_REGISTRY:
        if domain.name.lower() == key:
            return domain
    return None


def get_exact_command_patterns() -> Dict[str, str]:
    return {
        r"^\s*(download|export)\s+(attendance|payroll|gst|leave)\b": "report",
        r"^\s*(show|list|get)\s+my\s+(attendance|leave|payslip|salary)\b": "general",
        r"^\s*(cancel|withdraw)\s+(my\s+)?leave\b": "general",
        r"^\s*(apply|request|submit)\s+(a\s+)?leave\b": "general",
        r"^\s*(apply|request|submit)\s+(a\s+)?permission\b": "general",
        r"^\s*(how\s+is\s+salary\s+calculated|salary\s+calculation)\b": "formula",
        r"^\s*(what\s+is\s+gst|gst\s+calculation|calculate\s+gst)\b": "formula",
    }


def get_domain_clarifications() -> Dict[str, str]:
    return {domain.name: domain.clarification for domain in DOMAIN_REGISTRY}


def get_intent_clarification(domain_name: str, intent_name: str) -> str:
    domain = get_domain(domain_name)
    if not domain:
        return ""
    for intent in domain.intents:
        if intent.name.lower() == intent_name.strip().lower():
            return intent.clarification
    return ""


def iter_intents(domain_name: str | None = None) -> List[IntentDefinition]:
    intents: List[IntentDefinition] = []
    for domain in DOMAIN_REGISTRY:
        if domain_name and domain.name.lower() != domain_name.strip().lower():
            continue
        intents.extend(domain.intents)
    return intents


def iter_actions(domain_name: str | None = None, intent_name: str | None = None) -> List[ActionDefinition]:
    actions: List[ActionDefinition] = []
    for intent in iter_intents(domain_name):
        if intent_name and intent.name.lower() != intent_name.strip().lower():
            continue
        actions.extend(intent.actions)
    return actions

