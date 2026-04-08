"""
tasks.py — Three tasks (easy → medium → hard) for the Customer Support environment.

Each task is a dict with:
  - ticket          : the initial Ticket object
  - instructions    : natural language task objective
  - kb_snippets     : relevant knowledge base snippets
  - max_steps       : episode length limit
  - grader          : callable(history) → float in [0.0, 1.0]

Grader inputs
-------------
history : list of dicts, each:
  { "action": SupportAction, "reward": float, "step": int }
ticket  : the final (mutated) Ticket object
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import (
    ActionType,
    CustomerMessage,
    SupportAction,
    Ticket,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_action(history: List[Dict], action_type: ActionType) -> bool:
    return any(h["action"].action_type == action_type for h in history)


def _reply_texts(history: List[Dict]) -> List[str]:
    return [
        h["action"].reply_text or ""
        for h in history
        if h["action"].action_type in (ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.APPLY_TEMPLATE)
    ]


def _contains_keywords(texts: List[str], keywords: List[str]) -> int:
    """Return how many keywords appear in any of the texts (case-insensitive)."""
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined)


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# Password Reset (straightforward, one common resolution path)
# ---------------------------------------------------------------------------

_TICKET_1 = Ticket(
    ticket_id="TKT-001",
    subject="Cannot log in — forgot password",
    customer_name="Alice Johnson",
    customer_email="alice.johnson@example.com",
    priority=TicketPriority.LOW,
    category=TicketCategory.ACCOUNT,
    status=TicketStatus.OPEN,
    conversation=[
        CustomerMessage(
            sender="customer",
            content=(
                "Hi, I forgot my password and the reset email never arrived. "
                "I checked my spam folder too. My account email is alice.johnson@example.com. "
                "Please help!"
            ),
            timestamp=_now(),
        )
    ],
    metadata={"account_verified": True, "reset_attempts": 2},
)

_KB_1 = [
    "To trigger a manual password reset: go to Admin > Users > search email > click 'Force Reset'.",
    "If reset emails are not arriving, check the email allowlist. Common domains: gmail.com, outlook.com.",
    "Standard SLA for account issues: respond within 4 hours, resolve within 24 hours.",
]

_TEMPLATES_1 = [
    {
        "id": "pwd_reset_sent",
        "name": "Password Reset Sent",
        "preview": "Hi {name}, I've triggered a manual password reset. Please check your inbox within 5 minutes.",
    },
    {
        "id": "verify_email",
        "name": "Verify Email Address",
        "preview": "Hi {name}, could you confirm the email address associated with your account?",
    },
]


def _grader_1(history: List[Dict], ticket: Ticket) -> float:
    """
    Score breakdown (total = 1.0):
      0.30 — ticket categorized as ACCOUNT
      0.20 — priority set (any explicit set_priority action)
      0.30 — a reply / template was sent addressing the reset
      0.20 — ticket resolved or a clear resolution path given
    """
    score = 0.0

    # Categorization
    if ticket.category == TicketCategory.ACCOUNT:
        score += 0.30

    # Priority acknowledged
    if _has_action(history, ActionType.SET_PRIORITY):
        score += 0.20

    # Reply quality — should mention reset / email / password
    replies = _reply_texts(history)
    kw_hits = _contains_keywords(replies, ["reset", "password", "email", "inbox", "link"])
    score += min(0.30, 0.10 * kw_hits)

    # Resolution
    if ticket.status in (TicketStatus.RESOLVED,) or _has_action(history, ActionType.RESOLVE):
        score += 0.20

    return round(min(score, 1.0), 4)


TASK_1 = {
    "name": "password-reset-easy",
    "difficulty": "easy",
    "ticket": _TICKET_1,
    "instructions": (
        "A customer cannot log in because their password reset email never arrived. "
        "Your objectives:\n"
        "1. Categorize the ticket correctly.\n"
        "2. Set an appropriate priority.\n"
        "3. Reply to the customer with actionable next steps (trigger a manual reset or guide them).\n"
        "4. Resolve the ticket once the customer issue is addressed.\n"
        "You have access to templates and knowledge-base snippets. Use them."
    ),
    "kb_snippets": _KB_1,
    "available_templates": _TEMPLATES_1,
    "max_steps": 6,
    "grader": _grader_1,
}


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# Billing dispute with partial refund logic
# ---------------------------------------------------------------------------

_TICKET_2 = Ticket(
    ticket_id="TKT-002",
    subject="Charged twice for my subscription this month",
    customer_name="Bob Martinez",
    customer_email="bob.m@techcorp.io",
    priority=TicketPriority.MEDIUM,
    category=TicketCategory.BILLING,
    status=TicketStatus.OPEN,
    conversation=[
        CustomerMessage(
            sender="customer",
            content=(
                "Hello, I was charged $49.99 twice on April 3rd for my Pro subscription. "
                "Order IDs: ORD-9921 and ORD-9922. This is clearly a duplicate charge. "
                "I need a refund ASAP. This is unacceptable."
            ),
            timestamp=_now(),
        )
    ],
    metadata={
        "account_tier": "pro",
        "orders": [
            {"id": "ORD-9921", "amount": 49.99, "date": "2026-04-03", "status": "charged"},
            {"id": "ORD-9922", "amount": 49.99, "date": "2026-04-03", "status": "charged"},
        ],
        "previous_refunds": 0,
    },
)

_KB_2 = [
    "Duplicate charges: if two identical amounts hit on the same day, classify as billing error.",
    "Refund policy: full refund for duplicate charges within 30 days. Partial refunds require manager approval.",
    "To initiate refund: collect order ID, log in to Billing Portal > Refunds > New Refund.",
    "Always acknowledge the inconvenience before discussing resolution on billing disputes.",
    "Escalate if: refund > $200, customer tier is Enterprise, or more than 2 disputes in 90 days.",
]

_TEMPLATES_2 = [
    {
        "id": "billing_apology",
        "name": "Billing Apology",
        "preview": "Hi {name}, I sincerely apologize for the duplicate charge. I'm investigating this right now.",
    },
    {
        "id": "refund_initiated",
        "name": "Refund Initiated",
        "preview": "Hi {name}, I've initiated a refund of ${amount} for order {order_id}. It will appear in 3–5 business days.",
    },
    {
        "id": "request_order_details",
        "name": "Request Order Details",
        "preview": "Hi {name}, could you provide the order IDs and the charge amounts from your statement?",
    },
]


def _grader_2(history: List[Dict], ticket: Ticket) -> float:
    """
    Score breakdown (total = 1.0):
      0.15 — acknowledged the customer's frustration (apology / empathy in reply)
      0.25 — correctly identified it as BILLING / REFUND category
      0.25 — mentioned both order IDs or the specific refund amount in a reply
      0.20 — initiated refund or escalated with reason
      0.15 — resolved ticket or set status appropriately
    """
    score = 0.0

    replies = _reply_texts(history)

    # Empathy
    empathy_kws = ["apologize", "sorry", "inconvenience", "understand", "frustration"]
    if _contains_keywords(replies, empathy_kws) >= 1:
        score += 0.15

    # Correct category
    if ticket.category in (TicketCategory.BILLING, TicketCategory.REFUND):
        score += 0.25

    # Specificity — order IDs or amount mentioned
    specificity_kws = ["ORD-9921", "ORD-9922", "49.99", "duplicate", "refund"]
    hits = _contains_keywords(replies, specificity_kws)
    score += min(0.25, 0.05 * hits)

    # Action taken
    if _has_action(history, ActionType.ESCALATE) or _has_action(history, ActionType.RESOLVE):
        score += 0.20

    # Resolution
    if ticket.status == TicketStatus.RESOLVED or _has_action(history, ActionType.RESOLVE):
        score += 0.15

    return round(min(score, 1.0), 4)


TASK_2 = {
    "name": "billing-dispute-medium",
    "difficulty": "medium",
    "ticket": _TICKET_2,
    "instructions": (
        "A Pro-tier customer reports a duplicate charge of $49.99 on their subscription. "
        "They have provided two order IDs. Your objectives:\n"
        "1. Acknowledge their frustration empathetically.\n"
        "2. Categorize the ticket correctly (billing or refund).\n"
        "3. Reference the specific order IDs in your response.\n"
        "4. Initiate a refund or escalate if policy requires it.\n"
        "5. Resolve the ticket with a clear summary.\n"
        "Use the billing knowledge base and templates provided."
    ),
    "kb_snippets": _KB_2,
    "available_templates": _TEMPLATES_2,
    "max_steps": 8,
    "grader": _grader_2,
}


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# Multi-issue enterprise escalation (technical + billing + SLA breach risk)
# ---------------------------------------------------------------------------

_TICKET_3 = Ticket(
    ticket_id="TKT-003",
    subject="URGENT: API down, data loss, and billing overcharge — threatening churn",
    customer_name="Carol Wei",
    customer_email="carol.wei@bigenterprise.com",
    priority=TicketPriority.CRITICAL,
    category=TicketCategory.TECHNICAL,
    status=TicketStatus.OPEN,
    conversation=[
        CustomerMessage(
            sender="customer",
            content=(
                "This is completely unacceptable. Our production API integration has been "
                "returning 503 errors since 09:00 UTC today (April 8). We process 50,000 "
                "transactions/hour through your API. We also noticed our March invoice was "
                "$3,200 — our contract cap is $2,500. And last week's outage deleted 3 hours "
                "of our event logs. If this isn't escalated to engineering immediately, "
                "we are cancelling our Enterprise contract ($180,000/year)."
            ),
            timestamp=_now(),
        )
    ],
    metadata={
        "account_tier": "enterprise",
        "contract_value": 180000,
        "sla_tier": "platinum",
        "sla_response_minutes": 15,
        "open_incidents": ["INC-4421 (API 503)", "INC-4398 (data loss)"],
        "billing_overage": {"invoiced": 3200, "cap": 2500, "overage": 700},
        "churn_risk": "critical",
    },
)

_KB_3 = [
    "Platinum SLA: 15-minute initial response, 2-hour resolution target, 24/7 on-call engineering.",
    "Enterprise billing overages: any charge exceeding contract cap requires VP approval and credit.",
    "Data loss incidents: must be escalated to engineering AND legal within 1 hour.",
    "Churn risk > $100k ARR: immediately loop in Customer Success Manager (CSM) and VP of Sales.",
    "API 503 errors: check status.ourservice.com; if systemic, reference INC number in comms.",
    "When multiple issues exist, triage by severity: data loss > API outage > billing > general.",
    "Never promise specific resolution times unless confirmed by engineering.",
    "Compensation policy: SLA breach credits up to 10% of monthly fee per incident.",
]

_TEMPLATES_3 = [
    {
        "id": "enterprise_critical",
        "name": "Enterprise Critical Acknowledgment",
        "preview": "Hi {name}, I've escalated this to our on-call engineering and Customer Success team immediately.",
    },
    {
        "id": "billing_overage_credit",
        "name": "Billing Overage Credit",
        "preview": "Hi {name}, I've flagged the billing overage for VP approval. A credit will be applied.",
    },
    {
        "id": "data_loss_escalation",
        "name": "Data Loss Escalation",
        "preview": "Hi {name}, the data loss incident has been escalated to engineering and our legal team.",
    },
]


def _grader_3(history: List[Dict], ticket: Ticket) -> float:
    """
    Score breakdown (total = 1.0):
      0.20 — escalated to engineering (ESCALATE action used)
      0.15 — addressed all 3 issues in replies (API + billing + data loss)
      0.15 — mentioned SLA / compensation / credit
      0.15 — mentioned churn risk / retention / CSM / VP
      0.15 — correct prioritization (CRITICAL priority set)
      0.10 — professional, empathetic tone (apology + action language)
      0.10 — resolved or set clear next steps
    """
    score = 0.0
    replies = _reply_texts(history)

    # Escalation
    if _has_action(history, ActionType.ESCALATE):
        score += 0.20

    # All 3 issues addressed
    api_kws = ["503", "api", "incident", "outage", "INC-4421"]
    billing_kws = ["3200", "2500", "overage", "invoice", "cap", "credit"]
    data_kws = ["data loss", "logs", "INC-4398", "event logs"]
    issues_addressed = 0
    if _contains_keywords(replies, api_kws) >= 1:
        issues_addressed += 1
    if _contains_keywords(replies, billing_kws) >= 1:
        issues_addressed += 1
    if _contains_keywords(replies, data_kws) >= 1:
        issues_addressed += 1
    score += issues_addressed * 0.05  # 0.05 per issue, max 0.15

    # SLA / compensation
    sla_kws = ["sla", "credit", "compensation", "breach", "platinum", "10%"]
    if _contains_keywords(replies, sla_kws) >= 1:
        score += 0.15

    # Retention / churn
    retention_kws = ["csm", "customer success", "vp", "retain", "priority", "contract", "escalat"]
    if _contains_keywords(replies, retention_kws) >= 1:
        score += 0.15

    # Priority
    if ticket.priority == TicketPriority.CRITICAL or _has_action(history, ActionType.SET_PRIORITY):
        score += 0.15

    # Tone
    tone_kws = ["apologize", "sorry", "immediately", "urgently", "understand", "priority"]
    if _contains_keywords(replies, tone_kws) >= 2:
        score += 0.10

    # Resolution / next steps
    if _has_action(history, ActionType.RESOLVE) or ticket.status in (TicketStatus.ESCALATED, TicketStatus.RESOLVED):
        score += 0.10

    return round(min(score, 1.0), 4)


TASK_3 = {
    "name": "enterprise-escalation-hard",
    "difficulty": "hard",
    "ticket": _TICKET_3,
    "instructions": (
        "A critical enterprise customer ($180k/year contract) has reported three simultaneous issues: "
        "(1) API 503 errors since 09:00 UTC, (2) a billing overage of $700 above their contract cap, "
        "(3) data loss from last week's outage. They are threatening to cancel. Your objectives:\n"
        "1. Immediately escalate to engineering.\n"
        "2. Address ALL three issues in your reply — API, billing, and data loss.\n"
        "3. Reference SLA commitments and compensation policy.\n"
        "4. Loop in Customer Success / VP for churn risk.\n"
        "5. Set CRITICAL priority.\n"
        "6. Maintain professional, empathetic tone.\n"
        "7. Provide clear next steps and resolve or escalate appropriately.\n"
        "This task requires multi-issue triage and careful prioritization."
    ),
    "kb_snippets": _KB_3,
    "available_templates": _TEMPLATES_3,
    "max_steps": 10,
    "grader": _grader_3,
}


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {
    "password-reset-easy": TASK_1,
    "billing-dispute-medium": TASK_2,
    "enterprise-escalation-hard": TASK_3,
}
