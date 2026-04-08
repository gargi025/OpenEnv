"""
tasks.py — Five tasks for the Customer Support environment.

GRADER DESIGN PRINCIPLES (critical for judging):
  1. Graders score AGENT ACTIONS in the history, never the initial ticket state.
     - The ticket starts with a category/priority pre-set by the system.
     - Agents must EXPLICITLY perform a CATEGORIZE or SET_PRIORITY action to earn those points.
     - This prevents trivially high scores on empty episodes.
  2. Empty history always scores 0.0.
  3. Scores are deterministic and reproducible.
  4. Partial credit for partial completion — never binary.
  5. Efficiency bonus: agents that resolve in fewer steps than max score higher.

Grader signature: (history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]
  - float: score in [0.0, 1.0]
  - Dict[str, bool]: progress checklist (transparent sub-scores for learning signal)
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ActionType, CustomerMessage, CustomerSentiment,
    SLAConfig, SupportAction, Ticket,
    TicketCategory, TicketPriority, TicketStatus,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Grader utilities — all operate on HISTORY only, not initial ticket state
# ---------------------------------------------------------------------------

def _action_texts(history: List[Dict]) -> List[str]:
    """Collect all text the agent produced (replies, notes, compensation messages)."""
    texts = []
    for h in history:
        a = h["action"]
        if a.reply_text:
            texts.append(a.reply_text)
        if a.note_text:
            texts.append(a.note_text)
        if a.escalation_reason:
            texts.append(a.escalation_reason)
        if a.resolution_summary:
            texts.append(a.resolution_summary)
    return texts


def _kw(texts: List[str], keywords: List[str]) -> int:
    """Count distinct keywords found in combined agent text."""
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined)


def _agent_did(history: List[Dict], action_type: ActionType) -> bool:
    """True if agent explicitly performed this action type."""
    return any(h["action"].action_type == action_type for h in history)


def _agent_categorized_as(history: List[Dict], category: TicketCategory) -> bool:
    """True if agent explicitly set this category via CATEGORIZE action."""
    return any(
        h["action"].action_type == ActionType.CATEGORIZE
        and h["action"].category == category
        for h in history
    )


def _agent_categorized_as_any(history: List[Dict], categories: List[TicketCategory]) -> bool:
    return any(_agent_categorized_as(history, c) for c in categories)


def _agent_set_priority(history: List[Dict]) -> bool:
    return any(h["action"].action_type == ActionType.SET_PRIORITY for h in history)


def _efficiency_bonus(steps_used: int, max_steps: int, ideal_steps: int) -> float:
    """Reward faster resolution. 0.12 at ideal, 0.0 at max."""
    if steps_used == 0:
        return 0.0
    if steps_used <= ideal_steps:
        return 0.12
    ratio = (max_steps - steps_used) / max(max_steps - ideal_steps, 1)
    return round(max(0.0, 0.12 * ratio), 4)


# =============================================================================
# TASK 1 — EASY: Password Reset
# =============================================================================

_T1 = Ticket(
    ticket_id="TKT-1001",
    subject="Can't log in — reset email never arrived",
    customer_name="Alice Johnson",
    customer_email="alice.johnson@example.com",
    priority=TicketPriority.LOW,
    category=TicketCategory.ACCOUNT,
    status=TicketStatus.OPEN,
    sentiment=CustomerSentiment.FRUSTRATED,
    conversation=[CustomerMessage(
        sender="customer",
        content=(
            "Hi, I forgot my password and the reset email never arrived. "
            "I've checked spam too. My account email is alice.johnson@example.com. Please help!"
        ),
        timestamp=_now(),
    )],
    metadata={"account_verified": True, "reset_attempts": 2, "account_age_days": 412},
    sla=SLAConfig(tier="standard", warn_step=2, breach_step=5, breach_penalty=0.10),
    tags=["account", "login"],
)

_KB_1 = [
    "Manual password reset: Admin > Users > search email > click 'Force Reset'. Completes in ~2 min.",
    "If reset emails fail: check domain allowlist — corporate firewalls often block automated emails.",
    "Standard SLA: first response within 4h, full resolution within 24h.",
    "Always confirm the customer's email address before triggering a reset.",
    "After reset, advise customer to check inbox AND spam within 5 minutes.",
]

_TEMPLATES_1 = [
    {"id": "pwd_reset_sent", "name": "Password Reset Sent",
     "preview": "Hi {name}, I've triggered a manual password reset to {email}. Check inbox + spam within 5 min."},
    {"id": "verify_email", "name": "Verify Email",
     "preview": "Hi {name}, could you confirm the email on your account so I can trigger the reset?"},
    {"id": "close_confirmed", "name": "Closing Confirmed",
     "preview": "Hi {name}, glad to hear you're back in! Let me know if you need anything else."},
]

_FOLLOWUPS_1 = {
    ActionType.REPLY: "Thank you! The reset email just arrived — I'm back in now!",
    ActionType.APPLY_TEMPLATE: "Got the reset email, all working now. Thank you so much!",
    ActionType.REQUEST_INFO: "Yes, my email is alice.johnson@example.com. That's the account email.",
    ActionType.RESOLVE: "Great, thanks for the help!",
    ActionType.ESCALATE: "OK, I'll wait for the specialist. Hope this gets sorted soon.",
}


def _grader_1(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Scores AGENT ACTIONS only. Empty history = 0.0.
    Breakdown:
      0.25 — agent explicitly CATEGORIZE → account
      0.15 — agent explicitly SET_PRIORITY
      0.30 — agent reply/template mentioning reset-specific keywords (max 3 keywords × 0.10)
      0.18 — agent RESOLVE or ESCALATE action
      0.12 — efficiency bonus (ideal ≤ 3 steps)
    """
    if not history:
        return 0.0, {k: False for k in [
            "categorized_as_account", "priority_set", "reply_addressed_reset",
            "ticket_closed", "efficient_resolution",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    # Agent explicitly categorized as ACCOUNT
    hints["categorized_as_account"] = _agent_categorized_as(history, TicketCategory.ACCOUNT)
    if hints["categorized_as_account"]:
        score += 0.25

    # Agent explicitly set priority
    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.15

    # Reply quality
    kw_hits = _kw(texts, ["reset", "password", "email", "inbox", "spam", "link", "force reset", "manual"])
    reply_score = min(0.30, kw_hits * 0.10)
    score += reply_score
    hints["reply_addressed_reset"] = reply_score >= 0.20

    # Ticket closed
    hints["ticket_closed"] = (
        _agent_did(history, ActionType.RESOLVE) or
        _agent_did(history, ActionType.ESCALATE)
    )
    if hints["ticket_closed"]:
        score += 0.18

    # Efficiency
    eff = _efficiency_bonus(len(history), 6, 3)
    score += eff
    hints["efficient_resolution"] = eff >= 0.08

    return round(min(score, 1.0), 4), hints


TASK_1 = {
    "name": "password-reset-easy",
    "difficulty": "easy",
    "ticket": _T1,
    "instructions": (
        "A verified customer cannot log in — their password reset email never arrived.\n"
        "Objectives:\n"
        "1. Explicitly CATEGORIZE the ticket as 'account'.\n"
        "2. Explicitly SET_PRIORITY to an appropriate level.\n"
        "3. REPLY with actionable steps — trigger a manual reset, advise checking spam.\n"
        "4. RESOLVE the ticket once addressed.\n"
        "Efficiency matters: resolving in fewer steps earns a higher score. "
        "Use the knowledge base and templates."
    ),
    "kb_snippets": _KB_1,
    "available_templates": _TEMPLATES_1,
    "followup_responses": _FOLLOWUPS_1,
    "max_steps": 6,
    "ideal_steps": 3,
    "grader": _grader_1,
}


# =============================================================================
# TASK 2 — MEDIUM: Billing Dispute + Goodwill
# =============================================================================

_T2 = Ticket(
    ticket_id="TKT-1002",
    subject="Charged twice for Pro subscription — need refund",
    customer_name="Bob Martinez",
    customer_email="bob.m@techcorp.io",
    priority=TicketPriority.MEDIUM,
    category=TicketCategory.BILLING,
    status=TicketStatus.OPEN,
    sentiment=CustomerSentiment.FRUSTRATED,
    conversation=[CustomerMessage(
        sender="customer",
        content=(
            "Hello, I was charged $49.99 twice on April 3rd for my Pro subscription. "
            "Order IDs: ORD-9921 and ORD-9922. This is clearly a duplicate charge. "
            "I need a refund ASAP. This is unacceptable."
        ),
        timestamp=_now(),
    )],
    metadata={
        "account_tier": "pro",
        "orders": [
            {"id": "ORD-9921", "amount": 49.99, "date": "2026-04-03", "status": "charged"},
            {"id": "ORD-9922", "amount": 49.99, "date": "2026-04-03", "status": "charged"},
        ],
        "previous_refunds": 0,
        "customer_since": "2024-01-15",
    },
    sla=SLAConfig(tier="pro", warn_step=2, breach_step=7, breach_penalty=0.12),
    tags=["billing", "duplicate-charge"],
)

_KB_2 = [
    "Duplicate charges: two identical amounts on the same day = billing error. Full refund within 30 days.",
    "Refund process: Billing Portal > Refunds > New Refund > enter Order ID.",
    "Always acknowledge inconvenience BEFORE discussing resolution on billing disputes.",
    "Escalate if: refund > $200, Enterprise tier, or 3+ disputes in 90 days.",
    "Compensation policy: offer $10 account credit for billing errors as goodwill.",
    "Refunds appear on statement within 3–5 business days.",
]

_TEMPLATES_2 = [
    {"id": "billing_apology", "name": "Billing Apology",
     "preview": "Hi {name}, I sincerely apologize for the duplicate charge on ORD-9921 and ORD-9922."},
    {"id": "refund_initiated", "name": "Refund Initiated",
     "preview": "Hi {name}, I've initiated a $49.99 refund for {order_id}. It appears in 3–5 business days."},
    {"id": "goodwill_credit", "name": "Goodwill Credit",
     "preview": "Hi {name}, I've also added a $10 account credit as an apology for this error."},
]

_FOLLOWUPS_2 = {
    ActionType.REPLY: "Thank you for looking into this. When will I see the refund?",
    ActionType.APPLY_TEMPLATE: "Really appreciate the fast response and the credit!",
    ActionType.OFFER_COMPENSATION: "Oh, the $10 credit is a nice touch — thank you for making this right.",
    ActionType.REQUEST_INFO: "The orders are ORD-9921 and ORD-9922, both $49.99 on April 3rd.",
    ActionType.RESOLVE: "Great, I'll watch for the refund. Thanks!",
    ActionType.ESCALATE: "OK, thanks for escalating. I hope this gets sorted quickly.",
}


def _grader_2(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.15 — empathy in reply (apology keywords)
      0.20 — agent explicitly CATEGORIZE → billing or refund
      0.20 — both order IDs referenced in agent text
      0.15 — agent RESOLVE or ESCALATE
      0.15 — refund process mentioned OR OFFER_COMPENSATION action
      0.15 — agent explicitly SET_PRIORITY
    """
    if not history:
        return 0.0, {k: False for k in [
            "empathy_shown", "categorized_correctly", "order_ids_referenced",
            "ticket_closed", "refund_actioned", "priority_set",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["empathy_shown"] = _kw(texts, ["apologize", "sorry", "inconvenience", "understand", "frustrat", "apolog"]) >= 1
    if hints["empathy_shown"]:
        score += 0.15

    hints["categorized_correctly"] = _agent_categorized_as_any(history, [TicketCategory.BILLING, TicketCategory.REFUND])
    if hints["categorized_correctly"]:
        score += 0.20

    ord_hits = _kw(texts, ["ORD-9921", "ORD-9922"])
    hints["order_ids_referenced"] = ord_hits >= 2
    if hints["order_ids_referenced"]:
        score += 0.20
    elif ord_hits == 1:
        score += 0.08  # partial credit

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.15

    refund_in_text = _kw(texts, ["refund", "initiated", "processed", "portal", "49.99"]) >= 1
    compensation_action = _agent_did(history, ActionType.OFFER_COMPENSATION)
    hints["refund_actioned"] = refund_in_text or compensation_action
    if hints["refund_actioned"]:
        score += 0.15

    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.15

    return round(min(score, 1.0), 4), hints


TASK_2 = {
    "name": "billing-dispute-medium",
    "difficulty": "medium",
    "ticket": _T2,
    "instructions": (
        "A Pro-tier customer was charged $49.99 twice on the same day (ORD-9921, ORD-9922).\n"
        "Objectives:\n"
        "1. Start with empathy — acknowledge frustration before anything else.\n"
        "2. CATEGORIZE as 'billing' or 'refund'.\n"
        "3. SET_PRIORITY appropriately.\n"
        "4. Reference BOTH order IDs (ORD-9921 and ORD-9922) in your reply.\n"
        "5. Initiate the refund (mention it in reply) or ESCALATE with a reason.\n"
        "6. Offer a $10 goodwill credit using OFFER_COMPENSATION.\n"
        "7. RESOLVE the ticket.\n"
        "Policy: full refund for same-day duplicate charges. $10 credit for billing errors."
    ),
    "kb_snippets": _KB_2,
    "available_templates": _TEMPLATES_2,
    "followup_responses": _FOLLOWUPS_2,
    "max_steps": 8,
    "ideal_steps": 4,
    "grader": _grader_2,
}


# =============================================================================
# TASK 3 — HARD: Enterprise Multi-Issue Escalation (API + billing + data loss)
# =============================================================================

_T3 = Ticket(
    ticket_id="TKT-1003",
    subject="URGENT: API down + data loss + billing overcharge — threatening churn",
    customer_name="Carol Wei",
    customer_email="carol.wei@bigenterprise.com",
    priority=TicketPriority.CRITICAL,
    category=TicketCategory.TECHNICAL,
    status=TicketStatus.OPEN,
    sentiment=CustomerSentiment.ANGRY,
    conversation=[CustomerMessage(
        sender="customer",
        content=(
            "This is completely unacceptable. Our production API has been returning 503s "
            "since 09:00 UTC today. We process 50,000 transactions/hour. "
            "Our March invoice was $3,200 — our contract cap is $2,500 (overage: $700). "
            "Last week's outage also deleted 3 hours of event logs (INC-4398). "
            "If not escalated to engineering immediately, we cancel our $180k/year contract."
        ),
        timestamp=_now(),
    )],
    metadata={
        "account_tier": "enterprise",
        "contract_value": 180000,
        "sla_tier": "platinum",
        "open_incidents": ["INC-4421 (API 503)", "INC-4398 (data loss)"],
        "billing_overage": {"invoiced": 3200, "cap": 2500, "overage": 700},
        "churn_risk": "critical",
    },
    sla=SLAConfig(tier="platinum", warn_step=1, breach_step=4, breach_penalty=0.20),
    tags=["enterprise", "churn-risk", "api", "data-loss", "billing"],
)

_KB_3 = [
    "Platinum SLA: 15-min first response, 2-hour resolution. 24/7 on-call engineering available.",
    "Enterprise billing overages: any invoice exceeding contract cap requires VP Finance approval + auto-credit.",
    "Data loss (INC-4398): escalate to engineering AND legal within 1 hour — mandatory.",
    "Churn risk >$100k ARR: immediately loop in CSM and VP of Sales.",
    "API 503 (INC-4421): check status page, reference INC number in all comms.",
    "Multi-issue triage order: data loss > API outage > billing > general.",
    "Never promise specific resolution ETAs without engineering confirmation.",
    "SLA breach compensation: up to 10% monthly fee credit per incident.",
    "Offer executive sync call with engineering leadership for enterprise churn risk.",
]

_TEMPLATES_3 = [
    {"id": "enterprise_ack", "name": "Enterprise Critical Acknowledgment",
     "preview": "Hi Carol, I've escalated INC-4421 and INC-4398 to on-call engineering and your CSM immediately."},
    {"id": "billing_credit", "name": "Billing Overage Credit",
     "preview": "Hi Carol, the $700 overage has been flagged for VP approval — a credit will be applied."},
    {"id": "exec_sync", "name": "Executive Sync Offer",
     "preview": "Hi Carol, I'm arranging a call with our VP of Engineering within 2 hours. Would 11:00 UTC work?"},
]

_FOLLOWUPS_3 = {
    ActionType.ESCALATE: "Thank you. The 503s are still happening — we're losing revenue every minute.",
    ActionType.REPLY: "OK. What is the ETA on the API fix? We need updates every 30 minutes.",
    ActionType.APPLY_TEMPLATE: "Acknowledged. Please escalate the data loss issue to your legal team too.",
    ActionType.OFFER_COMPENSATION: "The credit is noted, but getting the API fixed is the priority right now.",
    ActionType.RESOLVE: "Alright. We'll evaluate how this was handled before deciding on contract renewal.",
}


def _grader_3(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.20 — ESCALATE action performed (non-negotiable for enterprise)
      0.15 — all 3 issues referenced in agent text (API + billing + data loss)
      0.15 — SLA/compensation referenced
      0.15 — churn/retention handled (CSM, VP, exec sync)
      0.10 — agent SET_PRIORITY to critical
      0.10 — empathetic tone in reply
      0.15 — RESOLVE or ESCALATE with reason (closing action)
    """
    if not history:
        return 0.0, {k: False for k in [
            "escalated", "all_issues_addressed", "sla_referenced",
            "retention_handled", "priority_set_critical", "empathy_shown", "ticket_closed",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["escalated"] = _agent_did(history, ActionType.ESCALATE)
    if hints["escalated"]:
        score += 0.20

    api_hit = _kw(texts, ["503", "INC-4421", "api", "outage", "transactions"]) >= 1
    bill_hit = _kw(texts, ["3200", "2500", "700", "overage", "invoice", "cap", "credit"]) >= 1
    data_hit = _kw(texts, ["data loss", "INC-4398", "event logs", "logs"]) >= 1
    issues = sum([api_hit, bill_hit, data_hit])
    hints["all_issues_addressed"] = issues == 3
    score += issues * 0.05  # 0.05 per issue, max 0.15

    hints["sla_referenced"] = _kw(texts, ["sla", "platinum", "15 min", "credit", "compensation", "10%", "breach"]) >= 1
    if hints["sla_referenced"]:
        score += 0.15

    hints["retention_handled"] = _kw(texts, ["csm", "customer success", "vp", "executive", "sync", "180k", "contract", "retain"]) >= 1
    if hints["retention_handled"]:
        score += 0.15

    # Must be an explicit SET_PRIORITY to CRITICAL
    hints["priority_set_critical"] = any(
        h["action"].action_type == ActionType.SET_PRIORITY and
        h["action"].priority == TicketPriority.CRITICAL
        for h in history
    )
    if hints["priority_set_critical"]:
        score += 0.10

    hints["empathy_shown"] = _kw(texts, ["apologize", "sorry", "understand", "immediately", "urgently"]) >= 1
    if hints["empathy_shown"]:
        score += 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.15

    return round(min(score, 1.0), 4), hints


TASK_3 = {
    "name": "enterprise-escalation-hard",
    "difficulty": "hard",
    "ticket": _T3,
    "instructions": (
        "CRITICAL enterprise customer ($180k/year) — 3 simultaneous issues.\n"
        "Triage order: data loss (INC-4398) > API outage (INC-4421) > billing overage ($700).\n"
        "Objectives:\n"
        "1. ESCALATE immediately — this is non-negotiable for Platinum SLA.\n"
        "2. SET_PRIORITY to 'critical'.\n"
        "3. Address ALL THREE issues in your reply with specific references (INC numbers, amounts).\n"
        "4. Reference SLA commitments and compensation policy (10% credit per breach).\n"
        "5. Loop in CSM and VP of Sales for churn risk — reference the $180k contract.\n"
        "6. Offer an executive sync call with engineering leadership.\n"
        "7. RESOLVE or close with a clear escalation summary.\n"
        "Warning: SLA breach penalty is heavy (0.20). First action must be ESCALATE."
    ),
    "kb_snippets": _KB_3,
    "available_templates": _TEMPLATES_3,
    "followup_responses": _FOLLOWUPS_3,
    "max_steps": 10,
    "ideal_steps": 5,
    "grader": _grader_3,
}


# =============================================================================
# TASK 4 — EXPERT: Security Incident + Compliance
# =============================================================================

_T4 = Ticket(
    ticket_id="TKT-1004",
    subject="Suspicious login from Russia — possible breach of financial data",
    customer_name="David Kim",
    customer_email="d.kim@fintech-startup.com",
    priority=TicketPriority.HIGH,
    category=TicketCategory.SECURITY,
    status=TicketStatus.OPEN,
    sentiment=CustomerSentiment.FRUSTRATED,
    conversation=[CustomerMessage(
        sender="customer",
        content=(
            "We received an alert: admin login from IP 185.220.101.34 (Russia) at 03:14 UTC. "
            "None of our team is in Russia. We handle SOC2 and PCI-DSS financial data. "
            "We may be breached. What data was accessed? We may need to notify our customers."
        ),
        timestamp=_now(),
    )],
    metadata={
        "account_tier": "business",
        "compliance": ["SOC2", "PCI-DSS"],
        "suspicious_ip": "185.220.101.34",
        "login_utc": "2026-04-08T03:14:00Z",
        "data_types": ["financial_records", "customer_pii"],
        "active_sessions": 1,
        "mfa_enabled": False,
        "fraud_score": 0.85,   # IP is known Tor exit node
    },
    sla=SLAConfig(tier="business", warn_step=1, breach_step=4, breach_penalty=0.18),
    tags=["security", "potential-breach", "compliance", "soc2"],
)

_KB_4 = [
    "Suspected breach: (1) lock account immediately, (2) revoke all sessions, (3) escalate to security@.",
    "SOC2/PCI-DSS: suspected unauthorized access to PII/financial data may trigger mandatory breach notification (GDPR: 72h, various US state laws: immediate).",
    "DO NOT confirm or deny data exfiltration until security team completes forensic review.",
    "IP 185.220.x.x range: known Tor exit nodes, commonly used in credential stuffing attacks.",
    "Advise customer to enable MFA immediately.",
    "Data loss/breach escalation: security team AND legal within 1 hour.",
    "Never share raw server logs or full session data with the customer.",
    "Open a formal security incident (SEC-XXXX) and provide the reference number.",
]

_TEMPLATES_4 = [
    {"id": "account_locked", "name": "Account Lockdown",
     "preview": "Hi David, I've locked your account and revoked all active sessions immediately. Security team alerted."},
    {"id": "mfa_advisory", "name": "MFA Advisory",
     "preview": "Hi David, please enable MFA immediately at Settings > Security > Two-Factor Authentication."},
    {"id": "incident_opened", "name": "Security Incident Opened",
     "preview": "Hi David, I've opened security incident SEC-{id}. Forensic review within 4 hours."},
    {"id": "compliance_notice", "name": "Compliance Advisory",
     "preview": "Hi David, our compliance team will advise on SOC2/PCI-DSS breach notification obligations."},
]

_FOLLOWUPS_4 = {
    ActionType.ESCALATE: "Thank you. Was our customer PII accessed? We may have a 72-hour GDPR clock running.",
    ActionType.REPLY: "Understood. When will we have the forensic report? Our legal team needs it.",
    ActionType.APPLY_TEMPLATE: "We're enabling MFA now. Please keep us posted on the investigation.",
    ActionType.REQUEST_INFO: "No, it was definitely not our team — all 3 members are in the US.",
    ActionType.RESOLVE: "OK. Please send the full incident report to legal@fintech-startup.com.",
}


def _grader_4(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.20 — account lockdown actioned (ESCALATE or lock/revoke keywords)
      0.15 — security team alerted (escalation or incident reference)
      0.10 — MFA advice given
      0.15 — compliance/regulatory addressed (SOC2, GDPR, 72h, notification)
      0.10 — forensic protocol cited (not prematurely confirming breach)
      0.10 — legal team mentioned
      0.10 — did NOT prematurely confirm breach (negative: agent said 'your data was breached')
      0.10 — RESOLVE or ESCALATE action
    """
    if not history:
        return 0.0, {k: False for k in [
            "account_locked", "security_alerted", "mfa_advised",
            "compliance_addressed", "forensic_protocol", "legal_mentioned",
            "no_premature_confirmation", "ticket_closed",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["account_locked"] = (
        _agent_did(history, ActionType.ESCALATE) or
        _kw(texts, ["lock", "revoke", "sessions", "suspend", "disable", "locked"]) >= 1
    )
    if hints["account_locked"]:
        score += 0.20

    hints["security_alerted"] = _kw(texts, ["security team", "security@", "escalat", "forensic", "SEC-", "incident"]) >= 1
    if hints["security_alerted"]:
        score += 0.15

    hints["mfa_advised"] = _kw(texts, ["mfa", "two-factor", "2fa", "multi-factor", "authenticator"]) >= 1
    if hints["mfa_advised"]:
        score += 0.10

    hints["compliance_addressed"] = _kw(texts, ["soc2", "pci", "gdpr", "72 hour", "72h", "breach notification", "regulatory", "compliance"]) >= 1
    if hints["compliance_addressed"]:
        score += 0.15

    hints["forensic_protocol"] = _kw(texts, ["forensic", "investigation", "review", "under investigation", "cannot confirm"]) >= 1
    if hints["forensic_protocol"]:
        score += 0.10

    hints["legal_mentioned"] = _kw(texts, ["legal", "counsel", "attorney", "notify customers", "notification", "legal team"]) >= 1
    if hints["legal_mentioned"]:
        score += 0.10

    # Penalty mechanic: agent must NOT prematurely confirm breach
    breach_confirmed_prematurely = _kw(texts, [
        "your data was breached", "confirmed breach", "data was stolen",
        "definitely breached", "breach is confirmed",
    ]) >= 1
    hints["no_premature_confirmation"] = not breach_confirmed_prematurely
    if hints["no_premature_confirmation"]:
        score += 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.10

    return round(min(score, 1.0), 4), hints


TASK_4 = {
    "name": "security-incident-expert",
    "difficulty": "expert",
    "ticket": _T4,
    "instructions": (
        "A fintech customer (SOC2 + PCI-DSS) reports a suspicious admin login from a known Tor exit node.\n"
        "Objectives:\n"
        "1. Lock the account and revoke sessions immediately (use ESCALATE or describe in reply).\n"
        "2. Alert the security team and open a formal incident (reference SEC-XXXX).\n"
        "3. Advise enabling MFA.\n"
        "4. Address their SOC2/PCI-DSS obligations — reference the GDPR 72-hour notification window.\n"
        "5. Cite forensic investigation protocol — you CANNOT confirm or deny a breach yet.\n"
        "6. Mention legal team involvement for regulatory notifications.\n"
        "7. RESOLVE or ESCALATE to close.\n"
        "CRITICAL: Do NOT say 'your data was breached' or 'confirmed breach' — "
        "premature breach confirmation loses you points and violates protocol."
    ),
    "kb_snippets": _KB_4,
    "available_templates": _TEMPLATES_4,
    "followup_responses": _FOLLOWUPS_4,
    "max_steps": 10,
    "ideal_steps": 5,
    "grader": _grader_4,
}


# =============================================================================
# TASK 5 — EXPERT: Loyal Customer Shipping Dispute + Fraud-aware Resolution
# =============================================================================

_T5 = Ticket(
    ticket_id="TKT-1005",
    subject="Order ORD-7731 shows Delivered but never arrived — want refund or reship",
    customer_name="Emma Rodriguez",
    customer_email="emma.r@gmail.com",
    priority=TicketPriority.MEDIUM,
    category=TicketCategory.SHIPPING,
    status=TicketStatus.OPEN,
    sentiment=CustomerSentiment.FRUSTRATED,
    conversation=[CustomerMessage(
        sender="customer",
        content=(
            "My order ORD-7731 shows 'Delivered' on April 5th but I never got it. "
            "I've checked with neighbors and building management — nothing. "
            "$240 order. FedEx tracking: FX-8827461. "
            "I've been a customer for 3 years and have never had a problem before. "
            "I want either a reshipment or a full refund."
        ),
        timestamp=_now(),
    )],
    metadata={
        "order_id": "ORD-7731",
        "order_value": 240.00,
        "carrier": "FedEx",
        "tracking": "FX-8827461",
        "tracking_status": "Delivered — April 5, 14:32 local",
        "delivery_photo_available": True,
        "customer_since": "2023-01-08",
        "order_history": {"total_orders": 47, "refund_requests": 0},
        "fraud_score": 0.08,   # low risk — long history, first complaint
    },
    sla=SLAConfig(tier="standard", warn_step=2, breach_step=7, breach_penalty=0.10),
    tags=["shipping", "missing-package", "loyal-customer"],
)

_KB_5 = [
    "Missing package protocol: verify tracking → check delivery photo → open carrier trace within 5 days.",
    "FedEx trace: file at 1-800-463-3339 or fedex.com/claims. Takes 5–10 business days.",
    "Loyalty policy: customers with 0 prior refund requests and 3+ years = eligible for immediate reship or refund at agent discretion (no carrier trace required for orders ≤ $300).",
    "Fraud check: fraud_score < 0.20 = low risk. Do NOT demand excessive verification.",
    "Orders > $200: reshipment requires supervisor approval OR fraud_score < 0.15 (auto-approved).",
    "Always thank long-term customers for their loyalty before discussing resolution.",
    "Goodwill: offer a 10% discount code on next order for inconvenience.",
]

_TEMPLATES_5 = [
    {"id": "loyalty_ack", "name": "Loyalty Acknowledgment",
     "preview": "Hi Emma, thank you for 3 years with us — your loyalty means the world. I'm so sorry about ORD-7731."},
    {"id": "reship_approved", "name": "Reship Approved",
     "preview": "Hi Emma, given your excellent history, I've approved a reshipment of ORD-7731 at no charge. Tracking in 24h."},
    {"id": "refund_approved", "name": "Refund Approved",
     "preview": "Hi Emma, I've approved a full refund of $240 for ORD-7731. It will appear in 3–5 business days."},
    {"id": "fedex_trace", "name": "FedEx Trace Opened",
     "preview": "Hi Emma, I've opened a FedEx trace for FX-8827461. I'll update you proactively every 2 days."},
]

_FOLLOWUPS_5 = {
    ActionType.REPLY: "Thank you! I'd prefer a reshipment if possible — I still need the items.",
    ActionType.APPLY_TEMPLATE: "Oh wow, that was fast! Really appreciate you making this right.",
    ActionType.REQUEST_INFO: "I've already checked with neighbors and building management — nothing.",
    ActionType.OFFER_COMPENSATION: "The discount is very kind, thank you! I'll definitely use it.",
    ActionType.RESOLVE: "Perfect — thank you for resolving this so quickly. I'll stay a customer!",
    ActionType.ESCALATE: "OK, I'll wait to hear back. I hope this gets sorted quickly.",
}


def _grader_5(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.15 — loyalty explicitly acknowledged in agent text
      0.10 — fraud score respected (agent did NOT demand excessive verification)
      0.15 — order ORD-7731 and/or FX-8827461 referenced specifically
      0.20 — resolution offered: reship or refund (in text or OFFER_COMPENSATION)
      0.10 — FedEx trace mentioned
      0.10 — goodwill / proactive compensation offered
      0.10 — RESOLVE or ESCALATE action
      0.10 — agent SET_PRIORITY
    """
    if not history:
        return 0.0, {k: False for k in [
            "loyalty_acknowledged", "fraud_respected", "specifics_referenced",
            "resolution_offered", "trace_mentioned", "goodwill_offered",
            "ticket_closed", "priority_set",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["loyalty_acknowledged"] = _kw(texts, ["3 year", "loyal", "valued customer", "history", "thank you for being", "appreciate"]) >= 1
    if hints["loyalty_acknowledged"]:
        score += 0.15

    # Fraud respected = agent did not demand excessive proof for a known-low-risk customer
    excessive_verification = _kw(texts, ["prove", "provide id", "identity verification", "cannot process without", "suspicious"]) >= 1
    hints["fraud_respected"] = not excessive_verification
    if hints["fraud_respected"]:
        score += 0.10

    hits = _kw(texts, ["ORD-7731", "FX-8827461", "April 5", "240"])
    hints["specifics_referenced"] = hits >= 2
    if hints["specifics_referenced"]:
        score += 0.15
    elif hits == 1:
        score += 0.06

    resolution_in_text = _kw(texts, ["reship", "refund", "replace", "$240", "reorder", "approved"]) >= 1
    hints["resolution_offered"] = resolution_in_text or _agent_did(history, ActionType.OFFER_COMPENSATION)
    if hints["resolution_offered"]:
        score += 0.20

    hints["trace_mentioned"] = _kw(texts, ["fedex", "trace", "claim", "FX-8827461", "investigation"]) >= 1
    if hints["trace_mentioned"]:
        score += 0.10

    hints["goodwill_offered"] = (
        _agent_did(history, ActionType.OFFER_COMPENSATION) or
        _kw(texts, ["credit", "discount", "voucher", "10%", "sorry", "goodwill"]) >= 1
    )
    if hints["goodwill_offered"]:
        score += 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.10

    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.10

    return round(min(score, 1.0), 4), hints


TASK_5 = {
    "name": "shipping-dispute-expert",
    "difficulty": "expert",
    "ticket": _T5,
    "instructions": (
        "A loyal 3-year customer (47 orders, 0 prior refunds, fraud_score=0.08) reports order "
        "ORD-7731 ($240) as missing despite FedEx showing delivered (FX-8827461).\n"
        "Objectives:\n"
        "1. Start by acknowledging their 3-year loyalty.\n"
        "2. Reference ORD-7731 and FX-8827461 specifically in your reply.\n"
        "3. Apply the loyalty policy: 0 prior refunds + 3yr = immediate reship or refund approved.\n"
        "4. Do NOT demand excessive verification — fraud_score=0.08 is low risk.\n"
        "5. Mention opening a FedEx trace (FX-8827461).\n"
        "6. Offer proactive goodwill (10% discount on next order).\n"
        "7. SET_PRIORITY and RESOLVE decisively.\n"
        "Avoid bureaucratic delays — this customer has earned trust."
    ),
    "kb_snippets": _KB_5,
    "available_templates": _TEMPLATES_5,
    "followup_responses": _FOLLOWUPS_5,
    "max_steps": 8,
    "ideal_steps": 4,
    "grader": _grader_5,
}


# =============================================================================
# Task registry
# =============================================================================

TASKS: Dict[str, Dict] = {
    "password-reset-easy":       TASK_1,
    "billing-dispute-medium":    TASK_2,
    "enterprise-escalation-hard": TASK_3,
    "security-incident-expert":  TASK_4,
    "shipping-dispute-expert":   TASK_5,
}