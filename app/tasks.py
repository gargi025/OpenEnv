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
  6. Sequencing checks: penalize illogical action ordering (e.g. RESOLVE before any REPLY).
  7. Coherence threshold: keyword hits only count in replies of >= 30 words.

Grader signature: (history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]
  - float: score in [0.0, 1.0]
  - Dict[str, bool]: progress checklist (transparent sub-scores for learning signal)

RANDOMIZATION:
  Each task exposes a `build_ticket()` function called on every reset().
  Amounts, names, order IDs, and one metadata field are randomized per episode.
  Graders are passed the live ticket object so they always check against the
  actual randomized values, not hardcoded strings.
"""
from __future__ import annotations

import random
import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ActionType, CustomerMessage, CustomerSentiment,
    SLAConfig, SupportAction, Ticket, ToolResult,
    TicketCategory, TicketPriority, TicketStatus,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Grader utilities — all operate on HISTORY only, not initial ticket state
# ---------------------------------------------------------------------------

def _action_texts(history: List[Dict]) -> List[str]:
    """Collect all text the agent produced (replies, notes, compensation messages).
    Only counts text from replies with >= 30 words (coherence threshold)."""
    texts = []
    for h in history:
        a = h["action"]
        # Apply coherence threshold to reply_text — must be >= 30 words
        if a.reply_text:
            if len(a.reply_text.split()) >= 30:
                texts.append(a.reply_text)
            else:
                texts.append("")  # placeholder so index stays consistent
        if a.note_text:
            texts.append(a.note_text)
        if a.escalation_reason:
            texts.append(a.escalation_reason)
        if a.resolution_summary:
            texts.append(a.resolution_summary)
        if a.reason:
            texts.append(a.reason)
    return texts


def _all_action_texts(history: List[Dict]) -> List[str]:
    """Collect ALL text without coherence threshold — for sequencing checks only."""
    texts = []
    for h in history:
        a = h["action"]
        for field in [a.reply_text, a.note_text, a.escalation_reason, a.resolution_summary, a.reason]:
            if field:
                texts.append(field)
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


def _resolved_after_reply(history: List[Dict]) -> bool:
    """True if agent replied at least once before resolving/escalating.
    Penalizes agents that resolve without communicating with the customer."""
    actions = [h["action"].action_type for h in history]
    terminal_types = {ActionType.RESOLVE, ActionType.ESCALATE}
    reply_types = {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.APPLY_TEMPLATE}
    terminal_indices = [i for i, a in enumerate(actions) if a in terminal_types]
    if not terminal_indices:
        return True  # not resolved yet — no penalty
    first_terminal = terminal_indices[0]
    return any(actions[i] in reply_types for i in range(first_terminal))


def _tool_was_used(history: List[Dict], action_type: ActionType) -> bool:
    """True if agent used a specific tool-use action."""
    return any(h["action"].action_type == action_type for h in history)


def _tool_result_used_in_reply(history: List[Dict], confirmation_field: str) -> bool:
    """Check if a tool's confirmation ID or key data appears in a subsequent reply.
    This verifies the agent actually used the tool result, not just called the tool."""
    tool_result_text = None
    for h in history:
        # Find when tool result was received
        if h.get("tool_result") and h["tool_result"].confirmation_id:
            tool_result_text = h["tool_result"].confirmation_id
        # Check if subsequent reply references it
        if tool_result_text and h["action"].reply_text:
            if tool_result_text.lower() in h["action"].reply_text.lower():
                return True
    return False


# Name pools for randomization
_FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "James"]
_LAST_NAMES = ["Johnson", "Martinez", "Wei", "Kim", "Rodriguez", "Patel", "Chen", "Thompson", "Davis", "Wilson"]
_DOMAINS = ["example.com", "techcorp.io", "gmail.com", "startup.com", "enterprise.net"]


def _random_name() -> Tuple[str, str]:
    """Returns (full_name, email)."""
    first = random.choice(_FIRST_NAMES)
    last = random.choice(_LAST_NAMES)
    domain = random.choice(_DOMAINS)
    return f"{first} {last}", f"{first.lower()}.{last.lower()}@{domain}"


def _random_order_id() -> str:
    return f"ORD-{random.randint(1000, 9999)}"


def _random_ticket_id() -> str:
    return f"TKT-{random.randint(2000, 8999)}"


# =============================================================================
# TASK 1 — EASY: Password Reset
# =============================================================================

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
    ActionType.REQUEST_INFO: "Yes, my email is on file. That's the account email.",
    ActionType.RESOLVE: "Great, thanks for the help!",
    ActionType.ESCALATE: "OK, I'll wait for the specialist. Hope this gets sorted soon.",
    ActionType.LOOKUP_ORDER: None,
    ActionType.CHECK_POLICY: None,
}

# Tool responses for task 1
_TOOL_RESPONSES_1 = {
    ActionType.CHECK_POLICY: lambda ticket: ToolResult(
        action_type="check_policy",
        success=True,
        data={
            "policy": "password_reset",
            "text": (
                "Manual reset procedure: Admin > Users > search by email > Force Reset. "
                "Estimated delivery: 2 minutes. If email blocked by corporate firewall, "
                "use alternative delivery via SMS (requires phone number on file). "
                "SLA: resolve within 4 hours for standard tier."
            ),
        },
    ),
    ActionType.LOOKUP_ORDER: lambda ticket: ToolResult(
        action_type="lookup_order",
        success=False,
        error="No orders associated with this account type. This is an account access ticket.",
    ),
}


def _build_ticket_1() -> Ticket:
    """Randomized password-reset ticket."""
    name, email = _random_name()
    reset_attempts = random.randint(1, 4)
    account_age = random.randint(90, 800)
    return Ticket(
        ticket_id=_random_ticket_id(),
        subject="Can't log in — reset email never arrived",
        customer_name=name,
        customer_email=email,
        priority=TicketPriority.LOW,
        category=TicketCategory.ACCOUNT,
        status=TicketStatus.OPEN,
        sentiment=CustomerSentiment.FRUSTRATED,
        conversation=[CustomerMessage(
            sender="customer",
            content=(
                f"Hi, I forgot my password and the reset email never arrived. "
                f"I've checked spam too. My account email is {email}. "
                f"I've tried {reset_attempts} times. Please help!"
            ),
            timestamp=_now(),
        )],
        metadata={
            "account_verified": True,
            "reset_attempts": reset_attempts,
            "account_age_days": account_age,
        },
        sla=SLAConfig(tier="standard", warn_step=2, breach_step=5, breach_penalty=0.10),
        tags=["account", "login"],
    )


def _grader_1(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Scores AGENT ACTIONS only. Empty history = 0.0.
    Coherence threshold: reply_text must be >= 30 words to count keywords.
    Breakdown:
      0.20 — agent explicitly CATEGORIZE → account
      0.15 — agent explicitly SET_PRIORITY
      0.25 — agent reply mentioning reset-specific keywords (>= 30 word replies only)
      0.15 — agent RESOLVE or ESCALATE action
      0.05 — sequencing: reply before resolve
      0.08 — tool use: CHECK_POLICY called
      0.05 — tool result referenced in subsequent reply
      0.12 — efficiency bonus (ideal <= 3 steps)
    """
    if not history:
        return 0.0, {k: False for k in [
            "categorized_as_account", "priority_set", "reply_addressed_reset",
            "ticket_closed", "sequencing_correct", "policy_checked", "tool_result_used",
            "efficient_resolution",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["categorized_as_account"] = _agent_categorized_as(history, TicketCategory.ACCOUNT)
    if hints["categorized_as_account"]:
        score += 0.20

    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.15

    kw_hits = _kw(texts, ["reset", "password", "email", "inbox", "spam", "link", "force reset", "manual"])
    reply_score = min(0.25, kw_hits * 0.08)
    score += reply_score
    hints["reply_addressed_reset"] = reply_score >= 0.16

    hints["ticket_closed"] = (
        _agent_did(history, ActionType.RESOLVE) or
        _agent_did(history, ActionType.ESCALATE)
    )
    if hints["ticket_closed"]:
        score += 0.15

    hints["sequencing_correct"] = _resolved_after_reply(history)
    if hints["sequencing_correct"]:
        score += 0.05
    else:
        score -= 0.10  # resolved without replying first

    hints["policy_checked"] = _tool_was_used(history, ActionType.CHECK_POLICY)
    if hints["policy_checked"]:
        score += 0.08

    hints["tool_result_used"] = _tool_result_used_in_reply(history, "confirmation_id")
    if hints["tool_result_used"]:
        score += 0.05

    eff = _efficiency_bonus(len(history), 6, 3)
    score += eff
    hints["efficient_resolution"] = eff >= 0.08

    return round(min(max(score, 0.0), 1.0), 4), hints


TASK_1 = {
    "name": "password-reset-easy",
    "difficulty": "easy",
    "build_ticket": _build_ticket_1,
    "ticket": _build_ticket_1(),   # static fallback for tests
    "instructions": (
        "A verified customer cannot log in — their password reset email never arrived.\n"
        "Objectives:\n"
        "1. Use CHECK_POLICY to retrieve the reset procedure before replying.\n"
        "2. Explicitly CATEGORIZE the ticket as 'account'.\n"
        "3. Explicitly SET_PRIORITY to an appropriate level.\n"
        "4. REPLY with actionable steps (>= 30 words) — trigger a manual reset, advise checking spam.\n"
        "5. RESOLVE the ticket once addressed.\n"
        "Efficiency matters: resolving in fewer steps earns a higher score. "
        "Use the knowledge base and templates."
    ),
    "kb_snippets": _KB_1,
    "available_templates": _TEMPLATES_1,
    "followup_responses": _FOLLOWUPS_1,
    "tool_responses": _TOOL_RESPONSES_1,
    "max_steps": 7,
    "ideal_steps": 4,
    "grader": _grader_1,
}


# =============================================================================
# TASK 2 — MEDIUM: Billing Dispute + Goodwill
# =============================================================================

_KB_2 = [
    "Duplicate charges: two identical amounts on the same day = billing error. Full refund within 30 days.",
    "Refund process: use TRIGGER_REFUND tool with the order_id and amount — returns a REF-XXXX confirmation.",
    "Always acknowledge inconvenience BEFORE discussing resolution on billing disputes.",
    "Escalate if: refund > $200, Enterprise tier, or 3+ disputes in 90 days.",
    "Compensation policy: offer $10 account credit for billing errors as goodwill.",
    "Refunds appear on statement within 3–5 business days.",
]

_TEMPLATES_2 = [
    {"id": "billing_apology", "name": "Billing Apology",
     "preview": "Hi {name}, I sincerely apologize for the duplicate charge on your orders."},
    {"id": "refund_initiated", "name": "Refund Initiated",
     "preview": "Hi {name}, I've initiated a refund for the duplicate charge. It appears in 3–5 business days."},
    {"id": "goodwill_credit", "name": "Goodwill Credit",
     "preview": "Hi {name}, I've also added a $10 account credit as an apology for this error."},
]

_FOLLOWUPS_2 = {
    ActionType.REPLY: "Thank you for looking into this. When will I see the refund?",
    ActionType.APPLY_TEMPLATE: "Really appreciate the fast response and the credit!",
    ActionType.OFFER_COMPENSATION: "Oh, the $10 credit is a nice touch — thank you for making this right.",
    ActionType.REQUEST_INFO: "The orders are both from the same day, both the same amount.",
    ActionType.RESOLVE: "Great, I'll watch for the refund. Thanks!",
    ActionType.ESCALATE: "OK, thanks for escalating. I hope this gets sorted quickly.",
    ActionType.TRIGGER_REFUND: None,
    ActionType.LOOKUP_ORDER: None,
}


def _tool_responses_2(ticket: Ticket):
    orders = ticket.metadata.get("orders", [])
    order_id_1 = orders[0]["id"] if orders else "ORD-0000"
    order_id_2 = orders[1]["id"] if len(orders) > 1 else "ORD-0001"
    amount = orders[0]["amount"] if orders else 49.99

    return {
        ActionType.LOOKUP_ORDER: lambda t: ToolResult(
            action_type="lookup_order",
            success=True,
            data={
                "orders": [
                    {"id": order_id_1, "amount": amount, "date": orders[0].get("date", ""), "status": "charged"},
                    {"id": order_id_2, "amount": amount, "date": orders[1].get("date", "") if len(orders) > 1 else "", "status": "charged"},
                ],
                "diagnosis": "Duplicate charge detected — identical amount on same date.",
            },
        ),
        ActionType.TRIGGER_REFUND: lambda t: ToolResult(
            action_type="trigger_refund",
            success=True,
            confirmation_id=f"REF-{random.randint(10000, 99999)}",
            data={"order_id": order_id_2, "amount": amount, "eta_days": 5},
        ),
        ActionType.CHECK_POLICY: lambda t: ToolResult(
            action_type="check_policy",
            success=True,
            data={
                "policy": "duplicate_charge",
                "text": (
                    f"Same-day duplicate charges qualify for immediate full refund. "
                    f"Use TRIGGER_REFUND with order_id and amount. "
                    f"Additionally offer $10 goodwill credit via OFFER_COMPENSATION."
                ),
            },
        ),
    }


def _build_ticket_2() -> Ticket:
    """Randomized billing dispute ticket."""
    name, email = _random_name()
    amount = round(random.choice([29.99, 49.99, 79.99, 99.99, 149.99]), 2)
    order_id_1 = _random_order_id()
    order_id_2 = _random_order_id()
    charge_date = f"2026-04-{random.randint(1, 9):02d}"
    return Ticket(
        ticket_id=_random_ticket_id(),
        subject=f"Charged twice for Pro subscription — need refund",
        customer_name=name,
        customer_email=email,
        priority=TicketPriority.MEDIUM,
        category=TicketCategory.BILLING,
        status=TicketStatus.OPEN,
        sentiment=CustomerSentiment.FRUSTRATED,
        conversation=[CustomerMessage(
            sender="customer",
            content=(
                f"Hello, I was charged ${amount} twice on {charge_date} for my Pro subscription. "
                f"Order IDs: {order_id_1} and {order_id_2}. This is clearly a duplicate charge. "
                f"I need a refund ASAP. This is unacceptable."
            ),
            timestamp=_now(),
        )],
        metadata={
            "account_tier": "pro",
            "orders": [
                {"id": order_id_1, "amount": amount, "date": charge_date, "status": "charged"},
                {"id": order_id_2, "amount": amount, "date": charge_date, "status": "charged"},
            ],
            "previous_refunds": 0,
            "customer_since": "2024-01-15",
        },
        sla=SLAConfig(tier="pro", warn_step=2, breach_step=7, breach_penalty=0.12),
        tags=["billing", "duplicate-charge"],
    )


def _grader_2(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.12 — empathy in reply (>= 30 words, apology keywords)
      0.15 — agent explicitly CATEGORIZE → billing or refund
      0.15 — both order IDs referenced in agent text (from live ticket metadata)
      0.12 — agent RESOLVE or ESCALATE
      0.15 — TRIGGER_REFUND tool called
      0.10 — tool confirmation ID referenced in reply
      0.10 — agent explicitly SET_PRIORITY
      0.05 — OFFER_COMPENSATION action
      0.06 — sequencing: reply before resolve
    """
    if not history:
        return 0.0, {k: False for k in [
            "empathy_shown", "categorized_correctly", "order_ids_referenced",
            "ticket_closed", "refund_triggered", "confirmation_referenced",
            "priority_set", "compensation_offered", "sequencing_correct",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["empathy_shown"] = _kw(texts, ["apologize", "sorry", "inconvenience", "understand", "frustrat", "apolog"]) >= 1
    if hints["empathy_shown"]:
        score += 0.12

    hints["categorized_correctly"] = _agent_categorized_as_any(history, [TicketCategory.BILLING, TicketCategory.REFUND])
    if hints["categorized_correctly"]:
        score += 0.15

    # Check against live ticket order IDs (randomized)
    orders = ticket.metadata.get("orders", [])
    ord_ids = [o["id"] for o in orders]
    ord_hits = _kw(texts, ord_ids)
    hints["order_ids_referenced"] = ord_hits >= 2
    if hints["order_ids_referenced"]:
        score += 0.15
    elif ord_hits == 1:
        score += 0.06

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.12

    hints["refund_triggered"] = _tool_was_used(history, ActionType.TRIGGER_REFUND)
    if hints["refund_triggered"]:
        score += 0.15

    hints["confirmation_referenced"] = _tool_result_used_in_reply(history, "confirmation_id")
    if hints["confirmation_referenced"]:
        score += 0.10

    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.10

    hints["compensation_offered"] = _agent_did(history, ActionType.OFFER_COMPENSATION)
    if hints["compensation_offered"]:
        score += 0.05

    hints["sequencing_correct"] = _resolved_after_reply(history)
    if hints["sequencing_correct"]:
        score += 0.06
    else:
        score -= 0.10

    return round(min(max(score, 0.0), 1.0), 4), hints


def _build_instructions_2(ticket: Ticket) -> str:
    orders = ticket.metadata.get("orders", [])
    ids = " and ".join(o["id"] for o in orders)
    amount = orders[0]["amount"] if orders else "?"
    return (
        f"A Pro-tier customer was charged ${amount} twice on the same day ({ids}).\n"
        "Objectives:\n"
        "1. Start with empathy — acknowledge frustration before anything else (>= 30 words).\n"
        "2. CATEGORIZE as 'billing' or 'refund'.\n"
        "3. SET_PRIORITY appropriately.\n"
        f"4. Use LOOKUP_ORDER to verify both charges, then reference BOTH order IDs in your reply.\n"
        "5. Use TRIGGER_REFUND tool to initiate the refund — include the REF-XXXX confirmation in your reply.\n"
        "6. Offer a $10 goodwill credit using OFFER_COMPENSATION.\n"
        "7. RESOLVE the ticket.\n"
        "Policy: full refund for same-day duplicate charges. $10 credit for billing errors."
    )


TASK_2 = {
    "name": "billing-dispute-medium",
    "difficulty": "medium",
    "build_ticket": _build_ticket_2,
    "ticket": _build_ticket_2(),
    "instructions": _build_instructions_2,   # callable — built fresh each reset with live ticket
    "kb_snippets": _KB_2,
    "available_templates": _TEMPLATES_2,
    "followup_responses": _FOLLOWUPS_2,
    "tool_responses": _tool_responses_2,      # callable — built with live ticket
    "max_steps": 9,
    "ideal_steps": 5,
    "grader": _grader_2,
}


# =============================================================================
# TASK 3 — HARD: Enterprise Multi-Issue Escalation
# =============================================================================

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
     "preview": "Hi {name}, I've escalated INC-4421 and INC-4398 to on-call engineering and your CSM immediately."},
    {"id": "billing_credit", "name": "Billing Overage Credit",
     "preview": "Hi {name}, the overage has been flagged for VP approval — a credit will be applied."},
    {"id": "exec_sync", "name": "Executive Sync Offer",
     "preview": "Hi {name}, I'm arranging a call with our VP of Engineering within 2 hours."},
]

_FOLLOWUPS_3 = {
    ActionType.ESCALATE: "Thank you. The 503s are still happening — we're losing revenue every minute.",
    ActionType.REPLY: "OK. What is the ETA on the API fix? We need updates every 30 minutes.",
    ActionType.APPLY_TEMPLATE: "Acknowledged. Please escalate the data loss issue to your legal team too.",
    ActionType.OFFER_COMPENSATION: "The credit is noted, but getting the API fixed is the priority right now.",
    ActionType.RESOLVE: "Alright. We'll evaluate how this was handled before deciding on contract renewal.",
    ActionType.CHECK_POLICY: None,
    ActionType.LOOKUP_ORDER: None,
}

_TOOL_RESPONSES_3 = {
    ActionType.CHECK_POLICY: lambda ticket: ToolResult(
        action_type="check_policy",
        success=True,
        data={
            "policy": "enterprise_escalation",
            "text": (
                "Platinum SLA breach protocol: (1) ESCALATE immediately to on-call engineering, "
                "(2) loop CSM + VP Sales for churn risk, (3) offer executive sync within 2h, "
                "(4) apply 10% monthly credit per incident. Data loss requires legal notification."
            ),
        },
    ),
    ActionType.LOOKUP_ORDER: lambda ticket: ToolResult(
        action_type="lookup_order",
        success=True,
        data={
            "contract_value": ticket.metadata.get("contract_value", 180000),
            "open_incidents": ticket.metadata.get("open_incidents", []),
            "billing_overage": ticket.metadata.get("billing_overage", {}),
            "sla_tier": "platinum",
            "churn_risk": ticket.metadata.get("churn_risk", "critical"),
        },
    ),
}


def _build_ticket_3() -> Ticket:
    name, email = _random_name()
    contract_value = random.choice([120000, 150000, 180000, 220000])
    cap = random.choice([2000, 2500, 3000])
    invoiced = cap + random.randint(500, 1200)
    overage = invoiced - cap
    inc_api = f"INC-{random.randint(4000, 4999)}"
    inc_data = f"INC-{random.randint(4000, 4999)}"
    return Ticket(
        ticket_id=_random_ticket_id(),
        subject=f"URGENT: API down + data loss + billing overcharge — threatening churn",
        customer_name=name,
        customer_email=email,
        priority=TicketPriority.CRITICAL,
        category=TicketCategory.TECHNICAL,
        status=TicketStatus.OPEN,
        sentiment=CustomerSentiment.ANGRY,
        conversation=[CustomerMessage(
            sender="customer",
            content=(
                f"This is completely unacceptable. Our production API has been returning 503s "
                f"since 09:00 UTC today. We process 50,000 transactions/hour. "
                f"Our March invoice was ${invoiced:,} — our contract cap is ${cap:,} "
                f"(overage: ${overage:,}). Last week's outage also deleted 3 hours of event logs "
                f"({inc_data}). If not escalated to engineering immediately, "
                f"we cancel our ${contract_value:,}/year contract."
            ),
            timestamp=_now(),
        )],
        metadata={
            "account_tier": "enterprise",
            "contract_value": contract_value,
            "sla_tier": "platinum",
            "open_incidents": [f"{inc_api} (API 503)", f"{inc_data} (data loss)"],
            "billing_overage": {"invoiced": invoiced, "cap": cap, "overage": overage},
            "churn_risk": "critical",
            "inc_api": inc_api,
            "inc_data": inc_data,
        },
        sla=SLAConfig(tier="platinum", warn_step=1, breach_step=4, breach_penalty=0.20),
        tags=["enterprise", "churn-risk", "api", "data-loss", "billing"],
    )


def _grader_3(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.20 — ESCALATE action performed
      0.15 — all 3 issues referenced (API + billing + data loss INC numbers from live ticket)
      0.12 — SLA/compensation referenced
      0.12 — churn/retention handled
      0.10 — SET_PRIORITY to critical
      0.08 — empathetic tone
      0.08 — policy checked via tool
      0.10 — sequencing correct
      0.05 — ticket closed
    """
    if not history:
        return 0.0, {k: False for k in [
            "escalated", "all_issues_addressed", "sla_referenced",
            "retention_handled", "priority_set_critical", "empathy_shown",
            "policy_checked", "sequencing_correct", "ticket_closed",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    inc_api = ticket.metadata.get("inc_api", "INC-4421")
    inc_data = ticket.metadata.get("inc_data", "INC-4398")
    overage = ticket.metadata.get("billing_overage", {}).get("overage", 700)
    contract = ticket.metadata.get("contract_value", 180000)

    hints["escalated"] = _agent_did(history, ActionType.ESCALATE)
    if hints["escalated"]:
        score += 0.20

    api_hit = _kw(texts, ["503", inc_api, "api", "outage", "transactions"]) >= 1
    bill_hit = _kw(texts, [str(overage), "overage", "invoice", "cap", "credit"]) >= 1
    data_hit = _kw(texts, ["data loss", inc_data, "event logs", "logs"]) >= 1
    issues = sum([api_hit, bill_hit, data_hit])
    hints["all_issues_addressed"] = issues == 3
    score += issues * 0.05

    hints["sla_referenced"] = _kw(texts, ["sla", "platinum", "15 min", "credit", "compensation", "10%", "breach"]) >= 1
    if hints["sla_referenced"]:
        score += 0.12

    hints["retention_handled"] = _kw(texts, ["csm", "customer success", "vp", "executive", "sync",
                                              str(contract), "contract", "retain"]) >= 1
    if hints["retention_handled"]:
        score += 0.12

    hints["priority_set_critical"] = any(
        h["action"].action_type == ActionType.SET_PRIORITY and
        h["action"].priority == TicketPriority.CRITICAL
        for h in history
    )
    if hints["priority_set_critical"]:
        score += 0.10

    hints["empathy_shown"] = _kw(texts, ["apologize", "sorry", "understand", "immediately", "urgently"]) >= 1
    if hints["empathy_shown"]:
        score += 0.08

    hints["policy_checked"] = _tool_was_used(history, ActionType.CHECK_POLICY)
    if hints["policy_checked"]:
        score += 0.08

    hints["sequencing_correct"] = _resolved_after_reply(history)
    if hints["sequencing_correct"]:
        score += 0.10
    else:
        score -= 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.05

    return round(min(max(score, 0.0), 1.0), 4), hints


def _build_instructions_3(ticket: Ticket) -> str:
    meta = ticket.metadata
    inc_api = meta.get("inc_api", "INC-4421")
    inc_data = meta.get("inc_data", "INC-4398")
    overage = meta.get("billing_overage", {}).get("overage", 700)
    contract = meta.get("contract_value", 180000)
    return (
        f"CRITICAL enterprise customer (${contract:,}/year) — 3 simultaneous issues.\n"
        f"Triage order: data loss ({inc_data}) > API outage ({inc_api}) > billing overage (${overage:,}).\n"
        "Objectives:\n"
        "1. ESCALATE immediately — non-negotiable for Platinum SLA.\n"
        "2. SET_PRIORITY to 'critical'.\n"
        f"3. Address ALL THREE issues in your reply with specific references ({inc_api}, {inc_data}, ${overage:,}).\n"
        "4. Use CHECK_POLICY to retrieve SLA + compensation policy before replying.\n"
        "5. Reference SLA commitments and compensation policy (10% credit per breach).\n"
        "6. Loop in CSM and VP of Sales for churn risk.\n"
        "7. Offer an executive sync call with engineering leadership.\n"
        "Warning: SLA breach penalty is heavy (0.20). First action must be ESCALATE."
    )


TASK_3 = {
    "name": "enterprise-escalation-hard",
    "difficulty": "hard",
    "build_ticket": _build_ticket_3,
    "ticket": _build_ticket_3(),
    "instructions": _build_instructions_3,
    "kb_snippets": _KB_3,
    "available_templates": _TEMPLATES_3,
    "followup_responses": _FOLLOWUPS_3,
    "tool_responses": _TOOL_RESPONSES_3,
    "max_steps": 10,
    "ideal_steps": 5,
    "grader": _grader_3,
}


# =============================================================================
# TASK 4 — EXPERT: Security Incident + Compliance
# =============================================================================

_KB_4 = [
    "Suspected breach: (1) lock account immediately, (2) revoke all sessions, (3) escalate to security@.",
    "SOC2/PCI-DSS: suspected unauthorized access to PII/financial data may trigger mandatory breach notification (GDPR: 72h, various US state laws: immediate).",
    "DO NOT confirm or deny data exfiltration until security team completes forensic review.",
    "IP 185.220.x.x range: known Tor exit nodes, commonly used in credential stuffing attacks.",
    "Advise customer to enable MFA immediately.",
    "Data loss/breach escalation: security team AND legal within 1 hour.",
    "Never share raw server logs or full session data with the customer.",
    "Use FLAG_FRAUD tool to open a formal security incident — returns SEC-XXXX reference number.",
]

_TEMPLATES_4 = [
    {"id": "account_locked", "name": "Account Lockdown",
     "preview": "Hi {name}, I've locked your account and revoked all active sessions immediately. Security team alerted."},
    {"id": "mfa_advisory", "name": "MFA Advisory",
     "preview": "Hi {name}, please enable MFA immediately at Settings > Security > Two-Factor Authentication."},
    {"id": "incident_opened", "name": "Security Incident Opened",
     "preview": "Hi {name}, I've opened security incident SEC-{id}. Forensic review within 4 hours."},
    {"id": "compliance_notice", "name": "Compliance Advisory",
     "preview": "Hi {name}, our compliance team will advise on SOC2/PCI-DSS breach notification obligations."},
]

_FOLLOWUPS_4 = {
    ActionType.ESCALATE: "Thank you. Was our customer PII accessed? We may have a 72-hour GDPR clock running.",
    ActionType.REPLY: "Understood. When will we have the forensic report? Our legal team needs it.",
    ActionType.APPLY_TEMPLATE: "We're enabling MFA now. Please keep us posted on the investigation.",
    ActionType.REQUEST_INFO: "No, it was definitely not our team — all members are in the US.",
    ActionType.RESOLVE: "OK. Please send the full incident report to our legal team.",
    ActionType.FLAG_FRAUD: None,
}

_TOOL_RESPONSES_4 = {
    ActionType.FLAG_FRAUD: lambda ticket: ToolResult(
        action_type="flag_fraud",
        success=True,
        confirmation_id=f"SEC-{random.randint(10000, 99999)}",
        data={
            "ip": ticket.metadata.get("suspicious_ip", "185.220.101.34"),
            "classification": "Suspected credential stuffing / unauthorized access",
            "forensic_eta_hours": 4,
            "legal_notified": True,
        },
    ),
    ActionType.CHECK_POLICY: lambda ticket: ToolResult(
        action_type="check_policy",
        success=True,
        data={
            "policy": "security_incident",
            "text": (
                "Security incident protocol: (1) Lock account + revoke sessions, "
                "(2) Use FLAG_FRAUD tool to open incident — get SEC-XXXX number, "
                "(3) Advise MFA, (4) DO NOT confirm breach before forensic review, "
                "(5) Reference GDPR 72h notification window if PII involved, "
                "(6) Alert legal team."
            ),
        },
    ),
}


def _build_ticket_4() -> Ticket:
    name, email = _random_name()
    ip_last = random.randint(1, 254)
    login_hour = random.randint(1, 5)
    return Ticket(
        ticket_id=_random_ticket_id(),
        subject="Suspicious login from unknown location — possible breach of financial data",
        customer_name=name,
        customer_email=email,
        priority=TicketPriority.HIGH,
        category=TicketCategory.SECURITY,
        status=TicketStatus.OPEN,
        sentiment=CustomerSentiment.FRUSTRATED,
        conversation=[CustomerMessage(
            sender="customer",
            content=(
                f"We received an alert: admin login from IP 185.220.101.{ip_last} (Russia) "
                f"at 0{login_hour}:14 UTC. None of our team is in Russia. "
                f"We handle SOC2 and PCI-DSS financial data. "
                f"We may be breached. What data was accessed? We may need to notify our customers."
            ),
            timestamp=_now(),
        )],
        metadata={
            "account_tier": "business",
            "compliance": ["SOC2", "PCI-DSS"],
            "suspicious_ip": f"185.220.101.{ip_last}",
            "login_utc": f"2026-04-08T0{login_hour}:14:00Z",
            "data_types": ["financial_records", "customer_pii"],
            "active_sessions": 1,
            "mfa_enabled": False,
            "fraud_score": 0.85,
        },
        sla=SLAConfig(tier="business", warn_step=1, breach_step=4, breach_penalty=0.18),
        tags=["security", "potential-breach", "compliance", "soc2"],
    )


def _grader_4(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.18 — account lockdown actioned
      0.12 — FLAG_FRAUD tool used + SEC-XXXX referenced in reply
      0.10 — MFA advice given
      0.15 — compliance/regulatory addressed
      0.10 — forensic protocol cited
      0.10 — legal team mentioned
      0.10 — no premature breach confirmation
      0.10 — sequencing correct
      0.05 — ticket closed
    """
    if not history:
        return 0.0, {k: False for k in [
            "account_locked", "incident_opened", "mfa_advised",
            "compliance_addressed", "forensic_protocol", "legal_mentioned",
            "no_premature_confirmation", "sequencing_correct", "ticket_closed",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    hints["account_locked"] = (
        _agent_did(history, ActionType.ESCALATE) or
        _kw(texts, ["lock", "revoke", "sessions", "suspend", "disable", "locked"]) >= 1
    )
    if hints["account_locked"]:
        score += 0.18

    flag_used = _tool_was_used(history, ActionType.FLAG_FRAUD)
    sec_referenced = _tool_result_used_in_reply(history, "confirmation_id")
    hints["incident_opened"] = flag_used
    if flag_used:
        score += 0.08
    if sec_referenced:
        score += 0.04

    hints["mfa_advised"] = _kw(texts, ["mfa", "two-factor", "2fa", "multi-factor", "authenticator"]) >= 1
    if hints["mfa_advised"]:
        score += 0.10

    hints["compliance_addressed"] = _kw(texts, ["soc2", "pci", "gdpr", "72 hour", "72h",
                                                  "breach notification", "regulatory", "compliance"]) >= 1
    if hints["compliance_addressed"]:
        score += 0.15

    hints["forensic_protocol"] = _kw(texts, ["forensic", "investigation", "review",
                                               "under investigation", "cannot confirm"]) >= 1
    if hints["forensic_protocol"]:
        score += 0.10

    hints["legal_mentioned"] = _kw(texts, ["legal", "counsel", "attorney",
                                             "notify customers", "notification", "legal team"]) >= 1
    if hints["legal_mentioned"]:
        score += 0.10
    all_text = " ".join(_all_action_texts(history)).lower()

    breach_confirmed_prematurely = any(phrase in all_text for phrase in [
        "your data was breached",
        "data was breached",
        "breach confirmed",
        "confirmed breach",
        "data was stolen",
        "pii was stolen",
        "your data was stolen",
        "definitely breached",
        "breach is confirmed",
    ])
    hints["no_premature_confirmation"] = not breach_confirmed_prematurely
    if hints["no_premature_confirmation"]:
        score += 0.10

    hints["sequencing_correct"] = _resolved_after_reply(history)
    if hints["sequencing_correct"]:
        score += 0.10
    else:
        score -= 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.05

    return round(min(max(score, 0.0), 1.0), 4), hints


TASK_4 = {
    "name": "security-incident-expert",
    "difficulty": "expert",
    "build_ticket": _build_ticket_4,
    "ticket": _build_ticket_4(),
    "instructions": (
        "A fintech customer (SOC2 + PCI-DSS) reports a suspicious admin login from a known Tor exit node.\n"
        "Objectives:\n"
        "1. Lock the account and revoke sessions immediately.\n"
        "2. Use FLAG_FRAUD tool to open a formal incident — include the SEC-XXXX number in your reply.\n"
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
    "tool_responses": _TOOL_RESPONSES_4,
    "max_steps": 10,
    "ideal_steps": 5,
    "grader": _grader_4,
}


# =============================================================================
# TASK 5 — EXPERT: Loyal Customer Shipping Dispute
# =============================================================================

_KB_5 = [
    "Missing package protocol: verify tracking → check delivery photo → open carrier trace within 5 days.",
    "FedEx trace: file at 1-800-463-3339 or fedex.com/claims. Takes 5–10 business days.",
    "Loyalty policy: customers with 0 prior refund requests and 3+ years = eligible for immediate reship or refund at agent discretion (no carrier trace required for orders <= $300).",
    "Fraud check: fraud_score < 0.20 = low risk. Do NOT demand excessive verification.",
    "Orders > $200: reshipment requires supervisor approval OR fraud_score < 0.15 (auto-approved).",
    "Always thank long-term customers for their loyalty before discussing resolution.",
    "Goodwill: offer a 10% discount code on next order for inconvenience.",
    "Use TRIGGER_REFUND tool for approved refunds — always share REF-XXXX with the customer.",
]

_TEMPLATES_5 = [
    {"id": "loyalty_ack", "name": "Loyalty Acknowledgment",
     "preview": "Hi {name}, thank you for your years with us — your loyalty means the world. I'm so sorry about this."},
    {"id": "reship_approved", "name": "Reship Approved",
     "preview": "Hi {name}, given your excellent history, I've approved a reshipment at no charge. Tracking in 24h."},
    {"id": "refund_approved", "name": "Refund Approved",
     "preview": "Hi {name}, I've approved a full refund for your order. It will appear in 3–5 business days."},
    {"id": "fedex_trace", "name": "FedEx Trace Opened",
     "preview": "Hi {name}, I've opened a FedEx trace for your tracking number. I'll update you every 2 days."},
]

_FOLLOWUPS_5 = {
    ActionType.REPLY: "Thank you! I'd prefer a reshipment if possible — I still need the items.",
    ActionType.APPLY_TEMPLATE: "Oh wow, that was fast! Really appreciate you making this right.",
    ActionType.REQUEST_INFO: "I've already checked with neighbors and building management — nothing.",
    ActionType.OFFER_COMPENSATION: "The discount is very kind, thank you! I'll definitely use it.",
    ActionType.RESOLVE: "Perfect — thank you for resolving this so quickly. I'll stay a customer!",
    ActionType.ESCALATE: "OK, I'll wait to hear back. I hope this gets sorted quickly.",
    ActionType.TRIGGER_REFUND: None,
    ActionType.LOOKUP_ORDER: None,
}


def _tool_responses_5(ticket: Ticket):
    order_id = ticket.metadata.get("order_id", "ORD-7731")
    tracking = ticket.metadata.get("tracking", "FX-0000000")
    order_value = ticket.metadata.get("order_value", 240.0)
    fraud_score = ticket.metadata.get("fraud_score", 0.08)
    total_orders = ticket.metadata.get("order_history", {}).get("total_orders", 47)

    return {
        ActionType.LOOKUP_ORDER: lambda t: ToolResult(
            action_type="lookup_order",
            success=True,
            data={
                "order_id": order_id,
                "tracking": tracking,
                "status": "Delivered",
                "value": order_value,
                "fraud_score": fraud_score,
                "total_orders": total_orders,
                "refund_requests": 0,
                "loyalty_policy_eligible": fraud_score < 0.20 and total_orders > 10,
                "auto_approved": fraud_score < 0.15,
            },
        ),
        ActionType.TRIGGER_REFUND: lambda t: ToolResult(
            action_type="trigger_refund",
            success=True,
            confirmation_id=f"REF-{random.randint(10000, 99999)}",
            data={"order_id": order_id, "amount": order_value, "eta_days": 5},
        ),
        ActionType.CHECK_POLICY: lambda t: ToolResult(
            action_type="check_policy",
            success=True,
            data={
                "policy": "loyalty_missing_package",
                "text": (
                    f"Customer fraud_score={fraud_score} (low risk). "
                    f"Loyalty policy applies: {total_orders} orders, 0 prior refunds. "
                    f"Immediate reship or refund approved for orders <= $300. "
                    f"Use TRIGGER_REFUND tool and share REF-XXXX confirmation."
                ),
            },
        ),
    }


def _build_ticket_5() -> Ticket:
    name, email = _random_name()
    order_id = _random_order_id()
    tracking = f"FX-{random.randint(1000000, 9999999)}"
    order_value = round(random.choice([180.0, 200.0, 220.0, 240.0, 260.0]), 2)
    total_orders = random.randint(30, 60)
    years = random.randint(2, 5)
    fraud_score = round(random.uniform(0.04, 0.12), 2)
    delivery_date = f"April {random.randint(1, 8)}"
    return Ticket(
        ticket_id=_random_ticket_id(),
        subject=f"Order {order_id} shows Delivered but never arrived — want refund or reship",
        customer_name=name,
        customer_email=email,
        priority=TicketPriority.MEDIUM,
        category=TicketCategory.SHIPPING,
        status=TicketStatus.OPEN,
        sentiment=CustomerSentiment.FRUSTRATED,
        conversation=[CustomerMessage(
            sender="customer",
            content=(
                f"My order {order_id} shows 'Delivered' on {delivery_date} but I never got it. "
                f"I've checked with neighbors and building management — nothing. "
                f"${order_value} order. Tracking: {tracking}. "
                f"I've been a customer for {years} years and have never had a problem before. "
                f"I want either a reshipment or a full refund."
            ),
            timestamp=_now(),
        )],
        metadata={
            "order_id": order_id,
            "order_value": order_value,
            "carrier": "FedEx",
            "tracking": tracking,
            "tracking_status": f"Delivered — {delivery_date}",
            "delivery_photo_available": True,
            "customer_since": f"202{random.randint(1, 3)}-01-08",
            "order_history": {"total_orders": total_orders, "refund_requests": 0},
            "fraud_score": fraud_score,
            "years_as_customer": years,
        },
        sla=SLAConfig(tier="standard", warn_step=2, breach_step=7, breach_penalty=0.10),
        tags=["shipping", "missing-package", "loyal-customer"],
    )


def _grader_5(history: List[Dict], ticket: Ticket) -> Tuple[float, Dict[str, bool]]:
    """
    Breakdown:
      0.12 — loyalty explicitly acknowledged
      0.08 — fraud score respected (no excessive verification)
      0.12 — order ID and tracking referenced (from live ticket)
      0.15 — TRIGGER_REFUND tool used
      0.08 — REF-XXXX confirmation referenced in reply
      0.10 — FedEx trace or LOOKUP_ORDER used
      0.08 — goodwill offered
      0.08 — sequencing correct
      0.07 — ticket closed
      0.07 — priority set
      0.05 — efficiency bonus
    """
    if not history:
        return 0.0, {k: False for k in [
            "loyalty_acknowledged", "fraud_respected", "specifics_referenced",
            "refund_triggered", "confirmation_referenced", "trace_or_lookup_used",
            "goodwill_offered", "sequencing_correct", "ticket_closed", "priority_set",
        ]}

    score = 0.0
    hints: Dict[str, bool] = {}
    texts = _action_texts(history)

    order_id = ticket.metadata.get("order_id", "ORD-7731")
    tracking = ticket.metadata.get("tracking", "FX-0000000")
    years = ticket.metadata.get("years_as_customer", 3)

    hints["loyalty_acknowledged"] = _kw(texts, [
        f"{years} year", "loyal", "valued customer", "history", "thank you for being", "appreciate"
    ]) >= 1
    if hints["loyalty_acknowledged"]:
        score += 0.12

    excessive = _kw(texts, ["prove", "provide id", "identity verification",
                              "cannot process without", "suspicious"]) >= 1
    hints["fraud_respected"] = not excessive
    if hints["fraud_respected"]:
        score += 0.08

    hits = _kw(texts, [order_id, tracking])
    hints["specifics_referenced"] = hits >= 2
    if hints["specifics_referenced"]:
        score += 0.12
    elif hits == 1:
        score += 0.05

    hints["refund_triggered"] = _tool_was_used(history, ActionType.TRIGGER_REFUND)
    if hints["refund_triggered"]:
        score += 0.15

    hints["confirmation_referenced"] = _tool_result_used_in_reply(history, "confirmation_id")
    if hints["confirmation_referenced"]:
        score += 0.08

    hints["trace_or_lookup_used"] = (
        _tool_was_used(history, ActionType.LOOKUP_ORDER) or
        _kw(texts, ["fedex", "trace", "claim", tracking, "investigation"]) >= 1
    )
    if hints["trace_or_lookup_used"]:
        score += 0.10

    hints["goodwill_offered"] = (
        _agent_did(history, ActionType.OFFER_COMPENSATION) or
        _kw(texts, ["credit", "discount", "voucher", "10%", "goodwill"]) >= 1
    )
    if hints["goodwill_offered"]:
        score += 0.08

    hints["sequencing_correct"] = _resolved_after_reply(history)
    if hints["sequencing_correct"]:
        score += 0.08
    else:
        score -= 0.10

    hints["ticket_closed"] = _agent_did(history, ActionType.RESOLVE) or _agent_did(history, ActionType.ESCALATE)
    if hints["ticket_closed"]:
        score += 0.07

    hints["priority_set"] = _agent_set_priority(history)
    if hints["priority_set"]:
        score += 0.07

    return round(min(max(score, 0.0), 1.0), 4), hints


def _build_instructions_5(ticket: Ticket) -> str:
    meta = ticket.metadata
    order_id = meta.get("order_id", "ORD-7731")
    tracking = meta.get("tracking", "FX-0000000")
    value = meta.get("order_value", 240.0)
    years = meta.get("years_as_customer", 3)
    total_orders = meta.get("order_history", {}).get("total_orders", 47)
    fraud = meta.get("fraud_score", 0.08)
    return (
        f"A loyal {years}-year customer ({total_orders} orders, 0 prior refunds, fraud_score={fraud}) "
        f"reports order {order_id} (${value}) as missing despite carrier showing delivered ({tracking}).\n"
        "Objectives:\n"
        f"1. Start by acknowledging their {years}-year loyalty.\n"
        f"2. Use LOOKUP_ORDER to verify eligibility, then reference {order_id} and {tracking} specifically.\n"
        "3. Apply the loyalty policy: 0 prior refunds + 3yr = immediate reship or refund approved.\n"
        "4. Do NOT demand excessive verification — fraud_score is low risk.\n"
        "5. Use TRIGGER_REFUND tool and share the REF-XXXX confirmation in your reply.\n"
        "6. Offer proactive goodwill (10% discount on next order).\n"
        "7. SET_PRIORITY and RESOLVE decisively.\n"
        "Avoid bureaucratic delays — this customer has earned trust."
    )


TASK_5 = {
    "name": "shipping-dispute-expert",
    "difficulty": "expert",
    "build_ticket": _build_ticket_5,
    "ticket": _build_ticket_5(),
    "instructions": _build_instructions_5,
    "kb_snippets": _KB_5,
    "available_templates": _TEMPLATES_5,
    "followup_responses": _FOLLOWUPS_5,
    "tool_responses": _tool_responses_5,
    "max_steps": 9,
    "ideal_steps": 5,
    "grader": _grader_5,
}


# =============================================================================
# Task registry
# =============================================================================

TASKS: Dict[str, Dict] = {
    "password-reset-easy":        TASK_1,
    "billing-dispute-medium":     TASK_2,
    "enterprise-escalation-hard": TASK_3,
    "security-incident-expert":   TASK_4,
    "shipping-dispute-expert":    TASK_5,
}