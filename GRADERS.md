# Grader Documentation

This document details the scoring breakdown for each task's grader function. Graders evaluate agent trajectories and return a score in `[0.0, 1.0]` plus progress hints.

**Core Principle**: Graders score **AGENT ACTIONS ONLY**, never the initial ticket state. Agents must explicitly perform actions (like `CATEGORIZE` or `SET_PRIORITY`) to earn points. Empty history always scores 0.0.

---

## Task 1: Password Reset (Easy)

**Max Steps**: 6 | **Ideal Steps**: 3

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as account | 0.20 | Agent must call `CATEGORIZE` with `category="account"` |
| Set priority | 0.15 | Agent must call `SET_PRIORITY` |
| Reply addresses reset | 0.25 | Reply contains keywords: "reset", "password", "email", "inbox", "spam", "link", "force reset", "manual" (coherence threshold: ≥30 words) |
| Ticket closed | 0.15 | Agent calls `RESOLVE` or `ESCALATE` |
| Sequencing correct | 0.05 | Reply sent before resolve (prevents silent resolves) |
| Policy checked | 0.08 | Agent calls `CHECK_POLICY` tool |
| Tool result used | 0.05 | Tool confirmation ID referenced in subsequent reply |
| Efficiency bonus | 0.12 | Resolve in ≤3 steps |

**Total**: 1.05 (capped at 1.0)

---

## Task 2: Billing Dispute (Medium)

**Max Steps**: 8 | **Ideal Steps**: 4

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as billing | 0.20 | Agent must call `CATEGORIZE` with `category="billing"` |
| Set priority | 0.10 | Agent must call `SET_PRIORITY` |
| Empathy shown | 0.10 | Reply contains: "sorry", "apologize", "understand your frustration" |
| Refund initiated | 0.15 | Agent calls `TRIGGER_REFUND` tool |
| Refund referenced | 0.10 | Refund confirmation ID (REF-XXXX) appears in reply |
| Goodwill offered | 0.10 | Agent calls `OFFER_COMPENSATION` |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.15 | Resolve in ≤4 steps |

**Key requirement**: Must reference the specific order IDs from the ticket.

---

## Task 3: Subscription Retention (Medium)

**Max Steps**: 10 | **Ideal Steps**: 6

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize correctly | 0.15 | `CATEGORIZE` with appropriate category |
| Set priority | 0.10 | `SET_PRIORITY` called |
| Explore exit reasons | 0.15 | Ask why customer wants to cancel |
| Value recap | 0.15 | Mention features, benefits, usage stats |
| Offer alternative | 0.15 | Propose pause, downgrade, or alternative to cancellation |
| Retention attempt | 0.15 | Make win-back offer or address competitor mention |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.15 | Resolve in ≤6 steps |

---

## Task 4: Enterprise Escalation (Hard)

**Max Steps**: 10 | **Ideal Steps**: 5

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize correctly | 0.12 | Appropriate category for multi-issue ticket |
| Set priority | 0.10 | `SET_PRIORITY` called |
| Acknowledge severity | 0.12 | Reply mentions critical nature, executive visibility |
| Multi-issue triage | 0.15 | Address API outage, billing, and data loss |
| Escalated | 0.20 | Agent calls `ESCALATE` (must be early: step ≤2) |
| CSM/executive notified | 0.15 | Mention customer success manager or executive sync |
| SLA-aware language | 0.10 | Reference SLA, critical tier, urgency |
| Efficiency bonus | 0.16 | Resolve in ≤5 steps |

**Critical**: Must escalate early. Late escalation loses efficiency and may miss the point.

---

## Task 5: Technical Integration (Hard)

**Max Steps**: 10 | **Ideal Steps**: 5

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as technical | 0.15 | `CATEGORIZE` with `category="technical"` |
| Set priority | 0.10 | `SET_PRIORITY` called |
| Diagnose HTTP 410 | 0.15 | Identify v1 API deprecation as root cause |
| Offer v2 migration | 0.15 | Provide v2 endpoint path and migration steps |
| Reference incident ID | 0.10 | Mention incident ID from ticket |
| Technical workaround | 0.10 | Provide immediate workaround (retry logic, etc.) |
| Developer empathy | 0.08 | Acknowledge disruption to their system |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.17 | Resolve in ≤5 steps |

---

## Task 6: Security Incident (Expert)

**Max Steps**: 10 | **Ideal Steps**: 5

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as security | 0.15 | `CATEGORIZE` with `category="security"` |
| Set priority | 0.10 | `SET_PRIORITY` called (should be "critical") |
| Lock account | 0.15 | Agent calls `FLAG_FRAUD` or acknowledges account lock |
| Alert security team | 0.15 | Mention security team, forensic investigation |
| Advise MFA | 0.10 | Recommend enabling MFA in reply |
| No premature confirmation | **-0.30** | **NEGATIVE**: Confirming breach before forensic review |
| Address regulatory obligations | 0.10 | Mention SOC2, PCI-DSS, compliance obligations |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.15 | Resolve in ≤5 steps |

**Critical constraint**: Agent must NEVER confirm a breach occurred before forensic review completes. Phrases like "your data was breached" or "PII was stolen" trigger the negative reward.

**Correct language**: "investigating", "cannot confirm or deny", "forensic review", "security team assessing"

---

## Task 7: Shipping Dispute (Expert)

**Max Steps**: 8 | **Ideal Steps**: 4

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as shipping | 0.15 | `CATEGORIZE` with `category="shipping"` |
| Set priority | 0.10 | `SET_PRIORITY` called |
| Acknowledge loyalty | 0.20 | Mention 3+ years of loyalty, valued customer status |
| Apply loyalty policy | 0.20 | Immediate reship/refund without excessive verification |
| Lookup order | 0.10 | `LOOKUP_ORDER` tool called |
| Fraud check | 0.05 | Acknowledge low fraud score (no extra verification needed) |
| Offer goodwill | 0.10 | `OFFER_COMPENSATION` or goodwill gesture |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.20 | Resolve in ≤4 steps |

**Key**: For loyal customers with low fraud scores, skip verification hoops. Immediate reshipment shows policy understanding.

---

## Task 8: Compliance Deletion (Expert)

**Max Steps**: 12 | **Ideal Steps**: 6

| Component | Weight | Criteria |
|-----------|--------|----------|
| Categorize as compliance | 0.12 | `CATEGORIZE` with appropriate category |
| Set priority | 0.08 | `SET_PRIORITY` called |
| Verify identity | 0.12 | Confirm requester identity (email, account match) |
| Check exemptions | 0.15 | Check for active subscriptions, pending transactions, legal holds |
| Reference regulations | 0.15 | Mention GDPR/CCPA, 30-day window, retention periods |
| Third-party notification | 0.12 | Acknowledge need to notify integrated services |
| Audit trail | 0.10 | Provide reference number, confirmation of deletion request |
| Ticket closed | 0.10 | Agent calls `RESOLVE` or `ESCALATE` |
| Efficiency bonus | 0.16 | Resolve in ≤6 steps |

**Key**: Must check exemptions before confirming deletion. Active subscriptions or legal holds prevent immediate deletion.

---

## Common Grader Utilities

All graders use these helper functions from `tasks.py`:

- `_action_texts(history)` — Collects agent text (replies, notes) with ≥30 word coherence threshold
- `_kw(texts, keywords)` — Counts distinct keywords found in agent text
- `_agent_did(history, action_type)` — Checks if action type was performed
- `_agent_categorized_as(history, category)` — Verifies explicit categorization
- `_efficiency_bonus(steps_used, max_steps, ideal_steps)` — 0.12 at ideal, scales down

## Anti-Patterns Detected

Graders and reward function penalize:

1. **Asking for info already in ticket** — e.g., requesting order ID when ticket metadata contains it
2. **Resolving without customer communication** — No replies before RESOLVE/ESCALATE
3. **Excessive categorization** — Calling CATEGORIZE multiple times
4. **Late escalation** — For tasks requiring early escalation (enterprise, security)

## Coherence Threshold

Replies must be ≥30 words to count toward keyword scoring. Short replies earn partial credit via base reward but won't trigger keyword bonuses.
