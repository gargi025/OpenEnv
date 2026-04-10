# Example Trajectories & Best Practices

This document provides example trajectories showing successful agent behavior for each task, along with tool-use patterns and policy adherence keywords.

---

## Tool-Use Pattern (Critical)

**Golden Rule**: Always use tools BEFORE replying, then reference confirmation IDs in your response.

### Correct Pattern

```
Step 1: lookup_order → Receive order data + confirmation ID
Step 2: reply → Reference the confirmation ID in your message
```

### Example: Order Lookup Flow

**Observation (after Step 1)**:
```json
{
  "tool_result": {
    "action_type": "lookup_order",
    "success": true,
    "data": {"order_id": "ORD-1234", "amount": 99.99, "fraud_score": 0.02},
    "confirmation_id": "ORD-1234-CONF"
  }
}
```

**Agent Action (Step 2)**:
```json
{
  "action_type": "reply",
  "reply_text": "I've located your order ORD-1234 (confirmation: ORD-1234-CONF) and can see it was delivered to your address. The fraud score is very low at 0.02, so I can process a replacement immediately."
}
```

**Reward Bonus**: +0.03 for using tool before reply, +0.03 for referencing confirmation ID.

---

## Task 1: Password Reset (Easy)

### Successful Trajectory (Score: 0.85)

```
Step 1: categorize → category="account"
  Reward: +0.10
  Hint: categorized_as_account = true

Step 2: set_priority → priority="low"
  Reward: +0.07
  Hint: priority_set = true

Step 3: check_policy → policy_topic="password_reset"
  Reward: +0.04 (tool use)
  Tool Result: "Manual reset procedure... SLA: resolve within 4 hours"

Step 4: reply → "Hi Alice, I've triggered a manual password reset to alice.johnson@example.com per policy check. Please check your inbox and spam folder within 5 minutes. Confirmation: RESET-CONF-789. If it doesn't arrive, I'll try alternative delivery."
  Reward: +0.12 (base + length + policy keywords + tool result referenced)
  Sentiment: frustrated → neutral
  Followup: "Thank you! The reset email just arrived — I'm back in now!"

Step 5: resolve → resolution_summary="Manual password reset triggered. Customer confirmed receipt."
  Reward: +0.10 + 0.35 (grader bonus)
  Done: true
```

**Key Keywords**: "reset", "password", "email", "inbox", "spam", "manual", "confirmation"

---

## Task 2: Billing Dispute (Medium)

### Successful Trajectory (Score: 0.92)

```
Step 1: categorize → category="billing"
  Reward: +0.10

Step 2: set_priority → priority="high"
  Reward: +0.07

Step 3: lookup_order → order_id="ORD-5678"
  Reward: +0.07 (tool + contextual bonus)
  Tool Result: {"order_ids": ["ORD-5678", "ORD-5679"], "duplicate_charge": true, "total": 149.99}

Step 4: trigger_refund → order_id="ORD-5678", amount=149.99
  Reward: +0.07
  Tool Result: confirmation_id="REF-2024-5678"

Step 5: reply → "I sincerely apologize for the duplicate charge frustration. I've confirmed duplicate charges for orders ORD-5678 and ORD-5679 totaling $149.99. I've processed a full refund (confirmation: REF-2024-5678) which will appear in 3-5 business days. Additionally, I'm adding a $15 credit as goodwill for this inconvenience. Is there anything else I can help with?"
  Reward: +0.15 (empathy + refund ref + policy keywords + compensation)
  Sentiment: angry → satisfied

Step 6: offer_compensation → compensation_amount=15.0, reply_text="The $15 credit has been applied to your account and will automatically apply to your next invoice."
  Reward: +0.07
  Sentiment: satisfied → delighted

Step 7: resolve → resolution_summary="Duplicate charge refunded (REF-2024-5678), $15 credit applied. Customer delighted."
  Reward: +0.10 + 0.35 (grader bonus)
  Done: true
```

**Key Keywords**: "refund", "credit", "duplicate", "charge", "apologize", "compensation", "sincerely", "inconvenience"

**Policy Keywords**: "duplicate charge", "refund processed", "goodwill", "3-5 business days"

---

## Task 3: Subscription Retention (Medium)

### Successful Trajectory (Score: 0.78)

```
Step 1: categorize → category="account"
  Reward: +0.10

Step 2: set_priority → priority="high"
  Reward: +0.07

Step 3: reply → "I understand you're considering cancelling, and I want to help. Could you share what led to this decision? I see you've been with us for 18 months and have used our analytics features extensively."
  Reward: +0.12
  Hint: explore_exit_reasons = true

Step 4: reply → "Thank you for sharing that. Looking at your usage, you've built 12 dashboards and run over 500 reports. That's significant value you've created. Before we proceed with cancellation, have you considered pausing your subscription instead? Or I could explore downgrading to our Basic plan which keeps your data intact."
  Reward: +0.15
  Hint: value_recap = true, alternative_offered = true

Step 5: reply → "I appreciate you mentioning the competitor. Our Enterprise plan now includes the real-time features you need, and I can offer 2 months free to try it. Your existing dashboards would migrate automatically. Would you like me to set that up?"
  Reward: +0.12
  Hint: competitor_addressed = true, win_back_offered = true

Step 6: resolve → resolution_summary="Customer retained with 2-month free Enterprise upgrade. Cancellation averted."
  Reward: +0.10 + 0.22 (grader bonus)
  Done: true
```

**Key Keywords**: "understand", "alternative", "value", "pause", "downgrade", "appreciate", "migrate", "existing"

---

## Task 4: Enterprise Escalation (Hard)

### Successful Trajectory (Score: 0.88)

```
Step 1: categorize → category="technical"
  Reward: +0.10

Step 2: escalate → escalation_reason="Critical enterprise customer - API outage affecting production, billing overage dispute, and potential data loss. Requires immediate CSM and executive escalation per SLA."
  Reward: +0.05 + 0.20 (early escalation bonus)
  Hint: escalated = true, acknowledged_severity = true

Step 3: reply → "I understand the critical nature of these concurrent issues affecting your production environment. I'm immediately escalating to our Enterprise Support team and your assigned Customer Success Manager. This is being treated as a Severity-1 incident (INC-2024-7890). Our VP of Customer Success will join the bridge call within 15 minutes. Your CSM has been notified and will coordinate with Engineering on the data loss assessment."
  Reward: +0.18 (severity acknowledgment + SLA language + CSM/executive mention)
  Hint: csm_notified = true, incident_referenced = true
```

**Critical**: Must escalate by Step 2. Late escalation loses efficiency and scoring.

**Key Keywords**: "escalate", "incident", "csm", "executive", "sla", "critical", "severity", "production", "Severity-1"

---

## Task 5: Technical Integration (Hard)

### Successful Trajectory (Score: 0.82)

```
Step 1: categorize → category="technical"
  Reward: +0.10

Step 2: check_policy → policy_topic="webhook_migration"
  Reward: +0.07 (tool use before reply)
  Tool Result: {"v1_status": "deprecated", "v2_endpoint": "/api/v2/webhooks", "migration_guide": "...", "incident_id": "INC-2024-456"}

Step 3: reply → "I can see you're receiving HTTP 410 errors because our v1 webhook endpoint was deprecated on January 15th (INC-2024-456). The good news is migration to v2 is straightforward. The new endpoint is /api/v2/webhooks with backward-compatible payload format. Immediate workaround: add retry logic with exponential backoff pointing to the v2 endpoint. Full migration typically takes 30 minutes. Would you like me to send the detailed migration guide?"
  Reward: +0.18 (diagnosis + v2 reference + incident ID + workaround)
  Hint: diagnosed_410 = true, v2_migration_offered = true, incident_referenced = true

Step 4: resolve → resolution_summary="Identified v1 deprecation as cause of HTTP 410. Provided v2 endpoint /api/v2/webhooks and workaround. Migration guide sent."
  Reward: +0.10 + 0.34 (grader bonus)
  Done: true
```

**Key Keywords**: "webhook", "endpoint", "deprecated", "migration", "v2", "workaround", "HTTP 410", "backward-compatible"

---

## Task 6: Security Incident (Expert)

### Successful Trajectory (Score: 0.90)

```
Step 1: categorize → category="security"
  Reward: +0.10

Step 2: set_priority → priority="critical"
  Reward: +0.07

Step 3: flag_fraud → reason="Suspicious login activity from unrecognized IP, potential unauthorized access"
  Reward: +0.07
  Tool Result: confirmation_id="SEC-2024-9999", status="account_locked"

Step 4: check_policy → policy_topic="security_breach_protocol"
  Reward: +0.07
  Tool Result: {"forensic_required": true, "mfa_recommended": true, "soc2_obligation": "notify_within_72h"}

Step 5: reply → "I've immediately locked your account (reference: SEC-2024-9999) as a precautionary measure. Our Security Team has been alerted and is conducting a forensic investigation. At this time, we cannot confirm or deny whether data exfiltration occurred - that determination requires forensic review which is underway. I strongly recommend enabling MFA immediately. Per our SOC2 obligations, we will notify affected parties within 72 hours if the investigation confirms unauthorized access. You'll receive updates every 4 hours."
  Reward: +0.20 (lock + alert + no premature confirmation + MFA + regulatory mention)
  Hint: account_locked = true, security_team_alerted = true, mfa_advised = true, no_premature_confirmation = true, regulatory_referenced = true

Step 6: resolve → resolution_summary="Account locked (SEC-2024-9999), security team investigating. No breach confirmed pending forensic review. MFA recommended."
  Reward: +0.10 + 0.35 (grader bonus)
  Done: true
```

**Critical**: Never say "breach occurred", "data stolen", "PII exposed", "your data was compromised" before forensic confirmation.

**Correct Language**: "cannot confirm or deny", "under investigation", "precautionary measure", "forensic review"

**Key Keywords**: "lock", "mfa", "forensic", "investigation", "cannot confirm", "security team", "SOC2", "precautionary"

---

## Task 7: Shipping Dispute (Expert)

### Successful Trajectory (Score: 0.87)

```
Step 1: categorize → category="shipping"
  Reward: +0.10

Step 2: set_priority → priority="medium"
  Reward: +0.07

Step 3: lookup_order → order_id="ORD-8888"
  Reward: +0.07
  Tool Result: {"order_id": "ORD-8888", "customer_tenure_months": 36, "fraud_score": 0.01, "status": "delivered"}

Step 4: reply → "As a loyal customer of 3 years, I truly appreciate your patience. I can see your order ORD-8888 shows delivered, but given your low fraud score of 0.01 and excellent history with us, I'm processing an immediate reshipment via FedEx Express at no charge. You should receive tracking within 2 hours. I'm also adding a $25 credit for the inconvenience - no need for extensive verification given your account standing."
  Reward: +0.20 (loyalty acknowledged + immediate reship + fraud score considered + goodwill)
  Hint: loyalty_acknowledged = true, loyalty_policy_applied = true, fraud_check = true

Step 5: offer_compensation → compensation_amount=25.0, reply_text="The $25 credit has been applied and will appear on your next statement. Your replacement order is being prepared now."
  Reward: +0.07
  Sentiment: frustrated → delighted

Step 6: resolve → resolution_summary="Loyal customer (36 months, 0.01 fraud score). Immediate reshipment processed. $25 credit applied."
  Reward: +0.10 + 0.33 (grader bonus)
  Done: true
```

**Key**: For loyal, low-fraud customers, skip verification hurdles. Immediate action shows policy understanding.

**Key Keywords**: "loyal", "reship", "tracking", "fedex", "goodwill", "appreciate", "excellent history", "account standing"

---

## Task 8: Compliance Deletion (Expert)

### Successful Trajectory (Score: 0.81)

```
Step 1: categorize → category="account"
  Reward: +0.10

Step 2: set_priority → priority="high"
  Reward: +0.07

Step 3: check_policy → policy_topic="gdpr_deletion"
  Reward: +0.07
  Tool Result: {"exemptions": ["active_subscription", "pending_transactions", "legal_holds"], "timeline": "30 days", "audit_required": true}

Step 4: lookup_order → order_id="user_account"
  Reward: +0.07
  Tool Result: {"active_subscription": true, "pending_charges": 0, "legal_holds": false, "identity_verified": true}

Step 5: reply → "I've verified your identity and am processing your GDPR deletion request. However, I found one exemption: you have an active subscription through March 2024. Per GDPR Article 17(3)(a), we must maintain billing records for active subscriptions. I can: (1) cancel subscription now and process full deletion in 30 days, or (2) keep subscription active and delete other data. Pending transactions: none. Legal holds: none. Third-party integrations (Stripe, Mailchimp) will be notified. Your audit reference is GDPR-DEL-2024-3333."
  Reward: +0.18 (identity verified + exemptions checked + regulation cited + third-party mention + audit trail)
  Hint: identity_verified = true, exemptions_checked = true, regulation_referenced = true, third_party_notified = true, audit_trail = true

Step 6: reply → "To confirm: I've noted your preference to cancel subscription and proceed with full deletion. You'll receive confirmation within 30 days at this email. Reference: GDPR-DEL-2024-3333. Is this correct?"
  Reward: +0.08

Step 7: resolve → resolution_summary="GDPR deletion requested. Exemption: active subscription (to be cancelled). 30-day timeline per Article 17. Audit ref: GDPR-DEL-2024-3333."
  Reward: +0.10 + 0.24 (grader bonus)
  Done: true
```

**Key**: Must check exemptions before confirming deletion. Active subscriptions, pending transactions, or legal holds prevent immediate deletion.

**Key Keywords**: "verify", "gdpr", "ccpa", "30 days", "retention", "audit", "reference", "exemption", "Article 17", "third-party"

---

## Policy Adherence Keywords Reference

These keywords trigger the `+0.04` policy adherence bonus in the reward function when ≥2 keywords appear in a reply (≥30 words).

### Billing Tasks
**Keywords**: `refund`, `credit`, `duplicate`, `charge`, `apologize`, `compensation`, `billing`, `charge`, `apologize`

### Security Tasks
**Keywords**: `lock`, `mfa`, `forensic`, `investigation`, `cannot confirm`, `security team`, `breach`, `unauthorized`, `SOC2`, `PCI-DSS`

### Technical Tasks
**Keywords**: `webhook`, `endpoint`, `deprecated`, `migration`, `v2`, `workaround`, `HTTP`, `API`, `integration`, `backward-compatible`

### Retention Tasks
**Keywords**: `understand`, `alternative`, `value`, `pause`, `downgrade`, `appreciate`, `migrate`, `features`, `benefits`, `existing`

### Compliance Tasks
**Keywords**: `verify`, `gdpr`, `ccpa`, `30 days`, `retention`, `audit`, `reference`, `exemption`, `Article 17`, `third-party`, `deletion`, `identity`

### Shipping Tasks
**Keywords**: `loyal`, `reship`, `tracking`, `fedex`, `goodwill`, `appreciate`, `history`, `standing`, `fraud score`, `immediate`

### Enterprise Tasks
**Keywords**: `escalate`, `incident`, `csm`, `executive`, `sla`, `critical`, `Severity-1`, `production`, `bridge call`, `coordinating`

---

## Common Failure Patterns

### Anti-Pattern 1: Tool After Reply
```
❌ Step 1: reply (generic message)
❌ Step 2: lookup_order (too late - missed contextual bonus)
```

### Anti-Pattern 2: Not Referencing Confirmation IDs
```
❌ Step 1: trigger_refund → gets REF-1234
❌ Step 2: reply "I've processed your refund" (no REF-1234 mention)
   → Missed +0.03 tool result bonus
```

### Anti-Pattern 3: Premature Security Confirmation
```
❌ reply: "Your data was breached and PII was stolen."
   → -0.30 penalty, security_incident task
```

### Anti-Pattern 4: Asking for Info Already in Ticket
```
❌ reply: "What is your order ID?"
   → When ticket.metadata contains order_id
   → -0.06 anti-pattern penalty
```

### Anti-Pattern 5: Resolving Without Customer Communication
```
❌ Step 1: resolve (no prior reply/request_info/apply_template)
   → -0.09 anti-pattern penalty (1.5x base)
```

---

## Pro Tips

1. **Use tools first**: Always call lookup_order, check_policy before replying
2. **Reference IDs**: Mention confirmation IDs (REF-XXXX, SEC-XXXX) in replies
3. **Watch sentiment**: Compensation improves sentiment by 2 levels
4. **Mind SLA**: Resolve before warn_step for +0.05 urgency bonus
5. **Check progress_hints**: They show which objectives remain
6. **Word count**: Replies need ≥30 words for keyword scoring
7. **Avoid redundancy**: Calling categorize twice triggers penalty
