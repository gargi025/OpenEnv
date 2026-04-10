
---
title: OpenEnv Customer Support RL
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Customer Support Ticket Resolution

An [OpenEnv] environment simulating real-world customer support ticket resolution. AI agents act as support specialists handling tickets across 8 tasks at 4 difficulty levels, featuring dense rewards, SLA pressure, dynamic customer sentiment, simulated multi-turn conversations, and trajectory-aware graders.

---

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t openenv-customer-support .
docker run -p 7860:7860 openenv-customer-support
```

### Validate Installation

```bash
# Check OpenEnv compliance
openenv validate

# Run tests
pytest tests/ -v
```

---

## Action Space

### Communication Actions
- `reply` — Send a response to the customer (requires `reply_text`)
- `request_info` — Ask customer for more information (requires `reply_text`)
- `apply_template` — Use a canned response template (requires `template_id`)
- `add_note` — Add internal note (not sent to customer)
- `offer_compensation` — Proactively offer credit/refund (requires `compensation_amount` and `reply_text`)

### Ticket Management Actions
- `categorize` — Set ticket category (requires `category`)
- `set_priority` — Set ticket priority (requires `priority`)
- `escalate` — Escalate to human agent (ends episode)
- `resolve` — Mark ticket resolved (ends episode)

### Tool-Use Actions (Agentic)
- `lookup_order` — Retrieve order status/metadata (requires `order_id`)
- `check_policy` — Query knowledge base for policy (requires `policy_topic`)
- `trigger_refund` — Process refund (requires `order_id`, `amount`)
- `flag_fraud` — Flag account for review (requires `reason`)

Tool actions cost a step and return structured data in `observation.tool_result`. Agents should use tools before replying and reference confirmation IDs (REF-XXXX, SEC-XXXX) in subsequent replies for bonus rewards.

---

## Observation Space

```python
{
  "ticket": Ticket,                    # Full ticket with conversation history
  "step_number": int,
  "max_steps": int,
  "steps_remaining": int,
  "available_templates": List[Dict],   # Canned response options
  "kb_snippets": List[str],            # Relevant knowledge base entries
  "last_action_feedback": str,         # Human-readable feedback
  "task_instructions": str,            # Natural language objectives
  "sla_status": "ok" | "warning" | "breached",
  "customer_sentiment": "angry" | "frustrated" | "neutral" | "satisfied" | "delighted",
  "customer_followup": str | None,     # Simulated customer reply
  "progress_hints": Dict[str, bool],   # Live grader checklist
  "tool_result": ToolResult | None     # Result from previous tool call
}
```

---

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|------------|-----------|-------------|
| `password-reset-easy` | Easy | 6 | Help customer with password reset. Must categorize as account, set priority, send actionable reply. |
| `billing-dispute-medium` | Medium | 8 | Handle duplicate charge dispute. Requires empathy, correct categorization, referencing order IDs, refund initiation. |
| `subscription-retention-medium` | Medium | 10 | Prevent customer churn. Explore exit reasons, provide value recap, offer alternatives (pause/downgrade). |
| `enterprise-escalation-hard` | Hard | 10 | Triage critical enterprise customer with API outage, billing overage, data loss. Requires multi-issue triage and SLA-aware escalation. |
| `technical-integration-hard` | Hard | 10 | Debug webhook/API failure (HTTP 410). Diagnose v1 deprecation, offer v2 migration path. |
| `security-incident-expert` | Expert | 10 | Handle potential security breach. Lock account, alert security, advise MFA. **Critical**: Must NOT prematurely confirm breach before forensic review. |
| `shipping-dispute-expert` | Expert | 8 | Resolve missing package for loyal customer. Apply loyalty policy (immediate reship/refund), avoid excessive verification. |
| `compliance-deletion-expert` | Expert | 12 | Process GDPR/CCPA deletion request. Verify identity, check exemptions, reference regulations, provide audit trail. |

---

## Reward System

Dense per-step rewards with terminal grader bonus:

| Action | Reward |
|--------|--------|
| Reply (base) | +0.08 (× sentiment multiplier) |
| Long reply (≥100 chars) | +0.04 |
| Correct categorize | +0.10 |
| Set priority | +0.07 |
| Tool use | +0.04 (+ contextual bonuses) |
| Compensation offer | +0.07 |
| Template applied | +0.05 |
| Sentiment improvement | +0.04 per level |
| SLA urgency bonus | +0.05 (resolve before warning) |
| Policy adherence | +0.04 |
| Terminal grader bonus | Up to +0.35 |
| Wrong category | -0.08 |
| Redundant actions | -0.04 |
| Short reply (<20 chars) | -0.02 |
| SLA breach | -0.10 to -0.20 |
| Anti-patterns | -0.06 |

**Key**: Graders score only AGENT ACTIONS, never initial ticket state. Empty history always scores 0.0.

---

## Running Baseline Inference

```bash
# Set environment variables
export SUPPORT_ENV_URL=http://localhost:7860  # Or your HF Space URL
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run all tasks
python inference.py

# Run single task
export TASK_NAME=password-reset-easy
python inference.py
```

Expected output format:
```
[START] task=password-reset-easy env=customer-support model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=categorize reward=0.10 done=false error=null
[STEP] step=2 action=set_priority reward=0.07 done=false error=null
...
[END] success=true steps=4 score=0.850 rewards=0.10,0.07,0.12,0.35
```

---

## Environment Design Principles

1. **Graders score actions only** — Initial ticket category/priority are set by system; agents must explicitly categorize/set priority to earn points.
2. **Empty history = 0.0** — Prevents trivial high scores.
3. **Tool-use first** — Agents should use tools (lookup_order, check_policy) before replying, then reference confirmation IDs.
4. **Sentiment dynamics** — Customer sentiment shifts based on agent actions. Compensation improves sentiment by 2 levels.
5. **SLA pressure** — Warning at 65% of breach_step, breach penalty escalates.
6. **Anti-pattern detection** — Penalizes asking for info already in ticket, resolving without customer communication.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start/restart episode. Optional: `task_name`, `session_id` |
| POST | `/step` | Execute action. Returns observation, reward, done |
| GET | `/state` | Inspect current state (no side effects) |
| GET | `/tasks` | List all tasks with metadata |
| GET | `/health` | Liveness probe |
| POST | `/reset_all` | Clear all sessions (for batch eval) |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## Deployment

### Hugging Face Spaces

1. Create new Space with Docker SDK
2. Push code to Space repository
3. Space will auto-build and deploy
4. Access endpoints at `https://{username}-{spacename}.hf.space`

### Validation Script

```bash
./validate-submission.sh https://your-space.hf.space
```

Validates:
1. HF Space responds to /reset
2. Docker build succeeds
3. `openenv validate` passes

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI server
│   ├── environment.py    # SupportEnvironment (episode engine)
│   ├── models.py         # Pydantic models
│   └── tasks.py          # Task definitions + graders
├── tests/
│   └── test_environment.py
├── server/
│   └── app.py            # Entry point wrapper
├── inference.py          # Baseline agent script
├── openenv.yaml          # OpenEnv spec
├── pyproject.toml        # Package config
├── requirements.txt
├── Dockerfile
├── validate-submission.sh
└── README.md
```

---

## License

MIT
