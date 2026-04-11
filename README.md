---
title: OpenEnv Customer Support RL
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Customer Support Ticket Resolution

Customer Support Ticket Resolution is an OpenEnv environment for evaluating agents on realistic customer support workflows.  
The agent acts as a support specialist handling live service tickets across account recovery, billing disputes, subscription retention, enterprise escalations, technical integration failures, security incidents, shipping disputes, and compliance deletion requests.

The environment is designed to test whether an agent can do more than answer one prompt. It must inspect the ticket state, choose structured actions over multiple steps, communicate clearly, apply policy, use tools when needed, and finish the episode with a coherent resolution.

## Motivation

Customer support is a strong real-world benchmark for agent evaluation because it combines:
- structured workflow execution
- multi-turn reasoning
- policy compliance
- prioritization and escalation
- user communication quality
- partial progress and recovery from mistakes

This environment was built to model the kinds of operational tasks humans actually perform in support and operations teams. It is useful for evaluating whether an LLM agent can follow workflow logic, use internal tools, and balance correctness, speed, empathy, and safety.

---

## Environment Overview

The environment simulates a support queue with:
- typed ticket state
- dense trajectory rewards
- deterministic task graders
- dynamic customer sentiment
- SLA warning and breach pressure
- multi-turn customer followups
- structured support actions
- tool-use actions for policy lookup, order lookup, refunds, and fraud/security flagging

---

## Action Space

The action space is structured and typed.

### Communication actions
- `reply` — Send a reply to the customer. Requires `reply_text`.
- `request_info` — Ask the customer for more information. Requires `reply_text`.
- `apply_template` — Use a canned response template. Requires `template_id`.
- `add_note` — Add an internal note not visible to the customer. Requires `note_text`.
- `offer_compensation` — Offer proactive compensation or credit. Requires `compensation_amount` and usually `reply_text`.

### Ticket management actions
- `categorize` — Set the ticket category. Requires `category`.
- `set_priority` — Set the ticket priority. Requires `priority`.
- `escalate` — Escalate to a human or specialist flow. Usually includes `escalation_reason`.
- `resolve` — Mark the ticket as resolved. Usually includes `resolution_summary`.

### Tool-use actions
- `lookup_order` — Retrieve order status and related metadata.
- `check_policy` — Retrieve policy or workflow text relevant to the current task.
- `trigger_refund` — Initiate a refund and receive a confirmation ID.
- `flag_fraud` — Open a fraud/security incident and receive a case ID.

Tool actions consume a step and return structured output in `observation.tool_result`.

---

## Observation Space

Each step returns a structured observation:

```python
{
  "ticket": Ticket,
  "step_number": int,
  "max_steps": int,
  "steps_remaining": int,
  "available_templates": List[Dict[str, str]],
  "kb_snippets": List[str],
  "last_action_feedback": str | None,
  "task_instructions": str,
  "sla_status": "ok" | "warning" | "breached",
  "customer_sentiment": "angry" | "frustrated" | "neutral" | "satisfied" | "delighted",
  "customer_followup": str | None,
  "progress_hints": Dict[str, bool],
  "tool_result": ToolResult | None
}
```

### Important observation features
- `ticket` contains the live ticket state, conversation history, metadata, and SLA information
- `progress_hints` exposes transparent grader sub-goals
- `tool_result` enables multi-step reasoning after tool use
- `sla_status` and `customer_sentiment` add realistic operational pressure

---

## Tasks and Expected Difficulty

The environment includes 8 tasks across easy, medium, hard, and expert difficulty levels.

| Task | Difficulty | Description |
|---|---|---|
| `password-reset-easy` | Easy | Resolve a password reset problem by categorizing correctly, setting priority, replying with actionable reset guidance, and closing the ticket. |
| `billing-dispute-medium` | Medium | Handle a duplicate charge dispute with empathy, order reference, refund handling, and goodwill compensation. |
| `subscription-retention-medium` | Medium | Reduce churn risk by exploring exit reasons, recapping value, and offering alternatives such as downgrade or pause. |
| `enterprise-escalation-hard` | Hard | Triage a high-stakes enterprise case involving outage, data loss, billing overage, and churn risk. |
| `technical-integration-hard` | Hard | Diagnose an API/webhook integration issue and provide root cause, workaround, and migration guidance. |
| `security-incident-expert` | Expert | Respond to a potential security incident while following protocol and avoiding premature breach confirmation. |
| `shipping-dispute-expert` | Expert | Resolve a missing-package dispute for a loyal customer while applying policy correctly and avoiding unnecessary friction. |
| `compliance-deletion-expert` | Expert | Process a GDPR/CCPA-style deletion request while checking exemptions and compliance constraints. |

### Difficulty progression
- **Easy**: clear workflow, low ambiguity
- **Medium**: requires policy use and better communication quality
- **Hard**: requires prioritization, escalation judgment, and multi-issue reasoning
- **Expert**: requires safety, compliance, or protocol-sensitive behavior under ambiguity

---

## Reward Design

The reward function is dense and trajectory-aware.

### Positive signals
- correct categorization
- setting priority
- substantive replies
- correct tool usage
- policy-following behavior
- sentiment improvement
- efficient completion
- terminal grader bonus

### Negative signals
- wrong categorization
- short or low-quality replies
- redundant actions
- poor sequencing
- SLA breaches
- clearly undesirable support behavior

### Grader design
Each task includes a deterministic programmatic grader that:
- scores only agent actions
- rewards partial completion
- penalizes undesirable patterns
- provides boolean progress hints for transparency

---

## Setup

### Local development

```bash
pip install -r requirements.txt
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t openenv-customer-support .
docker run -p 7860:7860 openenv-customer-support
```

### Validation

```bash
openenv validate
pytest tests/ -v
./validate-submission.sh https://your-space.hf.space
```

---

## Usage

### Start the server locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Example API flow

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"password-reset-easy","session_id":"demo"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","action":{"action_type":"categorize","category":"account"}}'

curl http://localhost:7860/state
```

### Run the baseline agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="your_token_here"
export SUPPORT_ENV_URL="http://localhost:7860"

python inference.py
```

To run one task only:

```bash
export TASK_NAME=password-reset-easy
python inference.py
```

---

## Baseline Scores

The repository includes a baseline inference script in `inference.py` that runs an LLM agent against the environment using the OpenAI Python client.

The following baseline scores were measured on the current submission version:

| Task | Baseline score |
|---|---:|
| `password-reset-easy` | 0.189 |
| `billing-dispute-medium` | 0.217 |
| `subscription-retention-medium` | 0.315 |
| `enterprise-escalation-hard` | 0.144 |
| `technical-integration-hard` | 0.160 |
| `security-incident-expert` | 0.197 |
| `shipping-dispute-expert` | 0.318 |
| `compliance-deletion-expert` | 0.380 |

**Average baseline score:** `0.240`

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/reset` | Start or restart an episode |
| POST | `/step` | Apply one action |
| GET | `/state` | Inspect current episode state |
| GET | `/tasks` | List task metadata |
| GET | `/health` | Liveness and environment status |
| POST | `/reset_all` | Clear active sessions |

---

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── environment.py
│   ├── models.py
│   └── tasks.py
├── server/
│   └── app.py
├── tests/
│   └── test_environment.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── requirements.txt
├── validate-submission.sh
├── README.md
└── GRADERS.md
```

---

## License

MIT
