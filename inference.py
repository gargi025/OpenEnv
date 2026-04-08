"""
inference.py — Baseline inference script for the Customer Support Ticket Resolution OpenEnv.

Environment variables required:
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  SUPPORT_ENV_URL  Base URL of the running environment (default: http://localhost:7860)
  TASK_NAME      Task to run   (default: runs all 3 tasks)

STDOUT format (per OpenEnv spec):
  [START] task=<name> env=customer-support model=<model>
  [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("SUPPORT_ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "customer-support"
SUCCESS_SCORE_THRESHOLD = 0.30   # normalized score in [0, 1]

ALL_TASKS = [
    "password-reset-easy",
    "billing-dispute-medium",
    "enterprise-escalation-hard",
]

TASK_NAME_OVERRIDE = os.getenv("TASK_NAME", "")  # set to run only one task

# ---------------------------------------------------------------------------
# Logging helpers (OpenEnv spec)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment client (thin HTTP wrapper)
# ---------------------------------------------------------------------------

class SupportEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.http = httpx.Client(timeout=30)

    def reset(self, task_name: str, session_id: str = "default") -> Dict[str, Any]:
        r = self.http.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "session_id": session_id},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        r = self.http.post(
            f"{self.base_url}/step",
            json={"action": action, "session_id": session_id},
        )
        r.raise_for_status()
        return r.json()

    def state(self, session_id: str = "default") -> Dict[str, Any]:
        r = self.http.get(f"{self.base_url}/state", params={"session_id": session_id})
        r.raise_for_status()
        return r.json()

    def close(self):
        self.http.close()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support agent. You will receive a customer ticket and must
resolve it step by step using the available action types.

Available action types:
  - reply          : Send a reply to the customer (requires reply_text)
  - categorize     : Set ticket category (requires category: billing|technical|account|general|refund|shipping)
  - set_priority   : Set ticket priority (requires priority: low|medium|high|critical)
  - escalate       : Escalate to human agent (requires escalation_reason)
  - resolve        : Mark ticket as resolved (requires resolution_summary)
  - request_info   : Ask customer for more information (requires reply_text)
  - apply_template : Use a canned response (requires template_id)

You must respond ONLY with a valid JSON object matching one action. Example:
{"action_type": "reply", "reply_text": "Hi Alice, I've triggered a manual password reset..."}

Do not include any explanation, markdown, or extra text — only the raw JSON object.
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, last_reward: float) -> str:
    ticket = obs["ticket"]
    conversation = ticket.get("conversation", [])
    conv_text = "\n".join(
        f"  [{m['sender'].upper()}]: {m['content']}" for m in conversation
    )
    templates = obs.get("available_templates", [])
    templates_text = "\n".join(
        f"  - {t['id']}: {t['preview']}" for t in templates
    ) or "  None"
    kb = obs.get("kb_snippets", [])
    kb_text = "\n".join(f"  • {s}" for s in kb) or "  None"

    return textwrap.dedent(f"""
        TASK INSTRUCTIONS:
        {obs['task_instructions']}

        TICKET #{ticket['ticket_id']} — {ticket['subject']}
        Customer: {ticket['customer_name']} <{ticket['customer_email']}>
        Status: {ticket['status']} | Priority: {ticket['priority']} | Category: {ticket['category']}

        CONVERSATION:
        {conv_text}

        KNOWLEDGE BASE:
        {kb_text}

        AVAILABLE TEMPLATES:
        {templates_text}

        STEP: {step} / {obs['max_steps']}
        LAST REWARD: {last_reward:.2f}
        LAST FEEDBACK: {obs.get('last_action_feedback') or 'None'}

        Choose your next action (JSON only):
    """).strip()


# ---------------------------------------------------------------------------
# Agent: call LLM, parse action
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    last_reward: float,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs, step, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Fallback: safe default action
        return {
            "action_type": "reply",
            "reply_text": (
                "Thank you for reaching out. I'm reviewing your ticket and will "
                "get back to you with a resolution shortly."
            ),
        }


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(
    env_client: SupportEnvClient,
    llm_client: OpenAI,
    task_name: str,
    session_id: str,
) -> None:
    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    try:
        # Reset environment
        reset_data = env_client.reset(task_name=task_name, session_id=session_id)
        obs = reset_data["observation"]
        max_steps = obs["max_steps"]
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            # Get action from LLM
            action = get_agent_action(llm_client, obs, step, last_reward)
            action_type = action.get("action_type", "unknown")

            # Send action to environment
            try:
                step_data = env_client.step(action=action, session_id=session_id)
            except httpx.HTTPStatusError as e:
                error_msg = str(e)
                log_step(step=step, action=action_type, reward=0.0, done=True, error=error_msg)
                break

            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            obs = step_data["observation"]
            feedback = obs.get("last_action_feedback", "")
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_type,
                reward=reward,
                done=done,
                error=feedback if "error" in (feedback or "").lower() else None,
            )

            if done:
                break

        # Compute final score (cumulative reward / theoretical max per-step reward)
        # Each step can earn up to ~0.45 max; we normalize over max_steps
        max_total = max_steps * 0.45
        score = sum(rewards) / max_total if max_total > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SupportEnvClient(base_url=ENV_URL)

    tasks_to_run = [TASK_NAME_OVERRIDE] if TASK_NAME_OVERRIDE else ALL_TASKS

    try:
        for i, task_name in enumerate(tasks_to_run):
            session_id = f"session_{i}"
            run_episode(
                env_client=env_client,
                llm_client=llm_client,
                task_name=task_name,
                session_id=session_id,
            )
    finally:
        env_client.close()


if __name__ == "__main__":
    main()
