"""
environment.py — Stateful episode engine for the Customer Support OpenEnv.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    ActionType, CustomerMessage, CustomerSentiment,
    ResetResult, StateResult, StepResult, ToolResult,
    SupportAction, SupportObservation, Ticket, TicketStatus,
)
from .tasks import TASKS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SENTIMENT_SCALE = [
    CustomerSentiment.ANGRY,
    CustomerSentiment.FRUSTRATED,
    CustomerSentiment.NEUTRAL,
    CustomerSentiment.SATISFIED,
    CustomerSentiment.DELIGHTED,
]

# Action types that are tool calls — they cost a step but don't communicate with customer
_TOOL_ACTION_TYPES = {
    ActionType.LOOKUP_ORDER,
    ActionType.CHECK_POLICY,
    ActionType.TRIGGER_REFUND,
    ActionType.FLAG_FRAUD,
    ActionType.SEND_REPLACEMENT,
    ActionType.SCHEDULE_CALLBACK,
}


def _shift_sentiment(current: CustomerSentiment, delta: int) -> CustomerSentiment:
    idx = _SENTIMENT_SCALE.index(current)
    return _SENTIMENT_SCALE[max(0, min(len(_SENTIMENT_SCALE) - 1, idx + delta))]


class SupportEnvironment:
    """
    One episode of the Customer Support Ticket Resolution environment.

    Implements the OpenEnv interface: reset() / step() / state()

    On each reset(), a fresh ticket is built via the task's build_ticket()
    function, randomizing amounts, names, and IDs for score variance.
    """

    # ---- Dense reward weights ----
    _R_REPLY_BASE       = 0.08
    _R_REPLY_LENGTH     = 0.04   # bonus for substantive replies (>= 100 chars)
    _R_CATEGORIZE_OK    = 0.10
    _R_PRIORITY         = 0.07
    _R_TEMPLATE         = 0.05
    _R_ESCALATE         = 0.05
    _R_NOTE             = 0.03
    _R_COMPENSATION     = 0.07
    _R_TOOL_USE         = 0.04   # reward for using a tool (agentic behavior)
    _R_SENTIMENT_UP     = 0.04
    _P_WRONG_CATEGORY   = -0.08
    _P_REDUNDANT        = -0.04
    _P_SHORT_REPLY      = -0.02
    _P_TOOL_REDUNDANT   = -0.03  # calling the same tool twice

    # ---- Advanced reward shaping weights ----
    _R_SENTIMENT_WEIGHTED = 0.03  # Additional reward based on sentiment difficulty
    _R_SLA_URGENCY_BONUS = 0.05   # Bonus for resolving before warn_step
    _R_POLICY_ADHERENCE = 0.04     # Bonus for policy-compliant keywords
    _R_CONTEXTUAL_TOOL = 0.03      # Bonus for using tool before replying
    _R_TOOL_RESULT_USED = 0.03    # Bonus for referencing tool result in reply
    _P_ANTI_PATTERN = -0.06       # Penalty for common mistakes

    # Policy-compliant keywords for different task types
    _POLICY_KEYWORDS = {
        "billing": ["refund", "credit", "duplicate", "charge", "apologize", "compensation"],
        "security": ["lock", "mfa", "forensic", "investigation", "cannot confirm", "security team"],
        "technical": ["webhook", "endpoint", "deprecated", "migration", "v2", "workaround"],
        "retention": ["understand", "alternative", "value", "pause", "downgrade", "appreciate"],
        "compliance": ["verify", "gdpr", "ccpa", "30 days", "retention", "audit", "reference"],
        "shipping": ["loyal", "reship", "tracking", "fedex", "goodwill", "appreciate"],
        "enterprise": ["escalate", "incident", "csm", "executive", "sla", "critical"],
    }

    def __init__(self, task_name: str) -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS)}")
        self.task_name = task_name
        self._task = TASKS[task_name]
        self._ticket: Ticket = None          # type: ignore[assignment]
        self._step = 0
        self._done = False
        self._history: List[Dict[str, Any]] = []
        self._cumulative_reward = 0.0
        self._action_counts: Dict[ActionType, int] = {}
        self._sla_breached = False
        self._first_reply_done = False
        self._progress_hints: Dict[str, bool] = {}
        self._sla_status = "ok"
        self._last_tool_result: Optional[ToolResult] = None
        self._tool_responses: Dict[ActionType, Callable] = {}
        self._instructions: str = ""
        self._last_tool_used: Optional[ActionType] = None  # Track tool-use for contextual rewards
        self._resolved_before_warn: bool = False  # Track SLA urgency bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        # Build a fresh randomized ticket on every reset
        build_fn = self._task.get("build_ticket")
        if build_fn:
            self._ticket = build_fn()
        else:
            self._ticket = copy.deepcopy(self._task["ticket"])

        # Resolve instructions — may be a callable that takes the live ticket
        instructions = self._task["instructions"]
        if callable(instructions):
            self._instructions = instructions(self._ticket)
        else:
            self._instructions = instructions

        # Resolve tool_responses — may be a callable that takes the live ticket
        tool_responses = self._task.get("tool_responses", {})
        if callable(tool_responses):
            self._tool_responses = tool_responses(self._ticket)
        else:
            self._tool_responses = tool_responses

        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0
        self._action_counts = {}
        self._sla_breached = False
        self._first_reply_done = False
        self._progress_hints = {}
        self._sla_status = "ok"
        self._last_tool_result = None
        self._last_tool_used = None
        self._resolved_before_warn = False

        return ResetResult(
            observation=self._build_obs(feedback=None, followup=None),
            info={"task": self.task_name, "max_steps": self._task["max_steps"]},
        )

    def step(self, action: SupportAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_obs("Episode finished.", None),
                reward=0.0, done=True,
                info={"error": "already_done"},
            )

        self._step += 1
        self._update_sla()
        sla_penalty = self._maybe_apply_sla_breach()

        # Handle tool-use actions separately
        if action.action_type in _TOOL_ACTION_TYPES:
            reward, feedback, tool_result = self._execute_tool(action)
            self._last_tool_result = tool_result
            reward += sla_penalty
            self._history.append({
                "step": self._step,
                "action": action,
                "reward": reward,
                "tool_result": tool_result,
            })
            self._cumulative_reward = round(self._cumulative_reward + reward, 4)
            self._action_counts[action.action_type] = self._action_counts.get(action.action_type, 0) + 1
            _, self._progress_hints = self._task["grader"](self._history, self._ticket)

            return StepResult(
                observation=self._build_obs(feedback, None),
                reward=min(max(round(reward, 4), -1.0), 1.0),
                done=False,
                info={
                    "step": self._step,
                    "cumulative_reward": self._cumulative_reward,
                    "tool_used": action.action_type.value,
                    "tool_success": tool_result.success if tool_result else False,
                },
            )

        reward, feedback = self._compute_reward(action)
        reward += sla_penalty

        followup = self._apply_action(action)

        self._history.append({
            "step": self._step,
            "action": action,
            "reward": reward,
            "tool_result": None,
        })
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._action_counts[action.action_type] = self._action_counts.get(action.action_type, 0) + 1
        self._last_tool_result = None  # clear after non-tool action

        terminal = action.action_type in {ActionType.RESOLVE, ActionType.ESCALATE}
        max_hit = self._step >= (self._task["max_steps"] - 1)

        if terminal or max_hit:
            self._done = True
            grader_score, self._progress_hints = self._task["grader"](self._history, self._ticket)
            bonus = round(grader_score * 0.35, 4)
            reward = round(reward + bonus, 4)
            self._cumulative_reward = round(self._cumulative_reward + bonus, 4)
            self._history[-1]["reward"] = reward
            feedback += f" | Episode done. Grader={grader_score:.3f} Bonus=+{bonus:.3f}"
        else:
            _, self._progress_hints = self._task["grader"](self._history, self._ticket)

        return StepResult(
            observation=self._build_obs(feedback, followup),
            reward=min(max(round(reward, 4), -1.0), 1.0),
            done=self._done,
            info={
                "step": self._step,
                "cumulative_reward": self._cumulative_reward,
                "ticket_status": self._ticket.status,
                "sla_status": self._sla_status,
                "sentiment": self._ticket.sentiment,
            },
        )

    def state(self) -> StateResult:
        scores: Dict[str, float] = {}
        hints: Dict[str, bool] = {}
        if self._history:
            score, hints = self._task["grader"](self._history, self._ticket)
            scores["current"] = score
        return StateResult(
            ticket=self._ticket,
            step_number=self._step,
            max_steps=self._task["max_steps"],
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            task_name=self.task_name,
            grader_scores=scores,
            progress_hints=hints,
            sla_status=self._sla_status,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, action: SupportAction) -> Tuple[float, str, Optional[ToolResult]]:
        """Execute a tool-use action. Returns (reward, feedback, tool_result)."""
        at = action.action_type
        prev = self._action_counts.get(at, 0)

        # Penalize calling the same tool twice
        if prev >= 1:
            result = ToolResult(
                action_type=at.value,
                success=False,
                error=f"Tool '{at.value}' already used this episode. Use results from the first call.",
            )
            return self._P_TOOL_REDUNDANT, f"Redundant tool call: {at.value}.", result

        handler = self._tool_responses.get(at)
        if handler:
            result = handler(self._ticket)
        else:
            result = ToolResult(
                action_type=at.value,
                success=False,
                error=f"Tool '{at.value}' not available for this task.",
            )

        if result.success:
            reward = self._R_TOOL_USE
            feedback = f"Tool '{at.value}' succeeded."
            if result.confirmation_id:
                feedback += f" Confirmation: {result.confirmation_id}."

            # Bonus for using tools in good sequence (tools before replies)
            if not self._first_reply_done:
                reward += self._R_CONTEXTUAL_TOOL
                feedback += " Contextual tool-use bonus (used before replying)."
        else:
            reward = 0.0
            feedback = f"Tool '{at.value}' failed: {result.error}"

        # Track tool use for contextual reward
        self._last_tool_used = at

        return reward, feedback, result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self, feedback: Optional[str], followup: Optional[str]) -> SupportObservation:
        max_s = self._task["max_steps"]
        return SupportObservation(
            ticket=copy.deepcopy(self._ticket),
            step_number=self._step,
            max_steps=max_s,
            steps_remaining=max(0, max_s - self._step - 1),
            available_templates=self._task.get("available_templates", []),
            kb_snippets=self._task.get("kb_snippets", []),
            last_action_feedback=feedback,
            task_instructions=self._instructions,
            sla_status=self._sla_status,
            customer_sentiment=self._ticket.sentiment,
            customer_followup=followup,
            progress_hints=copy.copy(self._progress_hints),
            tool_result=copy.deepcopy(self._last_tool_result),
        )

    def _update_sla(self) -> None:
        sla = self._ticket.sla
        if self._step >= sla.breach_step:
            self._sla_status = "breached"
        elif self._step >= sla.warn_step and not self._first_reply_done:
            self._sla_status = "warning"
        elif self._step >= int(sla.breach_step * 0.65):
            self._sla_status = "warning"
        else:
            self._sla_status = "ok"

    def _maybe_apply_sla_breach(self) -> float:
        if self._sla_status == "breached" and not self._sla_breached:
            self._sla_breached = True
            return -self._ticket.sla.breach_penalty
        return 0.0

    def _compute_reward(self, action: SupportAction) -> Tuple[float, str]:
        r = 0.0
        msgs: List[str] = []
        at = action.action_type
        prev = self._action_counts.get(at, 0)

        # Track anti-patterns
        anti_pattern_penalty = self._check_anti_patterns(action)
        if anti_pattern_penalty < 0:
            r += anti_pattern_penalty
            msgs.append("Anti-pattern detected.")

        if prev >= 2 and at not in {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.ADD_NOTE}:
            r += self._P_REDUNDANT
            msgs.append(f"Redundant {at} (×{prev}).")

        if at in {ActionType.REPLY, ActionType.REQUEST_INFO}:
            text = (action.reply_text or "").strip()
            if len(text) < 20:
                r += self._P_SHORT_REPLY
                msgs.append("Reply too short.")
            else:
                # Base reply reward with sentiment weighting
                sentiment_idx = _SENTIMENT_SCALE.index(self._ticket.sentiment)
                sentiment_multiplier = 1.0 + (sentiment_idx * 0.15)  # Harder to please angry customers = higher reward
                base_reply = self._R_REPLY_BASE * sentiment_multiplier
                r += round(base_reply, 4)

                if len(text) >= 100:
                    r += self._R_REPLY_LENGTH
                msgs.append(f"Reply sent (len={len(text)}, sentiment={self._ticket.sentiment.value}).")

                # Check for policy adherence
                policy_bonus = self._check_policy_adherence(text)
                if policy_bonus > 0:
                    r += policy_bonus
                    msgs.append("Policy keywords detected.")

                # Check if tool result is referenced in reply
                if self._last_tool_used and self._last_tool_result:
                    if self._last_tool_result.confirmation_id and self._last_tool_result.confirmation_id.lower() in text.lower():
                        r += self._R_TOOL_RESULT_USED
                        msgs.append("Tool confirmation referenced.")

                # Reset contextual tool tracking after reply
                self._last_tool_used = None

            if not self._first_reply_done and len(text) >= 20:
                self._first_reply_done = True

        elif at == ActionType.APPLY_TEMPLATE:
            r += self._R_TEMPLATE
            msgs.append(f"Template: {action.template_id}.")
            if not self._first_reply_done:
                self._first_reply_done = True

        elif at == ActionType.CATEGORIZE:
            if action.category is None:
                msgs.append("CATEGORIZE missing category field.")
            elif action.category == self._task["ticket"].category:
                r += self._R_CATEGORIZE_OK
                msgs.append(f"Correct category: {action.category}.")
            else:
                r += self._P_WRONG_CATEGORY
                msgs.append(f"Wrong category ({action.category} ≠ {self._task['ticket'].category}).")

        elif at == ActionType.SET_PRIORITY:
            if action.priority:
                r += self._R_PRIORITY
                msgs.append(f"Priority → {action.priority}.")

        elif at == ActionType.ESCALATE:
            r += self._R_ESCALATE
            msgs.append("Escalated.")

        elif at == ActionType.ADD_NOTE:
            if action.note_text and len(action.note_text.strip()) > 10:
                r += self._R_NOTE
                msgs.append("Internal note added.")

        elif at == ActionType.OFFER_COMPENSATION:
            r += self._R_COMPENSATION
            msgs.append(f"Compensation offered (${action.compensation_amount}).")

        elif at == ActionType.RESOLVE:
            msgs.append("Resolve received — grader bonus at episode end.")
            # Check SLA urgency bonus
            if self._step < self._ticket.sla.warn_step:
                r += self._R_SLA_URGENCY_BONUS
                msgs.append("SLA urgency bonus (resolved before warning).")
                self._resolved_before_warn = True

        return round(min(max(r, -1.0), 1.0), 4), " | ".join(msgs) or "Action processed."

    def _check_anti_patterns(self, action: SupportAction) -> float:
        """Check for common support mistakes and return penalty."""
        penalty = 0.0
        at = action.action_type

        # Anti-pattern: Asking for info already in the ticket
        if at in {ActionType.REPLY, ActionType.REQUEST_INFO} and action.reply_text:
            text = action.reply_text.lower()
            ticket_content = " ".join([m.content for m in self._ticket.conversation]).lower()

            # Check if asking for order ID when already provided
            if "order id" in text or "what is your order" in text:
                for meta_key in ["order_id", "orders"]:
                    if meta_key in self._ticket.metadata:
                        penalty = self._P_ANTI_PATTERN
                        break

            # Check if asking for email when already in ticket
            if "email address" in text and self._ticket.customer_email:
                penalty = self._P_ANTI_PATTERN

        # Anti-pattern: Resolving without any customer communication
        if at == ActionType.RESOLVE:
            agent_replies = [h for h in self._history if h["action"].action_type in
                           {ActionType.REPLY, ActionType.APPLY_TEMPLATE, ActionType.OFFER_COMPENSATION}]
            if not agent_replies:
                penalty = self._P_ANTI_PATTERN * 1.5  # Higher penalty

        # Anti-pattern: Escalating too early (before basic triage)
        if at == ActionType.ESCALATE and self._step <= 1:
            if not any(h["action"].action_type in {ActionType.CATEGORIZE, ActionType.SET_PRIORITY}
                       for h in self._history):
                penalty = self._P_ANTI_PATTERN * 0.5  # Smaller penalty

        return penalty

    def _check_policy_adherence(self, text: str) -> float:
        """Check if reply contains policy-compliant keywords for the task type."""
        if not text:
            return 0.0

        text_lower = text.lower()
        task_name = self.task_name.lower()

        # Determine which policy keywords to check
        keywords = []
        for category, kw_list in self._POLICY_KEYWORDS.items():
            if category in task_name:
                keywords = kw_list
                break

        if not keywords:
            return 0.0

        # Count keyword hits
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 2:
            return self._R_POLICY_ADHERENCE
        return 0.0

    def _apply_action(self, action: SupportAction) -> Optional[str]:
        """Mutate ticket state; return simulated customer followup or None."""
        at = action.action_type
        followups = self._task.get("followup_responses", {})
        followup: Optional[str] = None

        if at in {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.APPLY_TEMPLATE}:
            text = action.reply_text or f"[Template applied: {action.template_id}]"
            self._ticket.conversation.append(
                CustomerMessage(sender="agent", content=text, timestamp=_now())
            )
            if self._ticket.status == TicketStatus.OPEN:
                self._ticket.status = TicketStatus.IN_PROGRESS
            if at == ActionType.REQUEST_INFO:
                self._ticket.status = TicketStatus.WAITING_CUSTOMER

            resp = followups.get(at)
            if resp:
                self._ticket.conversation.append(
                    CustomerMessage(sender="customer", content=resp, timestamp=_now())
                )
                followup = resp
                self._ticket.sentiment = _shift_sentiment(self._ticket.sentiment, +1)

        elif at == ActionType.CATEGORIZE and action.category:
            self._ticket.category = action.category

        elif at == ActionType.SET_PRIORITY and action.priority:
            self._ticket.priority = action.priority

        elif at == ActionType.ESCALATE:
            self._ticket.status = TicketStatus.ESCALATED
            if action.escalation_reason:
                self._ticket.metadata["escalation_reason"] = action.escalation_reason
            followup = followups.get(ActionType.ESCALATE)

        elif at == ActionType.RESOLVE:
            self._ticket.status = TicketStatus.RESOLVED
            if action.resolution_summary:
                self._ticket.metadata["resolution_summary"] = action.resolution_summary
            followup = followups.get(ActionType.RESOLVE)

        elif at == ActionType.ADD_NOTE and action.note_text:
            self._ticket.internal_notes.append(action.note_text)

        elif at == ActionType.OFFER_COMPENSATION:
            text = action.reply_text or f"I'd like to offer a ${action.compensation_amount} credit as an apology."
            self._ticket.conversation.append(
                CustomerMessage(sender="agent", content=text, timestamp=_now())
            )
            self._ticket.metadata["compensation_offered"] = action.compensation_amount
            followup = followups.get(ActionType.OFFER_COMPENSATION)
            self._ticket.sentiment = _shift_sentiment(self._ticket.sentiment, +2)

        if action.tags:
            for tag in action.tags:
                if tag not in self._ticket.tags:
                    self._ticket.tags.append(tag)

        return followup