"""
environment.py — Stateful episode engine for the Customer Support OpenEnv.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ActionType, CustomerMessage, CustomerSentiment,
    ResetResult, StateResult, StepResult,
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


def _shift_sentiment(current: CustomerSentiment, delta: int) -> CustomerSentiment:
    idx = _SENTIMENT_SCALE.index(current)
    return _SENTIMENT_SCALE[max(0, min(len(_SENTIMENT_SCALE) - 1, idx + delta))]


class SupportEnvironment:
    """
    One episode of the Customer Support Ticket Resolution environment.

    Implements the OpenEnv interface: reset() / step() / state()
    """

    # ---- Dense reward weights ----
    _R_REPLY_BASE       = 0.08
    _R_REPLY_LENGTH     = 0.04   # bonus for substantive replies (>100 chars)
    _R_CATEGORIZE_OK    = 0.10
    _R_PRIORITY         = 0.07
    _R_TEMPLATE         = 0.05
    _R_ESCALATE         = 0.05
    _R_NOTE             = 0.03
    _R_COMPENSATION     = 0.07
    _R_SENTIMENT_UP     = 0.04   # each time sentiment improves one level
    _P_WRONG_CATEGORY   = -0.08
    _P_REDUNDANT        = -0.04
    _P_SHORT_REPLY      = -0.02

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        self._ticket = copy.deepcopy(self._task["ticket"])
        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0
        self._action_counts = {}
        self._sla_breached = False
        self._first_reply_done = False
        self._progress_hints = {}
        self._sla_status = "ok"
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

        reward, feedback = self._compute_reward(action)
        reward += sla_penalty

        followup = self._apply_action(action)

        self._history.append({"step": self._step, "action": action, "reward": reward})
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._action_counts[action.action_type] = self._action_counts.get(action.action_type, 0) + 1

        terminal = action.action_type in {ActionType.RESOLVE, ActionType.ESCALATE}
        max_hit = self._step >= self._task["max_steps"]

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
            reward=min(max(round(reward, 4), 0.0), 1.0),
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
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self, feedback: Optional[str], followup: Optional[str]) -> SupportObservation:
        max_s = self._task["max_steps"]
        return SupportObservation(
            ticket=copy.deepcopy(self._ticket),
            step_number=self._step,
            max_steps=max_s,
            steps_remaining=max(0, max_s - self._step),
            available_templates=self._task.get("available_templates", []),
            kb_snippets=self._task.get("kb_snippets", []),
            last_action_feedback=feedback,
            task_instructions=self._task["instructions"],
            sla_status=self._sla_status,
            customer_sentiment=self._ticket.sentiment,
            customer_followup=followup,
            progress_hints=copy.copy(self._progress_hints),
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

        # Redundancy
        if prev >= 2 and at not in {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.ADD_NOTE}:
            r += self._P_REDUNDANT
            msgs.append(f"Redundant {at} (×{prev}).")

        if at in {ActionType.REPLY, ActionType.REQUEST_INFO}:
            text = (action.reply_text or "").strip()
            if len(text) < 20:
                r += self._P_SHORT_REPLY
                msgs.append("Reply too short.")
            else:
                r += self._R_REPLY_BASE
                if len(text) >= 100:
                    r += self._R_REPLY_LENGTH
                msgs.append(f"Reply sent (len={len(text)}).")
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

        return round(r, 4), " | ".join(msgs) or "Action processed."

    def _apply_action(self, action: SupportAction) -> Optional[str]:
        """Mutate ticket state; return simulated customer followup or None."""
        at = action.action_type
        followups = self._task.get("followup_responses", {})
        followup: Optional[str] = None
        prev_sentiment = self._ticket.sentiment

        if at in {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.APPLY_TEMPLATE}:
            text = action.reply_text or f"[Template applied: {action.template_id}]"
            self._ticket.conversation.append(
                CustomerMessage(sender="agent", content=text, timestamp=_now())
            )
            if self._ticket.status == TicketStatus.OPEN:
                self._ticket.status = TicketStatus.IN_PROGRESS
            if at == ActionType.REQUEST_INFO:
                self._ticket.status = TicketStatus.WAITING_CUSTOMER

            # Simulate customer reply
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
            # Compensation jumps sentiment by 2
            self._ticket.sentiment = _shift_sentiment(self._ticket.sentiment, +2)

        if action.tags:
            for tag in action.tags:
                if tag not in self._ticket.tags:
                    self._ticket.tags.append(tag)

        return followup