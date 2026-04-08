"""
environment.py — Stateful Customer Support environment engine.

Holds runtime state for one episode. Called by the FastAPI routes.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    ActionType,
    CustomerMessage,
    ResetResult,
    StateResult,
    StepResult,
    SupportAction,
    SupportObservation,
    Ticket,
    TicketStatus,
)
from .tasks import TASKS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupportEnvironment:
    """
    One episode of the Customer Support environment.

    Usage:
        env = SupportEnvironment(task_name="password-reset-easy")
        reset_result = env.reset()
        step_result  = env.step(action)
        state_result = env.state()
    """

    # Reward weights — used to produce dense, per-step rewards
    _REWARD_REPLY = 0.10          # any meaningful reply/request
    _REWARD_CATEGORIZE = 0.15     # correct categorize action
    _REWARD_PRIORITY = 0.10       # any explicit priority setting
    _REWARD_TEMPLATE = 0.05       # using a provided template
    _REWARD_RESOLVE_BONUS = 0.20  # grader-gated resolve bonus
    _PENALTY_WRONG_CAT = -0.05   # wrong category
    _PENALTY_REDUNDANT = -0.03   # same action type repeated unnecessarily

    def __init__(self, task_name: str) -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS)}")
        self.task_name = task_name
        self._task = TASKS[task_name]

        # These are set properly in reset()
        self._ticket: Ticket = None  # type: ignore[assignment]
        self._step: int = 0
        self._done: bool = False
        self._history: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._action_type_counts: Dict[ActionType, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        """Start a fresh episode."""
        # Deep-copy so repeated resets don't share mutable state
        self._ticket = copy.deepcopy(self._task["ticket"])
        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0
        self._action_type_counts = {}

        obs = self._build_observation(last_feedback=None)
        return ResetResult(observation=obs, info={"task": self.task_name})

    def step(self, action: SupportAction) -> StepResult:
        """Advance the episode by one agent action."""
        if self._done:
            obs = self._build_observation(last_feedback="Episode already done.")
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "already_done"})

        self._step += 1
        reward, feedback = self._compute_reward(action)

        # Apply action side-effects to the ticket
        self._apply_action(action, feedback)

        # Record history
        self._history.append({"step": self._step, "action": action, "reward": reward})
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._action_type_counts[action.action_type] = (
            self._action_type_counts.get(action.action_type, 0) + 1
        )

        # Episode termination conditions
        terminal_actions = {ActionType.RESOLVE, ActionType.ESCALATE}
        max_steps_reached = self._step >= self._task["max_steps"]
        agent_resolved = action.action_type in terminal_actions

        if agent_resolved or max_steps_reached:
            self._done = True
            # Final grader bonus
            grader_score = self._task["grader"](self._history, self._ticket)
            # Map grader score to a bonus (up to 0.30 extra on top of step rewards)
            bonus = round(grader_score * 0.30, 4)
            reward = round(reward + bonus, 4)
            self._cumulative_reward = round(self._cumulative_reward + bonus, 4)
            feedback += f" [Episode ended. Grader score: {grader_score:.3f}. Bonus: +{bonus:.3f}]"
            self._history[-1]["reward"] = reward  # update last step reward

        obs = self._build_observation(last_feedback=feedback)
        return StepResult(
            observation=obs,
            reward=min(max(reward, 0.0), 1.0),
            done=self._done,
            info={
                "step": self._step,
                "cumulative_reward": self._cumulative_reward,
                "ticket_status": self._ticket.status,
            },
        )

    def state(self) -> StateResult:
        """Return current episode state (for inspection / debugging)."""
        grader_scores = {}
        if self._history:
            grader_scores["current"] = self._task["grader"](self._history, self._ticket)

        return StateResult(
            ticket=self._ticket,
            step_number=self._step,
            max_steps=self._task["max_steps"],
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            task_name=self.task_name,
            grader_scores=grader_scores,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self, last_feedback: Optional[str]) -> SupportObservation:
        return SupportObservation(
            ticket=copy.deepcopy(self._ticket),
            step_number=self._step,
            max_steps=self._task["max_steps"],
            available_templates=self._task.get("available_templates", []),
            kb_snippets=self._task.get("kb_snippets", []),
            last_action_feedback=last_feedback,
            task_instructions=self._task["instructions"],
        )

    def _compute_reward(self, action: SupportAction) -> tuple[float, str]:
        """
        Dense reward function.
        Returns (reward_float, human_readable_feedback).
        """
        reward = 0.0
        feedbacks: List[str] = []

        at = action.action_type
        prev_count = self._action_type_counts.get(at, 0)

        # --- Redundancy penalty ---
        if prev_count >= 2 and at not in {ActionType.REPLY, ActionType.REQUEST_INFO}:
            reward += self._PENALTY_REDUNDANT
            feedbacks.append(f"Redundant {at} action (used {prev_count} times already).")

        # --- Reply / request_info ---
        if at in {ActionType.REPLY, ActionType.REQUEST_INFO}:
            text = action.reply_text or ""
            if len(text.strip()) < 20:
                feedbacks.append("Reply too short — provide more detail.")
            else:
                reward += self._REWARD_REPLY
                feedbacks.append("Good reply sent.")

        # --- Template use ---
        elif at == ActionType.APPLY_TEMPLATE:
            reward += self._REWARD_TEMPLATE
            feedbacks.append(f"Template '{action.template_id}' applied.")

        # --- Categorize ---
        elif at == ActionType.CATEGORIZE:
            if action.category is not None:
                expected = self._task["ticket"].category
                if action.category == expected:
                    reward += self._REWARD_CATEGORIZE
                    feedbacks.append(f"Correct category: {action.category}.")
                else:
                    reward += self._PENALTY_WRONG_CAT
                    feedbacks.append(f"Category mismatch (set {action.category}, expected {expected}).")
            else:
                feedbacks.append("CATEGORIZE action requires 'category' field.")

        # --- Priority ---
        elif at == ActionType.SET_PRIORITY:
            if action.priority is not None:
                reward += self._REWARD_PRIORITY
                feedbacks.append(f"Priority set to {action.priority}.")
            else:
                feedbacks.append("SET_PRIORITY action requires 'priority' field.")

        # --- Escalate ---
        elif at == ActionType.ESCALATE:
            reward += 0.05
            feedbacks.append("Ticket escalated.")

        # --- Resolve ---
        elif at == ActionType.RESOLVE:
            # Resolve bonus depends on grader — computed in step()
            feedbacks.append("Resolve action received — computing grader score.")

        return round(reward, 4), " | ".join(feedbacks) if feedbacks else "Action processed."

    def _apply_action(self, action: SupportAction, feedback: str) -> None:
        """Mutate ticket state based on the action."""
        at = action.action_type

        if at in {ActionType.REPLY, ActionType.REQUEST_INFO, ActionType.APPLY_TEMPLATE}:
            text = action.reply_text or f"[Template: {action.template_id}]"
            self._ticket.conversation.append(
                CustomerMessage(sender="agent", content=text, timestamp=_now())
            )
            if self._ticket.status == TicketStatus.OPEN:
                self._ticket.status = TicketStatus.IN_PROGRESS
            if at == ActionType.REQUEST_INFO:
                self._ticket.status = TicketStatus.WAITING_CUSTOMER

        elif at == ActionType.CATEGORIZE and action.category:
            self._ticket.category = action.category

        elif at == ActionType.SET_PRIORITY and action.priority:
            self._ticket.priority = action.priority

        elif at == ActionType.ESCALATE:
            self._ticket.status = TicketStatus.ESCALATED
            if action.escalation_reason:
                self._ticket.metadata["escalation_reason"] = action.escalation_reason

        elif at == ActionType.RESOLVE:
            self._ticket.status = TicketStatus.RESOLVED
            if action.resolution_summary:
                self._ticket.metadata["resolution_summary"] = action.resolution_summary
