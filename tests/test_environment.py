"""
tests/test_environment.py — Unit tests for the Customer Support OpenEnv environment.

Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.environment import SupportEnvironment
from app.models import ActionType, SupportAction, TicketCategory, TicketPriority, TicketStatus
from app.tasks import TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_env():
    env = SupportEnvironment("password-reset-easy")
    env.reset()
    return env


@pytest.fixture
def medium_env():
    env = SupportEnvironment("billing-dispute-medium")
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = SupportEnvironment("enterprise-escalation-hard")
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------

def test_all_tasks_present():
    assert set(TASKS.keys()) == {
        "password-reset-easy",
        "billing-dispute-medium",
        "enterprise-escalation-hard",
    }


def test_tasks_have_required_fields():
    required = {"name", "difficulty", "ticket", "instructions", "kb_snippets",
                "available_templates", "max_steps", "grader"}
    for name, task in TASKS.items():
        assert required.issubset(task.keys()), f"Task '{name}' missing fields"


def test_graders_are_callable():
    for name, task in TASKS.items():
        assert callable(task["grader"]), f"Task '{name}' grader is not callable"


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    env = SupportEnvironment("password-reset-easy")
    result = env.reset()
    assert result.observation is not None
    assert result.observation.step_number == 0
    assert result.observation.ticket is not None


def test_reset_clears_state():
    env = SupportEnvironment("password-reset-easy")
    env.reset()
    env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="I will help you reset your password right away."
    ))
    env.reset()
    state = env.state()
    assert state.step_number == 0
    assert state.cumulative_reward == 0.0


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

def test_reply_action_advances_step(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I'll trigger a manual password reset for you immediately."
    ))
    assert result.observation.step_number == 1
    assert result.reward >= 0.0
    assert not result.done


def test_categorize_correct_gives_reward(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.CATEGORIZE,
        category=TicketCategory.ACCOUNT,
    ))
    assert result.reward > 0.0
    assert easy_env.state().ticket.category == TicketCategory.ACCOUNT


def test_categorize_wrong_gives_penalty(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.CATEGORIZE,
        category=TicketCategory.BILLING,  # wrong for task 1
    ))
    assert result.reward < 0.05  # should be penalized or zero


def test_set_priority_gives_reward(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.SET_PRIORITY,
        priority=TicketPriority.MEDIUM,
    ))
    assert result.reward > 0.0
    assert easy_env.state().ticket.priority == TicketPriority.MEDIUM


def test_resolve_ends_episode(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.RESOLVE,
        resolution_summary="Password reset triggered. Customer confirmed access restored.",
    ))
    assert result.done
    assert easy_env.state().ticket.status == TicketStatus.RESOLVED


def test_escalate_ends_episode(easy_env):
    result = easy_env.step(SupportAction(
        action_type=ActionType.ESCALATE,
        escalation_reason="Customer issue requires senior support.",
    ))
    assert result.done
    assert easy_env.state().ticket.status == TicketStatus.ESCALATED


def test_step_after_done_returns_zero(easy_env):
    easy_env.step(SupportAction(
        action_type=ActionType.RESOLVE,
        resolution_summary="Done."
    ))
    result = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Any follow up needed?"
    ))
    assert result.reward == 0.0
    assert result.done


def test_max_steps_terminates_episode():
    env = SupportEnvironment("password-reset-easy")
    env.reset()
    max_steps = TASKS["password-reset-easy"]["max_steps"]
    for i in range(max_steps):
        result = env.step(SupportAction(
            action_type=ActionType.REPLY,
            reply_text=f"Step {i+1}: Still working on your issue, we appreciate your patience."
        ))
    assert result.done


# ---------------------------------------------------------------------------
# Reward range tests
# ---------------------------------------------------------------------------

def test_rewards_in_valid_range(easy_env):
    actions = [
        SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT),
        SupportAction(action_type=ActionType.SET_PRIORITY, priority=TicketPriority.LOW),
        SupportAction(action_type=ActionType.REPLY,
                      reply_text="Hi Alice, I've triggered a manual password reset. Check your inbox."),
        SupportAction(action_type=ActionType.RESOLVE,
                      resolution_summary="Manual password reset sent. Customer confirmed."),
    ]
    for action in actions:
        result = easy_env.step(action)
        assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
        if result.done:
            break


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

def test_grader_returns_float_in_range():
    for name, task in TASKS.items():
        # Empty history → score should be 0
        score = task["grader"]([], task["ticket"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Grader '{name}' returned {score}"


def test_grader_easy_full_path():
    """Simulate a good episode for task 1 and verify score > 0.5."""
    from app.tasks import TASK_1
    import copy

    ticket = copy.deepcopy(TASK_1["ticket"])
    ticket.category = TicketCategory.ACCOUNT
    ticket.status = TicketStatus.RESOLVED

    history = [
        {"step": 1, "action": SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT), "reward": 0.15},
        {"step": 2, "action": SupportAction(action_type=ActionType.SET_PRIORITY, priority=TicketPriority.LOW), "reward": 0.10},
        {"step": 3, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Hi Alice, I've triggered a manual password reset. Please check your inbox and email for the reset link."
        ), "reward": 0.10},
        {"step": 4, "action": SupportAction(action_type=ActionType.RESOLVE, resolution_summary="Password reset sent."), "reward": 0.10},
    ]
    score = TASK_1["grader"](history, ticket)
    assert score >= 0.5, f"Expected >= 0.5, got {score}"


def test_grader_hard_empty_history():
    from app.tasks import TASK_3
    import copy
    ticket = copy.deepcopy(TASK_3["ticket"])
    score = TASK_3["grader"]([], ticket)
    assert score == 0.0 or score < 0.3


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

def test_state_matches_step_count(easy_env):
    for i in range(3):
        easy_env.step(SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Working on your ticket — please hold while I investigate the issue."
        ))
        state = easy_env.state()
        assert state.step_number == i + 1


def test_state_task_name_correct():
    for task_name in TASKS:
        env = SupportEnvironment(task_name)
        env.reset()
        assert env.state().task_name == task_name
