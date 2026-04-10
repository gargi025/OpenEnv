"""
tests/test_environment.py — Full test suite for Customer Support OpenEnv v2.
Run: pytest tests/ -v
"""
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.environment import SupportEnvironment, _shift_sentiment, _SENTIMENT_SCALE
from app.models import (
    ActionType, CustomerSentiment, SupportAction,
    TicketCategory, TicketPriority, TicketStatus,
)
from app.tasks import TASKS


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def easy_env():
    e = SupportEnvironment("password-reset-easy"); e.reset(); return e

@pytest.fixture
def medium_env():
    e = SupportEnvironment("billing-dispute-medium"); e.reset(); return e

@pytest.fixture
def hard_env():
    e = SupportEnvironment("enterprise-escalation-hard"); e.reset(); return e

@pytest.fixture
def security_env():
    e = SupportEnvironment("security-incident-expert"); e.reset(); return e

@pytest.fixture
def shipping_env():
    e = SupportEnvironment("shipping-dispute-expert"); e.reset(); return e


# ===========================================================================
# Task registry
# ===========================================================================

def test_eight_tasks_registered():
    assert set(TASKS.keys()) == {
        "password-reset-easy", "billing-dispute-medium",
        "enterprise-escalation-hard", "security-incident-expert",
        "shipping-dispute-expert", "technical-integration-hard",
        "compliance-deletion-expert", "subscription-retention-medium",
    }

def test_required_fields_present():
    required = {
        "name", "difficulty", "ticket", "instructions", "kb_snippets",
        "available_templates", "max_steps", "ideal_steps", "grader", "followup_responses",
    }
    for name, task in TASKS.items():
        missing = required - task.keys()
        assert not missing, f"Task '{name}' missing: {missing}"

def test_difficulty_levels():
    diffs = {TASKS[t]["difficulty"] for t in TASKS}
    assert diffs == {"easy", "medium", "hard", "expert"}

def test_ideal_steps_less_than_max():
    for name, task in TASKS.items():
        assert task["ideal_steps"] < task["max_steps"], f"{name}: ideal must be < max"

def test_kb_snippets_nonempty():
    for name, task in TASKS.items():
        assert len(task["kb_snippets"]) >= 4, f"{name}: need ≥4 KB snippets"

def test_templates_nonempty():
    for name, task in TASKS.items():
        assert len(task["available_templates"]) >= 2, f"{name}: need ≥2 templates"


# ===========================================================================
# CRITICAL: Grader invariants — judges will check these
# ===========================================================================

def test_all_graders_score_near_zero_on_empty_history():
    """Graders must return a score strictly in (0, 1) — evaluator rejects 0.0 and 1.0.
    On empty history the score must be effectively zero (< 0.01) but not exactly 0.0."""
    for name, task in TASKS.items():
        score, hints = task["grader"]([], task["ticket"])
        assert score < 0.01, f"Task '{name}' scored {score} on empty history — grader is broken"
        assert score > 0.0, f"Task '{name}' returned exactly 0.0 — evaluator requires strictly > 0"

def test_all_graders_return_float_in_range():
    for name, task in TASKS.items():
        score, hints = task["grader"]([], task["ticket"])
        assert isinstance(score, float)
        assert 0.0 < score < 1.0, f"Task '{name}' score {score} must be strictly between 0 and 1"

def test_all_graders_return_dict_hints():
    for name, task in TASKS.items():
        score, hints = task["grader"]([], task["ticket"])
        assert isinstance(hints, dict)
        assert len(hints) >= 4, f"{name}: need ≥4 hint keys for progress tracking"

def test_all_hints_are_false_on_empty_history():
    """Every hint must start False when no actions taken."""
    for name, task in TASKS.items():
        _, hints = task["grader"]([], task["ticket"])
        for key, val in hints.items():
            assert val is False, f"Task '{name}' hint '{key}' is True on empty history"

def test_grader_score_increases_with_good_actions():
    """Score must be strictly higher after good actions than on empty history."""
    from app.tasks import TASK_1
    ticket = copy.deepcopy(TASK_1["ticket"])
    ticket.category = TicketCategory.ACCOUNT  # simulate env mutation
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT), "reward": 0.10},
        {"step": 2, "action": SupportAction(
            action_type=ActionType.SET_PRIORITY, priority=TicketPriority.LOW), "reward": 0.07},
        {"step": 3, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Hi Alice, I have triggered a manual password reset to alice.johnson@example.com. "
                       "Please check your inbox and spam folder within 5 minutes."), "reward": 0.10},
        {"step": 4, "action": SupportAction(
            action_type=ActionType.RESOLVE,
            resolution_summary="Manual password reset triggered. Issue resolved."), "reward": 0.10},
    ]
    score, hints = TASK_1["grader"](history, ticket)
    assert score > 0.50, f"Good episode should score > 0.50, got {score}"
    assert hints["categorized_as_account"]
    assert hints["priority_set"]
    assert hints["ticket_closed"]

def test_grader_hard_escalation_required():
    """Enterprise task: no ESCALATE action → escalated hint must be False."""
    from app.tasks import TASK_3
    ticket = copy.deepcopy(TASK_3["ticket"])
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Hi Carol, I understand your frustration with the API outage."), "reward": 0.08},
    ]
    _, hints = TASK_3["grader"](history, ticket)
    assert not hints["escalated"], "escalated hint must be False without ESCALATE action"

def test_grader_security_no_premature_breach():
    """Security task: premature breach confirmation loses the point."""
    from app.tasks import TASK_4
    ticket = copy.deepcopy(TASK_4["ticket"])
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Your data was breached and your customer PII was stolen."), "reward": 0.08},
    ]
    _, hints = TASK_4["grader"](history, ticket)
    assert not hints["no_premature_confirmation"]

def test_grader_security_no_premature_breach_ok_on_correct_language():
    from app.tasks import TASK_4
    ticket = copy.deepcopy(TASK_4["ticket"])
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="We are conducting a forensic investigation and cannot confirm or deny data exfiltration yet."), "reward": 0.08},
    ]
    _, hints = TASK_4["grader"](history, ticket)
    assert hints["no_premature_confirmation"]

def test_grader_shipping_loyalty_only_if_agent_says_it():
    """Loyalty hint must be False if agent never mentioned loyalty."""
    from app.tasks import TASK_5
    ticket = copy.deepcopy(TASK_5["ticket"])
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.REPLY,
            reply_text="Hi, I'll look into your missing package right away."), "reward": 0.08},
    ]
    _, hints = TASK_5["grader"](history, ticket)
    assert not hints["loyalty_acknowledged"]

def test_grader_deterministic():
    """Same inputs must always produce same outputs."""
    from app.tasks import TASK_2
    ticket = copy.deepcopy(TASK_2["ticket"])
    history = [
        {"step": 1, "action": SupportAction(
            action_type=ActionType.CATEGORIZE, category=TicketCategory.BILLING), "reward": 0.10},
    ]
    s1, h1 = TASK_2["grader"](history, ticket)
    s2, h2 = TASK_2["grader"](history, ticket)
    assert s1 == s2
    assert h1 == h2


# ===========================================================================
# Reset
# ===========================================================================

def test_reset_step_zero(easy_env):
    assert easy_env.state().step_number == 0

def test_reset_cumulative_zero(easy_env):
    assert easy_env.state().cumulative_reward == 0.0

def test_reset_progress_hints_empty(easy_env):
    assert easy_env.state().progress_hints == {}

def test_reset_steps_remaining(easy_env):
    r = easy_env.reset()
    assert r.observation.steps_remaining == 6

def test_reset_clears_after_episode(easy_env):
    easy_env.step(SupportAction(action_type=ActionType.RESOLVE, resolution_summary="Done."))
    easy_env.reset()
    assert easy_env.state().step_number == 0
    assert not easy_env.state().done


# ===========================================================================
# Step mechanics
# ===========================================================================

def test_step_increments_step(easy_env):
    easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I will trigger a password reset to your email right now."))
    assert easy_env.state().step_number == 1

def test_steps_remaining_decrements(easy_env):
    r = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I will trigger a password reset to your email right now."))
    assert r.observation.steps_remaining == 5

def test_resolve_ends_episode(easy_env):
    r = easy_env.step(SupportAction(action_type=ActionType.RESOLVE, resolution_summary="Resolved."))
    assert r.done
    assert easy_env.state().ticket.status == TicketStatus.RESOLVED

def test_escalate_ends_episode(easy_env):
    r = easy_env.step(SupportAction(action_type=ActionType.ESCALATE, escalation_reason="Needs review."))
    assert r.done
    assert easy_env.state().ticket.status == TicketStatus.ESCALATED

def test_max_steps_terminates():
    env = SupportEnvironment("password-reset-easy")
    env.reset()
    result = None
    for i in range(6):
        result = env.step(SupportAction(
            action_type=ActionType.REPLY,
            reply_text=f"Step {i+1}: I'm still investigating your issue. Bear with me please."))
    assert result.done

def test_step_after_done_zero_reward(easy_env):
    easy_env.step(SupportAction(action_type=ActionType.RESOLVE, resolution_summary="Done."))
    r = easy_env.step(SupportAction(action_type=ActionType.REPLY, reply_text="Anything else?"))
    assert r.reward == 0.0
    assert r.done

def test_all_step_rewards_in_range(easy_env):
    actions = [
        SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT),
        SupportAction(action_type=ActionType.SET_PRIORITY, priority=TicketPriority.LOW),
        SupportAction(action_type=ActionType.REPLY,
                      reply_text="Hi Alice, I've triggered a manual password reset. Check your inbox."),
        SupportAction(action_type=ActionType.RESOLVE, resolution_summary="Reset sent. Resolved."),
    ]
    for a in actions:
        r = easy_env.step(a)
        assert -1.0 <= r.reward <= 1.0, f"Reward out of range: {r.reward}"
        if r.done:
            break

def test_correct_categorize_gives_positive_reward(easy_env):
    r = easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT))
    assert r.reward > 0

def test_wrong_categorize_penalizes_cumulative(easy_env):
    r = easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.BILLING))
    assert r.info["cumulative_reward"] < 0

def test_short_reply_penalized(easy_env):
    r = easy_env.step(SupportAction(action_type=ActionType.REPLY, reply_text="Hi"))
    assert r.reward < 0.05

def test_long_reply_higher_than_short(easy_env):
    env2 = SupportEnvironment("password-reset-easy"); env2.reset()
    r_long = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I've triggered a manual password reset to alice.johnson@example.com. "
                   "Please check your inbox and spam folder within 5 minutes. "
                   "If it still doesn't arrive, please let me know and I'll try an alternative delivery method."))
    r_short = env2.step(SupportAction(
        action_type=ActionType.REPLY, reply_text="Hi Alice, I triggered a reset."))
    assert r_long.reward >= r_short.reward

def test_redundant_action_eventually_penalized(easy_env):
    for _ in range(3):
        easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT))
    r = easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT))
    assert r.info["cumulative_reward"] < 0.40  # redundancy penalty reduces overall score

def test_add_note_earns_reward(easy_env):
    r = easy_env.step(SupportAction(
        action_type=ActionType.ADD_NOTE,
        note_text="Customer verified. Reset triggered manually via admin panel."))
    assert r.reward > 0

def test_offer_compensation_earns_reward(medium_env):
    r = medium_env.step(SupportAction(
        action_type=ActionType.OFFER_COMPENSATION,
        reply_text="I'm adding a $10 account credit as a goodwill gesture for this billing error.",
        compensation_amount=10.0))
    assert r.reward > 0


# ===========================================================================
# Sentiment & followup (new features)
# ===========================================================================

def test_sentiment_shift_up():
    s = _shift_sentiment(CustomerSentiment.FRUSTRATED, +1)
    assert s == CustomerSentiment.NEUTRAL

def test_sentiment_shift_down():
    s = _shift_sentiment(CustomerSentiment.NEUTRAL, -1)
    assert s == CustomerSentiment.FRUSTRATED

def test_sentiment_clamps_at_delighted():
    s = _shift_sentiment(CustomerSentiment.DELIGHTED, +5)
    assert s == CustomerSentiment.DELIGHTED

def test_sentiment_clamps_at_angry():
    s = _shift_sentiment(CustomerSentiment.ANGRY, -5)
    assert s == CustomerSentiment.ANGRY

def test_reply_improves_sentiment(easy_env):
    before = easy_env.state().ticket.sentiment
    easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I've triggered a manual password reset. Please check your inbox."))
    after = easy_env.state().ticket.sentiment
    assert _SENTIMENT_SCALE.index(after) >= _SENTIMENT_SCALE.index(before)

def test_compensation_improves_sentiment_by_two(medium_env):
    before = medium_env.state().ticket.sentiment
    medium_env.step(SupportAction(
        action_type=ActionType.OFFER_COMPENSATION,
        reply_text="I'm adding a $10 credit as an apology for the duplicate charge.",
        compensation_amount=10.0))
    after = medium_env.state().ticket.sentiment
    assert _SENTIMENT_SCALE.index(after) >= _SENTIMENT_SCALE.index(before) + 2 or \
           after == CustomerSentiment.DELIGHTED

def test_customer_followup_in_observation(easy_env):
    r = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I've triggered a manual password reset to your email."))
    assert r.observation.customer_followup is not None
    assert len(r.observation.customer_followup) > 5

def test_followup_appended_to_conversation(easy_env):
    easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I've triggered a manual password reset to alice.johnson@example.com."))
    ticket = easy_env.state().ticket
    senders = [m.sender for m in ticket.conversation]
    assert senders.count("agent") >= 1
    assert senders.count("customer") >= 2  # original + followup


# ===========================================================================
# SLA
# ===========================================================================

def test_sla_starts_ok(easy_env):
    assert easy_env.state().sla_status == "ok"

def test_sla_status_in_observation(easy_env):
    r = easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, I've triggered a manual password reset to your email."))
    assert r.observation.sla_status in ("ok", "warning", "breached")

def test_sla_can_breach_at_breach_step():
    """Force an episode to breach by running many steps without closing."""
    env = SupportEnvironment("enterprise-escalation-hard")  # breach_step=4
    env.reset()
    for i in range(5):
        r = env.step(SupportAction(
            action_type=ActionType.ADD_NOTE,
            note_text=f"Note {i}: still investigating the multiple issues."))
        if r.observation.sla_status == "breached":
            break
    assert env.state().sla_status == "breached"


# ===========================================================================
# Progress hints
# ===========================================================================

def test_progress_hints_empty_before_any_step(easy_env):
    assert easy_env.state().progress_hints == {}

def test_progress_hints_populated_after_step(easy_env):
    easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT))
    hints = easy_env.state().progress_hints
    assert isinstance(hints, dict)
    assert len(hints) > 0

def test_correct_action_flips_hint(easy_env):
    easy_env.step(SupportAction(action_type=ActionType.CATEGORIZE, category=TicketCategory.ACCOUNT))
    hints = easy_env.state().progress_hints
    assert hints.get("categorized_as_account") is True


# ===========================================================================
# State
# ===========================================================================

def test_state_task_name(easy_env):
    assert easy_env.state().task_name == "password-reset-easy"

def test_state_step_matches(easy_env):
    easy_env.step(SupportAction(
        action_type=ActionType.REPLY,
        reply_text="Hi Alice, triggering password reset now."))
    assert easy_env.state().step_number == 1

def test_state_all_tasks():
    for name in TASKS:
        env = SupportEnvironment(name)
        env.reset()
        assert env.state().task_name == name

def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        SupportEnvironment("nonexistent-xyz")