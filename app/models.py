"""
models.py — Typed Pydantic models for the Customer Support Ticket Resolution environment.

Observation  : what the agent sees each step
Action       : what the agent can do
StepResult   : returned by step()
ResetResult  : returned by reset()
StateResult  : returned by state()
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    REFUND = "refund"
    SHIPPING = "shipping"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ActionType(str, Enum):
    REPLY = "reply"                    # Send a reply to the customer
    CATEGORIZE = "categorize"          # Set ticket category
    SET_PRIORITY = "set_priority"      # Set ticket priority
    ESCALATE = "escalate"              # Escalate to a human agent
    RESOLVE = "resolve"                # Mark ticket as resolved
    REQUEST_INFO = "request_info"      # Ask customer for more info
    APPLY_TEMPLATE = "apply_template"  # Use a canned response template


# ---------------------------------------------------------------------------
# Ticket & Message models
# ---------------------------------------------------------------------------

class CustomerMessage(BaseModel):
    sender: str = Field(..., description="'customer' or 'agent'")
    content: str = Field(..., description="Message body text")
    timestamp: str = Field(..., description="ISO-8601 timestamp string")


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    customer_name: str
    customer_email: str
    priority: TicketPriority
    category: TicketCategory
    status: TicketStatus
    conversation: List[CustomerMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class SupportAction(BaseModel):
    """
    The agent emits exactly one SupportAction per step.

    action_type   : one of ActionType
    reply_text    : required when action_type is REPLY or REQUEST_INFO
    category      : required when action_type is CATEGORIZE
    priority      : required when action_type is SET_PRIORITY
    template_id   : required when action_type is APPLY_TEMPLATE
    escalation_reason : optional note when action_type is ESCALATE
    resolution_summary : optional summary when action_type is RESOLVE
    """
    action_type: ActionType
    reply_text: Optional[str] = None
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    template_id: Optional[str] = None
    escalation_reason: Optional[str] = None
    resolution_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class SupportObservation(BaseModel):
    """Everything the agent can observe at a given step."""
    ticket: Ticket
    step_number: int
    max_steps: int
    available_templates: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {id, name, preview} canned response templates"
    )
    kb_snippets: List[str] = Field(
        default_factory=list,
        description="Relevant knowledge-base snippets for this ticket"
    )
    last_action_feedback: Optional[str] = Field(
        None,
        description="Human-readable feedback on the previous action"
    )
    task_instructions: str = Field(
        ...,
        description="Natural-language description of what the agent must accomplish"
    )


# ---------------------------------------------------------------------------
# Step / Reset / State result models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: SupportObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: SupportObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    ticket: Ticket
    step_number: int
    max_steps: int
    cumulative_reward: float
    done: bool
    task_name: str
    grader_scores: Dict[str, float] = Field(default_factory=dict)
