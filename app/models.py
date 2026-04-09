"""
models.py — Typed Pydantic models for the Customer Support Ticket Resolution OpenEnv.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    SECURITY = "security"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class CustomerSentiment(str, Enum):
    """Tracks how the customer feels. Shifts dynamically with agent behaviour."""
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    DELIGHTED = "delighted"


class ActionType(str, Enum):
    # --- Communication actions ---
    REPLY = "reply"
    REQUEST_INFO = "request_info"
    APPLY_TEMPLATE = "apply_template"
    ADD_NOTE = "add_note"
    # --- Ticket management actions ---
    CATEGORIZE = "categorize"
    SET_PRIORITY = "set_priority"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    OFFER_COMPENSATION = "offer_compensation"
    # --- Tool-use actions (agentic) ---
    # Agent explicitly calls these to retrieve structured data.
    # Each costs one step and returns data in the next observation's tool_result field.
    LOOKUP_ORDER = "lookup_order"        # args: order_id — returns order status/metadata
    CHECK_POLICY = "check_policy"        # args: policy_topic — returns policy text
    TRIGGER_REFUND = "trigger_refund"    # args: order_id, amount — returns confirmation ID
    FLAG_FRAUD = "flag_fraud"            # args: reason — returns case number


class SLAConfig(BaseModel):
    tier: str = "standard"
    # Steps before warning / breach (not wall-clock time, since agents run at variable speed)
    warn_step: int = 3
    breach_step: int = 7
    breach_penalty: float = 0.12


class CustomerMessage(BaseModel):
    sender: str   # 'customer' | 'agent' | 'system'
    content: str
    timestamp: str


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    customer_name: str
    customer_email: str
    # NOTE: priority and category here are the INITIAL values set by the system.
    # Agents must observe them and decide whether to change them.
    priority: TicketPriority
    category: TicketCategory
    status: TicketStatus
    sentiment: CustomerSentiment = CustomerSentiment.NEUTRAL
    conversation: List[CustomerMessage] = Field(default_factory=list)
    internal_notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sla: SLAConfig = Field(default_factory=SLAConfig)
    tags: List[str] = Field(default_factory=list)


class SupportAction(BaseModel):
    """One action emitted by the agent per step."""
    action_type: ActionType
    # Communication
    reply_text: Optional[str] = None
    note_text: Optional[str] = None
    template_id: Optional[str] = None
    # Ticket management
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    escalation_reason: Optional[str] = None
    resolution_summary: Optional[str] = None
    compensation_amount: Optional[float] = None
    tags: Optional[List[str]] = None
    # Tool-use args
    order_id: Optional[str] = None          # for LOOKUP_ORDER, TRIGGER_REFUND
    policy_topic: Optional[str] = None      # for CHECK_POLICY
    amount: Optional[float] = None          # for TRIGGER_REFUND
    reason: Optional[str] = None            # for FLAG_FRAUD


class ToolResult(BaseModel):
    """Structured result returned after a tool-use action."""
    action_type: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    confirmation_id: Optional[str] = None   # REF-XXXX for refunds, SEC-XXXX for fraud flags
    error: Optional[str] = None


class SupportObservation(BaseModel):
    """Full observation returned to the agent each step."""
    ticket: Ticket
    step_number: int
    max_steps: int
    steps_remaining: int
    available_templates: List[Dict[str, str]] = Field(default_factory=list)
    kb_snippets: List[str] = Field(default_factory=list)
    last_action_feedback: Optional[str] = None
    task_instructions: str
    sla_status: str = "ok"   # 'ok' | 'warning' | 'breached'
    customer_sentiment: CustomerSentiment = CustomerSentiment.NEUTRAL
    # Simulated customer reply generated after agent speaks
    customer_followup: Optional[str] = None
    # Live checklist of grader sub-objectives — gives the agent a learning signal
    progress_hints: Dict[str, bool] = Field(default_factory=dict)
    # Tool result from previous step (None if last action was not a tool call)
    tool_result: Optional[ToolResult] = None


class StepResult(BaseModel):
    observation: SupportObservation
    reward: float = Field(..., ge=-1.0, le=1.0)
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
    progress_hints: Dict[str, bool] = Field(default_factory=dict)
    sla_status: str = "ok"