"""
main.py — FastAPI server implementing the OpenEnv HTTP interface.

Endpoints
---------
POST /reset          — start or restart an episode
POST /step           — send one action, get observation + reward
GET  /state          — inspect current episode state (no side effects)
GET  /tasks          — list available tasks
GET  /health         — liveness probe
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .environment import SupportEnvironment
from .models import (
    ResetResult,
    StateResult,
    StepResult,
    SupportAction,
)
from .tasks import TASKS

# ---------------------------------------------------------------------------
# Session store — one environment per session_id (in-memory, single process)
# ---------------------------------------------------------------------------

_sessions: Dict[str, SupportEnvironment] = {}
_DEFAULT_TASK = "password-reset-easy"


# ---------------------------------------------------------------------------
# Request/response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action: SupportAction
    session_id: Optional[str] = "default"


class StateRequest(BaseModel):
    session_id: Optional[str] = "default"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm a default session so /reset immediately works
    _sessions["default"] = SupportEnvironment(_DEFAULT_TASK)
    yield
    _sessions.clear()


app = FastAPI(
    title="Customer Support Ticket Resolution — OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on real-world customer support ticket resolution tasks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.get("/tasks")
async def list_tasks():
    return {
        name: {
            "difficulty": task["difficulty"],
            "max_steps": task["max_steps"],
            "instructions": task["instructions"],
        }
        for name, task in TASKS.items()
    }


@app.post("/reset", response_model=ResetResult)
async def reset(req: ResetRequest = None):
    """
    Start or restart an episode.
    If task_name is omitted, the previously configured task is reused
    (or the default task on first call).
    """
    if req is None:
        req = ResetRequest()

    session_id = req.session_id or "default"
    task_name = req.task_name

    if task_name:
        if task_name not in TASKS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown task '{task_name}'. Available: {list(TASKS)}",
            )
        _sessions[session_id] = SupportEnvironment(task_name)
    elif session_id not in _sessions:
        _sessions[session_id] = SupportEnvironment(_DEFAULT_TASK)

    env = _sessions[session_id]
    return env.reset()


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest):
    """Advance the episode by one agent action."""
    session_id = req.session_id or "default"
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session '{session_id}'. Call /reset first.",
        )
    return env.step(req.action)


@app.get("/state", response_model=StateResult)
async def state(session_id: str = Query(default="default")):
    """Return current episode state without advancing it."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session '{session_id}'. Call /reset first.",
        )
    return env.state()
