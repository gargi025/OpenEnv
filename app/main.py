"""
main.py — FastAPI server implementing the OpenEnv HTTP interface.

Endpoints
---------
POST /reset       start or restart an episode
POST /step        send one action, receive observation + reward
GET  /state       inspect current state (no side effects)
GET  /tasks       list all tasks with metadata
GET  /health      liveness probe
POST /reset_all   clear all sessions (for batch eval)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .environment import SupportEnvironment
from .models import ResetResult, StateResult, StepResult, SupportAction
from .tasks import TASKS

_sessions: Dict[str, SupportEnvironment] = {}
_DEFAULT_TASK = "password-reset-easy"


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    action: SupportAction
    session_id: Optional[str] = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = SupportEnvironment(_DEFAULT_TASK)
    env.reset()
    _sessions["default"] = env
    yield
    _sessions.clear()


app = FastAPI(
    title="Customer Support Ticket Resolution — OpenEnv",
    description=(
        "OpenEnv-compliant RL environment for customer support ticket resolution. "
        "Five tasks across four difficulty levels (easy → medium → hard → expert). "
        "Features: dense rewards, SLA pressure, dynamic customer sentiment, "
        "simulated multi-turn conversations, and trajectory-aware graders."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {
        "message": "Customer Support Environment API",
        "endpoints": ["/reset", "/step", "/state"]
    }
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "tasks": list(TASKS.keys()),
        "task_count": len(TASKS),
        "active_sessions": len(_sessions),
        "difficulties": {n: t["difficulty"] for n, t in TASKS.items()},
    }


@app.get("/tasks")
async def list_tasks():
    return {
        name: {
            "difficulty": task["difficulty"],
            "max_steps": task["max_steps"],
            "ideal_steps": task.get("ideal_steps"),
            "instructions": task["instructions"],
            "kb_snippets": len(task.get("kb_snippets", [])),
            "templates": len(task.get("available_templates", [])),
            "sla_tier": task["ticket"].sla.tier,
            "sla_breach_step": task["ticket"].sla.breach_step,
            "sla_breach_penalty": task["ticket"].sla.breach_penalty,
            "initial_sentiment": task["ticket"].sentiment,
        }
        for name, task in TASKS.items()
    }


@app.post("/reset", response_model=ResetResult)
async def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    sid = req.session_id or "default"
    task_name = req.task_name
    if task_name:
        if task_name not in TASKS:
            raise HTTPException(422, detail=f"Unknown task '{task_name}'. Available: {list(TASKS)}")
        _sessions[sid] = SupportEnvironment(task_name)
    elif sid not in _sessions:
        _sessions[sid] = SupportEnvironment(_DEFAULT_TASK)
    return _sessions[sid].reset()


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest):
    sid = req.session_id or "default"
    env = _sessions.get(sid)
    if env is None:
        raise HTTPException(400, detail=f"No session '{sid}'. Call /reset first.")
    return env.step(req.action)


@app.get("/state", response_model=StateResult)
async def state(session_id: str = Query(default="default")):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(400, detail=f"No session '{session_id}'. Call /reset first.")
    return env.state()


@app.post("/reset_all")
async def reset_all():
    cleared = list(_sessions.keys())
    _sessions.clear()
    env = SupportEnvironment(_DEFAULT_TASK)
    env.reset()
    _sessions["default"] = env
    return {"cleared": cleared, "status": "ok"}
