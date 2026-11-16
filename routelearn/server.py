"""FastAPI server.   Run:  uvicorn routelearn.server:app --reload

Without ANTHROPIC_API_KEY it uses a mock provider and relies on POST /feedback for learning.
With a key it calls real models and samples an LLM judge (ROUTELEARN_JUDGE_RATE, default 0.1).
"""
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ARMS
from .features import FEATURE_NAMES, extract
from .gateway import Gateway
from .policies import BanditPolicy
from .providers import AnthropicProvider, LLMJudge, MockProvider

STATE_PATH = os.getenv("ROUTELEARN_STATE", "state.npz")
SAVE_EVERY = 25

policy = BanditPolicy(lam=float(os.getenv("ROUTELEARN_LAMBDA", 0.6)), gamma=0.995)
if os.path.exists(STATE_PATH):
    policy.load(STATE_PATH)

live = bool(os.getenv("ANTHROPIC_API_KEY"))
gateway = Gateway(policy, AnthropicProvider() if live else MockProvider(),
                  LLMJudge() if live else None,
                  judge_rate=float(os.getenv("ROUTELEARN_JUDGE_RATE", 0.1)))
app = FastAPI(title="RouteLearn", version="0.1.0")


class ChatReq(BaseModel):
    prompt: str
    lam: float | None = Field(None, ge=0, description="cost/quality dial; higher = cheaper")


class FeedbackReq(BaseModel):
    request_id: str
    score: float = Field(..., ge=0, le=1, description="0 = bad answer, 1 = great answer")


class ConfigReq(BaseModel):
    lam: float | None = Field(None, ge=0)
    judge_rate: float | None = Field(None, ge=0, le=1)
    cascade_tau: float | None = Field(None, ge=0, le=1)


def _persist():
    if sum(policy.model.counts) % SAVE_EVERY == 0:
        policy.save(STATE_PATH)


@app.post("/chat")
def chat(req: ChatReq):
    if req.lam is not None:
        policy.lam = req.lam
    r = gateway.handle(req.prompt)
    _persist()
    return {"request_id": r.request_id, "answer": r.text, "model": r.model, "tier": ARMS[r.arm].name,
            "cost": r.cost, "escalations": r.escalations, "judged_quality": r.judged_quality,
            "why": r.diagnostics}


@app.post("/explain")
def explain(req: ChatReq):
    """Dry-run: show how each tier scores for this prompt without calling any model."""
    _, diag = policy.choose(extract(req.prompt))
    return {"tiers": [a.name for a in ARMS], **diag,
            "features": dict(zip(FEATURE_NAMES, np.round(extract(req.prompt), 3).tolist()))}


@app.post("/feedback")
def feedback(req: FeedbackReq):
    if not gateway.feedback(req.request_id, req.score):
        raise HTTPException(404, "unknown or already-used request_id")
    _persist()
    return {"ok": True}


@app.post("/config")
def config(req: ConfigReq):
    if req.lam is not None:
        policy.lam = req.lam
    if req.judge_rate is not None:
        gateway.judge_rate = req.judge_rate
    if "cascade_tau" in req.model_fields_set:
        gateway.cascade_tau = req.cascade_tau
    return {"lam": policy.lam, "judge_rate": gateway.judge_rate, "cascade_tau": gateway.cascade_tau}


@app.get("/stats")
def stats():
    return {**gateway.summary(), "lam": policy.lam, "observations_per_tier":
            dict(zip([a.name for a in ARMS], policy.model.counts.tolist())), "live_models": live}
