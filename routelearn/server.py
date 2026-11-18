"""FastAPI server. Run with: uvicorn routelearn.server:app --reload"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .config import ARMS, JUDGE_ARM, JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT, est_cost
from .features import FEATURE_NAMES, extract
from .gateway import Gateway
from .policies import BanditPolicy
from .providers import AnthropicProvider, LLMJudge, MockProvider
from .storage import QuotaExceeded, SQLiteStateStore

STATE_PATH = os.getenv("ROUTELEARN_STATE", "state.npz")
DATABASE_PATH = os.getenv("ROUTELEARN_DATABASE", "routelearn.sqlite3")
RATE_LIMIT_PER_MINUTE = int(os.getenv("ROUTELEARN_RATE_LIMIT_PER_MINUTE", "60"))
DAILY_SPEND_LIMIT = float(os.getenv("ROUTELEARN_DAILY_SPEND_LIMIT", "10"))
ALLOW_TIER_OVERRIDE = os.getenv("ROUTELEARN_ALLOW_TIER_OVERRIDE", "").lower() in {
    "1", "true", "yes"
}
API_KEYS = tuple(
    key.strip() for key in os.getenv("ROUTELEARN_API_KEYS", "").split(",") if key.strip()
)

CHAT_EXAMPLES = {
    "new_conversation": {
        "summary": "Start a chat",
        "value": {"prompt": "What is TCP?"},
    },
    "follow_up": {
        "summary": "Continue a conversation",
        "value": {
            "prompt": "ok, do it",
            "history": [
                {"role": "user", "content": "Compare these architecture options."},
                {"role": "assistant", "content": "Option A is simpler; option B scales better."},
            ],
        },
    },
    "test_medium": {
        "summary": "Test the medium tier",
        "description": "Requires ROUTELEARN_ALLOW_TIER_OVERRIDE=true.",
        "value": {
            "prompt": "Compare PostgreSQL and MongoDB for a small inventory service.",
            "force_tier": "medium",
        },
    },
    "test_high": {
        "summary": "Test the high (large) tier",
        "description": "Requires ROUTELEARN_ALLOW_TIER_OVERRIDE=true.",
        "value": {
            "prompt": "Design a fault-tolerant payment system and analyze its failure modes.",
            "force_tier": "large",
        },
    },
}
STREAM_EXAMPLES = {
    "stream_answer": {
        "summary": "Stream a response",
        "value": {"prompt": "Explain a TCP three-way handshake in three steps."},
    },
    "test_medium": {
        "summary": "Stream from the medium tier",
        "description": "Requires ROUTELEARN_ALLOW_TIER_OVERRIDE=true.",
        "value": {"prompt": "Compare SQL and NoSQL databases.", "force_tier": "medium"},
    },
    "test_high": {
        "summary": "Stream from the high (large) tier",
        "description": "Requires ROUTELEARN_ALLOW_TIER_OVERRIDE=true.",
        "value": {
            "prompt": "Design a highly available payment service with idempotent retries.",
            "force_tier": "large",
        },
    },
}
EXPLAIN_EXAMPLES = {
    "contextual_routing": {
        "summary": "Explain a follow-up routing decision",
        "value": {
            "prompt": "Implement the safer option",
            "history": [
                {"role": "user", "content": "Analyze the security trade-offs."},
                {"role": "assistant", "content": "The safer option is to validate inputs."},
            ],
            "force_tier": "large",
        },
        "description": "Set ROUTELEARN_ALLOW_TIER_OVERRIDE=true to select a tier explicitly.",
    },
}
FEEDBACK_EXAMPLES = {
    "rate_answer": {
        "summary": "Submit answer feedback",
        "description": "Replace the request ID with one returned by /chat.",
        "value": {"request_id": "f00ba41234cd", "score": 0.9},
    },
}
CONFIG_EXAMPLES = {
    "prefer_lower_cost": {
        "summary": "Prefer lower-cost tiers",
        "value": {"lam": 0.8},
    },
    "enable_cascade": {
        "summary": "Enable answer escalation",
        "value": {"cascade_tau": 0.6},
    },
    "adjust_judge_rate": {
        "summary": "Judge half of the responses",
        "value": {"judge_rate": 0.5},
    },
}


def _create_runtime() -> tuple[BanditPolicy, Gateway, SQLiteStateStore, bool]:
    live = bool(os.getenv("ANTHROPIC_API_KEY"))
    policy = BanditPolicy(lam=float(os.getenv("ROUTELEARN_LAMBDA", 0.6)), gamma=0.995)
    if os.path.exists(STATE_PATH):
        policy.load(STATE_PATH)
    store = SQLiteStateStore(DATABASE_PATH)
    store.initialize_policy(policy)
    policy.store = store
    gateway = Gateway(
        policy,
        AnthropicProvider() if live else MockProvider(),
        LLMJudge() if live else None,
        judge_rate=float(os.getenv("ROUTELEARN_JUDGE_RATE", 0.1)),
        store=store,
    )
    return policy, gateway, store, live


policy, gateway, store, live = _create_runtime()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatReq(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    lam: float | None = Field(None, ge=0, description="cost/quality dial; higher = cheaper")
    force_tier: Literal["small", "medium", "large"] | None = Field(
        None,
        description="Testing override; requires ROUTELEARN_ALLOW_TIER_OVERRIDE=true. "
                    "Use 'large' for the high tier.",
    )

    @model_validator(mode="after")
    def validate_context_size(self):
        if sum(len(message.content) for message in self.history) + len(self.prompt) > 20000:
            raise ValueError("prompt and history must contain at most 20,000 characters")
        return self


class FeedbackReq(BaseModel):
    request_id: str
    score: float = Field(..., ge=0, le=1, description="0 = bad answer, 1 = great answer")


class ConfigReq(BaseModel):
    lam: float | None = Field(None, ge=0)
    judge_rate: float | None = Field(None, ge=0, le=1)
    cascade_tau: float | None = Field(None, ge=0, le=1)


def _estimate_request_cost(gw: Gateway, req: ChatReq) -> float:
    prompt_chars = len(req.prompt.encode("utf-8")) + sum(
        len(message.content.encode("utf-8")) for message in req.history
    )
    input_tokens = max(1, prompt_chars)
    max_tokens = getattr(gw.provider, "max_tokens", 1024)
    model_cost = sum(est_cost(arm, input_tokens, max_tokens) for arm in ARMS)
    judge_input_tokens = max(JUDGE_TOKENS_IN, min(15000, prompt_chars + 7000))
    judge_cost = 3 * est_cost(JUDGE_ARM, judge_input_tokens, JUDGE_TOKENS_OUT)
    return model_cost + judge_cost


def create_app(gw: Gateway | None = None, state_store: SQLiteStateStore | None = None,
               *, rate_limit_per_minute: int = RATE_LIMIT_PER_MINUTE,
               daily_spend_limit: float = DAILY_SPEND_LIMIT,
               api_keys: tuple[str, ...] = API_KEYS,
               allow_tier_override: bool = ALLOW_TIER_OVERRIDE) -> FastAPI:
    if rate_limit_per_minute < 0:
        raise ValueError("rate_limit_per_minute cannot be negative")
    if not math.isfinite(daily_spend_limit) or daily_spend_limit < 0:
        raise ValueError("daily_spend_limit must be a finite non-negative amount")
    active_gateway = gw or gateway
    active_store = state_store or store
    app = FastAPI(title="RouteLearn", version="0.2.0")

    def quota_identity(api_key: str | None) -> str:
        if api_keys:
            if not api_key or not any(hmac.compare_digest(api_key, key) for key in api_keys):
                raise HTTPException(status_code=401, detail="A valid X-API-Key is required")
            return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return "shared"

    def forced_arm(req: ChatReq) -> int | None:
        if req.force_tier is None:
            return None
        if not allow_tier_override:
            raise HTTPException(
                status_code=403,
                detail="Tier override is disabled; set ROUTELEARN_ALLOW_TIER_OVERRIDE=true "
                       "for testing.",
            )
        return next(index for index, arm in enumerate(ARMS) if arm.name == req.force_tier)

    def reserve(req: ChatReq, api_key: str | None) -> tuple[str, str]:
        identity = quota_identity(api_key)
        try:
            reservation = active_store.reserve_usage(
                identity,
                rate_limit_per_minute,
                daily_spend_limit,
                _estimate_request_cost(active_gateway, req),
                time.time(),
            )
            return identity, reservation
        except QuotaExceeded as error:
            detail = ("Request rate limit exceeded" if error.quota == "rate_limit"
                      else "Daily spend limit would be exceeded")
            headers = {"Retry-After": "60"} if error.quota == "rate_limit" else None
            raise HTTPException(status_code=429, detail=detail, headers=headers) from error

    def update_lambda(lam: float | None) -> None:
        if lam is not None:
            active_gateway.policy.set_lam(lam)

    @app.post("/chat", openapi_extra={
        "requestBody": {"content": {"application/json": {"examples": CHAT_EXAMPLES}}}
    })
    def chat(req: ChatReq, x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        force_arm = forced_arm(req)
        identity, reservation = reserve(req, x_api_key)
        try:
            update_lambda(req.lam)
            history = [message.model_dump() for message in req.history]
            result = active_gateway.handle(
                req.prompt, history=history, identity=identity, force_arm=force_arm
            )
            active_store.settle_usage(reservation, result.cost)
        except Exception:
            active_store.release_usage(reservation)
            raise
        return {
            "request_id": result.request_id,
            "answer": result.text,
            "model": result.model,
            "tier": ARMS[result.arm].name,
            "cost": result.cost,
            "escalations": result.escalations,
            "judged_quality": result.judged_quality,
            "why": result.diagnostics,
        }

    @app.post("/chat/stream", openapi_extra={
        "requestBody": {"content": {"application/json": {"examples": STREAM_EXAMPLES}}}
    })
    def chat_stream(req: ChatReq, x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        force_arm = forced_arm(req)
        identity, reservation = reserve(req, x_api_key)
        try:
            update_lambda(req.lam)
            history = [message.model_dump() for message in req.history]
        except Exception:
            active_store.release_usage(reservation)
            raise

        def events():
            try:
                for event, value in active_gateway.stream(
                    req.prompt,
                    history=history,
                    identity=identity,
                    force_arm=force_arm,
                ):
                    if event == "delta":
                        yield f"data: {json.dumps({'type': 'delta', 'text': value})}\n\n"
                    else:
                        result = value
                        active_store.settle_usage(reservation, result.cost)
                        payload = {
                            "type": "done",
                            "request_id": result.request_id,
                            "model": result.model,
                            "tier": ARMS[result.arm].name,
                            "cost": result.cost,
                            "escalations": result.escalations,
                            "judged_quality": result.judged_quality,
                            "why": result.diagnostics,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
            except GeneratorExit:
                active_store.release_usage(reservation)
                raise
            except Exception:
                active_store.release_usage(reservation)
                raise

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/explain", openapi_extra={
        "requestBody": {"content": {"application/json": {"examples": EXPLAIN_EXAMPLES}}}
    })
    def explain(req: ChatReq, x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        quota_identity(x_api_key)
        force_arm = forced_arm(req)
        selected_arm, diagnostics = active_gateway.policy.choose(extract(req.prompt, [
            message.model_dump() for message in req.history
        ]))
        if force_arm is not None:
            selected_arm = force_arm
        return {
            "tiers": [arm.name for arm in ARMS],
            "selected_tier": ARMS[selected_arm].name,
            "forced_tier": req.force_tier,
            **diagnostics,
            "features": dict(zip(
                FEATURE_NAMES,
                np.round(extract(req.prompt, [
                    message.model_dump() for message in req.history
                ]), 3).tolist(),
            )),
        }

    @app.post("/feedback", openapi_extra={
        "requestBody": {"content": {"application/json": {"examples": FEEDBACK_EXAMPLES}}}
    })
    def feedback(req: FeedbackReq, x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        identity = quota_identity(x_api_key)
        if not active_gateway.feedback(req.request_id, req.score, identity):
            raise HTTPException(404, "unknown or already-used request_id")
        return {"ok": True}

    @app.post("/config", openapi_extra={
        "requestBody": {"content": {"application/json": {"examples": CONFIG_EXAMPLES}}}
    })
    def config(req: ConfigReq, x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        quota_identity(x_api_key)
        update_lambda(req.lam)
        if req.judge_rate is not None:
            active_gateway.judge_rate = req.judge_rate
        if "cascade_tau" in req.model_fields_set:
            active_gateway.cascade_tau = req.cascade_tau
        return {
            "lam": active_gateway.policy.lam,
            "judge_rate": active_gateway.judge_rate,
            "cascade_tau": active_gateway.cascade_tau,
        }

    @app.get("/stats")
    def stats(x_api_key: str | None = Header(
        None, description="Required when ROUTELEARN_API_KEYS is configured."
    )):
        quota_identity(x_api_key)
        active_gateway.policy.refresh()
        return {
            **active_gateway.summary(),
            "lam": active_gateway.policy.lam,
            "observations_per_tier": dict(zip(
                [arm.name for arm in ARMS],
                active_gateway.policy.model.counts.tolist(),
            )),
            "live_models": live if active_gateway is gateway else hasattr(
                active_gateway.provider, "client"
            ),
            "rate_limit_per_minute": rate_limit_per_minute,
            "daily_spend_limit": daily_spend_limit,
        }

    return app


app = create_app()
