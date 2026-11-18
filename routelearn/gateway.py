"""The Gateway ties everything together: route -> call -> (sampled) judge -> learn -> (optional) cascade."""
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .config import ARMS, N_ARMS
from .features import extract


@dataclass
class Result:
    request_id: str
    text: str
    arm: int
    model: str
    cost: float                       # total spend incl. judge calls and cascade retries
    latency: float
    first_arm: int                    # what the policy originally chose
    judged_quality: float | None = None
    true_quality: float | None = None  # simulator only
    escalations: int = 0
    diagnostics: dict = field(default_factory=dict)


class Gateway:
    def __init__(self, policy, provider, judge=None, judge_rate: float = 0.3,
                 cascade_tau: float | None = None, seed: int = 0, pending_cap: int = 10_000,
                 store=None):
        self.policy, self.provider, self.judge = policy, provider, judge
        self.judge_rate, self.cascade_tau = judge_rate, cascade_tau
        self.rng = np.random.default_rng(seed)
        self.pending: OrderedDict = OrderedDict()  # request_id -> (arm, x, identity)
        self.pending_cap = pending_cap
        self.log: list[Result] = []
        self.store = store

    def handle(self, prompt: str, meta: dict | None = None,
               history: list[dict[str, str]] | None = None,
               identity: str = "shared", force_arm: int | None = None) -> Result:
        result = None
        for event, value in self.stream(
            prompt,
            meta,
            history,
            stream_output=False,
            identity=identity,
            force_arm=force_arm,
        ):
            if event == "result":
                result = value
        if result is None:
            raise RuntimeError("Gateway completed without a result")
        return result

    def stream(self, prompt: str, meta: dict | None = None,
               history: list[dict[str, str]] | None = None,
               stream_output: bool = True,
               identity: str = "shared",
               force_arm: int | None = None) -> Iterator[tuple[str, str | Result]]:
        x = extract(prompt, history)
        arm, diag = self.policy.choose(x, meta)
        if force_arm is not None:
            if not 0 <= force_arm < N_ARMS:
                raise ValueError(f"Invalid forced tier index: {force_arm}")
            arm = force_arm
            diag = {**diag, "forced_tier": ARMS[arm].name}
        first_arm, cost, escalations, judged = arm, 0.0, 0, None
        t0 = time.perf_counter()
        while True:
            if stream_output and hasattr(self.provider, "stream"):
                buffered: list[str] = []
                defer_chunks = self.cascade_tau is not None and arm < N_ARMS - 1
                provider_stream = iter(
                    self.provider.stream(arm, prompt, meta, history=history)
                )
                try:
                    while True:
                        try:
                            chunk = next(provider_stream)
                        except StopIteration as finished:
                            comp = finished.value
                            break
                        if defer_chunks:
                            buffered.append(chunk)
                        else:
                            yield "delta", chunk
                finally:
                    close_stream = getattr(provider_stream, "close", None)
                    if close_stream is not None:
                        close_stream()
            else:
                comp = self.provider.complete(arm, prompt, meta, history=history)
                if stream_output:
                    buffered = [comp.text]
                    defer_chunks = True
            cost += comp.cost
            can_escalate = self.cascade_tau is not None and arm < N_ARMS - 1
            do_judge = self.judge is not None and (can_escalate or self.rng.random() < self.judge_rate)
            if do_judge:
                judged = self.judge.score(prompt, comp, history=history)
                cost += self.judge.cost
                self.policy.update(arm, x, judged)          # learn from every judged attempt
                if can_escalate and judged < self.cascade_tau:
                    arm, escalations = arm + 1, escalations + 1
                    continue
            if stream_output and defer_chunks:
                for chunk in buffered:
                    yield "delta", chunk
            break
        rid = uuid.uuid4().hex[:12]
        if self.store is not None:
            self.store.save_pending(rid, arm, x, identity)
        else:
            self.pending[rid] = (arm, x, identity)
            if len(self.pending) > self.pending_cap:
                self.pending.popitem(last=False)
        res = Result(rid, comp.text, arm, ARMS[arm].model, cost,
                     time.perf_counter() - t0, first_arm,
                     judged, comp.hidden_quality, escalations, diag)
        self.log.append(res)
        yield "result", res

    def feedback(self, request_id: str, quality: float, identity: str = "shared") -> bool:
        """Delayed human feedback (thumbs up/down or 0-1 score) for a past request."""
        if self.store is not None:
            item = self.store.consume_pending(request_id, identity)
        else:
            item = self.pending.get(request_id)
        if item is None:
            return False
        arm, x = item[:2]
        if self.store is None and item[2] != identity:
            return False
        if self.store is None:
            del self.pending[request_id]
        self.policy.update(arm, x, float(np.clip(quality, 0, 1)))
        return True

    def summary(self) -> dict:
        n = len(self.log)
        if not n:
            return {"requests": 0}
        share = [sum(r.arm == a for r in self.log) / n for a in range(N_ARMS)]
        return {"requests": n, "tier_share": dict(zip([a.name for a in ARMS], np.round(share, 3))),
                "avg_cost": float(np.mean([r.cost for r in self.log])),
                "escalation_rate": float(np.mean([r.escalations > 0 for r in self.log]))}
