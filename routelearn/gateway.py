"""The Gateway ties everything together: route -> call -> (sampled) judge -> learn -> (optional) cascade."""
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

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
                 cascade_tau: float | None = None, seed: int = 0, pending_cap: int = 10_000):
        self.policy, self.provider, self.judge = policy, provider, judge
        self.judge_rate, self.cascade_tau = judge_rate, cascade_tau
        self.rng = np.random.default_rng(seed)
        self.pending: OrderedDict = OrderedDict()  # request_id -> (arm, x) for delayed user feedback
        self.pending_cap = pending_cap
        self.log: list[Result] = []

    def handle(self, prompt: str, meta: dict | None = None) -> Result:
        x = extract(prompt)
        arm, diag = self.policy.choose(x, meta)
        first_arm, cost, escalations, judged = arm, 0.0, 0, None
        t0 = time.time()
        while True:
            comp = self.provider.complete(arm, prompt, meta)
            cost += comp.cost
            can_escalate = self.cascade_tau is not None and arm < N_ARMS - 1
            do_judge = self.judge is not None and (can_escalate or self.rng.random() < self.judge_rate)
            if do_judge:
                judged = self.judge.score(prompt, comp)
                cost += self.judge.cost
                self.policy.update(arm, x, judged)          # learn from every judged attempt
                if can_escalate and judged < self.cascade_tau:
                    arm, escalations = arm + 1, escalations + 1
                    continue
            break
        rid = uuid.uuid4().hex[:12]
        self.pending[rid] = (arm, x)
        if len(self.pending) > self.pending_cap:
            self.pending.popitem(last=False)
        res = Result(rid, comp.text, arm, ARMS[arm].model, cost, time.time() - t0, first_arm,
                     judged, comp.hidden_quality, escalations, diag)
        self.log.append(res)
        return res

    def feedback(self, request_id: str, quality: float) -> bool:
        """Delayed human feedback (thumbs up/down or 0-1 score) for a past request."""
        item = self.pending.pop(request_id, None)
        if item is None:
            return False
        arm, x = item
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
