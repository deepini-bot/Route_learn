"""Providers (who answers) and judges (how good was the answer)."""
import math
import re
import time
from dataclasses import dataclass

import numpy as np

from .config import ARMS, JUDGE_ARM, JUDGE_COST, Arm, est_cost


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    cost: float
    latency: float
    hidden_quality: float | None = None  # simulator ground truth; None for real providers


# ---------------------------------------------------------------- simulator
class SimProvider:
    """Quality of arm a on a prompt with latent difficulty d:  sigmoid(K * (capability_a - d)) + noise.

    `degrade=(arm, at_step, delta)` simulates a SILENT model regression: from `at_step` on, that
    arm's capability shifts by `delta` (e.g. a provider-side model update made the cheap tier worse).
    """
    K = 7.0

    def __init__(self, seed: int = 0, noise: float = 0.06, degrade: tuple | None = None):
        self.rng = np.random.default_rng(seed)
        self.noise, self.degrade = noise, degrade

    @classmethod
    def expected_quality(cls, arm_idx: int, difficulty: float, t: int = 0,
                         degrade: tuple | None = None) -> float:
        cap = ARMS[arm_idx].capability
        if degrade and arm_idx == degrade[0] and t >= degrade[1]:
            cap += degrade[2]
        return 1.0 / (1.0 + math.exp(-cls.K * (cap - difficulty)))

    def complete(self, arm_idx: int, prompt: str, meta: dict | None = None) -> Completion:
        q = self.expected_quality(arm_idx, meta["difficulty"], meta.get("t", 0), self.degrade)
        q = float(np.clip(q + self.rng.normal(0, self.noise), 0, 1))
        return Completion(f"[sim:{ARMS[arm_idx].name}]", 200, 400, est_cost(ARMS[arm_idx]), 0.0, q)


class SimJudge:
    """Noisy view of the hidden quality (LLM judges are imperfect)."""
    cost = JUDGE_COST

    def __init__(self, seed: int = 1, noise: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def score(self, prompt: str, comp: Completion) -> float:
        return float(np.clip(comp.hidden_quality + self.rng.normal(0, self.noise), 0, 1))


# ------------------------------------------------------------------ real APIs
class MockProvider:
    """Canned answers so the API server runs without any key."""

    def complete(self, arm_idx: int, prompt: str, meta=None) -> Completion:
        arm = ARMS[arm_idx]
        return Completion(f"[mock answer from {arm.model}]", max(1, len(prompt) // 4), 50,
                          est_cost(arm), 0.0)


class AnthropicProvider:
    def __init__(self, max_tokens: int = 1024):
        import anthropic  # lazy import: only needed for real calls
        self.client, self.max_tokens = anthropic.Anthropic(), max_tokens

    def complete(self, arm_idx: int, prompt: str, meta=None) -> Completion:
        arm: Arm = ARMS[arm_idx]
        t0 = time.time()
        r = self.client.messages.create(model=arm.model, max_tokens=self.max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
        return Completion(r.content[0].text, tin, tout,
                          est_cost(arm, tin, tout), time.time() - t0)


class LLMJudge:
    """Cheap LLM-as-judge: rates an answer 0-10; returned as 0-1."""
    cost = JUDGE_COST

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def score(self, prompt: str, comp: Completion) -> float:
        msg = ("Rate how well the ANSWER addresses the REQUEST on a 0-10 scale "
               "(10 = correct, complete, well-reasoned). Reply with the number only.\n\n"
               f"REQUEST:\n{prompt[:3000]}\n\nANSWER:\n{comp.text[:4000]}")
        r = self.client.messages.create(model=JUDGE_ARM.model, max_tokens=5,
                                        messages=[{"role": "user", "content": msg}])
        m = re.search(r"\d+(\.\d+)?", r.content[0].text)
        return min(float(m.group()) / 10, 1.0) if m else 0.5
