"""Providers (who answers) and judges (how good was the answer)."""
import math
import re
import time
from dataclasses import dataclass
from typing import Generator

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

    def complete(self, arm_idx: int, prompt: str, meta: dict | None = None,
                 history: list[dict[str, str]] | None = None) -> Completion:
        q = self.expected_quality(arm_idx, meta["difficulty"], meta.get("t", 0), self.degrade)
        q = float(np.clip(q + self.rng.normal(0, self.noise), 0, 1))
        return Completion(f"[sim:{ARMS[arm_idx].name}]", 200, 400, est_cost(ARMS[arm_idx]), 0.0, q)

    def stream(self, arm_idx: int, prompt: str, meta: dict | None = None,
               history: list[dict[str, str]] | None = None) -> Generator[str, None, Completion]:
        comp = self.complete(arm_idx, prompt, meta)
        yield comp.text
        return comp


class SimJudge:
    """Noisy view of the hidden quality (LLM judges are imperfect)."""
    cost = JUDGE_COST

    def __init__(self, seed: int = 1, noise: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def score(self, prompt: str, comp: Completion,
              history: list[dict[str, str]] | None = None) -> float:
        return float(np.clip(comp.hidden_quality + self.rng.normal(0, self.noise), 0, 1))


# ------------------------------------------------------------------ real APIs
class MockProvider:
    """Canned answers so the API server runs without any key."""

    def complete(self, arm_idx: int, prompt: str, meta=None,
                 history: list[dict[str, str]] | None = None) -> Completion:
        arm = ARMS[arm_idx]
        return Completion(f"[mock answer from {arm.model}]", max(1, len(prompt) // 4), 50,
                          est_cost(arm), 0.0)

    def stream(self, arm_idx: int, prompt: str, meta=None,
               history: list[dict[str, str]] | None = None) -> Generator[str, None, Completion]:
        comp = self.complete(arm_idx, prompt, meta, history)
        for start in range(0, len(comp.text), 12):
            yield comp.text[start:start + 12]
        return comp


class AnthropicProvider:
    def __init__(self, max_tokens: int = 1024):
        import anthropic  # lazy import: only needed for real calls
        self.client, self.max_tokens = anthropic.Anthropic(), max_tokens

    @staticmethod
    def _messages(prompt: str, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        return [*(history or []), {"role": "user", "content": prompt}]

    def complete(self, arm_idx: int, prompt: str, meta=None,
                 history: list[dict[str, str]] | None = None) -> Completion:
        arm: Arm = ARMS[arm_idx]
        t0 = time.perf_counter()
        r = self.client.messages.create(model=arm.model, max_tokens=self.max_tokens,
                                        messages=self._messages(prompt, history))
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
        return Completion("".join(block.text for block in r.content if hasattr(block, "text")), tin, tout,
                          est_cost(arm, tin, tout), time.perf_counter() - t0)

    def stream(self, arm_idx: int, prompt: str, meta=None,
               history: list[dict[str, str]] | None = None) -> Generator[str, None, Completion]:
        arm: Arm = ARMS[arm_idx]
        t0 = time.perf_counter()
        with self.client.messages.stream(
            model=arm.model,
            max_tokens=self.max_tokens,
            messages=self._messages(prompt, history),
        ) as stream:
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()
        tin, tout = response.usage.input_tokens, response.usage.output_tokens
        return Completion("".join(block.text for block in response.content
                                  if hasattr(block, "text")), tin, tout,
                          est_cost(arm, tin, tout), time.perf_counter() - t0)


class LLMJudge:
    """Cheap LLM-as-judge: rates an answer 0-10; returned as 0-1."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self._last_cost = JUDGE_COST

    @property
    def cost(self) -> float:
        return self._last_cost

    def score(self, prompt: str, comp: Completion,
              history: list[dict[str, str]] | None = None) -> float:
        msg = ("Rate how well the ANSWER addresses the REQUEST on a 0-10 scale "
               "(10 = correct, complete, well-reasoned). Reply with the number only.\n\n"
               f"CONVERSATION:\n{_format_history(history)}\n\n"
               f"REQUEST:\n{prompt[:3000]}\n\nANSWER:\n{comp.text[:4000]}")
        r = self.client.messages.create(model=JUDGE_ARM.model, max_tokens=5,
                                        messages=[{"role": "user", "content": msg}])
        text = r.content[0].text
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text)
        if not m or not 0 <= float(m.group(1)) <= 10:
            raise ValueError(f"Judge returned an invalid score: {r.content[0].text!r}")
        self._last_cost = est_cost(
            JUDGE_ARM, r.usage.input_tokens, r.usage.output_tokens
        )
        return float(m.group(1)) / 10


def _format_history(history: list[dict[str, str]] | None) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in history or [])
