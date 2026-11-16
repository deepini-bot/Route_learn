"""Synthetic prompt workload with a hidden difficulty and optional concept drift.

The router only ever sees TEXT features; difficulty is latent. Prompt style correlates with
difficulty only noisily (deliberate label noise), so a fixed hand-written heuristic is imperfect.

Drift: from `drift_at` onward, code-related prompts become harder (e.g. users start pasting
bigger codebases). A static router cannot notice; a discounted bandit re-learns.
"""
import numpy as np

TOPICS = ["photosynthesis", "the French Revolution", "a hash map", "TCP congestion control",
          "inflation", "a binary search tree", "neural networks", "the water cycle",
          "database indexing", "supply chains", "black holes", "a rate limiter",
          "protein folding", "compound interest", "the Roman Empire", "a message queue"]
SIMPLE = ["What is {t}?", "Define {t}.", "Who is associated with {t}?", "Give one fact about {t}."]
MEDIUM = ["Explain how {t} works with an example.", "Summarize the main ideas of {t}.",
          "Write a short email about {t}.", "Describe the difference between {t} and a related concept."]
HARD = ["Analyze the trade-offs involved in {t} and evaluate alternative strategies.",
        "Design a scalable architecture around {t} and compare the options step by step.",
        "Derive a formal argument about {t} and critique its assumptions.",
        "Debug and refactor an implementation involving {t}, then optimize it."]
FILL = ["Consider the following context carefully.", "The audience is a mixed technical group.",
        "Earlier attempts were inconsistent.", "Latency and budget both matter here.",
        "Be specific about edge cases.", "Assume the reader is new to the area.",
        "Keep the tone professional."]
CODE_ADD = [" Use a Python function and an SQL query.", " Include an API and an algorithm.",
            " Show a class and a regex where useful."]
REASON_ADD = [" Compare the trade-offs.", " Analyze why this happens and evaluate fixes.",
              " Prove each step."]
CONSTRAINT_ADD = [" You must use exactly three paragraphs.", " Do not exceed one page."]
MULTI_ADD = [" First outline it, then implement it, and finally review it."]


def _sample_difficulty(rng) -> float:
    r = rng.random()
    if r < 0.40:
        return float(rng.beta(2, 8))       # easy
    if r < 0.75:
        return float(rng.beta(5, 5))       # medium
    return float(rng.beta(8, 3))           # hard


def make_prompt(rng, d0: float):
    pool_idx = int(np.clip(round(2 * d0 + rng.normal(0, 0.45)), 0, 2))   # noisy style choice
    base = [SIMPLE, MEDIUM, HARD][pool_idx]
    p = base[rng.integers(len(base))].format(t=TOPICS[rng.integers(len(TOPICS))])
    for _ in range(int(np.clip(round(d0 * 5 + rng.normal(0, 1.5)), 0, 8))):
        p += " " + FILL[rng.integers(len(FILL))]
    is_code = rng.random() < 0.30
    if is_code:
        p += CODE_ADD[rng.integers(len(CODE_ADD))]
    if rng.random() < d0 ** 1.3:
        p += REASON_ADD[rng.integers(len(REASON_ADD))]
    if rng.random() < 0.35 * d0:
        p += CONSTRAINT_ADD[rng.integers(len(CONSTRAINT_ADD))]
    if rng.random() < 0.5 * d0:
        p += MULTI_ADD[rng.integers(len(MULTI_ADD))]
    return p, is_code


def make_workload(n: int, seed: int = 0, drift_at: int | None = None, drift_shift: float = 0.30):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d0 = _sample_difficulty(rng)
        prompt, is_code = make_prompt(rng, d0)
        d = d0 + (drift_shift if (drift_at is not None and i >= drift_at and is_code) else 0.0)
        out.append((prompt, {"difficulty": float(np.clip(d, 0, 1)), "is_code": is_code, "t": i}))
    return out
