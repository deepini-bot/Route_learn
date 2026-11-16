"""Model tiers ("arms") and cost model.

Prices are $/1M tokens and are PLACEHOLDERS: update them from the provider's pricing page.
`capability` is used ONLY by the simulator (it never influences real routing).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    price_in: float
    price_out: float
    capability: float  # simulator only


ARMS = [
    Arm("small", "claude-haiku-5-5", 1.0, 5.0, 0.38),
    Arm("medium", "claude-sonnet-5-5", 3.0, 15.0, 0.64),
    Arm("large", "claude-opus-5-5", 5.0, 25.0, 0.92),
]
N_ARMS = len(ARMS)

AVG_TOKENS_IN, AVG_TOKENS_OUT = 200, 400          # used for planning + simulation
JUDGE_ARM = ARMS[0]                               # judge with the cheapest model
JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT = 800, 10


def est_cost(arm: Arm, tin: int = AVG_TOKENS_IN, tout: int = AVG_TOKENS_OUT) -> float:
    return (tin * arm.price_in + tout * arm.price_out) / 1e6


JUDGE_COST = est_cost(JUDGE_ARM, JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT)
# Cost of each arm relative to the largest model. The bandit treats this as KNOWN
# (only quality is uncertain), which makes the cost/quality dial free to change at runtime.
COST_NORM = [est_cost(a) / est_cost(ARMS[-1]) for a in ARMS]
