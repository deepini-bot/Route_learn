"""Cold-start mitigation: seed the bandit from a small offline log of judged prompts.

Every prompt is answered by EVERY tier and judged once (full-information feedback), which is
something online bandit feedback never gives you. It costs real money, so the caller should
account for it (see `warm_start_cost`).
"""
from .config import ARMS, JUDGE_COST, N_ARMS, est_cost
from .features import extract


def warm_start(policy, workload, provider, judge) -> int:
    for prompt, meta in workload:
        x = extract(prompt)
        for a in range(N_ARMS):
            comp = provider.complete(a, prompt, meta)
            policy.update(a, x, judge.score(prompt, comp))
    return len(workload)


def warm_start_cost(n_prompts: int) -> float:
    return n_prompts * (sum(est_cost(a) for a in ARMS) + N_ARMS * JUDGE_COST)
