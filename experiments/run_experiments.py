"""Reproduces every number/plot in the README.   Run:  python experiments/run_experiments.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from routelearn.config import ARMS, COST_NORM, N_ARMS, est_cost
from routelearn.gateway import Gateway
from routelearn.policies import AlwaysArm, BanditPolicy, Oracle, StaticHeuristic
from routelearn.providers import SimJudge, SimProvider
from routelearn.warmstart import warm_start, warm_start_cost
from routelearn.workload import make_workload

N, DEGRADE_AT, SEEDS, LAM, WARM_N = 4000, 2000, 5, 0.6, 100
DEGRADE = (0, DEGRADE_AT, -0.25)     # the cheap tier silently gets worse halfway through
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LARGE_COST = est_cost(ARMS[-1])
COST = np.array(COST_NORM)


def exp_reward(arm, d, lam, t=0, degrade=DEGRADE):
    return SimProvider.expected_quality(arm, d, t, degrade) - lam * COST[arm]


def run(policy_fn, seed, lam=LAM, judge_rate=0.3, cascade=None, n=N, degrade=DEGRADE, warm=False):
    work = make_workload(n, seed)
    prov, judge = SimProvider(seed, degrade=degrade), SimJudge(seed + 100)
    policy = policy_fn()
    extra = 0.0
    if warm:   # offline log collected BEFORE deployment (different seed, no degradation yet)
        warm_set = make_workload(WARM_N, seed + 5000)
        warm_start(policy, warm_set, SimProvider(seed + 6000), judge)
        extra = warm_start_cost(WARM_N) / LARGE_COST / n        # amortised one-time cost
    gw = Gateway(policy, prov, judge, judge_rate=judge_rate, cascade_tau=cascade, seed=seed)
    arms, qual, cost, regret = [], [], [], []
    for prompt, meta in work:
        r = gw.handle(prompt, meta)
        d, t = meta["difficulty"], meta["t"]
        best = max(exp_reward(a, d, lam, t, degrade) for a in range(N_ARMS))
        arms.append(r.arm)
        qual.append(r.true_quality)
        cost.append(r.cost / LARGE_COST + extra)         # relative to always-large, incl. judge+cascade
        regret.append(best - exp_reward(r.first_arm, d, lam, t, degrade))   # regret of the ROUTING decision
    return {"arm": np.array(arms), "quality": np.array(qual), "cost": np.array(cost),
            "regret": np.array(regret)}


def agg(runs, key, sl=slice(None)):
    return float(np.mean([r[key][sl].mean() for r in runs]))


def policies(lam=LAM):
    return {
        "always_small": (lambda: AlwaysArm(0, "always_small"), {}),
        "always_medium": (lambda: AlwaysArm(1, "always_medium"), {}),
        "always_large": (lambda: AlwaysArm(2, "always_large"), {}),
        "static_heuristic": (lambda: StaticHeuristic(), {}),
        "bandit (cold, gamma=1)": (lambda: BanditPolicy(lam=lam, gamma=1.0), {"judge_rate": 0.3}),
        "bandit (cold, gamma=0.995)": (lambda: BanditPolicy(lam=lam, gamma=0.995), {"judge_rate": 0.3}),
        "bandit (warm-start, gamma=0.995)": (lambda: BanditPolicy(lam=lam, gamma=0.995), {"judge_rate": 0.3, "warm": True}),
        "oracle": (lambda: Oracle(lam, lambda a, d, t: SimProvider.expected_quality(a, d, t, DEGRADE)), {}),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    BW = "bandit (warm-start, gamma=0.995)"

    # ---------------- 1. head-to-head with a silent regression of the small tier ----------------
    runs = {}
    for name, (fn, kw) in policies().items():
        runs[name] = [run(fn, s, **kw) for s in range(SEEDS)]
    pre, post = slice(0, DEGRADE_AT), slice(DEGRADE_AT, None)
    rows = []
    for name, rs in runs.items():
        q, c = agg(rs, "quality"), agg(rs, "cost")
        rows.append({"policy": name, "quality": q, "rel_cost": c, "reward": q - LAM * c,
                     "cum_regret": float(np.mean([r["regret"].sum() for r in rs])),
                     "post_quality": agg(rs, "quality", post), "post_cost": agg(rs, "cost", post),
                     "post_regret": float(np.mean([r["regret"][post].sum() for r in rs]))})
    results["head_to_head"] = rows

    # ---------------- 2. Pareto frontier (no degradation): the lambda dial ----------------
    pareto = {"bandit (warm-start)": [], "static_heuristic": []}
    for lam in [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4]:
        rs = [run(lambda: BanditPolicy(lam=lam, gamma=0.995), s, lam=lam, degrade=None, warm=True, n=3000)
              for s in range(3)]
        pareto["bandit (warm-start)"].append((agg(rs, "cost"), agg(rs, "quality"), lam))
    for shift in [-0.9, -0.5, -0.2, 0.0, 0.3, 0.7, 1.2, 2.0]:
        rs = [run(lambda: StaticHeuristic(0.9 + shift, 2.2 + shift), s, judge_rate=0.0, degrade=None, n=3000)
              for s in range(3)]
        pareto["static_heuristic"].append((agg(rs, "cost"), agg(rs, "quality"), shift))
    fixed = {n: (agg(runs[n], "cost"), agg(runs[n], "quality")) for n in ["always_small", "always_medium", "always_large"]}
    results["pareto"] = {k: v for k, v in pareto.items()}

    # ---------------- 3. ablations (warm-started bandit, with degradation) ----------------
    abl = []
    def add(label, **kw):
        rs = [run(lambda: BanditPolicy(lam=LAM, gamma=0.995), s, warm=True, **kw) for s in range(3)]
        q, c = agg(rs, "quality"), agg(rs, "cost")
        abl.append({"ablation": label, "quality": q, "rel_cost": c, "reward": q - LAM * c,
                    "post_quality": agg(rs, "quality", post)})
    for jr in [1.0, 0.3, 0.1, 0.03]:
        add(f"judge_rate={jr}", judge_rate=jr)
    for tau in [0.5, 0.7]:
        add(f"judge_rate=0.3 + cascade tau={tau}", judge_rate=0.3, cascade=tau)
    results["ablations"] = abl

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- markdown summary ----------------
    md = [f"## Head-to-head: N={N}, cheap tier silently degrades at request {DEGRADE_AT}, {SEEDS} seeds, lambda={LAM}\n",
          "Cost is relative to always-large (includes judge calls, cascade retries and amortised warm-start).\n",
          "| policy | quality | cost | reward | cum. regret | post-degradation quality | post-degradation regret |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['policy']} | {r['quality']:.3f} | {r['rel_cost']:.2f}x | {r['reward']:.3f} | "
                  f"{r['cum_regret']:.0f} | {r['post_quality']:.3f} | {r['post_regret']:.0f} |")
    md += ["\n## Ablations (warm-started bandit, gamma=0.995, 3 seeds)\n",
           "| setting | quality | cost | reward | post-degradation quality |", "|---|---|---|---|---|"]
    for r in abl:
        md.append(f"| {r['ablation']} | {r['quality']:.3f} | {r['rel_cost']:.2f}x | {r['reward']:.3f} | {r['post_quality']:.3f} |")
    md += ["\n## Cost/quality frontier (no degradation, 3 seeds)\n", "| policy | setting | cost | quality |", "|---|---|---|---|"]
    for name, pts in pareto.items():
        for c, q, k in sorted(pts):
            md.append(f"| {name} | {'lambda' if 'bandit' in name else 'threshold shift'}={k} | {c:.2f}x | {q:.3f} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---------------- plots ----------------
    plt.figure(figsize=(7.5, 4.2))
    for name in ["static_heuristic", "bandit (cold, gamma=1)", "bandit (cold, gamma=0.995)", BW, "always_large", "always_small"]:
        plt.plot(np.mean([np.cumsum(r["regret"]) for r in runs[name]], axis=0), label=name)
    plt.axvline(DEGRADE_AT, color="gray", ls="--"); plt.text(DEGRADE_AT + 30, 20, "small tier degrades", color="gray")
    plt.xlabel("requests"); plt.ylabel("cumulative regret vs oracle"); plt.legend(fontsize=8)
    plt.title("Learning, cold start, and adapting to a silent regression")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "regret.png"), dpi=140); plt.close()

    plt.figure(figsize=(6.2, 4.6))
    for name, pts in pareto.items():
        pts = sorted(pts)
        plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name)
    for name, (c, q) in fixed.items():
        plt.scatter(c, q, marker="x", s=70, label=name)
    plt.xlabel("cost relative to always-large"); plt.ylabel("quality"); plt.legend(fontsize=8)
    plt.title("Cost/quality frontier (sweeping the lambda dial)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pareto.png"), dpi=140); plt.close()

    plt.figure(figsize=(7.5, 4.2)); w = 150
    arms = np.stack([r["arm"] for r in runs[BW]])
    for a in range(N_ARMS):
        plt.plot(np.convolve((arms == a).mean(axis=0), np.ones(w) / w, mode="valid"), label=ARMS[a].name)
    plt.axvline(DEGRADE_AT - w // 2, color="gray", ls="--")
    plt.xlabel("requests"); plt.ylabel("tier share (rolling)"); plt.legend()
    plt.title("Router reroutes away from the degraded small tier")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "tier_share.png"), dpi=140); plt.close()


if __name__ == "__main__":
    main()
