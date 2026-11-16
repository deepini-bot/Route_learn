"""60-second demo (no API key needed):  python examples/demo.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from routelearn.config import ARMS
from routelearn.features import extract
from routelearn.gateway import Gateway
from routelearn.policies import BanditPolicy
from routelearn.providers import SimJudge, SimProvider
from routelearn.workload import make_workload

policy = BanditPolicy(lam=0.6, gamma=0.995)
gw = Gateway(policy, SimProvider(0), SimJudge(1), judge_rate=0.3)

work = make_workload(1500, seed=7)
for p, m in work:
    gw.handle(p, m)

def share(rs):
    return {a.name: round(sum(r.arm == i for r in rs) / len(rs), 2) for i, a in enumerate(ARMS)}

print("tier share, first 200 requests :", share(gw.log[:200]))
print("tier share, last 200 requests  :", share(gw.log[-200:]))
print("(the router never saw ground-truth difficulty: it learned from judge feedback only)\n")

for prompt in ["What is photosynthesis?",
               "Write a short email about inflation. Be specific about edge cases.",
               "Design a scalable architecture around a message queue and compare the trade-offs "
               "step by step. First outline it, then implement it, and finally review it. Prove each step."]:
    _, diag = policy.choose(extract(prompt))
    print(f"PROMPT: {prompt[:90]}")
    for a, q, u, s in zip(ARMS, diag["quality_est"], diag["uncertainty"], diag["scores"]):
        print(f"   {a.name:7} est.quality={q:.2f} (+/-{u:.2f})  net score={s:+.2f}")
    print(f"   -> routed to: {ARMS[int(np.argmax(diag['scores']))].name}\n")
