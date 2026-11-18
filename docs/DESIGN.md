# Design notes & interview talking points

## Problem framing
Routing is a **contextual bandit**: context = prompt features, arms = model tiers, reward = answer
quality (judge/user) with cost handled separately. You only observe the quality of the tier you chose
(partial feedback), so you must *explore*.

## Key decisions (and the trade-off behind each)
1. **Learn quality only; treat cost as known.** Selection is `argmax q̂ + α·σ − λ·cost`. Because cost
   is deterministic, `λ` is a live product knob rather than a training-time constant.
2. **LinUCB over deep models.** Interpretable (`/explain` shows q̂ and σ per tier), sample-efficient,
   microsecond inference, closed-form updates. Cost: a linear model on hand-built features. A
   frozen-embedding variant is the obvious upgrade.
3. **Discounting (`gamma<1`).** Old data is forgotten so the router tracks silent regressions. Cost: more
   variance in stationary settings.
4. **Sampled judging.** Judging every response can add more cost than routing saves. The ablation shows
   10% sampling retains the benefit.
5. **Warm start.** Cold start cost ~500 requests of elevated regret. Seeding from a small full-information
   log reduces it, but the log costs money, so it is *amortised into reported cost*.
6. **Cascade as optional safety net.** Shown to trade cost for quality roughly along the same frontier
   as lowering `lambda`: reported as such, not oversold.

## Evaluation integrity
- Same code path for simulation and production (`Gateway`), only the provider/judge swap.
- Regret is computed against a true-quality oracle; cost includes judge, retries and warm-start.
- Multi-seed means; ablations for judge rate and cascade; a stationary frontier where the static
  baseline does as well or better, reported rather than hidden.
- An early version had no benefit from discounting; the drift scenario was too weak. I replaced it with a
  more realistic one (silent cheap-tier regression) instead of tuning until it "won".

## Likely interview questions
- *Why a bandit and not a classifier?* No labels exist for "right tier". You only see outcomes of what you
  chose, and the world drifts.
- *What breaks it?* A biased/noisy judge (it optimises what the judge likes); feature blind spots;
  non-stationary judge. Mitigate with periodic human-labelled audits and the `/feedback` channel.
- *Why not always use the cheap model and cascade?* Cascade pays for the failed attempt plus the judge; see ablation.
- *How would you scale it?* SQLite shares bandit updates across worker processes on one host. For
  multi-host deployments, move the state and quota transactions to a shared database/service (for
  example Redis with transactional updates); the current implementation does not provide that backend.
