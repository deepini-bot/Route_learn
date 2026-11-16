## Head-to-head: N=4000, cheap tier silently degrades at request 2000, 5 seeds, lambda=0.6

Cost is relative to always-large (includes judge calls, cascade retries and amortised warm-start).

| policy | quality | cost | reward | cum. regret | post-degradation quality | post-degradation regret |
|---|---|---|---|---|---|---|
| always_small | 0.326 | 0.22x | 0.192 | 877 | 0.206 | 649 |
| always_medium | 0.703 | 0.62x | 0.328 | 299 | 0.701 | 118 |
| always_large | 0.898 | 1.02x | 0.284 | 450 | 0.897 | 192 |
| static_heuristic | 0.685 | 0.63x | 0.309 | 383 | 0.630 | 272 |
| bandit (cold, gamma=1) | 0.697 | 0.62x | 0.326 | 313 | 0.705 | 191 |
| bandit (cold, gamma=0.995) | 0.707 | 0.61x | 0.340 | 260 | 0.740 | 132 |
| bandit (warm-start, gamma=0.995) | 0.734 | 0.69x | 0.320 | 214 | 0.746 | 113 |
| oracle | 0.834 | 0.71x | 0.405 | 0 | 0.855 | 0 |

## Ablations (warm-started bandit, gamma=0.995, 3 seeds)

| setting | quality | cost | reward | post-degradation quality |
|---|---|---|---|---|
| judge_rate=1.0 | 0.745 | 0.75x | 0.293 | 0.772 |
| judge_rate=0.3 | 0.739 | 0.70x | 0.321 | 0.749 |
| judge_rate=0.1 | 0.731 | 0.67x | 0.327 | 0.734 |
| judge_rate=0.03 | 0.722 | 0.67x | 0.320 | 0.695 |
| judge_rate=0.3 + cascade tau=0.5 | 0.812 | 0.88x | 0.286 | 0.836 |
| judge_rate=0.3 + cascade tau=0.7 | 0.853 | 1.00x | 0.250 | 0.866 |

## Cost/quality frontier (no degradation, 3 seeds)

| policy | setting | cost | quality |
|---|---|---|---|
| bandit (warm-start) | lambda=1.4 | 0.29x | 0.442 |
| bandit (warm-start) | lambda=1.0 | 0.32x | 0.467 |
| bandit (warm-start) | lambda=0.8 | 0.46x | 0.568 |
| bandit (warm-start) | lambda=0.6 | 0.67x | 0.728 |
| bandit (warm-start) | lambda=0.45 | 0.79x | 0.807 |
| bandit (warm-start) | lambda=0.3 | 0.91x | 0.858 |
| bandit (warm-start) | lambda=0.15 | 1.01x | 0.884 |
| bandit (warm-start) | lambda=0.0 | 1.08x | 0.895 |
| static_heuristic | threshold shift=2.0 | 0.28x | 0.496 |
| static_heuristic | threshold shift=1.2 | 0.38x | 0.575 |
| static_heuristic | threshold shift=0.7 | 0.47x | 0.649 |
| static_heuristic | threshold shift=0.3 | 0.53x | 0.694 |
| static_heuristic | threshold shift=0.0 | 0.60x | 0.738 |
| static_heuristic | threshold shift=-0.2 | 0.66x | 0.778 |
| static_heuristic | threshold shift=-0.5 | 0.72x | 0.812 |
| static_heuristic | threshold shift=-0.9 | 0.78x | 0.840 |
