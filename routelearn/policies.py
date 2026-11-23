"""Routing policies. All share one interface so they can be benchmarked head-to-head."""
import threading

import numpy as np

from .bandit import LinUCB
from .config import COST_NORM, N_ARMS
from .features import DIM, IDX

COMPLEXITY_PRIOR_BONUS = 0.6


class Policy:
    name = "base"

    def choose(self, x: np.ndarray, meta: dict | None = None):
        """Returns (arm_index, diagnostics_dict)."""
        raise NotImplementedError

    def update(self, arm: int, x: np.ndarray, quality: float) -> None:
        pass


class AlwaysArm(Policy):
    def __init__(self, arm: int, label: str):
        self.arm, self.name = arm, label

    def choose(self, x, meta=None):
        return self.arm, {}


class StaticHeuristic(Policy):
    """The 'usual' hand-written router: fixed weights, fixed thresholds, never learns."""
    name = "static_heuristic"

    def __init__(self, t_small: float = 0.9, t_medium: float = 2.2):
        self.t = (t_small, t_medium)

    @staticmethod
    def score(x: np.ndarray) -> float:
        g = lambda n: x[IDX[n]]
        return (1.0 * g("n_reason") + 0.8 * g("n_code") + 0.6 * g("log_words") + 0.5 * g("multi_part")
                + 0.5 * g("math") + 0.4 * g("explain_task") + 0.3 * g("constraints")
                - 0.8 * g("simple_pattern"))

    def choose(self, x, meta=None):
        s = self.score(x)
        return (0 if s < self.t[0] else 1 if s < self.t[1] else 2), {"score": s}


class BanditPolicy(Policy):
    def __init__(self, lam: float = 0.4, alpha: float = 0.5, gamma: float = 1.0,
                 label: str = "bandit", store=None):
        self.lam, self.name = lam, label
        self.model = LinUCB(N_ARMS, DIM, alpha=alpha, gamma=gamma)
        self.cost = np.array(COST_NORM)
        self.complexity_policy = StaticHeuristic(t_medium=3.0)
        self.store = store
        self._lock = threading.RLock()

    def choose(self, x, meta=None):
        with self._lock:
            if self.store is not None:
                self.store.refresh_policy(self)
            mu, sigma = self.model.predict(x)
            scores = mu + self.model.alpha * sigma - self.lam * self.cost
            prior_arm, prior = self.complexity_policy.choose(x, meta)
            cold_start = not np.any(self.model.counts)
            scores[prior_arm] += COMPLEXITY_PRIOR_BONUS
            return int(np.argmax(scores)), {
                "quality_est": mu.tolist(),
                "uncertainty": sigma.tolist(),
                "scores": scores.tolist(),
                "cold_start": bool(cold_start),
                "complexity_prior_tier": ("small", "medium", "large")[prior_arm],
                "complexity_prior_score": prior["score"],
            }

    def update(self, arm, x, quality):
        with self._lock:
            if self.store is not None:
                self.store.update_policy(self, arm, x, quality)
            else:
                self.model.update(arm, x, quality)

    def persist(self) -> None:
        with self._lock:
            if self.store is not None:
                self.store.save_policy(self)

    def set_lam(self, lam: float) -> None:
        with self._lock:
            if self.store is not None:
                self.store.set_lambda(self, lam)
            else:
                self.lam = lam

    def refresh(self) -> None:
        with self._lock:
            if self.store is not None:
                self.store.refresh_policy(self)

    # persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, **self.model.state(), lam=self.lam)

    def load(self, path: str) -> None:
        d = np.load(path)
        self.model.load_state({"A": d["A"], "b": d["b"], "counts": d["counts"]})
        self.lam = float(d["lam"])


class Oracle(Policy):
    """Simulator-only upper bound: knows the true expected quality of every arm."""
    name = "oracle"

    def __init__(self, lam: float, quality_fn):
        self.lam, self.quality_fn, self.cost = lam, quality_fn, np.array(COST_NORM)

    def choose(self, x, meta=None):
        q = np.array([self.quality_fn(a, meta["difficulty"], meta.get("t", 0)) for a in range(N_ARMS)])
        return int(np.argmax(q - self.lam * self.cost)), {}
