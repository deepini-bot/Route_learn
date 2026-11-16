"""Routing policies. All share one interface so they can be benchmarked head-to-head."""
import numpy as np

from .bandit import LinUCB
from .config import COST_NORM, N_ARMS
from .features import DIM, IDX


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
    def __init__(self, lam: float = 0.4, alpha: float = 0.5, gamma: float = 1.0, label: str = "bandit"):
        self.lam, self.name = lam, label
        self.model = LinUCB(N_ARMS, DIM, alpha=alpha, gamma=gamma)
        self.cost = np.array(COST_NORM)

    def choose(self, x, meta=None):
        mu, sigma = self.model.predict(x)
        scores = mu + self.model.alpha * sigma - self.lam * self.cost
        return int(np.argmax(scores)), {"quality_est": mu.tolist(), "uncertainty": sigma.tolist(),
                                        "scores": scores.tolist()}

    def update(self, arm, x, quality):
        self.model.update(arm, x, quality)

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
