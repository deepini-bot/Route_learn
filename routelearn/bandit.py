"""Discounted LinUCB: one ridge-regression quality model per arm, with optimism for exploration.

Design choice: the bandit learns QUALITY q(x, arm) only. Cost is known, so selection is
    argmax_a  [ q_hat(x,a) + alpha * sigma(x,a) ]  -  lambda * cost_norm[a]
and `lambda` (the cost/quality dial) can be changed at runtime without retraining.

`gamma < 1` exponentially forgets old observations, so the router tracks workload drift.
"""
import numpy as np


class LinUCB:
    def __init__(self, n_arms: int, dim: int, alpha: float = 0.5, ridge: float = 1.0,
                 gamma: float = 1.0):
        self.n_arms, self.dim, self.alpha, self.ridge, self.gamma = n_arms, dim, alpha, ridge, gamma
        self.A = np.stack([ridge * np.eye(dim) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self.counts = np.zeros(n_arms, dtype=int)

    def predict(self, x: np.ndarray):
        """Returns (mean quality, uncertainty) per arm."""
        mu, sigma = np.zeros(self.n_arms), np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta = np.linalg.solve(self.A[a], self.b[a])
            mu[a] = float(np.clip(theta @ x, 0.0, 1.0))
            sigma[a] = float(np.sqrt(max(x @ np.linalg.solve(self.A[a], x), 0.0)))
        return mu, sigma

    def update(self, arm: int, x: np.ndarray, quality: float):
        if self.gamma < 1.0:  # forget old data, but never shrink below the ridge prior
            eye = self.ridge * np.eye(self.dim)
            self.A[arm] = self.gamma * (self.A[arm] - eye) + eye
            self.b[arm] *= self.gamma
        self.A[arm] += np.outer(x, x)
        self.b[arm] += quality * x
        self.counts[arm] += 1

    def state(self) -> dict:
        return {"A": self.A, "b": self.b, "counts": self.counts}

    def load_state(self, s) -> None:
        self.A, self.b, self.counts = s["A"], s["b"], s["counts"]
