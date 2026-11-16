"""Run with:  python -m unittest discover -s tests -v   (or pytest)"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routelearn.bandit import LinUCB
from routelearn.features import DIM, FEATURE_NAMES, IDX, extract
from routelearn.gateway import Gateway
from routelearn.policies import AlwaysArm, BanditPolicy, Oracle, StaticHeuristic
from routelearn.providers import SimJudge, SimProvider
from routelearn.workload import make_workload


class TestFeatures(unittest.TestCase):
    def test_shape_and_bias(self):
        x = extract("hello")
        self.assertEqual(x.shape, (DIM,))
        self.assertEqual(x[IDX["bias"]], 1.0)

    def test_hard_prompt_scores_higher(self):
        easy = extract("What is photosynthesis?")
        hard = extract("Analyze the trade-offs and design an architecture, then compare options step by step.")
        self.assertGreater(hard[IDX["n_reason"]], easy[IDX["n_reason"]])
        self.assertEqual(easy[IDX["simple_pattern"]], 1.0)
        self.assertGreater(StaticHeuristic.score(hard), StaticHeuristic.score(easy))

    def test_keywords_use_word_boundaries(self):
        # "capital" contains "api" as a substring; it must NOT count as code
        self.assertEqual(extract("What is the capital of France?")[IDX["n_code"]], 0.0)


class TestBandit(unittest.TestCase):
    def test_learns_arm_quality(self):
        m, rng = LinUCB(2, 3), np.random.default_rng(0)
        for _ in range(300):
            x = np.array([1.0, rng.random(), rng.random()])
            m.update(0, x, 0.2)
            m.update(1, x, 0.8)
        mu, _ = m.predict(np.array([1.0, 0.5, 0.5]))
        self.assertAlmostEqual(mu[0], 0.2, delta=0.05)
        self.assertAlmostEqual(mu[1], 0.8, delta=0.05)

    def test_uncertainty_shrinks_with_data(self):
        m = LinUCB(1, 2)
        x = np.array([1.0, 0.5])
        s0 = m.predict(x)[1][0]
        for _ in range(100):
            m.update(0, x, 0.5)
        self.assertLess(m.predict(x)[1][0], s0)

    def test_discounting_adapts_faster(self):
        x = np.array([1.0, 0.3])
        plain, disc = LinUCB(1, 2, gamma=1.0), LinUCB(1, 2, gamma=0.98)
        for m in (plain, disc):
            for _ in range(500):
                m.update(0, x, 0.9)      # model used to be great...
            for _ in range(60):
                m.update(0, x, 0.1)      # ...then silently regressed
        self.assertLess(disc.predict(x)[0][0], plain.predict(x)[0][0])


class TestGateway(unittest.TestCase):
    def _gw(self, policy, **kw):
        return Gateway(policy, SimProvider(0), SimJudge(1), **kw)

    def test_oracle_picks_cheapest_sufficient_arm(self):
        pol = Oracle(0.6, SimProvider.expected_quality)
        gw = self._gw(pol, judge_rate=0.0)
        easy = gw.handle("What is X?", {"difficulty": 0.02, "t": 0})
        hard = gw.handle("Prove it.", {"difficulty": 0.70, "t": 1})   # hard but solvable by large
        self.assertLess(easy.arm, hard.arm)

    def test_cost_includes_judge(self):
        gw = self._gw(AlwaysArm(0, "s"), judge_rate=1.0)
        r = gw.handle("hi", {"difficulty": 0.1, "t": 0})
        self.assertGreater(r.cost, SimProvider(0).complete(0, "hi", {"difficulty": 0.1, "t": 0}).cost)

    def test_cascade_escalates_on_low_quality(self):
        gw = self._gw(AlwaysArm(0, "s"), judge_rate=0.0, cascade_tau=0.6)
        r = gw.handle("hard thing", {"difficulty": 0.9, "t": 0})     # small tier fails here
        self.assertGreaterEqual(r.escalations, 1)
        self.assertGreater(r.arm, r.first_arm)

    def test_delayed_feedback_updates_bandit(self):
        pol = BanditPolicy()
        gw = Gateway(pol, SimProvider(0), judge=None)
        r = gw.handle("hello", {"difficulty": 0.2, "t": 0})
        before = pol.model.counts.sum()
        self.assertTrue(gw.feedback(r.request_id, 0.9))
        self.assertEqual(pol.model.counts.sum(), before + 1)
        self.assertFalse(gw.feedback(r.request_id, 0.9))             # can only be used once

    def test_persistence_roundtrip(self):
        pol = BanditPolicy(lam=0.7)
        gw = self._gw(pol, judge_rate=1.0)
        for p, m in make_workload(50, 0):
            gw.handle(p, m)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.npz")
            pol.save(path)
            pol2 = BanditPolicy()
            pol2.load(path)
        self.assertTrue(np.allclose(pol.model.A, pol2.model.A))
        self.assertEqual(pol2.lam, 0.7)


class TestWorkload(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual([p for p, _ in make_workload(20, 3)], [p for p, _ in make_workload(20, 3)])

    def test_difficulty_in_range(self):
        self.assertTrue(all(0 <= m["difficulty"] <= 1 for _, m in make_workload(200, 1)))


if __name__ == "__main__":
    unittest.main()
