#!/usr/bin/env python3
"""
Tests for util/training.py: the opponent split used by sb-train.py's selfplay-edax mode.
"""

import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.training import parse_edax_mix, mixed_block_plan


class TestTrainingMix(unittest.TestCase):

    def test_parse_edax_mix(self):
        self.assertEqual(parse_edax_mix("1, 2,3", "0.1,0.1, 0.2"), ([1, 2, 3], [0.1, 0.1, 0.2]))
        with self.assertRaises(ValueError):
            parse_edax_mix("1,2", "0.5")
        with self.assertRaises(ValueError):
            parse_edax_mix("1,x", "0.5,0.5")

    def test_plan_order_and_total(self):
        plan = mixed_block_plan(100_000, 0.6, [1, 2, 3], [0.1, 0.1, 0.1], 0.1)
        self.assertEqual([(o, d) for o, d, _ in plan],
                         [("Self", None), ("Edax", 1), ("Edax", 2), ("Edax", 3), ("Random", None)])
        self.assertEqual(sum(t for _, _, t in plan), 100_000)
        self.assertEqual(plan[0][2], 60_000)
        self.assertEqual(plan[1][2], 10_000)

    def test_ratios_are_normalized(self):
        a = mixed_block_plan(10_000, 6, [2], [3], 1)
        b = mixed_block_plan(10_000, 0.6, [2], [0.3], 0.1)
        self.assertEqual(a, b)

    def test_rounding_leftover_keeps_total_exact(self):
        plan = mixed_block_plan(99_999, 1, [1, 2], [1, 1], 1)
        self.assertEqual(sum(t for _, _, t in plan), 99_999)

    def test_zero_ratios_dropped(self):
        plan = mixed_block_plan(1_000, 0.7, [1, 2], [0.3, 0.0], 0.0)
        self.assertEqual([(o, d) for o, d, _ in plan], [("Self", None), ("Edax", 1)])
        self.assertEqual(sum(t for _, _, t in plan), 1_000)

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            mixed_block_plan(1_000, 0, [1], [0], 0)


class TestVecEnvTraining(unittest.TestCase):
    """set_opponent over a DummyVecEnv, and get_model's n_steps override."""

    def setUp(self):
        import tempfile
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv
        from util.reversi import build_reversi
        from util.util import get_model, mask_fn
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make = lambda: ActionMasker(build_reversi(opponent="Random"), mask_fn)
        self.env = DummyVecEnv([make for _ in range(3)])
        self.get_model, self.path = get_model, os.path.join(self.tmp.name, "m")
        self.model = get_model(self.path, self.env, model_type="resnet", channels=8, blocks=1, n_steps=64)

    def test_new_model_n_steps(self):
        self.assertEqual(self.model.n_steps, 64)
        self.assertEqual(self.model.rollout_buffer.buffer_size * self.model.n_envs, 64 * 3)

    def test_loaded_model_n_steps_override(self):
        self.model.save(self.path)
        loaded = self.get_model(self.path, self.env, n_steps=32)
        self.assertEqual(loaded.n_steps, 32)
        self.assertEqual(loaded.n_envs, 3)

    def test_set_opponent_all_envs_share_reloaded_model(self):
        import torch
        from util.training import set_opponent
        opp_path = os.path.join(self.tmp.name, "opp")
        self.model.save(opp_path)
        set_opponent(self.env, "Model", opp_path, reload=True)
        opponents = [e.unwrapped.opponent for e in self.env.envs]
        self.assertTrue(all(o is opponents[0] for o in opponents))
        # refresh: overwrite the file with changed weights; all envs get the new shared opponent
        with torch.no_grad():
            next(self.model.policy.parameters()).add_(1.0)
        self.model.save(opp_path)
        set_opponent(self.env, "Model", opp_path, reload=True)
        refreshed = [e.unwrapped.opponent for e in self.env.envs]
        self.assertIsNot(refreshed[0], opponents[0])
        self.assertTrue(all(o is refreshed[0] for o in refreshed))
        w_old = next(opponents[0].model.policy.parameters()).detach().cpu()
        w_new = next(refreshed[0].model.policy.parameters()).detach().cpu()
        self.assertTrue(torch.allclose(w_new, w_old + 1.0))

    def test_set_opponent_single_env(self):
        from util.training import set_opponent
        from util.opponents import RandomOpponent
        single = self.env.envs[0]
        set_opponent(single, "Random", None)
        self.assertIsInstance(single.unwrapped.opponent, RandomOpponent)

    def test_short_vectorized_learn(self):
        self.model.learn(3 * 64 * 2)
        self.assertGreaterEqual(self.model.num_timesteps, 3 * 64 * 2)


if __name__ == '__main__':
    unittest.main()
