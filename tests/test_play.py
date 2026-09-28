#!/usr/bin/env python3
"""
Tests for util/play.py's random_opening (randomized evaluation starts).
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.play import random_opening
from util.util import BLACK


class TestRandomOpening(unittest.TestCase):

    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)

    def test_black_to_move_after_opening(self):
        for seed in range(20):
            self.env.reset()
            self.assertTrue(random_opening(self.env, 8, np.random.default_rng(seed)))
            self.assertEqual(self.env.player, BLACK)
            self.assertTrue(self.env.has_valid(self.env.board, BLACK))
            self.assertIsNone(self.env.get_winner(self.env.board))
            # 8 plies without passes add 8 stones to the starting 4
            self.assertGreaterEqual(int((self.env.board != 0).sum()), 12)

    def test_same_seed_same_opening(self):
        boards = []
        for _ in range(2):
            self.env.reset()
            random_opening(self.env, 8, np.random.default_rng(7))
            boards.append(self.env.board.copy())
        np.testing.assert_array_equal(boards[0], boards[1])

    def test_openings_vary(self):
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(20):
            self.env.reset()
            random_opening(self.env, 8, rng)
            seen.add(self.env.board.tobytes())
        self.assertGreater(len(seen), 15)

    def test_step_works_after_opening(self):
        """The env's normal two-ply step continues correctly from the opening position."""
        self.env.reset()
        random_opening(self.env, 8, np.random.default_rng(3))
        action = int(self.env.all_valid_actions(self.env.board)[0])
        obs, reward, term, trunc, info = self.env.step(action)
        if not term:
            self.assertEqual(self.env.player, BLACK)


if __name__ == '__main__':
    unittest.main()
