#!/usr/bin/env python3
"""Tests for util/levels.py: every strength level gives legal moves, and the descriptions match."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.levels import LEVELS, STRONGEST, describe, make_player, PolicySampler
from util.reversi import ReversiEnvCNN
from util.search import legal_moves, SearchPlayer
from tests.test_start_positions import sample_positions


class TestLevels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch
        from util.util import get_model
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as d:
            cls.model = get_model(os.path.join(d, "m"), ReversiEnvCNN(opponent="Random"), model_type="resnet",
                                  channels=8, blocks=1)
        cls.positions = [b for b in sample_positions(n_games=2, seed=31) if len(legal_moves(b))][::3]

    def test_levels_play_legal_moves(self):
        for level in range(1, 6):                      # the cheap levels; higher ones only differ in budget
            player = make_player(self.model, level, seed=0)
            for b, m in zip(self.positions, player.choose_many(self.positions)):
                self.assertIn(m, set(int(x) for x in legal_moves(b)), f"level {level}")

    def test_player_kinds(self):
        self.assertIsInstance(make_player(self.model, 1), PolicySampler)
        self.assertIsInstance(make_player(self.model, 3), SearchPlayer)
        top = make_player(self.model, STRONGEST)
        self.assertEqual((top.mcts.sims, top.solve_empties), (800, 18))       # web_play.py --strong

    def test_descriptions(self):
        self.assertEqual(sorted(LEVELS), list(range(1, 11)))
        self.assertIn("top move", describe(3))
        self.assertIn("MCTS 800", describe(10))
        sims = [LEVELS[n].get("mcts_sims", 0) for n in range(4, 11)]
        self.assertEqual(sims, sorted(sims))                                  # budgets rise with level


if __name__ == "__main__":
    unittest.main()
