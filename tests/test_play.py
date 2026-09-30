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


class TestPairedStarts(unittest.TestCase):
    """sb-play's model-vs-model mode relies on play() giving the same sequence of starting
    positions for a given seed, whatever the models do, so the color-swapped halves pair up."""

    def starts_seen(self, model_seed, **play_kwargs):
        import tempfile, torch
        from util.util import get_model
        from util.play import play
        from tests.test_start_positions import sample_positions
        torch.manual_seed(model_seed)
        with tempfile.TemporaryDirectory() as d:
            env = ReversiEnvCNN(opponent="Random", start_positions=sample_positions(n_games=5))
            model = get_model(os.path.join(d, "m"), env, model_type="resnet", channels=8, blocks=1)
            inner = model.get_env().envs[0].unwrapped
            seen, original_reset = [], inner.reset

            def recording_reset(**kwargs):
                obs, info = original_reset(**kwargs)
                seen.append(inner.board.copy())
                return obs, info
            inner.reset = recording_reset
            wins, draws = play(model, 6, None, False, False, return_draws=True, **play_kwargs)
            self.assertLessEqual(wins + draws, 6)
            return seen

    def test_same_seed_same_start_sequence(self):
        a = self.starts_seen(model_seed=1, seed=11)
        b = self.starts_seen(model_seed=2, seed=11)  # different model, different games
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_different_seed_different_starts(self):
        a = self.starts_seen(model_seed=1, seed=11)
        b = self.starts_seen(model_seed=1, seed=12)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, b)))


class TestStartSequence(unittest.TestCase):
    """play(start_sequence=...) starts game i from position i, each once and in order."""

    def test_each_position_once_in_order(self):
        import tempfile
        from util.util import get_model
        from util.play import play
        from util.search import legal_moves
        from tests.test_start_positions import sample_positions
        positions = sample_positions(n_games=3, seed=5)[:12]
        env = ReversiEnvCNN(opponent="Random")
        with tempfile.TemporaryDirectory() as d:
            model = get_model(os.path.join(d, "m"), env, model_type="resnet", channels=8, blocks=1)
        firsts, new_game = [], [False]
        original_reset = env.reset

        def reset(**kwargs):            # a reset (loop or end-of-game) marks the next move as a game start
            new_game[0] = True
            return original_reset(**kwargs)
        env.reset = reset

        def chooser(board):
            if new_game[0]:
                firsts.append(np.asarray(board).reshape(64).astype(np.int8).tobytes())
                new_game[0] = False
            return int(legal_moves(np.asarray(board).reshape(64))[0])

        play(model, len(positions), None, True, False, env=env, chooser=chooser, start_sequence=positions)
        self.assertEqual(firsts, [p.tobytes() for p in positions])


if __name__ == '__main__':
    unittest.main()
