#!/usr/bin/env python3
"""
Tests for varied-start training: ReversiEnvCNN(start_positions=...) and
make_start_positions.extract_start_positions.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import BLACK
from make_start_positions import extract_start_positions


def standard_board():
    env = ReversiEnvCNN(opponent="Random")
    board, _ = env.reset()
    return board.copy()


def sample_positions(n_games=10, seed=0):
    """Mover's-perspective positions with BLACK to move, from random games."""
    env = ReversiEnvCNN(opponent="Random")
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(n_games):
        board, _ = env.reset()
        board = board.copy()
        player = BLACK
        while env.get_winner(board) is None:
            if not env.has_valid(board, player):
                player = -player
                continue
            positions.append((board * player).reshape(64).astype(np.int8))
            env.player = player
            a = int(rng.choice(env._all_valid_actions(board, player)))
            env.get_next_state(board, (0, a // 8, a % 8))
            player = -player
    return np.unique(np.array(positions), axis=0)


class TestStartPositions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        all_positions = sample_positions()
        stones = (all_positions != 0).sum(axis=1)
        cls.positions = all_positions[(stones >= 8) & (stones <= 20)]

    def make_env(self, **kwargs):
        return ReversiEnvCNN(opponent="Random", start_positions=self.positions, **kwargs)

    def test_default_is_standard_opening(self):
        env = ReversiEnvCNN(opponent="Random")
        board, _ = env.reset()
        np.testing.assert_array_equal(board, standard_board())

    def test_reset_uses_start_positions(self):
        env = self.make_env()
        known = {p.tobytes() for p in self.positions}
        seen = set()
        for i in range(30):
            board, _ = env.reset(seed=i) if i == 0 else env.reset()
            flat = board.reshape(64).astype(np.int8)
            self.assertIn(flat.tobytes(), known)
            self.assertEqual(env.player, BLACK)
            self.assertTrue(env.has_valid(board, BLACK))
            seen.add(flat.tobytes())
        self.assertGreater(len(seen), 20)

    def test_start_positions_are_copied(self):
        """Playing a game must not modify the stored start positions."""
        before = self.positions.copy()
        env = self.make_env()
        env.reset(seed=0)
        env.step(int(env.all_valid_actions(env.board)[0]))
        np.testing.assert_array_equal(self.positions, before)

    def test_standard_start_prob_one(self):
        env = self.make_env(standard_start_prob=1.0)
        for _ in range(5):
            board, _ = env.reset()
            np.testing.assert_array_equal(board, standard_board())

    def test_seed_reproducible(self):
        a, _ = self.make_env().reset(seed=123)
        b, _ = self.make_env().reset(seed=123)
        np.testing.assert_array_equal(a, b)

    def test_games_after_the_first_also_vary(self):
        """step() resets internally at game end; the next game must start from a stored position."""
        env = self.make_env()
        env.reset(seed=1)
        known = {p.tobytes() for p in self.positions}
        rng = np.random.default_rng(0)
        games = 0
        while games < 3:
            action = int(rng.choice(env.all_valid_actions(env.board)))
            obs, reward, term, trunc, info = env.step(action)
            if term:
                games += 1
                self.assertIn(obs.reshape(64).astype(np.int8).tobytes(), known)
                self.assertEqual(env.player, BLACK)

    def test_load_from_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "starts.npy")
            np.save(path, self.positions)
            env = ReversiEnvCNN(opponent="Random", start_positions=path)
            self.assertEqual(env.start_positions.shape, (len(self.positions), 1, 8, 8))

    def test_extract_start_positions(self):
        states = sample_positions(n_games=5, seed=3)
        dataset = [{'state': s.reshape(1, 8, 8), 'action': 0} for s in states]
        dataset += dataset[:10]  # duplicates are removed
        out = extract_start_positions(dataset, 8, 12)
        stones = (out != 0).sum(axis=1)
        self.assertTrue(((stones >= 8) & (stones <= 12)).all())
        self.assertEqual(len(out), len(np.unique(out, axis=0)))
        self.assertEqual(out.dtype, np.int8)
        self.assertEqual(out.shape[1], 64)


if __name__ == '__main__':
    unittest.main()
