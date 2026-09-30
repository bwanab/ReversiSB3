#!/usr/bin/env python3
"""
Tests for util/search.py: move generation and application against the env, and the negamax
search against a brute-force reference built only from env calls.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import BLACK
from util.search import legal_moves, play_move, search, SearchPlayer
from tests.test_start_positions import sample_positions

ENV = ReversiEnvCNN(opponent="Random")


def env_moves(board):
    return np.flatnonzero(ENV._get_valid(board.reshape(1, 8, 8), BLACK))


def env_play(board, move):
    b = board.reshape(1, 8, 8).copy()
    ENV.player = BLACK
    ENV.get_next_state(b, (0, move // 8, move % 8))
    return -b.reshape(64)


def disc_eval(boards):
    """Test evaluator: disc difference / 64 from each board's side-to-move perspective."""
    return (boards == 1).sum(axis=1) / 64 - (boards == -1).sum(axis=1) / 64


def reference(board, depth):
    """Brute-force negamax using only env calls; same rules as util.search."""
    moves = env_moves(board)
    if len(moves) == 0:
        if len(env_moves(-board)) == 0:
            return float(np.sign((board == 1).sum() - (board == -1).sum()))
        return -reference(-board, depth)
    if depth == 0:
        return float(disc_eval(board[None])[0])
    return max(-reference(env_play(board, m), depth - 1) for m in moves)


class TestSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.positions = sample_positions(n_games=6, seed=4)  # all game phases, side to move = 1

    def test_legal_moves_match_env(self):
        for b in self.positions:
            np.testing.assert_array_equal(legal_moves(b), env_moves(b))

    def test_play_move_matches_env(self):
        for b in self.positions[::3]:
            for m in legal_moves(b):
                np.testing.assert_array_equal(play_move(b, m), env_play(b, m))

    def test_search_matches_reference(self):
        """Root scores equal the brute-force negamax, at depths 1 and 2, including endgames
        (passes and finished games)."""
        for depth in (1, 2):
            for b in self.positions[::4]:
                if len(legal_moves(b)) == 0:
                    continue
                best, scores = search(b, depth, disc_eval)
                for m, s in scores.items():
                    self.assertAlmostEqual(s, -reference(env_play(b, m), depth - 1), places=9)
                self.assertEqual(scores[best], max(scores.values()))

    def test_takes_winning_final_move(self):
        """Last empty square: the move that ends the game in a win scores exactly +1."""
        b = np.ones(64, dtype=np.int8)
        b[0], b[1] = 0, -1          # our move at 0 flips the opponent stone at 1 -> all ours
        best, scores = search(b, 2, lambda x: np.zeros(len(x)))
        self.assertEqual(best, 0)
        self.assertEqual(scores[0], 1.0)

    def test_depth1_negates_leaf_values(self):
        """At depth 1 each root move's score is minus the evaluator's value of the resulting
        position (the opponent's view), unless the game ends there."""
        b = self.positions[5]
        _, scores = search(b, 1, disc_eval)
        for m, s in scores.items():
            child = play_move(b, m)
            if len(legal_moves(child)) or len(legal_moves(-child)):
                expected = -disc_eval(child[None])[0] if len(legal_moves(child)) else disc_eval(-child[None])[0]
                self.assertAlmostEqual(s, expected)

    def test_root_moves_subset(self):
        b = self.positions[10]
        moves = legal_moves(b)[:2]
        best, scores = search(b, 2, disc_eval, root_moves=moves)
        self.assertEqual(set(scores), set(int(m) for m in moves))

    def test_search_player_with_model(self):
        """SearchPlayer (value head + optional policy top-k) always returns a legal move."""
        import tempfile
        from util.util import get_model
        with tempfile.TemporaryDirectory() as d:
            model = get_model(os.path.join(d, "m"), ENV, model_type="resnet", channels=8, blocks=1)
            for top_k in (None, 2):
                player = SearchPlayer(model, depth=2, top_k=top_k)
                for b in self.positions[::7]:
                    if len(legal_moves(b)):
                        self.assertIn(player(b), set(legal_moves(b)))


if __name__ == '__main__':
    unittest.main()
