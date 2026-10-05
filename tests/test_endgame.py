#!/usr/bin/env python3
"""
Tests for the exact endgame solver (solver/endgame.c via util/endgame.py): move generation and
flips against util/search.py, and exact scores against a brute-force Python negamax.
Skipped if the library hasn't been built (solver/build.sh).
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.endgame import _LIB_PATH
from util.search import legal_moves, play_move
from tests.test_start_positions import sample_positions

BUILT = os.path.exists(_LIB_PATH)


def final_score(board):
    """Official final score for the side to move: disc difference, empties to the winner."""
    p, o = int((board == 1).sum()), int((board == -1).sum())
    e = 64 - p - o
    return p - o + e if p > o else (p - o - e if o > p else 0)


def brute(board):
    """Exact negamax over every line (small endgames only)."""
    moves = legal_moves(board)
    if len(moves) == 0:
        if len(legal_moves(-board)) == 0:
            return final_score(board)
        return -brute(-board)
    return max(-brute(play_move(board, int(m))) for m in moves)


@unittest.skipUnless(BUILT, f"endgame solver not built ({_LIB_PATH}); run solver/build.sh")
class TestEndgameSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.positions = sample_positions(n_games=8, seed=21)       # all phases, side to move = 1

    def test_moves_match(self):
        from util.endgame import moves
        for b in self.positions:
            self.assertEqual(moves(b), [int(m) for m in legal_moves(b)])

    def test_flips_match(self):
        from util.endgame import flips
        for b in self.positions[::3]:
            for m in legal_moves(b):
                after = -play_move(b, int(m))               # back to the mover's view
                expected = sorted(int(i) for i in np.flatnonzero((b == -1) & (after == 1)))
                self.assertEqual(flips(b, int(m)), expected)

    def test_exact_scores_small_endgames(self):
        from util.endgame import solve
        late = [b for b in self.positions if (b == 0).sum() <= 8][:16]     # brute force is slow above ~8
        self.assertGreaterEqual(len(late), 10)
        for b in late:
            score, best, _ = solve(b)
            self.assertEqual(score, brute(b))
            if len(legal_moves(b)):
                self.assertEqual(-brute(play_move(b, best)), score)    # the best move achieves it
            else:
                self.assertEqual(best, -1)

    def test_window_bounds(self):
        """A null window around 0 classifies win / draw / loss correctly."""
        from util.endgame import solve
        for b in [b for b in self.positions if (b == 0).sum() <= 8][:12]:
            exact = brute(b)
            s, _, _ = solve(b, -1, 1)
            self.assertEqual(np.sign(s), np.sign(exact))


    def test_search_player_uses_solver_late(self):
        """With solve_empties, late positions get a move that achieves the exact value."""
        import tempfile
        from util.util import get_model
        from util.search import SearchPlayer
        from util.reversi import ReversiEnvCNN
        env = ReversiEnvCNN(opponent="Random")
        with tempfile.TemporaryDirectory() as d:
            model = get_model(os.path.join(d, "m"), env, model_type="resnet", channels=8, blocks=1)
        late = [b for b in self.positions if 0 < (b == 0).sum() <= 8 and len(legal_moves(b))][:8]
        early = [b for b in self.positions if (b == 0).sum() > 30][:3]
        player = SearchPlayer(model, depth=1, solve_empties=8)
        chosen = player.choose_many(early + late)
        for b, m in zip(late, chosen[len(early):]):
            self.assertEqual(-brute(play_move(b, m)), brute(b))
        for b, m in zip(early, chosen[:len(early)]):
            self.assertIn(m, set(int(x) for x in legal_moves(b)))


    def test_solving_evaluator(self):
        """Late leaves get tanh(exact score / 16); the others the wrapped evaluator's value."""
        from util.search import solving_evaluator
        late = [b for b in self.positions if (b == 0).sum() <= 8][:5]
        early = [b for b in self.positions if (b == 0).sum() > 30][:3]
        boards = np.array(early + late)
        out = solving_evaluator(lambda bs: np.full(len(bs), 0.123), 8)(boards)
        np.testing.assert_allclose(out[:len(early)], 0.123)
        np.testing.assert_allclose(out[len(early):], np.tanh(np.array([brute(b) for b in late]) / 16.0))


if __name__ == '__main__':
    unittest.main()
