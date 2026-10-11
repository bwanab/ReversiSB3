#!/usr/bin/env python3
"""Tests for util/lockstep.play_match (two of our players against each other) and util/hints.py."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.lockstep import play_match
from util.reversi import ReversiEnvCNN
from util.search import legal_moves, play_move
from tests.test_start_positions import sample_positions


def first_move(boards):
    return [int(legal_moves(b)[0]) for b in boards]


def greedy(boards):
    """Most discs after the move (a simple deterministic player)."""
    return [max((int(m) for m in legal_moves(b)), key=lambda m: int((play_move(b, m) == -1).sum())) for b in boards]


def reference_game(choose_a, choose_b, start, a_first):
    """One game, sequentially; final margin from A's view."""
    b, a_to_move = np.asarray(start, dtype=np.int8).copy(), a_first
    while True:
        if not len(legal_moves(b)):
            if not len(legal_moves(-b)):
                d = int((b == 1).sum() - (b == -1).sum())
                return d if a_to_move else -d
            b, a_to_move = -b, not a_to_move
            continue
        m = (choose_a if a_to_move else choose_b)([b])[0]
        b, a_to_move = play_move(b, m), not a_to_move


class TestMatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.starts = [b for b in sample_positions(n_games=3, seed=21) if len(legal_moves(b))][::6][:6]

    def test_matches_reference_games(self):
        diffs = play_match(greedy, first_move, self.starts)
        n = len(self.starts)
        for i, s in enumerate(self.starts):
            self.assertEqual(diffs[i], reference_game(greedy, first_move, s, True))
            self.assertEqual(diffs[n + i], reference_game(greedy, first_move, s, False))

    def test_same_player_mirrors(self):
        diffs = play_match(greedy, greedy, self.starts)
        n = len(self.starts)
        np.testing.assert_array_equal(diffs[:n], -diffs[n:])        # same games, colors swapped


class TestHints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch
        from util.util import get_model
        torch.manual_seed(6)
        with tempfile.TemporaryDirectory() as d:
            cls.model = get_model(os.path.join(d, "m"), ReversiEnvCNN(opponent="Random"), model_type="resnet",
                                  channels=8, blocks=1)

    def test_scores_and_player(self):
        from util.endgame import solve
        from util.hints import MoveScorer, ScorePlayer
        boards = [b for b in sample_positions(n_games=4, seed=22) if len(legal_moves(b)) > 1]
        late = [b for b in boards if (b == 0).sum() <= 12][:4]
        early = [b for b in boards if (b == 0).sum() > 30][:3]
        scorer = MoveScorer(self.model, sims=40)
        for b, (margins, exact) in zip(late + early, scorer.margins(late + early)):
            self.assertEqual(set(margins), set(int(m) for m in legal_moves(b)))
            self.assertEqual(exact, (b == 0).sum() <= 18)
            if exact:
                for m, v in margins.items():
                    self.assertEqual(v, -solve(play_move(b, m))[0])
        player = ScorePlayer(self.model, sims=40)
        for b, m in zip(late, player.choose_many(late)):
            self.assertEqual(-solve(play_move(b, m))[0], solve(b)[0])     # a best move near the end


if __name__ == "__main__":
    unittest.main()
