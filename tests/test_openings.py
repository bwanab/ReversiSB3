#!/usr/bin/env python3
"""
Tests for util/openings.py (moves.txt parsing and replay) and make_opening_positions.py.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.openings import parse_moves, load_openings, replay, canonical, book_positions
from util.reversi import ReversiEnvCNN
from util.util import BLACK
from util.board_features import PERMS
from make_opening_positions import opening_positions

MOVES_TXT = os.path.join(os.path.dirname(__file__), '..', 'moves.txt')


class TestOpenings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.openings = load_openings(MOVES_TXT)

    def test_parse_moves(self):
        # f5 = row 5, column f -> row index 4, col 5; BLACK's opening moves are 19, 26, 37, 44
        self.assertEqual(parse_moves("F5"), [37])
        self.assertEqual(parse_moves("C4c3D3"), [26, 18, 19])

    def test_all_lines_replay_legally(self):
        """replay() asserts legality of every move; run it over the whole file."""
        self.assertGreater(len(self.openings), 300)
        for moves, name in self.openings:
            steps = list(replay(moves))
            self.assertEqual(len(steps), len(moves) + 1)

    def test_replay_perspective(self):
        """Positions are from the side to move: after one move, the mover (WHITE) has 1 stone
        (+1 values) and the opponent 4."""
        steps = list(replay(parse_moves("F5d6")))
        pos, move = steps[1]
        self.assertEqual(move, parse_moves("d6")[0])
        self.assertEqual(int((pos == 1).sum()), 1)
        self.assertEqual(int((pos == -1).sum()), 4)

    def test_canonical_merges_symmetries(self):
        flat = np.arange(64, dtype=np.int8) % 3 - 1
        keys = {canonical(flat[p]) for p in PERMS}
        self.assertEqual(len(keys), 1)

    def test_book_positions_have_legal_replies(self):
        env = ReversiEnvCNN(opponent="Random")
        book = book_positions(self.openings)
        for key, replies in book.items():
            pos = np.frombuffer(key, dtype=np.int8).reshape(1, 8, 8)
            for move in replies:
                self.assertTrue(env.is_valid(pos, BLACK, np.array([0, move // 8, move % 8])))

    def test_opening_positions(self):
        positions = opening_positions(self.openings)
        env = ReversiEnvCNN(opponent="Random")
        self.assertGreater(len(positions), 50)
        self.assertEqual(len({canonical(p) for p in positions}), len(positions))
        for p in positions:
            self.assertTrue(env.has_valid(p.reshape(1, 8, 8), BLACK))


if __name__ == '__main__':
    unittest.main()
