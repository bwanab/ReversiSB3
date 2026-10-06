#!/usr/bin/env python3
"""
Egaroucid client (util/egaroucid_client.py): legal moves from positions in all game phases, either
side to move. Skipped if Egaroucid for Console isn't built.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.egaroucid_client import DEFAULT_PATH
from util.search import legal_moves
from tests.test_start_positions import sample_positions

BUILT = os.path.exists(os.environ.get("EGAROUCID", DEFAULT_PATH))


@unittest.skipUnless(BUILT, "Egaroucid for Console not built (see CLAUDE.md, 'Egaroucid')")
class TestEgaroucidClient(unittest.TestCase):

    def test_legal_moves(self):
        from util.egaroucid_client import EgaroucidClient
        client = EgaroucidClient(threads=1)
        try:
            positions = [b for b in sample_positions(n_games=3, seed=41) if len(legal_moves(b))]
            for level in (1, 4):
                for b in positions:
                    r = client.analyze(b, level)
                    self.assertIn(r["move"], set(int(m) for m in legal_moves(b)))
        finally:
            client.close()


    def test_labeling(self):
        """label_boards(teacher="egaroucid"): every legal move scored and nothing else; best move and
        score are the best of them."""
        import numpy as np
        from label_positions import label_boards
        positions = [b for b in sample_positions(n_games=2, seed=43) if len(legal_moves(b))][::6][:8]
        for b, (best, score, ms) in zip(positions, label_boards(np.array(positions), 4, True, 2, teacher="egaroucid")):
            self.assertEqual(set(np.flatnonzero(~np.isnan(ms))), set(int(m) for m in legal_moves(b)))
            self.assertEqual(best, int(np.nanargmax(ms)))
            self.assertEqual(score, int(np.nanmax(ms)))


if __name__ == '__main__':
    unittest.main()
