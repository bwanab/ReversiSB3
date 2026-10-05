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
from util.search import legal_moves, play_move, search, search_many, SearchPlayer
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


class TestBatchedSearch(unittest.TestCase):
    """search_many / choose_many give the same moves as one-at-a-time search; the lockstep
    driver plays complete, legal games."""

    @classmethod
    def setUpClass(cls):
        import tempfile, torch
        from util.util import get_model
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = get_model(os.path.join(cls.tmp.name, "m"), ENV, model_type="resnet", channels=8, blocks=1)
        cls.positions = [b for b in sample_positions(n_games=4, seed=9) if len(legal_moves(b))][::3]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_search_many_matches_search(self):
        many = search_many(self.positions, 2, disc_eval)
        for b, (best, scores) in zip(self.positions, many):
            self.assertEqual((best, scores), search(b, 2, disc_eval))

    def test_choose_many_matches_single(self):
        """With the same root candidates, batched search scores every move like one-at-a-time
        search (to float tolerance: batched and single network calls differ by ~1e-7), and
        choose_many plays the batched best move. (Root candidates themselves come from one batched
        policy call; this untrained network's near-uniform logits make single-call rankings
        differ, so they're computed once and shared here.)"""
        from util.search import policy_top_moves
        for depth, top_k in ((1, None), (2, 3)):
            player = SearchPlayer(self.model, depth=depth, top_k=top_k)
            roots = policy_top_moves(self.model, self.positions, top_k) if top_k else [None] * len(self.positions)
            batched = search_many(self.positions, depth, player.evaluate, roots)
            for b, r, (best, scores) in zip(self.positions, roots, batched):
                _, single = search(b, depth, player.evaluate, r)
                self.assertEqual(set(single), set(scores))
                for m in scores:
                    self.assertAlmostEqual(scores[m], single[m], places=5)
            self.assertEqual(player.choose_many(self.positions), [best for best, _ in batched])

    def test_chunked_network_calls(self):
        """Network calls split into chunks give the same results as one call (the split guards
        against MPS returning wrong values for batches over ~65k positions)."""
        from util.search import policy_logits, value_evaluator
        boards = np.array(self.positions * 3)
        np.testing.assert_allclose(policy_logits(self.model, boards, max_batch=7),
                                   policy_logits(self.model, boards), atol=1e-5)
        np.testing.assert_allclose(value_evaluator(self.model, max_batch=5)(boards),
                                   value_evaluator(self.model)(boards), atol=1e-5)

    def test_top_move_is_policy_argmax(self):
        from util.search import policy_logits
        player = SearchPlayer(self.model, depth=0)
        logits = policy_logits(self.model, self.positions)
        for b, lg, m in zip(self.positions, logits, player.choose_many(self.positions)):
            moves = legal_moves(b)
            self.assertEqual(m, int(moves[np.argmax(lg[moves])]))

    def test_lockstep_games_complete(self):
        """Against a stand-in 'Edax' that plays its first legal move, every game ends, and the
        result equals the final disc difference from the model's side."""
        from util.lockstep import play_vs_edax

        class FirstMove:
            def analyze(self, board, depth):
                return {"move": int(legal_moves(np.asarray(board).reshape(64))[0])}
        player = SearchPlayer(self.model, depth=0)
        recorded = []
        diffs = play_vs_edax(player.choose_many, self.positions[:6], 1, client=FirstMove(),
                             on_model_move=lambda g, b, m: recorded.append((g, b, m)))
        self.assertEqual(len(diffs), 6)
        self.assertTrue(all(-64 <= d <= 64 for d in diffs))
        # the hook sees every model move, each legal in the recorded position
        self.assertEqual({g for g, _, _ in recorded}, set(range(6)))
        for _, b, m in recorded:
            self.assertIn(m, set(int(x) for x in legal_moves(b)))

    def test_sb_play_starts_reproducible(self):
        from util.lockstep import sb_play_starts
        a, b = sb_play_starts(5, 8, 42), sb_play_starts(5, 8, 42)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertTrue(all(len(legal_moves(x)) for x in a))


class TestPrunedSearch(unittest.TestCase):
    """search_many_pruned: identical to search_many when nothing is pruned, and equal to a
    brute-force pruned negamax built from env calls when it is."""

    @classmethod
    def setUpClass(cls):
        cls.positions = [b for b in sample_positions(n_games=5, seed=11) if len(legal_moves(b))][::4]

    def test_no_pruning_matches_search_many(self):
        from util.search import search_many_pruned
        all_moves = lambda boards, k: [legal_moves(b) for b in boards]
        for depth in (1, 2, 3):
            pruned = search_many_pruned(self.positions, depth, disc_eval, all_moves, 64)
            full = search_many(self.positions, depth, disc_eval)
            for (b1, s1), (b2, s2) in zip(pruned, full):
                self.assertEqual(set(s1), set(s2))
                for m in s1:
                    self.assertAlmostEqual(s1[m], s2[m], places=9)

    def test_pruning_matches_reference(self):
        """Keep the first k legal moves at every node (a deterministic stand-in for the policy)."""
        from util.search import search_many_pruned
        first_k = lambda boards, k: [legal_moves(b)[:k] for b in boards]

        def ref(board, depth, k):
            moves = env_moves(board)
            if len(moves) == 0:
                if len(env_moves(-board)) == 0:
                    return float(np.sign((board == 1).sum() - (board == -1).sum()))
                return -ref(-board, depth, k)
            if depth == 0:
                return float(disc_eval(board[None])[0])
            return max(-ref(env_play(board, m), depth - 1, k) for m in moves[:k])

        for depth, k in ((2, 2), (3, 2), (3, 3)):
            for b, (best, scores) in zip(self.positions, search_many_pruned(self.positions, depth, disc_eval, first_k, k)):
                self.assertEqual(set(scores), set(int(m) for m in env_moves(b)[:k]))
                for m, sc in scores.items():
                    self.assertAlmostEqual(sc, -ref(env_play(b, m), depth - 1, k), places=9)

    def test_search_player_prune_all(self):
        import tempfile
        from util.util import get_model
        with tempfile.TemporaryDirectory() as d:
            model = get_model(os.path.join(d, "m"), ENV, model_type="resnet", channels=8, blocks=1)
        player = SearchPlayer(model, depth=3, top_k=2, prune_all=True)
        for b, m in zip(self.positions, player.choose_many(self.positions)):
            self.assertIn(m, set(int(x) for x in legal_moves(b)))


class TestSearchCollection(unittest.TestCase):
    """collect_positions.search_moves: the search's move, or (explore) a policy top-k move."""

    def test_explore_extremes(self):
        import tempfile
        from util.util import get_model
        from util.search import policy_top_moves
        from collect_positions import search_moves
        positions = [b for b in sample_positions(n_games=3, seed=13) if len(legal_moves(b))][::5]
        with tempfile.TemporaryDirectory() as d:
            model = get_model(os.path.join(d, "m"), ENV, model_type="resnet", channels=8, blocks=1)
        player = SearchPlayer(model, depth=2, top_k=3, prune_all=True)
        rng = np.random.default_rng(0)
        self.assertEqual(search_moves(player, model, positions, 3, 0.0, rng), player.choose_many(positions))
        tops = policy_top_moves(model, positions, 3)
        for m, top in zip(search_moves(player, model, positions, 3, 1.0, rng), tops):
            self.assertIn(m, set(int(x) for x in top))


if __name__ == '__main__':
    unittest.main()
