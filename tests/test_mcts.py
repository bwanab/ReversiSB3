#!/usr/bin/env python3
"""
Tests for util/mcts.py: exact bookkeeping between every edge and its subtree (value negation through
moves and passes, virtual loss, collisions), best moves with perfect values and with exact leaf
solving, batch independence, official final scores, and the combined policy/value network call.
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.search import legal_moves, play_move, policy_logits, policy_value, value_evaluator, SearchPlayer
from util.mcts import MCTS, Node, PASS, final_value
from util.endgame import solve
from tests.test_start_positions import sample_positions

ENV = ReversiEnvCNN(opponent="Random")


def flat_net(boards):
    """No knowledge: equal priors, value 0."""
    return np.zeros((len(boards), 64)), np.zeros(len(boards))


def hashed_net(boards):
    """Deterministic, board-dependent priors and values (for comparing runs)."""
    rng = np.random.default_rng(7)
    proj = rng.normal(size=(64, 64))
    b = np.asarray(boards, dtype=np.float64)
    return b @ proj, np.tanh(b @ proj[:, 0] / 8)


def move_scores(board):
    """Exact score of every legal move (side to move's view)."""
    return {int(m): -solve(play_move(board, int(m)))[0] for m in legal_moves(board)}


def random_positions(n, empties, seed):
    """Positions with `empties` empty squares and a legal move, from random games."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = np.zeros(64, dtype=np.int8)
        b[[27, 36]], b[[28, 35]] = -1, 1
        while (b == 0).sum() > empties:
            moves = legal_moves(b)
            if len(moves) == 0:
                if len(legal_moves(-b)) == 0:
                    break
                b = -b
                continue
            b = play_move(b, int(rng.choice(moves)))
        if (b == 0).sum() == empties and len(legal_moves(b)) > 1:
            out.append(b)
    return out


class TestMCTS(unittest.TestCase):

    def test_final_value_official_score(self):
        b = np.zeros(64, dtype=np.int8)
        b[:40], b[40:60] = 1, -1                     # 40-20 with 4 empties -> 44-20 = +24
        self.assertAlmostEqual(final_value(b), np.tanh(24 / 16))
        self.assertAlmostEqual(final_value(-b), -np.tanh(24 / 16))

    def _check_edge(self, parent, i):
        """Edge parent -> child i: its visits and value total (parent's view) must equal what the
        child's subtree backed up: the child's own value once plus its edges' totals, negated."""
        child = parent.children[i]
        n, w = parent.n[i], parent.w[i]
        if child is None:
            self.assertEqual(n, 0)
            return
        if child.exact:
            self.assertAlmostEqual(w, -n * child.value, places=9)
            return
        own = 0 if parent.moves[i] == PASS else 1      # a pass wrapper's child was expanded with it
        self.assertEqual(n, own + child.n.sum())
        self.assertAlmostEqual(w, -(own * child.value + child.w.sum()), places=9)
        for j in range(len(child.moves)):
            self._check_edge(child, j)

    def test_backup_bookkeeping(self):
        """Exact identities between every edge and its subtree (value negation, passes, virtual loss
        removed, collisions undone), after searches with many leaves per round."""
        boards = random_positions(6, 10, seed=8) + random_positions(6, 30, seed=9)
        for parallel in (1, 8):
            mcts = MCTS(hashed_net, sims=400, parallel=parallel, leaf_solve_empties=6)
            mcts.run(boards)
            for root in mcts.last_roots:
                self.assertLessEqual(root.n.sum(), 400)
                for i in range(len(root.moves)):
                    self._check_edge(root, i)

    def test_perfect_values_choose_good_moves(self):
        """With the exact value as the network's value, MCTS mostly plays best moves. Not always:
        Q is an average over the subtree, including the opponent's weaker replies (measured 2026-10-07:
        80-88% best moves on 10-empty positions for c_puct 0.5-2, 100-300 simulations)."""
        def exact_net(boards):
            return np.zeros((len(boards), 64)), np.array([np.tanh(solve(b)[0] / 16) for b in boards])
        regrets = []
        for b in random_positions(40, 10, seed=3):
            scores = move_scores(b)
            best, _ = MCTS(exact_net, sims=300, parallel=1).run([b])[0]
            regrets.append(max(scores.values()) - scores[best])
        self.assertGreaterEqual(np.mean(np.array(regrets) == 0), 0.75)
        self.assertLess(np.mean(regrets), 1.5)

    def test_exact_leaf_solving(self):
        """Every child solved exactly: the most visited move is a best move (or within 0.05 of it on
        the tanh scale, where big wins are nearly equal)."""
        for b in random_positions(10, 12, seed=5):
            scores = move_scores(b)
            mcts = MCTS(flat_net, sims=200, parallel=4, leaf_solve_empties=64)
            best, visits = mcts.run([b])[0]
            self.assertGreaterEqual(np.tanh(scores[best] / 16), np.tanh(max(scores.values()) / 16) - 0.05,
                                    (scores, visits))
            self.assertEqual(mcts.evaluations, 1)           # only the root used the network

    def test_batch_independence(self):
        """With one leaf per root per round, searching boards together equals searching them alone."""
        boards = [b for b in sample_positions(n_games=2, seed=11) if len(legal_moves(b)) > 1][::5]
        together = MCTS(hashed_net, sims=100, parallel=1).run(boards)
        alone = [MCTS(hashed_net, sims=100, parallel=1).run([b])[0] for b in boards]
        self.assertEqual(together, alone)

    def test_visits_and_legality(self):
        boards = [b for b in sample_positions(n_games=2, seed=12) if len(legal_moves(b)) > 1][::4]
        for b, (best, visits) in zip(boards, MCTS(hashed_net, sims=120, parallel=8).run(boards)):
            self.assertEqual(set(visits), set(int(m) for m in legal_moves(b)))
            self.assertIn(best, visits)
            self.assertLessEqual(sum(visits.values()), 120)
            self.assertGreater(sum(visits.values()), 60)        # few collisions
            self.assertEqual(visits[best], max(visits.values()))

    def test_single_move_needs_no_search(self):
        for b in sample_positions(n_games=6, seed=13):
            if len(legal_moves(b)) == 1:
                mcts = MCTS(hashed_net, sims=100)
                self.assertEqual(mcts.run([b])[0][0], int(legal_moves(b)[0]))
                self.assertEqual(mcts.evaluations, 1)
                return
        self.skipTest("no single-move position sampled")

    def test_pass_node(self):
        """A position whose side to move must pass becomes a PASS node wrapping the opponent's node."""
        for b in random_positions(200, 10, seed=21):
            for m in legal_moves(b):
                c = play_move(b, int(m))
                if len(legal_moves(c)) == 0 and len(legal_moves(-c)) > 0:
                    node = MCTS(hashed_net)._make_nodes([c])[0]
                    self.assertEqual(list(node.moves), [PASS])
                    child = node.children[0]
                    np.testing.assert_array_equal(child.board, -c)
                    self.assertAlmostEqual(node.value, -child.value)
                    return
        self.skipTest("no pass position found")


class TestPolicyValue(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile, torch
        from util.util import get_model
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as d:
            cls.model = get_model(os.path.join(d, "m"), ENV, model_type="resnet", channels=8, blocks=1)
        cls.positions = np.array([b for b in sample_positions(n_games=3, seed=19) if len(legal_moves(b))])

    def test_matches_separate_calls(self):
        logits, values = policy_value(self.model, self.positions)
        np.testing.assert_allclose(logits, policy_logits(self.model, self.positions), atol=1e-4)
        np.testing.assert_allclose(values, value_evaluator(self.model)(self.positions), atol=1e-4)

    def test_search_player_mcts(self):
        player = SearchPlayer(self.model, mcts_sims=40, mcts_parallel=4)
        for b, m in zip(self.positions, player.choose_many(list(self.positions))):
            self.assertIn(m, set(int(x) for x in legal_moves(b)))


if __name__ == "__main__":
    unittest.main()
