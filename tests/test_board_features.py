#!/usr/bin/env python3
"""
Tests for util/board_features.py: feature planes, board symmetries, and the
ReversiCNN input_planes option.
"""

import unittest
import tempfile
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.board_features import board_to_planes, random_symmetry, PERMS, INV_PERMS
from util.util import get_model, BLACK, WHITE


def random_positions(env, n_games=20, seed=0):
    """Positions from random games, each paired with the player to move."""
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
            positions.append((board.copy(), player))
            env.player = player
            actions = env._all_valid_actions(board, player)
            a = int(rng.choice(actions))
            env.get_next_state(board, (0, a // 8, a % 8))
            player = -player
    return positions


class TestBoardFeatures(unittest.TestCase):

    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        self.positions = random_positions(self.env)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_planes_match_env(self):
        """Stone planes and both legal-move planes agree with the env on random positions."""
        boards = np.stack([b * p for b, p in self.positions]).astype(np.float32)  # mover's view
        planes = board_to_planes(torch.as_tensor(boards)).numpy()
        for i, (board, player) in enumerate(self.positions):
            view = board * player
            np.testing.assert_array_equal(planes[i, 0], (view[0] == 1))
            np.testing.assert_array_equal(planes[i, 1], (view[0] == -1))
            np.testing.assert_array_equal(planes[i, 2], (view[0] == 0))
            np.testing.assert_array_equal(planes[i, 3].flatten(), self.env._get_valid(board, player))
            np.testing.assert_array_equal(planes[i, 4].flatten(), self.env._get_valid(board, -player))

    def test_opening_planes(self):
        board, _ = self.env.reset()
        planes = board_to_planes(torch.as_tensor(board[None]).float())[0]
        self.assertEqual(sorted(np.flatnonzero(planes[3].numpy())), [19, 26, 37, 44])
        self.assertEqual(sorted(np.flatnonzero(planes[4].numpy())), [20, 29, 34, 43])

    def test_symmetry_tables(self):
        """The 8 symmetries are distinct, include the identity, and match numpy's rot90/fliplr."""
        self.assertEqual(len({tuple(p) for p in PERMS}), 8)
        np.testing.assert_array_equal(PERMS[0], np.arange(64))
        board = np.arange(64).reshape(8, 8)
        np.testing.assert_array_equal(board.flatten()[PERMS[1]], np.rot90(board).flatten())
        np.testing.assert_array_equal(board.flatten()[PERMS[4]], np.fliplr(board).flatten())
        for k in range(8):
            np.testing.assert_array_equal(INV_PERMS[k][PERMS[k]], np.arange(64))

    def test_symmetry_preserves_legality(self):
        """A legal move stays legal (and the stone planes follow) under random symmetries."""
        views, actions = [], []
        for board, player in self.positions:
            views.append(board * player)
            actions.append(self.env._all_valid_actions(board, player)[0])
        states = torch.as_tensor(np.stack(views)).float()
        actions = torch.as_tensor(np.array(actions))
        torch.manual_seed(0)
        new_states, new_actions = random_symmetry(states, actions)
        legal = board_to_planes(new_states)[:, 3].reshape(-1, 64)
        self.assertTrue(bool(legal[torch.arange(len(actions)), new_actions].all()))
        # Stone counts are unchanged
        self.assertTrue(torch.equal(new_states.sum(dim=(1, 2, 3)), states.sum(dim=(1, 2, 3))))

    def test_cnn_input_planes(self):
        """A planes model accepts the unchanged (1, 8, 8) observation and survives save/load."""
        path = os.path.join(self.tmp.name, "planes_model")
        model = get_model(path, self.env, input_planes=True)
        extractor = model.policy.features_extractor
        self.assertEqual(extractor.cnn[0].in_channels, 5)
        board, _ = self.env.reset()
        action, _ = model.predict(board, action_masks=self.env.get_valid(board) == 1)
        self.assertIn(int(action), [19, 26, 37, 44])

        model.save(path)
        loaded = get_model(path, self.env)
        self.assertTrue(loaded.policy.features_extractor.input_planes)
        obs = torch.as_tensor(board[None]).float()
        with torch.no_grad():
            a = model.policy.features_extractor(obs.to(model.device))
            b = loaded.policy.features_extractor(obs.to(loaded.device))
        self.assertTrue(torch.allclose(a.cpu(), b.cpu()))

    def test_cnn_default_unchanged(self):
        model = get_model(os.path.join(self.tmp.name, "board_model"), self.env)
        self.assertEqual(model.policy.features_extractor.cnn[0].in_channels, 1)
        self.assertFalse(model.policy.features_extractor.input_planes)


if __name__ == '__main__':
    unittest.main()
