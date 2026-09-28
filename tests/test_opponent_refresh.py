#!/usr/bin/env python3
"""
Regression test: a self-play opponent saved to the same file must actually refresh.

get_opponent() caches ModelOpponents per model file, and sb-train.py's self-play modes
overwrite one temp file at every refresh, so without reload=True the opponent stayed
frozen at the first save for the whole run.
"""

import unittest
import tempfile
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import get_model


def first_weight(opponent):
    return next(opponent.model.policy.parameters()).detach().cpu().clone()


class TestOpponentRefresh(unittest.TestCase):

    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "temp_opponent")
        self.model = get_model(self.path + "_train", self.env, model_type="resnet", channels=8, blocks=1)

    def save_with_offset(self, offset):
        with torch.no_grad():
            next(self.model.policy.parameters()).add_(offset)
        self.model.save(self.path)

    def test_reload_picks_up_new_weights(self):
        self.save_with_offset(0.0)
        self.env.set_opponent("Model", self.path, reload=True)
        before = first_weight(self.env.opponent)

        self.save_with_offset(1.0)  # "refresh": overwrite the same file
        self.env.set_opponent("Model", self.path, reload=True)
        after = first_weight(self.env.opponent)
        self.assertTrue(torch.allclose(after, before + 1.0))

    def test_without_reload_is_cached(self):
        """Documents the cache: without reload the old weights are kept."""
        self.save_with_offset(0.0)
        self.env.set_opponent("Model", self.path)
        first = self.env.opponent
        self.save_with_offset(1.0)
        self.env.set_opponent("Model", self.path)
        self.assertIs(self.env.opponent, first)


if __name__ == '__main__':
    unittest.main()
