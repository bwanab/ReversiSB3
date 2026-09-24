"""
Tests that util.util.set_learning_rate updates the learning rate correctly and that the
new rate persists across save/load and learn() calls.
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from util.reversi import build_reversi
from util.reversi_cnn import ReversiCNN
from util.util import mask_fn, get_model, set_learning_rate, linear_schedule


def optimizer_lrs(model):
    """The learning rates the optimizer will actually use (not just the model attribute)."""
    return [param_group['lr'] for param_group in model.policy.optimizer.param_groups]


class TestLearningRateUpdates(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env = ActionMasker(build_reversi(opponent="Random"), mask_fn)
        # get_model creates a fresh CNN model when the file doesn't exist
        self.model = get_model(os.path.join(self.tmpdir.name, "lr_model"), self.env,
                               learning_rate=2e-5, device="cpu")

    def tearDown(self):
        self.tmpdir.cleanup()

    def assertLearningRate(self, model, lr):
        for actual in optimizer_lrs(model):
            self.assertAlmostEqual(actual, lr, places=12)
        self.assertAlmostEqual(model.learning_rate, lr, places=12)
        self.assertAlmostEqual(model.lr_schedule(1.0), lr, places=12)

    def test_basic_update(self):
        set_learning_rate(self.model, 1e-5)
        self.assertLearningRate(self.model, 1e-5)

    def test_repeated_updates(self):
        for lr in (5e-6, 3e-5, 1e-5):
            set_learning_rate(self.model, lr)
            self.assertLearningRate(self.model, lr)

    def test_update_after_save_and_reload(self):
        path = os.path.join(self.tmpdir.name, "lr_model_reloaded")
        self.model.save(path)
        reloaded = MaskablePPO.load(path, env=self.env, device="cpu")
        set_learning_rate(reloaded, 1.5e-5)
        self.assertLearningRate(reloaded, 1.5e-5)

    def test_persists_across_learn(self):
        # SB3 recomputes the optimizer LR from lr_schedule during learn(), so this is the
        # case set_learning_rate's lr_schedule override exists for
        set_learning_rate(self.model, 1.5e-5)
        self.model.learn(total_timesteps=self.model.n_steps, reset_num_timesteps=False, progress_bar=False)
        self.assertLearningRate(self.model, 1.5e-5)

    def test_overrides_schedule(self):
        policy_kwargs = dict(features_extractor_class=ReversiCNN,
                             features_extractor_kwargs=dict(features_dim=256),
                             normalize_images=False)
        model = MaskablePPO("CnnPolicy", self.env, policy_kwargs=policy_kwargs,
                            learning_rate=linear_schedule(2e-5), device="cpu", verbose=0)
        set_learning_rate(model, 5e-6)
        self.assertLearningRate(model, 5e-6)
        # the constant must also hold at other points of training progress
        self.assertAlmostEqual(model.lr_schedule(0.5), 5e-6, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
