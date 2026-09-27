#!/usr/bin/env python3
"""
Tests for util/reversi_resnet.py: the residual-trunk policy with spatial heads.
"""

import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.reversi_resnet import ReversiResNet, SpatialHeads, ReversiResNetPolicy
from util.util import get_model


class TestResNetPolicy(unittest.TestCase):

    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = get_model(os.path.join(self.tmp.name, "resnet"), self.env, model_type="resnet",
                               channels=16, blocks=2)
        self.policy = self.model.policy
        board, _ = self.env.reset()
        self.board = board

    def obs(self, n=4):
        boards = np.repeat(self.board[None], n, axis=0)
        return torch.as_tensor(boards).float().to(self.policy.device)

    def test_structure(self):
        self.assertIsInstance(self.policy, ReversiResNetPolicy)
        self.assertIsInstance(self.policy.features_extractor, ReversiResNet)
        self.assertIsInstance(self.policy.mlp_extractor, SpatialHeads)
        self.assertIsInstance(self.policy.action_net, nn.Identity)
        self.assertEqual(self.policy.features_extractor.features_dim, 16 * 64)
        self.assertEqual(len(self.policy.features_extractor.blocks), 2)
        # The optimizer covers exactly the policy's parameters
        opt_params = {id(p) for g in self.policy.optimizer.param_groups for p in g['params']}
        self.assertEqual(opt_params, {id(p) for p in self.policy.parameters()})

    def test_bc_forward_path_shapes(self):
        """bc_train.py's path: extract_features -> mlp_extractor -> action_net / value_net."""
        self.policy.eval()
        features = self.policy.extract_features(self.obs())
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
        self.assertEqual(self.policy.action_net(latent_pi).shape, (4, 64))
        self.assertEqual(self.policy.value_net(latent_vf).shape, (4, 1))

    def test_initial_policy_near_uniform(self):
        self.policy.eval()
        with torch.no_grad():
            features = self.policy.extract_features(self.obs(1))
            logits = self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))
        self.assertLess(logits.abs().max().item(), 0.5)

    def test_logits_are_per_square(self):
        """Each logit is read off its own square: perturbing the trunk output at one square
        changes that square's logit and no other."""
        torch.manual_seed(0)
        self.policy.eval()
        heads = self.policy.mlp_extractor
        changed_any = False
        with torch.no_grad():
            nn.init.normal_(heads.policy_head[2].weight)  # undo the tiny init so changes are visible
            features = self.policy.extract_features(self.obs(1))
            base = heads.forward_actor(features)
            # The head's ReLU can zero a square's output for a given perturbation, so try several
            for _ in range(10):
                fmap = features.view(1, 16, 8, 8).clone()
                fmap[0, :, 2, 3] = torch.randn(16, device=fmap.device) * 3
                diff = (base - heads.forward_actor(fmap.flatten(1))).abs()[0] > 1e-6
                changed = np.flatnonzero(diff.cpu().numpy()).tolist()
                self.assertIn(changed, ([], [2 * 8 + 3]))
                changed_any = changed_any or changed == [2 * 8 + 3]
        self.assertTrue(changed_any)

    def test_predict_respects_mask(self):
        mask = self.env.get_valid(self.board) == 1
        for _ in range(20):
            action, _ = self.model.predict(self.board, action_masks=mask, deterministic=False)
            self.assertIn(int(action), [19, 26, 37, 44])

    def test_bc_step_reduces_loss(self):
        """A few supervised steps on one batch reduce the policy loss (training mode, BatchNorm on)."""
        self.policy.train()
        states = torch.as_tensor(np.stack([self.board] * 8)).float().to(self.policy.device)
        actions = torch.full((8,), 19, device=self.policy.device)
        opt = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            features = self.policy.extract_features(states)
            latent_pi, _ = self.policy.mlp_extractor(features)
            loss = F.cross_entropy(self.policy.action_net(latent_pi), actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_save_load_round_trip(self):
        """Weights, BatchNorm running stats, and trunk size survive save/load."""
        self.policy.train()
        with torch.no_grad():
            self.policy.extract_features(self.obs(8) * 0.5)  # update BatchNorm running stats
        self.policy.eval()
        path = os.path.join(self.tmp.name, "saved")
        self.model.save(path)
        loaded = get_model(path, self.env)
        self.assertIsInstance(loaded.policy, ReversiResNetPolicy)
        self.assertEqual(len(loaded.policy.features_extractor.blocks), 2)
        loaded.policy.eval()
        with torch.no_grad():
            obs = self.obs(2)
            a, _, _ = self.policy.evaluate_actions(obs, torch.tensor([19, 26], device=obs.device))
            b, _, _ = loaded.policy.evaluate_actions(obs.to(loaded.device),
                                                     torch.tensor([19, 26], device=loaded.device))
        self.assertTrue(torch.allclose(a.cpu(), b.cpu(), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
