"""
Residual-trunk policy with spatial heads for MaskablePPO.

The network maps the (1, 8, 8) board to 5 feature planes (util/board_features.py),
runs a residual conv trunk that keeps the 8x8 layout, and reads the policy off each
square with 1x1 convs (one logit per square) instead of flattening into a large
fully-connected layer. The value head summarizes the same trunk output.

It plugs into SB3's policy pipeline:
    features_extractor = ReversiResNet   (trunk; output C*64, flattened in square order)
    mlp_extractor      = SpatialHeads    (reshapes to C x 8 x 8; policy -> 64 logits, value -> 64 latent)
    action_net         = Identity        (the policy head already produces per-square logits)
    value_net          = Linear(64, 1)   (SB3's default)
The trunk is shared by both heads (AlphaZero style).
"""

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from util.board_features import board_to_planes, N_PLANES

BOARD_SQUARES = 64
VALUE_HIDDEN = 64


def conv_bn(in_ch, out_ch, kernel_size):
    # No conv bias: BatchNorm's shift makes it redundant
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_bn(channels, channels, 3)
        self.conv2 = conv_bn(channels, channels, 3)

    def forward(self, x):
        y = th.relu(self.conv1(x))
        y = self.conv2(y)
        return th.relu(x + y)


class ReversiResNet(BaseFeaturesExtractor):
    """Residual trunk. Output is (N, channels * 64): the C x 8 x 8 map, flattened."""

    def __init__(self, observation_space: gym.spaces.Box, channels: int = 64, blocks: int = 6):
        super().__init__(observation_space, features_dim=channels * BOARD_SQUARES)
        self.channels = channels
        self.stem = nn.Sequential(conv_bn(N_PLANES, channels, 3), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.blocks(self.stem(board_to_planes(observations)))
        return x.flatten(1)


class SpatialHeads(nn.Module):
    """Policy and value heads over the trunk's C x 8 x 8 map (SB3 mlp_extractor interface)."""

    def __init__(self, features_dim: int):
        super().__init__()
        self.channels = features_dim // BOARD_SQUARES
        self.latent_dim_pi = BOARD_SQUARES   # one logit per square, in row * 8 + col order
        self.latent_dim_vf = VALUE_HIDDEN
        self.policy_head = nn.Sequential(
            conv_bn(self.channels, 2, 1), nn.ReLU(),
            nn.Conv2d(2, 1, 1),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            conv_bn(self.channels, 1, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SQUARES, VALUE_HIDDEN), nn.ReLU(),
        )

    def _to_map(self, features):
        return features.view(-1, self.channels, 8, 8)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_head(self._to_map(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_head(self._to_map(features))

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


class ReversiResNetPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy using ReversiResNet + SpatialHeads.

    Trunk size is set with policy_kwargs=dict(features_extractor_kwargs=dict(channels=..., blocks=...)).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", ReversiResNet)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialHeads(self.features_dim).to(self.device)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # SB3 put a dense Linear(64, 64) on top of the policy head; that would mix the
        # squares again, so the head's per-square logits are used directly.
        self.action_net = nn.Identity()
        if self.ortho_init:
            # SB3 initializes action_net with gain 0.01 so the starting policy is near
            # uniform; apply that to the layer that now produces the logits.
            final = self.mlp_extractor.policy_head[2]
            nn.init.orthogonal_(final.weight, gain=0.01)
            nn.init.zeros_(final.bias)
        # Rebuild the optimizer over the final parameter set
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
