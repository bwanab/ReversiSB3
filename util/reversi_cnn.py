import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import gymnasium as gym
from util.board_features import board_to_planes, N_PLANES

class ReversiCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, input_planes: bool = False):
        """input_planes: expand the (1, 8, 8) -1/0/1 board into own / opponent / empty /
        own legal moves / opponent legal moves planes before the first conv. The
        observation itself is unchanged, so envs and opponents need no changes."""
        super().__init__(observation_space, features_dim)
        self.input_planes = input_planes

        n_input_channels = N_PLANES if input_planes else observation_space.shape[0]  # 1 for the raw board

        self.cnn = nn.Sequential(
            # First conv block - learn basic patterns
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Second conv block - learn complex patterns
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Third conv block - strategic understanding
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(self._encode(sample)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        return board_to_planes(observations) if self.input_planes else observations

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(self._encode(observations)))
