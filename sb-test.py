import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, board_player_from_state

import numpy as np
import os.path

#from stable_baselines3 import DQN
#from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.callbacks import BaseCallback

class FullRoundTripCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super(FullRoundTripCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        state = env.board
        _, player = board_player_from_state(state)
        while player == -1:
            actions = env.all_valid_actions(state)
            action = np.random.choice(actions)
            # obs, rewards, terminated, _truncated, _info = env.step(action)
            obs, rewards, term, info = self.training_env.step([action])
            state = obs[0]
            _, player = board_player_from_state(state)
        return True


# Create environment
env = gym.make("Reversi-v0")

# def sample_factory(env):
#     def get_sample():
#         return np.random.choice(env.all_valid_actions(env.board))
#     return get_sample


# env.action_space.sample = sample_factory(env)

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask + [1], dtype=np.int8)

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

file = 'ppo_reversi'


# Instantiate the agent
if os.path.isfile(file + ".zip"):
    model = MaskablePPO.load(file, env=env)
    model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
else:
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
# Train the agent and display a progress bar
frtCB = FullRoundTripCallback()
model.learn(total_timesteps=int(10e5), progress_bar=True, callback=frtCB)
# Save the agent
model.save(file)
model.policy.save(file + "_policy.zip")
