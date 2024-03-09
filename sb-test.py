import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, random_action, board_player_from_state

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
class Opponent():
    def get_action(self, env, state):
        pass

class RAIOpponent(Opponent):
    def get_action(self, env, state):
        return reversi_ai_action(env, state)

class RandomOpponent(Opponent):
    def get_action(self, env, state):
        return random_action(env, state)

class FullRoundTripCallback(BaseCallback):
    def __init__(self, episodes = 100_000, verbose: int = 1, opponent = RandomOpponent):
        self.threshold = 1
        self.r_factor = 1 / episodes
        self.opponent = opponent
        super(FullRoundTripCallback, self).__init__(verbose)

    def get_threshold(self):
        rval = self.threshold
        self.threshold -= self.r_factor
        if self.threshold < 0:
            self.threshold = 0.0
        return rval
    
    def get_action(self, env, state):
        # r_factor = self.get_threshold()
        # r_val = np.random.random(1)[0]
        # if r_val < r_factor:
        #     return get_random_action(state, shape)
        # else:
        #     return get_reversi_ai_action(state, shape)
        #return get_reversi_ai_action(state, shape)
        return self.opponent.get_action(env, state)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        state = env.board
        _, player = board_player_from_state(state)
        while player == -1:
            action = self.get_action(env, state)
            # obs, rewards, terminated, _truncated, _info = env.step(action)
            obs, rewards, term, info = self.training_env.step(action)
            state = obs[0]
            _, player = board_player_from_state(state)
        return True

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-e", "--episodes", default=10_000)
    parser.add_argument("-m", "--model", default = "ppo_reversi_test")
    parser.add_argument("-o", "--opponent", default="Random")
    args = parser.parse_args()

    if args.opponent == "Random":
        opponent = RandomOpponent()
    else:
        opponent = RAIOpponent()

    
    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        mask = env.get_valid(env.board).reshape(64).tolist()
        return np.array(mask + [1], dtype=np.int8)

    env = ActionMasker(gym.make("Reversi-v0"), mask_fn)  # Wrap to enable masking

    file = args.model

    # Instantiate the agent
    if os.path.isfile(file + ".zip"):
        model = MaskablePPO.load(file, env=env)
        model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    else:
        # lrs = lambda x: 0.003
        # net_arch = dict(pi=[128, 512, 64], vf=[128, 512, 64])
        # policy = MaskableActorCriticPolicy(env.observation_space, env.action_space, lrs, net_arch=net_arch)
        # model = MaskablePPO(policy, env, verbose=1)
        model = MaskablePPO(MaskableActorCriticPolicy, env, tensorboard_log=file + ".log")
    # Train the agent and display a progress bar
    episodes = int(args.episodes)
    frtCB = FullRoundTripCallback(episodes=episodes, opponent=opponent)
    model.learn(total_timesteps=int(episodes), progress_bar=True, callback=frtCB)
    # Save the agent
    model.save(file)
    model.policy.save(file + "_policy.zip")
