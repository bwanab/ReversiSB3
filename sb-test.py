import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, random_action, board_player_from_state, get_opponent, play, mask_fn, ModelOpponent, RandomOpponent

import numpy as np
import os.path
from time import time

#from stable_baselines3 import DQN
#from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback

def round_trip(training_env, state, opponent):
    env = training_env.envs[0]
    _, player = board_player_from_state(state)
    while player == -1:
        action = opponent.get_action(env, state)
        obs, rewards, term, info = training_env.step(action)
        state = obs[0]
        _, player = board_player_from_state(state)
    return True

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
        return self.opponent.get_action(env, state)

    def _on_step(self) -> bool:
        state = env.board
        return round_trip(self.training_env, state, self.opponent)


class PlayCallback(BaseCallback):
    def __init__(self, model, env, file, episodes = 100, opponent = RandomOpponent, verbose: bool = False):
        self.episodes = episodes
        self.verbose = verbose
        self.opponent = opponent
        self.env = env
        self.model = model
        self.logger = Logger("./" + file + ".log", TensorBoardOutputFormat("./" + file + ".log"))
        super(PlayCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        black_wins = play(self.model, self.env, self.episodes, self.opponent, self.verbose)
        self.logger.record(key="black_wins", value=black_wins)
        print(f"black_wins: {black_wins}")
        return True


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-p", "--epochs", default=1)
    parser.add_argument("-e", "--episodes", default=10_000)
    parser.add_argument("-m", "--model", default = "ppo_reversi_256")
    parser.add_argument("-o", "--opponent", default="Model") # training opponent
    parser.add_argument("-t", "--test_opponent", default="Random") # test opponent
    args = parser.parse_args()

    
    env = ActionMasker(gym.make("Reversi-v0"), mask_fn)  # Wrap to enable masking

    file = args.model

    opponent = get_opponent(args.opponent, file, env)

    # Instantiate the agent
    if os.path.isfile(file + ".zip"):
        model = MaskablePPO.load(file, env=env)
        model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    else:
        # lrs = lambda x: 0.003
        # net_arch = dict(pi=[128, 512, 64], vf=[128, 512, 64])
        # policy = MaskableActorCriticPolicy(env.observation_space, env.action_space, lrs, net_arch=net_arch)
        # model = MaskablePPO(policy, env, verbose=1)
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))

        model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=file + ".log")
    # Train the agent and display a progress bar
    episodes = int(args.episodes)
    epochs = int(args.epochs)
    frtCB = FullRoundTripCallback(episodes=episodes, opponent=opponent)
    # checkpointCB = CheckpointCallback(
    #     save_freq=10000,
    #     save_path="./",
    #     name_prefix=file,
    #     save_replay_buffer=True,
    #     save_vecnormalize=True,
    #     )
    # playCB = PlayCallback(model, env, 100, opponent, False)
    for i in range(epochs):
        test_opponent = get_opponent(args.test_opponent, file, env)
        playCB = PlayCallback(model, env, file, 100, test_opponent, False)
        everyNCB = EveryNTimesteps(n_steps=10_000, callback=playCB)
        model.learn(total_timesteps=int(episodes), progress_bar=True, callback=[frtCB, everyNCB])
        # Save the agent
        model.save(file)
        model.policy.save(file + "_policy.zip")
