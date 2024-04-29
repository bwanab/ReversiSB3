import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, random_action, board_player_from_state, get_opponent, play, mask_fn, ModelOpponent, RandomOpponent, get_model

import numpy as np
import os.path
from time import time

#from stable_baselines3 import DQN
#from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat, configure
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback

def round_trip(training_env, opponent):
    env = training_env.envs[0]
    state = env.board
    _, player = board_player_from_state(state)
    while player == -1:
        action = opponent.get_action(env, state)
        obs, rewards, term, info = training_env.step(action)
        state = obs[0]
        _, player = board_player_from_state(state)
    return True

class FullRoundTripCallback(BaseCallback):
    def __init__(self, file, env, net_width, episodes = 100_000, verbose: int = 1, opponentName = "Random"):
        self.file = file
        self.env = env
        self.net_width = net_width
        self.threshold = 1
        self.r_factor = 1 / episodes
        self.opponentName = opponentName
        self.update_opponent()
        super(FullRoundTripCallback, self).__init__(verbose)

    def get_threshold(self):
        rval = self.threshold
        self.threshold -= self.r_factor
        if self.threshold < 0:
            self.threshold = 0.0
        return rval
    
    def update_opponent(self):
        self.opponent = get_opponent(self.opponentName, file=self.file, env=self.env, net_width=self.net_width)

    def get_action(self, env, state):
        return self.opponent.get_action(env, state)

    def _on_step(self) -> bool:
        return round_trip(self.env, self.opponent)


class PlayCallback(BaseCallback):
    def __init__(self, model, file, frtCB, episodes = 100, opponent = RandomOpponent, verbose: bool = False):
        self.episodes = episodes
        self.verbose = verbose
        self.opponent = opponent
        self.model = model
        self.file = file
        self.logger = Logger("./" + file + ".log", TensorBoardOutputFormat("./" + file + ".log"))
        self.frtCB = frtCB
        super(PlayCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        black_wins = play(self.model, self.episodes, self.opponent, True, self.verbose)
        self.logger.record(key="black_wins", value=black_wins)
        self.model.save(file)
        self.model.policy.save(file + "_policy.zip")
        self.frtCB.update_opponent()
        return True


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-p", "--epochs", default=1)
    parser.add_argument("-e", "--episodes", default=100000)
    parser.add_argument("-m", "--model", default = "reversi_ppo")
    parser.add_argument("-o", "--opponent", default="Model") # training opponent
    parser.add_argument("-t", "--test_opponent", default="Random") # test opponent
    parser.add_argument("-w", "--net_width", default="512")
    args = parser.parse_args()

    
    env = ActionMasker(gym.make("Reversi-v0"), mask_fn)  # Wrap to enable masking

    file = "models/" + args.model + "_5layer_03LR_" + args.net_width
    net_width = int(args.net_width)
    # opponent = get_opponent(args.opponent, file=file, env=env, net_width=net_width)
    
    model = get_model(file, env, net_width=net_width)
    new_logger = configure("models/temp/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    # Train the agent and display a progress bar
    episodes = int(args.episodes) * 60
    epochs = int(args.epochs)
    frtCB = FullRoundTripCallback(file, model.get_env(), net_width, episodes=episodes, opponentName=args.opponent)
    for i in range(epochs):
        print(f"--------- Epoch: {i + 1} -----------")
        test_opponent = get_opponent(args.test_opponent, file=file, env=env, net_width=net_width)
        playCB = PlayCallback(model, file, frtCB, 100, test_opponent, False)
        everyNCB = EveryNTimesteps(n_steps=10000, callback=playCB)
        model.learn(total_timesteps=episodes, progress_bar=True, callback=[frtCB, everyNCB])
        # Save the agent
        model.save(file)
        model.policy.save(file + "_policy.zip")
