import gymnasium as gym
from util.reversi import build_reversi
from util.util import mask_fn, get_model
from util.play import play
from util.opponents import get_opponent, RandomOpponent

import numpy as np
import os.path
from time import time

import torch
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat, configure
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback

# Setup device
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ MPS (Apple Silicon GPU) is available and will be used")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ CUDA GPU is available and will be used")
else:
    device = "cpu"
    print("✓ Using CPU")

def round_trip(training_env, opponent):
    env = training_env.envs[0]
    state = env.board
    player = env.player
    while player == -1:
        action = opponent.get_action(env, state)
        obs, rewards, term, info = training_env.step(action)
        state = obs[0]
        player = env.player
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
    def __init__(self, model, file, episodes = 100, opponent = RandomOpponent, verbose: bool = False):
        self.episodes = episodes
        self.verbose = verbose
        self.opponent = opponent
        self.model = model
        self.file = file
        self.logger = Logger("./" + file + ".log", TensorBoardOutputFormat("./" + file + ".log"))
        super(PlayCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        black_wins = play(self.model, self.episodes, self.opponent, True, self.verbose)
        self.logger.record(key="black_wins", value=black_wins)
        self.model.save(self.file)
        return True


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-p", "--epochs", default=20)
    parser.add_argument("-e", "--episodes", default=500_000)
    parser.add_argument("-m", "--model", default = "dorkJ")
    parser.add_argument("-o", "--opponent", default="Model") # training opponent
    parser.add_argument("-r", "--opp_model", default="models/dorkJ_CNN_test")
    parser.add_argument("-t", "--test_opponent", default="Random") # test opponent
    parser.add_argument("-w", "--net_width", default="512")
    args = parser.parse_args()

    file = "models/" + args.model + "_CNN_test"
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix=args.model + "_CNN_test",
    )

    env = ActionMasker(build_reversi(opponent=args.opponent, opponent_model=args.opp_model), mask_fn)  # Wrap to enable masking

    net_width = int(args.net_width)
    # opponent = get_opponent(args.opponent, file=file, env=env, net_width=net_width)
    model = get_model(file, env, net_width=net_width, model_type="cnn", device=device)
    # new_logger = configure("models/temp/", ["stdout", "csv", "tensorboard"])
    # model.set_logger(new_logger)
    # Train the agent and display a progress bar
    episodes = int(args.episodes)
    epochs = int(args.epochs)
    # frtCB = FullRoundTripCallback(file, model.get_env(), net_width, episodes=episodes, opponentName=args.opponent)
    test_opponent = get_opponent(args.test_opponent, file=file, env=env, net_width=net_width)
    playCB = PlayCallback(model, file, 100, test_opponent, False)
    everyNCB = EveryNTimesteps(n_steps=10000, callback=playCB)
    model.learn(total_timesteps=episodes, 
                progress_bar=True, 
                callback=[everyNCB, checkpoint_cb], 
                tb_log_name=args.model, 
                reset_num_timesteps=False)
    # model.learning_rate = 2e-5
    # env.set_opponent("Model", opponent_model=file, )
    # model.learn(total_timesteps=10000, 
    #             progress_bar=True, 
    #             callback=[everyNCB, checkpoint_cb], 
    #             tb_log_name=args.model, 
    #             reset_num_timesteps=False)

    # Save the agent
    model.save(file)
