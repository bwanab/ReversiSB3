import random
import numpy as np
import sys
import os

import gymnasium as gym
from boardgame2.env import board_player_from_state, strfboard
from boardgame2 import EMPTY
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback

import torch as th

def get_model(file, env, net_width=256):
    if os.path.isfile(file + ".zip"):
        model = MaskablePPO.load(file, env=env)
        model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    else:
        # lrs = lambda x: 0.003
        # net_arch = dict(pi=[128, 512, 64], vf=[128, 512, 64])
        # policy = MaskableActorCriticPolicy(env.observation_space, env.action_space, lrs, net_arch=net_arch)
        # model = MaskablePPO(policy, env, verbose=1)
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[net_width, net_width], vf=[net_width, net_width]))

        model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=file + ".log")
    return model

class Opponent():
    def get_action(self, env, state):
        pass

class ModelOpponent(Opponent):
    def __init__(self, **kwargs):
        file = kwargs.get('file')
        env = kwargs.get('env')
        net_width=kwargs.get('net_width')
        self.deterministic = kwargs.get('deterministic')
        if self.deterministic == None:
            self.deterministic = False
        self.model = get_model(file, env, net_width=net_width)
        self.vec_env = self.model.get_env()
        self.obs = self.vec_env.reset()
    def get_action(self, env, state):
        action, _ = self.model.predict(state, action_masks=mask_fn(env), deterministic=self.deterministic)
        return np.array([action])

class RAIOpponent(Opponent):
    def get_action(self, env, state):
        return reversi_ai_action(env, state)

class RandomOpponent(Opponent):
    def get_action(self, env, state):
        return random_action(env, state)

def get_opponent(s, **kwargs):
    if s == "Random":
        opponent = RandomOpponent()
    elif s == "RAI":
        opponent = RAIOpponent()
    else:
        opponent = ModelOpponent(**kwargs)
    return opponent


"""
return a random action from the valid possible actions
"""
def random_action(env, state):
    return np.array([random.choice(env.all_valid_actions(state))])

rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}
render_cell_map = {-1: 'x', 0: '.', 1: 'o'}

def reversi_ai_action(env, state):
    board, player, rai_board = build_rai_board(state)

    try:
        rai = ReversiAI()
        r_ai_player = rai_cell_map[player]
        coord = rai.get_next_move(rai_board, r_ai_player)
        x = coord.x
        y = coord.y
        rval = [np.ravel_multi_index([x, y], env.board_shape)]
    except GameHasEndedError:
        rval = random_action(env, state)
    return np.array(rval)

def build_rai_board(state):
    board, player = board_player_from_state(state)
    rai_board = []
    for x in range(8):
        rai_board.append([])
        for y in range(8):
            rai_board[x].append(rai_cell_map[board[x, y]])
    return board,player,rai_board


"""
this is a dupe of the render in boardgame2 only using the obs instead of the env since
where it's needed here the env isn't available
"""
def render(obs):
    """See gym.Env.render()."""
    outfile = sys.stdout
    board, _ = board_player_from_state(obs)
    s = strfboard(board, render_cell_map)
    outfile.write(s)
    outfile.write('\n\n')
    return outfile

def mask_fn(env: gym.Env) -> np.ndarray:
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask + [0], dtype=np.int8)

def get_action(model: MaskablePPO, obs, mask, deterministic=False, verbose=False):
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    if verbose:
        actions = [x for x in range(64) if mask[x] > 0]
        evals = model.policy.evaluate_actions(th.Tensor(obs), th.Tensor(actions))
        return action, evals[1], actions
    else:
        return action, None, None


def play(model, env, num_games, opponent, deterministic, verbose):
    vec_env = model.get_env()
    obs = vec_env.reset()
    black_wins = 0

    if verbose:
        np.set_printoptions(precision=3, suppress=True)

    for i in range(num_games):
        term = False

        while not term:
            if verbose:
                render(obs[0])
            _,player = board_player_from_state(obs[0])
            if player == -1:
                action = opponent.get_action(env, obs[0])

            else:
                action, probs, actions = get_action(model, obs, mask_fn(env), deterministic=deterministic, verbose=verbose)
                if verbose:
                    m = th.nn.Softmax(dim=0)
                    print(action, actions, m(probs).detach().numpy())
            obs, rewards, term, info = vec_env.step(action)
            if term:
                term_obs = info[0]['terminal_observation']
                black_score = sum(term_obs == 1)
                white_score = sum(term_obs == -1)
                if verbose:
                    print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards[0]}")
                    render(term_obs)
                black_wins += black_score > white_score
    return black_wins
