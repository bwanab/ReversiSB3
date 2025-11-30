import random
import numpy as np
import sys
import os
import copy
import csv
import itertools
from typing import Callable

import gymnasium as gym
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

from sb3_contrib import MaskablePPO
from util.reversi_cnn import ReversiCNN
#from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import torch as th

EMPTY = 0
BLACK = 1
WHITE = -1

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def slow_entropy_decay(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * (0.3 + 0.7 * progress_remaining)
    return func

def set_learning_rate(model: MaskablePPO, lr: float) -> None:
    """Update the learning rate of a model's optimizer.

    CRITICAL: Updates three places to ensure LR persists across learn() calls:
    1. The optimizer's param_groups (actual LR used in training)
    2. model.learning_rate (the attribute)
    3. model.lr_schedule (the function SB3 calls during learn())

    Parameters
    ----------
    model : MaskablePPO
        The model to update
    lr : float
        The new learning rate

    Example
    -------
    >>> set_learning_rate(model, 1e-5)
    >>> model.learn(total_timesteps=100000, reset_num_timesteps=False)
    >>> # LR will remain 1e-5 throughout training
    """
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = lr
    model.learning_rate = lr
    # CRITICAL: Also update lr_schedule to prevent reset during learn()
    if hasattr(model, 'lr_schedule'):
        model.lr_schedule = lambda _: lr

def get_model(file, env, net_width=256, learning_rate = 1e-5, model_type="cnn", device="cpu"):
    if os.path.isfile(file + ".zip"):
        model = MaskablePPO.load(file, env=env)
        #### turns out, this is redundant since policy is always saved with model
        # model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
        # Update learning rate if provided and different from saved model
        if learning_rate != model.learning_rate:
            set_learning_rate(model, learning_rate)
            print(f"✓ Updated learning rate to {learning_rate}")
    elif model_type == "cnn":
        policy_kwargs = dict(
            features_extractor_class=ReversiCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False
            )
        model = MaskablePPO("CnnPolicy", 
                            env, 
                            policy_kwargs=policy_kwargs, 
                            tensorboard_log=file + ".log",
                            device=device,
                            batch_size=256,
                            n_steps=2048,
                            learning_rate=learning_rate,
                            ent_coef=0.03,
                            n_epochs=5,             # Reduced from 15 to avoid overfitting
                            gae_lambda=0.90,
                            gamma=0.98,
                            clip_range=0.1,          # Increased from 0.01 to allow policy updates
                            verbose=1
        )
    else:
        # lrs = lambda x: 0.003
        # net_arch = dict(pi=[128, 512, 64], vf=[128, 512, 64])
        # policy = MaskableActorCriticPolicy(env.observation_space, env.action_space, lrs, net_arch=net_arch)
        # model = MaskablePPO(policy, env, verbose=1)
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[net_width, net_width, net_width, net_width, net_width], vf=[net_width, net_width, net_width, net_width, net_width]))


        #model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=file + ".log", gamma=1.0, learning_rate=learning_rate, ent_coef=0.01)
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log=file + ".log", gamma=1.0, learning_rate=learning_rate, ent_coef=0.01)
    return model



def add_notation(s):
    rval = "  abcdefgh\n1 "
    index = 0
    rval += s.replace("\n", "\n$ ")
    for i in range(2,9):
        rval = rval.replace("$", str(i), 1)
    return rval

def strfboard(board: np.array, render_characters: str='+ox', end: str='\n') -> str:
    """Format a board as a string

    Parameters
    ----
    board : np.array
    render_characters : str="+ox"
        - character at position 0 represents empty;
        - character at position 1 represents BLACK;
        - character at position -1 represents WHITE.
    end : str

    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[1]):
        for y in range(board.shape[2]):
            c = render_characters[board[0, x, y]]
            s += c
        s += end
    return s[:-len(end)]


def render(obs):
    render_cell_map = {-1: 'x', 0: '.', 1: 'o'}

    """See gym.Env.render()."""
    outfile = sys.stdout
    board = obs
    s = strfboard(board, render_cell_map)
    s = add_notation(s)
    outfile.write(s)
    outfile.write('\n\n')
    return outfile

def mask_fn(env: gym.Env) -> np.ndarray:
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask, dtype=bool)

def get_action(model: MaskablePPO, obs, mask, deterministic=False, verbose=False):
    if np.all(mask == 0):
        return np.array(-8), th.Tensor([]), np.array([])
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    if verbose:
        actions = np.where(mask > 0)[0]
        evals = model.policy.evaluate_actions(th.Tensor(np.reshape(obs, (1, *obs.shape))), th.Tensor(actions))
        return action, evals[1], actions
    else:
        return action, None, None

def get_move_notation(env, player, action):
    letters = {1: "ABCDEFGH", -1: "abcdefgh"}
    _, row, col = np.unravel_index(action, env.board.shape)
    return letters[player][col] + str(row + 1)

def get_move_db():
    with open('moves.txt') as csvfile:
        movereader = csv.reader(csvfile, delimiter='|')
        moves = [row for row in movereader]
    moves.reverse()
    return moves

def count_players(per_player, board):
    old_player = dict(per_player)
    per_player[1] = len(np.where(board == 1)[0])
    per_player[-1] = len(np.where(board == -1)[0])
    return {1: per_player[1] - old_player[1], -1: per_player[-1] - old_player[-1]}


def get_pos_score(board, player, action) -> int:
    """
    Parameters
    ----
    board : np.array    
    action : np.array   location

    Returns
    ----
    score : int     the score given by number of opponents captured
    """
    
    if not is_index(board, action):
        return False

    _, x, y = np.unravel_index(action, board.shape)
    if board[0, x, y] != EMPTY:
        return 0

    for dx in [-1, 0, 1]:  # loop on the 8 directions
        for dy in [-1, 0, 1]:
            if (dx, dy) == (0, 0):
                continue
            xx, yy = x, y
            for count in itertools.count():
                xx, yy = xx + dx, yy + dy
                if xx < 0 or xx >= board.shape[1] or yy < 0 or yy >= board.shape[2]:
                    break
                if not is_index(board, (0, xx, yy)):
                    break
                if board[0, xx, yy] == EMPTY:
                    break
                if board[0, xx, yy] == -player:
                    continue
                if count:  # and is player
                    return count
                break
    return 0

def get_scores(env, state):
    """Get all valid locations for the current state.

    Parameters
    ----
    state : (np.array, int)    board and player

    Returns
    ----
    valid : np.array     current valid place for the player
    """
    board = state
    player = env.player
    max = len(board) - 1
    scores = []
    # for x in range(board.shape[1]):
    #     for y in range(board.shape[2]):
    for action in env.all_valid_actions(board):
        score = get_pos_score(board, player, action)
        # add 2 for getting a corner, 1 for getting an edge
        _, x, y = np.unravel_index(action, board.shape)
        if score > 0:
            if ((x == 0) or (x == max)):
                if ((y == 0) or (y == max)):
                    score += 2
                else:
                    score += 1
            elif ((y == 0) or (y == max)):
                if ((x == 0) or (x == max)):
                    score += 2
                else:
                    score += 1

            scores.append((score, x, y))
    return sorted(scores, reverse=True)

def is_index(board: np.array, location) -> str:
    """Check whether a location is a valid index of the board

    Parameters:
    ----
    board : np.array 2D
    location : int 

    Returns
    ----
    is_index : bool
    """
    if isinstance(location, int) or isinstance(location, np.integer):
        if location < 0 or location >= board.size:
            return False
        _, x, y = np.unravel_index(location, board.shape)
    else:
        _, x, y = location

    return x in range(board.shape[1]) and y in range(board.shape[2])
