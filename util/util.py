import random
import numpy as np
import sys
import os
import copy
import csv

import gymnasium as gym
from boardgame2.env import board_player_from_state, strfboard
from boardgame2 import EMPTY
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import torch as th

def get_model(file, env, net_width=256, learning_rate = 0.01):
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


        model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log=file + ".log", gamma=1.0, learning_rate=learning_rate)
    return model

class Opponent():
    def get_action(self, env, state):
        pass

class ModelOpponent(Opponent):
    def __init__(self, **kwargs):
        file = kwargs.get('file')
        env = kwargs.get('env')
        net_width=kwargs.get('net_width')
        self.deterministic = kwargs.get('deterministic', False)
        self.verbose = kwargs.get('verbose', False)
        self.model = get_model(file, env, net_width=net_width)
        self.vec_env = self.model.get_env()
        self.obs = self.vec_env.reset()
        self.alt_env = copy.deepcopy(self.vec_env.envs[0])
    def alt_get_action(self, env, state):
        _, player = board_player_from_state(state)
        alt_state = copy.copy(state) * player
        self.alt_env.board = alt_state
        # action, _ = self.model.predict(state, action_masks=mask_fn(self.alt_env), deterministic=self.deterministic)
        action, _, _ = get_action(self.model, alt_state, mask_fn(self.alt_env), deterministic=self.deterministic, verbose=self.verbose)
        return action
    def get_action(self, env, state):
        #action, _ = self.model.predict(state, action_masks=mask_fn(env), deterministic=self.deterministic)
        #action, _ = self.model.predict(state, action_masks=mask_fn(env), deterministic=True)
        alt_action = self.alt_get_action(env, state)
        return np.array([alt_action])

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


def add_notation(s):
    rval = "  abcdefgh\n1 "
    index = 0
    rval += s.replace("\n", "\n$ ")
    for i in range(2,9):
        rval = rval.replace("$", str(i), 1)
    return rval

"""
this is a dupe of the render in boardgame2 only using the obs instead of the env since
where it's needed here the env isn't available
"""
def render(obs):
    """See gym.Env.render()."""
    outfile = sys.stdout
    board, _ = board_player_from_state(obs)
    s = strfboard(board, render_cell_map)
    s = add_notation(s)
    outfile.write(s)
    outfile.write('\n\n')
    return outfile

def mask_fn(env: gym.Env) -> np.ndarray:
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask + [0], dtype=np.int8)

def get_action(model: MaskablePPO, obs, mask, deterministic=False, verbose=False):
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    if verbose:
        actions = np.where(mask > 0)[0]
        evals = model.policy.evaluate_actions(th.Tensor(obs), th.Tensor(actions))
        return action, evals[1], actions
    else:
        return action, None, None

def get_move_notation(env, player, action):
    letters = {1: "ABCDEFGH", -1: "abcdefgh"}
    row, col = np.unravel_index(action[0], env.board_shape)
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

def play(model, env, num_games, opponent, deterministic, verbose):
    vec_env = model.get_env()
    obs = vec_env.reset()
    black_wins = 0

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        move_db = get_move_db()

    for i in range(num_games):
        moves = []
        term = False

        per_player = {1: 2, -1: 2}
        while not term:
            board, player = board_player_from_state(obs[0])
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
            if verbose:
                mn = get_move_notation(env, player, action)
                moves.append(mn)
                print(mn)
            obs, rewards, term, info = vec_env.step(action)
            if verbose:
                b,_ = board_player_from_state(obs[0])
                per_player_diff = count_players(per_player, b)
                print(per_player_diff)

            if term:
                term_obs = info[0]['terminal_observation']
                black_score = sum(term_obs == 1)
                white_score = sum(term_obs == -1)
                if verbose:
                    print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards[0]}")
                    render(term_obs)
                    moves_str = "".join(moves)
                    print(moves_str)
                    for m, desc in move_db:
                        if m == moves_str[:len(m)]:
                            print(m, desc)
                black_wins += black_score > white_score
    return black_wins
