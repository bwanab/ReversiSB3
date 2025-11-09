import random
import numpy as np
import sys
import os
import copy
import csv
import itertools

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

def get_model(file, env, net_width=256, learning_rate = 0.0003, model_type="cnn", device="cpu"):
    if os.path.isfile(file + ".zip"):
        model = MaskablePPO.load(file, env=env)
        #### turns out, this is redundant since policy is always saved with model
        # model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
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
                            batch_size=128 if device == "cpu" else 256,
                            n_steps = 2048
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

# class Opponent():
#     def __init__(self):
#         self.player = -1
        
#     def get_action(self, env, state):
#         pass

# class Human(Opponent):
#     def __init__(self):
#         super().__init__()

#     def get_action(self, env, state):
#         render(state)
#         for (score, y, x) in get_scores(env, state):
#             print(f"{'abcdefgh'[x]}{y+1}: {score}")
#         while True:
#             t = input()
#             x = "abcdefgh".find(t[0])
#             if x >= 0:
#                 y = int(t[1]) - 1
#                 if 0 <= y < 8:
#                     return np.array([np.ravel_multi_index([y,x], (8,8))])

# class ModelOpponent(Opponent):
#     def __init__(self, **kwargs):
#         super().__init__()
#         file = kwargs.get('file')
#         env = kwargs.get('env')
#         net_width=kwargs.get('net_width')
#         self.deterministic = kwargs.get('deterministic', False)
#         self.verbose = kwargs.get('verbose', False)
#         self.model = get_model(file, env, net_width=net_width)
#         self.vec_env = self.model.get_env()
#         self.obs = self.vec_env.reset()
#         self.alt_env = copy.deepcopy(self.vec_env.envs[0])
#     def alt_get_action(self, env, state):
#         player = self.player
#         alt_state = copy.copy(state) * player
#         self.alt_env.board = alt_state
#         # action, _ = self.model.predict(state, action_masks=mask_fn(self.alt_env), deterministic=self.deterministic)
#         action, _, _ = get_action(self.model, alt_state, mask_fn(self.alt_env), deterministic=self.deterministic, verbose=self.verbose)
#         return action
#     def get_action(self, env, state):
#         #action, _ = self.model.predict(state, action_masks=mask_fn(env), deterministic=self.deterministic)
#         #action, _ = self.model.predict(state, action_masks=mask_fn(env), deterministic=True)
#         alt_action = self.alt_get_action(env, state)
#         return np.array([alt_action])

# class RAIOpponent(Opponent):
#     def __init__(self):
#         super().__init__()

#     def get_action(self, env, state):
#         player = self.player
#         alt_state = copy.copy(state) * player
#         return reversi_ai_action(env, alt_state)

# class RandomOpponent(Opponent):
#     def __init__(self):
#         super().__init__()

#     def get_action(self, env, state):
#         return random_action(env, state)

# def get_opponent(s, **kwargs):
#     if s == "Random":
#         opponent = RandomOpponent()
#     elif s == "RAI":
#         opponent = RAIOpponent()
#     elif s == "Human":
#         opponent = Human()
#     else:
#         opponent = ModelOpponent(**kwargs)
#     return opponent


# """
# return a random action from the valid possible actions
# """
# def random_action(env, state):
#     actions = env.all_valid_actions(state)
#     if len(actions) == 0:
#         return np.array([])
#     else:
#         return np.array([random.choice(actions)])

# rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}
# render_cell_map = {-1: 'x', 0: '.', 1: 'o'}

# def reversi_ai_action(env, state):
#     board, player, rai_board = build_rai_board(state, env.player)

#     try:
#         rai = ReversiAI()
#         r_ai_player = rai_cell_map[player]
#         coord = rai.get_next_move(rai_board, r_ai_player)
#         x = coord.x
#         y = coord.y
#         rval = [np.ravel_multi_index([x, y], env.board_shape)]
#     except GameHasEndedError:
#         rval = random_action(env, state)
#     return np.array(rval)

# def build_rai_board(state, player):
#     board = state
#     rai_board = []
#     for x in range(board.shape[1]):
#         rai_board.append([])
#         for y in range(board.shape[2]):
#             rai_board[x].append(rai_cell_map[board[0, x, y]])
#     return board,player,rai_board


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
    return np.array(mask, dtype=np.int8)

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

# def play(model, num_games, opponent, deterministic, verbose):
#     vec_env = model.get_env()
#     env = vec_env.envs[0].unwrapped
#     # obs = vec_env.reset()
#     obs, _ = env.reset()
#     black_wins = 0

#     if verbose:
#         np.set_printoptions(precision=3, suppress=True)
#         move_db = get_move_db()

#     for i in range(num_games):
#         moves = []
#         term = False

#         per_player = {1: 2, -1: 2}
#         while not term:
#             board = obs
#             player = env.player
#             if verbose:
#                 render(board)
#             player = env.player
#             if player == -1:
#                 action = opponent.get_action(env, board)

#             else:
#                 action, probs, actions = get_action(model, board, mask_fn(env), deterministic=deterministic, verbose=verbose)
#                 if verbose:
#                     m = th.nn.Softmax(dim=0)
#                     print(action, actions, m(probs).detach().numpy())
#             if verbose:
#                 mn = get_move_notation(env, player, action)
#                 moves.append(mn)
#                 print(mn)
#             obs, rewards, term, _truncated, info = env.step(action)
#             if verbose:
#                 b = obs
#                 per_player_diff = count_players(per_player, b)
#                 print(per_player_diff)

#             if term:
#                 term_obs = info['terminal_observation']
#                 black_score = np.sum(term_obs == 1)
#                 white_score = np.sum(term_obs == -1)
#                 if verbose:
#                     print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards[0]}")
#                     render(term_obs)
#                     moves_str = "".join(moves)
#                     print(moves_str)
#                     for m, desc in move_db:
#                         if m == moves_str[:len(m)]:
#                             print(m, desc)
#                 black_wins += black_score > white_score
#     return black_wins

# def alt_play(env, num_games, black_player: Opponent, white_player: Opponent):
#     black_wins = 0
#     for i in range(num_games):
#         term = False
#         obs = env.reset()[0]
#         while not term:
#             player = env.player
#             if player == 1:
#                 action = black_player.get_action(env, obs)
#             else:
#                 action = white_player.get_action(env, obs)
            
#             obs, score, term, something, info = env.step(action[0])
#             if (score != 0) and (term == False):
#                 print("score != and term false!") 
#             if term:
#                 # term_obs = info[0]['terminal_observation']
#                 term_obs = obs
#                 black_score = sum(term_obs == 1)
#                 white_score = sum(term_obs == -1)
#                 black_wins += black_score > white_score
#     return black_wins

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

    x, y = np.unravel_index(action, board.shape)
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
    for x in range(board.shape[1]):
        for y in range(board.shape[2]):
            score = get_pos_score(board, player, np.ravel_multi_index((0, x, y), board.shape))
            # add 2 for getting a corner, 1 for getting an edge
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

# just for test

# import gymnasium as gym
# import boardgame2

# env = gym.make("Reversi-v0")
# wo = RAIOpponent()
# bo = RandomOpponent()
# print("black wins: ", alt_play(env, 10, bo, wo))