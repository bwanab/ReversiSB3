import random
import numpy as np
import sys


from boardgame2.env import board_player_from_state, strfboard
from boardgame2 import EMPTY
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

class Opponent():
    def get_action(self, env, state):
        pass

class RAIOpponent(Opponent):
    def get_action(self, env, state):
        return reversi_ai_action(env, state)

class RandomOpponent(Opponent):
    def get_action(self, env, state):
        return random_action(env, state)

def get_opponent(s):
    if s == "Random":
        opponent = RandomOpponent()
    else:
        opponent = RAIOpponent()
    return opponent


"""
return a random action from the valid possible actions
"""
def random_action(env, state):
    return np.array([random.choice(env.all_valid_actions(state))])

rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}
render_cell_map = {-1: 'x', 0: ' ', 1: 'o'}

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
    outfile.write('\n')
    return outfile
