import random
import numpy as np

from boardgame2.env import board_player_from_state
from boardgame2 import EMPTY
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

"""
return a random action from the valid possible actions
"""
def random_action(env, state):
    return random.choice(env.all_valid_actions(state))

rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}

def reversi_ai_action(env, state):
    board, player, rai_board = build_rai_board(state)

    try:
        rai = ReversiAI()
        r_ai_player = rai_cell_map[player]
        coord = rai.get_next_move(rai_board, r_ai_player)
        x = coord.x
        y = coord.y
        rval = np.ravel_multi_index([x, y], env.board_shape)
    except GameHasEndedError:
        rval = random_action(env, state)
    return rval

def build_rai_board(state):
    board, player = board_player_from_state(state)
    rai_board = []
    for x in range(8):
        rai_board.append([])
        for y in range(8):
            rai_board[x].append(rai_cell_map[board[x, y]])
    return board,player,rai_board
