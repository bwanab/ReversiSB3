from util.util import render, get_scores, get_model, mask_fn, BLACK
import copy
import numpy as np
import random
from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

class Opponent():
    def __init__(self):
        self.player = -1
        
    def get_action(self, env, state):
        pass

class Human(Opponent):
    def __init__(self):
        super().__init__()

    def get_action(self, env, state):
        render(state)
        for (score, y, x) in get_scores(env, state):
            print(f"{'abcdefgh'[x]}{y+1}: {score}")
        while True:
            t = input()
            x = "abcdefgh".find(t[0])
            if x >= 0:
                y = int(t[1]) - 1
                if 0 <= y < 8:
                    return np.array([np.ravel_multi_index([y,x], (8,8))])

class ModelOpponent(Opponent):
    def __init__(self, **kwargs):
        super().__init__()
        file = kwargs.get('opponent_model')
        env = kwargs.get('env')
        net_width=kwargs.get('net_width')
        self.verbose = kwargs.get('verbose', False)
        self.model = get_model(file, env, net_width=net_width)
        self.alt_env = copy.deepcopy(env)
        self.alt_env.player = BLACK

    def get_action(self, env, state):
        # the idea here is that the model is trained to behave like BLACK, but here
        # it is actually playing WHITE.
        #
        # we simulate this by reversing the board (state) values,
        # Note that the player of alt_env is set in the constructor to be BLACK.
        # Thus, when the model.predict is invoked, the model believes the state of the
        # board and player are BLACK and gives its view of the best play BLACK could make
        # which when translated back should be the best play WHITE would make given the
        # actual game state.
        alt_state = state * self.player
        self.alt_env.board = alt_state
        action, _ = self.model.predict(alt_state, action_masks=mask_fn(self.alt_env), deterministic=False)
        return np.array(action)

class RAIOpponent(Opponent):
    def __init__(self):
        super().__init__()

    def get_action(self, env, state):
        player = self.player
        # alt_state = copy.copy(state) * player
        alt_state = copy.deepcopy(state)
        return reversi_ai_action(env, alt_state)

class RandomOpponent(Opponent):
    def __init__(self):
        super().__init__()

    def get_action(self, env, state):
        return random_action(env, state)

opponent_map = {}
def get_opponent(s, **kwargs):
    if s == "Random":
        opponent = RandomOpponent()
    elif s == "RAI":
        opponent = RAIOpponent()
    elif s == "Human":
        opponent = Human()
    else:
        file = kwargs.get("opponent_model")
        opponent = opponent_map.get(file)
        if opponent is None:
            opponent = ModelOpponent(**kwargs)
            opponent_map[file] = opponent
    return opponent


"""
return a random action from the valid possible actions
"""
def random_action(env, state):
    actions = env.all_valid_actions(state)
    if len(actions) == 0:
        return np.array()
    else:
        return np.array(random.choice(actions))

rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}

def reversi_ai_action(env, state):
    board, player, rai_board = build_rai_board(state, env.player)

    try:
        rai = ReversiAI()
        r_ai_player = rai_cell_map[player]
        coord = rai.get_next_move(rai_board, r_ai_player)
        x = coord.x
        y = coord.y
        rval = np.ravel_multi_index([0, x, y], env.board.shape)
    except GameHasEndedError:
        rval = random_action(env, state)
    return np.array(rval)

def build_rai_board(state, player):
    board = state
    rai_board = []
    for x in range(board.shape[1]):
        rai_board.append([])
        for y in range(board.shape[2]):
            rai_board[x].append(rai_cell_map[board[0, x, y]])
    return board,player,rai_board
