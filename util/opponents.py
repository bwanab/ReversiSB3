from util.util import render, get_scores, get_model, mask_fn, BLACK
import copy
import numpy as np
import random
import subprocess
import os
import re
import select
import time
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
        valid = env.all_valid_actions(state)
        while True:
            t = input().strip().lower()
            if len(t) == 2 and t[0] in "abcdefgh" and t[1] in "12345678":
                action = np.ravel_multi_index([int(t[1]) - 1, "abcdefgh".find(t[0])], (8,8))
                # an illegal move would be treated as a resignation, so re-prompt instead
                if action in valid:
                    return np.array(action)
            print("Enter a legal move, e.g. d3")

class ModelOpponent(Opponent):
    def __init__(self, **kwargs):
        super().__init__()
        file = kwargs.get('opponent_model')
        env = kwargs.get('env')
        net_width=kwargs.get('net_width')
        self.verbose = kwargs.get('verbose', False)
        self.model = get_model(file, env, net_width=net_width)
        self.alt_env = copy.deepcopy(env.unwrapped)
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
    def __init__(self, **kwargs):
        self.depth = kwargs.get("depth", 2)
        super().__init__()

    def get_action(self, env, state):
        player = self.player
        # alt_state = copy.copy(state) * player
        alt_state = copy.deepcopy(state)
        return reversi_ai_action(env, alt_state, self.depth)

class RandomOpponent(Opponent):
    def __init__(self):
        super().__init__()

    def get_action(self, env, state):
        return random_action(env, state)

class EdaxOpponent(Opponent):
    """Opponent using Edax engine via client/server architecture.

    Edax runs in a standalone server process. This opponent connects via
    Unix socket for move requests. This architecture is completely safe for
    multiprocessing (SB3 parallel environments).

    Start the server first:
        python edax_server.py

    Performance:
    - Depth 4: ~0.01s per move (~100 moves/sec)
    - Depth 6: ~0.1s per move (~10 moves/sec)
    - Depth 10: ~5-10s per move
    """

    def __init__(self, **kwargs):
        super().__init__()
        from util.edax_client import EdaxClient

        self.depth = kwargs.get("depth", 6)
        socket_path = kwargs.get("socket_path", "/tmp/edax_server.sock")

        # Create client (lightweight, multiprocessing-safe)
        self.edax = EdaxClient(socket_path=socket_path, depth=self.depth)

    def get_action(self, env, state):
        """Get Edax's move for current position.

        This is a stateless implementation - no synchronization or move
        history tracking needed. Just pass the current board state to Edax
        server and get back the best move.

        Args:
            env: Game environment (not used, state contains all info)
            state: Current board state (numpy array shape (1, 8, 8) or (3, 8, 8))
                   From environment perspective: Black=1, White=-1
                   We flip it so opponent (White) sees their pieces as 1

        Returns:
            np.array: Single-element array containing action number (0-63)
        """
        try:
            # Flip board perspective: opponent's pieces become 1
            # Since self.player = -1 for opponents, this flips White to 1
            alt_state = state * self.player

            # Get Edax's move via client (server communication)
            move = self.edax.get_move(alt_state)

            if move is None:
                # No legal moves (shouldn't happen if env is correct)
                # Fall back to random valid move
                return random_action(env, state)

            return np.array([move])

        except Exception as e:
            print(f"WARNING: Edax failed: {e}, using random")
            import traceback
            traceback.print_exc()
            return random_action(env, state)

    def __del__(self):
        """Cleanup - client is stateless, nothing to clean up."""
        pass

opponent_map = {}
def get_opponent(s, **kwargs):
    if s == "Random":
        opponent = RandomOpponent()
    elif s == "RAI":
        opponent = RAIOpponent(**kwargs)
    elif s == "Edax":
        opponent = EdaxOpponent(**kwargs)
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

def reversi_ai_action(env, state, depth = 2):
    board, player, rai_board = build_rai_board(state, env.player)

    try:
        rai = ReversiAI()
        r_ai_player = rai_cell_map[player]
        coord = rai.get_next_move(rai_board, r_ai_player, depth)

        # Check if coord is None (no valid moves)
        if coord is None:
            rval = random_action(env, state)
        else:
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
