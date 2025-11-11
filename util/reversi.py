import itertools
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from util.util import is_index, EMPTY, BLACK, WHITE
from util.opponents import RandomOpponent, get_opponent

class ReversiEnvCNN(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    reward_range = (-1, 1)

    PASS = np.array([0, -1, 0])
    RESIGN = np.array([0, -1, -1])

    def __init__(self, board_shape=8, illegal_action_mode: str='resign',
            render_characters: str='+ox', allow_pass: bool=True, 
            render_mode='human'):
        """Create a board game.

        Parameters
        ----
        board_shape: int or tuple    shape of the board
            - int: the same as (int, int)
            - tuple: in the form of (int, int), the two dimension of the board
        illegal_action_mode: str  What to do when the agent makes an illegal place.
            - 'resign': invalid location equivalent to resign
            - 'pass': invalid location equivalent to pass
        render_characters: str with length 3. characters used to render ('012', ' ox', etc)
        allow_pass: bool=True
            - True:  allow pass
            - False: not allow pass
        """
        self.allow_pass = allow_pass

        if illegal_action_mode == 'resign':
            self.illegal_equivalent_action = self.RESIGN
        elif illegal_action_mode == 'pass':
            self.illegal_equivalent_action = self.PASS
        else:
            raise ValueError()

        self.render_characters = {player : render_characters[player] for player \
                in [EMPTY, BLACK, WHITE]}

        self.board = np.zeros((1, board_shape, board_shape)).astype(np.int8)
        assert self.board.size > 1  # Invalid board shape

        self.observation_space = spaces.Box(low=-1, high=1, shape=self.board.shape, dtype=np.int8)
        self.action_space = spaces.Discrete(board_shape * board_shape)    # -8 results in self.PASS
        self.player = BLACK
        self.actual_player = BLACK
        self.opponent = get_opponent("Random")

    # static 
    def build_reversi():
        env = gym.make("ReversiCNN-v0")
        return env
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        _, x, y = (s // 2 for s in self.board.shape)
        board = self.board
        np.copyto(board, 0)
        board[0, x - 1, y - 1] = board[0, x, y] = -1
        board[0, x - 1, y] = board[0, x, y - 1] = 1
        self.player = BLACK
        self.actual_player = BLACK
        return board, {}

    def get_sample(self, state):
        a = self.all_valid_actions(state)
        if len(a) == 0:
            return ReversiEnvCNN.PASS
        else:
            return a[np.random.randint(len(a))]

    def get_valid(self, state):
        return self._get_valid(state, self.player)
    
    def _get_valid(self, state, player):
        """Get all valid locations for the current state.

        Parameters
        ----
        state : (np.array, int)    board and player

        Returns
        ----
        valid : np.array     current valid place for the player
        """
        board = state
        valid = np.zeros_like(board, dtype=np.int8)
        for x in range(board.shape[1]):
            for y in range(board.shape[2]):
                valid[0, x, y] = self.is_valid(state, player, np.array([0, x, y]))
        return valid.flatten()

    def all_valid_actions(self, state):
        return self._all_valid_actions(state, self.player)
    
    def _all_valid_actions(self, state, player):
        a = self._get_valid(state, player)
        return np.where(a == 1)[0]

    def has_valid(self, state, player) -> bool:
        """Check whether there are valid locations for current state.

        Parameters
        ----
        state : (np.array, int)    board and player

        Returns
        ----
        has_valid : bool
        """
        board = state
        for x in range(board.shape[1]):
            for y in range(board.shape[2]):
                if self.is_valid(state, player, np.array([0, x, y])):
                    return True
        return False

    def is_valid(self, state, player, action) -> bool:
        """
        Parameters
        ----
        state : (np.array, int)    board and player
        action : np.array   location

        Returns
        ----
        valid : bool     whether the current action is a valid action
        """
        
        if np.all(action == self.PASS):
            return True
        
        board = state

        if not is_index(board, action):
            return False

        if isinstance(action, int) or isinstance(action, np.integer):
            x, y = np.unravel_index(action, self.board.shape)
        else:
            _, x, y = action

        if board[0, x, y] != EMPTY:
            return False

        for dx in [-1, 0, 1]:  # loop on the 8 directions
            for dy in [-1, 0, 1]:
                if (dx, dy) == (0, 0):
                    continue
                xx, yy = x, y
                for count in itertools.count():
                    xx, yy = xx + dx, yy + dy
                    if xx < 0 or xx >= self.board.shape[1] or yy < 0 or yy >= self.board.shape[2]:
                        break
                    if not is_index(board, (0, xx, yy)):
                        break
                    if board[0, xx, yy] == EMPTY:
                        break
                    if board[0, xx, yy] == -player:
                        continue
                    if count:  # and is player
                        return True
                    break
        return False

    def step(self, action):
        """See gym.Env.step().

        Parameters
        ----
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float        the winner or zero
        termination : bool    whether the game end or not
        truncation : bool=False
        info : dict={}
        """
        m_act = (0, action // 8, action % 8)
        next_state, reward, termination, info = self.next_step(self.board, m_act)
        if termination:
            # if terminated on BLACK's move return now
            return next_state, reward, termination, False, info
        ## at this point the player has been set to WHITE (-1)
        if self.player == 1:
            print("something's wrong")
        while len(self._all_valid_actions(self.board, WHITE)) > 0:
            f_act = self.opponent.get_action(self, self.board)
            f_act = (0, f_act // 8, f_act % 8)
            next_state, reward, termination, info = self.next_step(self.board, f_act)
            if termination:
                return next_state, reward, termination, False, info
            if len(self._all_valid_actions(next_state, BLACK)) > 0:
                break
            self.player = -1
        self.player = 1
        return next_state, reward, termination, False, info

    def next_step(self, state, action):
        """Get the next observation, reward, termination, and info.

        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float               the winner or zeros
        termination : bool           whether the game end or not
        info : {'valid' : np.array}    a dict shows the valid place for the next player
        """
        if not self.is_valid(state, self.player, action):
            action = self.illegal_equivalent_action
        if np.array_equal(action, self.RESIGN):
            return state, -self.player, True, {}
        
        state = self.get_next_state(state, action)
        winner = self.get_winner(state)
        if winner is not None:
            terminal_state = copy.deepcopy(state)
            state, info = self.reset()
            info['terminal_observation'] = terminal_state
            return state, winner, True, info

        return state, 0., False, {}

    def get_next_state(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """

        board = state
        player = self.player

        if np.all(action == self.PASS):
            #self.player = -self.player
            return board

        if self.is_valid(state, player, action):
            _, x, y = action
            board[0, x, y] = player
            for dx in [-1, 0, 1]:  # loop on the 8 directions
                for dy in [-1, 0, 1]:
                    if (dx, dy) == (0, 0):
                        continue
                    xx, yy = x, y
                    for count in itertools.count():
                        xx, yy = xx + dx, yy + dy
                        if xx < 0 or xx >= self.board.shape[1] or yy < 0 or yy >= self.board.shape[2]:
                            break
                        if not is_index(board, (0, xx, yy)):
                            break
                        if board[0, xx, yy] == EMPTY:
                            break
                        if board[0, xx, yy] == player:
                            for i in range(count+1):  # overwrite
                                board[0, x + i * dx, y + i * dy] = player
                            break
        self.player = -player
        return board

    def get_winner(self, state):
        """Check whether the game has ended. If so, who is the winner.

        Parameters
        ----
        state : (np.array, int)   board and player. only board info is used

        Returns
        ----
        winner : None or int
            - None       The game is not ended and the winner is not determined.
            - env.BLACK  The game is ended with the winner BLACK.
            - env.WHITE  The game is ended with the winner WHITE.
            - env.EMPTY  The game is ended tie.
        """
        board  = state
        for player in [BLACK, WHITE]:
            if self.has_valid(board, player):
                return None
        if np.sum(board == 1) == np.sum(board == -1):
            return EMPTY
        return np.sign(np.nansum(board))


register(
        id='ReversiCNN-v0',
        entry_point='util.reversi:ReversiEnvCNN',
        )
