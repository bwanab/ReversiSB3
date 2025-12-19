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
    """Opponent using Edax engine via subprocess.

    Edax is a fast C-based Othello/Reversi engine that can search much deeper
    than Python-based engines. Typical performance:
    - Depth 4: ~0.01s per move
    - Depth 6: ~0.1s per move
    - Depth 10: ~5-10s per move
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.depth = kwargs.get("depth", 6)

        # Find edax executable
        # Try relative path from project root first
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        edax_path = kwargs.get("edax_path",
                               os.path.join(project_root, "../edax-reversi/bin/mEdax-native"))

        if not os.path.exists(edax_path):
            raise FileNotFoundError(f"Edax executable not found at: {edax_path}")

        self.edax_path = edax_path
        self.edax_dir = os.path.dirname(edax_path)
        self.process = None
        self._start_edax()

    def _start_edax(self):
        """Start Edax subprocess."""
        self.process = subprocess.Popen(
            [self.edax_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            cwd=self.edax_dir  # Run from bin directory so it finds data/
        )

        # Give Edax time to start and load book
        time.sleep(0.5)

        # Send init command to ensure clean state
        self._send_command("init")
        self._read_until_prompt(timeout=10.0)

        # Set level (search depth)
        self._send_command(f"level {self.depth}")
        self._read_until_prompt()

    def _send_command(self, cmd):
        """Send command to Edax."""
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _read_until_prompt(self, timeout=5.0):
        """Read output until we see the '>' prompt.

        Args:
            timeout: Maximum time to wait for prompt (seconds)

        Returns:
            str: All output received
        """
        output = []
        start_time = time.time()

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Edax did not respond within {timeout} seconds")

            # Check if data is available (non-blocking)
            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)

            if ready:
                line = self.process.stdout.readline()
                if not line:
                    break
                output.append(line)
                if line.strip() == '>':
                    break

        return ''.join(output)

    def _board_to_edax_string(self, state, player):
        """Convert board state to Edax 64-character string format.

        Args:
            state: Board array of shape (3, 8, 8) where channel 0 contains:
                   1 = black, -1 = white, 0 = empty
            player: Current player (1 = black, -1 = white)

        Returns:
            tuple: (board_string, player_char)
                board_string: 64-char string of board state
                player_char: 'X' for black, 'O' for white
        """
        board = state[0]  # Extract the piece channel
        board_str = ""

        # Edax board order: A1, B1, C1, ..., H1, A2, B2, ..., H8
        # Our board: [row][col] where row 0 = rank 1
        for row in range(8):
            for col in range(8):
                cell = board[row, col]
                if cell == 1:
                    board_str += 'X'  # Black
                elif cell == -1:
                    board_str += 'O'  # White
                else:
                    board_str += '-'  # Empty

        player_char = 'X' if player == 1 else 'O'
        return board_str, player_char

    def _edax_to_action(self, move_str):
        """Convert Edax move notation (e.g., 'D6') to action number (0-63).

        Args:
            move_str: Move in format like 'D6', 'E3', etc.

        Returns:
            int: Action number (0-63)
        """
        if not move_str or len(move_str) < 2:
            return None

        col = ord(move_str[0].upper()) - ord('A')  # A=0, B=1, ..., H=7
        row = int(move_str[1]) - 1  # 1-indexed to 0-indexed

        if 0 <= col < 8 and 0 <= row < 8:
            return row * 8 + col
        return None

    def get_action(self, env, state):
        """Get Edax's move for the current position.

        Args:
            env: The game environment
            state: Current board state

        Returns:
            np.array: Single-element array containing action number
        """
        try:
            # Convert board to Edax format
            board_str, player_char = self._board_to_edax_string(state, env.player)

            # Set position
            self._send_command(f"setboard {board_str} {player_char}")
            self._read_until_prompt()

            # Set mode to computer vs computer (mode 1)
            self._send_command("mode 1")
            self._read_until_prompt()

            # Make Edax play
            self._send_command("go")
            output = self._read_until_prompt()

            # Undo the move so board state doesn't persist in Edax
            self._send_command("undo")
            self._read_until_prompt()

            # Parse move from output
            move = self._parse_edax_move(output)

            if move is None:
                print(f"WARNING: Failed to parse Edax move, using random")
                rand_action = random_action(env, state)
                return np.array([rand_action]) if np.ndim(rand_action) == 0 else rand_action

            action = self._edax_to_action(move)

            if action is None:
                print(f"WARNING: Failed to convert move {move} to action, using random")
                rand_action = random_action(env, state)
                return np.array([rand_action]) if np.ndim(rand_action) == 0 else rand_action

            return np.array([action])

        except Exception as e:
            print(f"WARNING: Edax execution failed: {e}, using random")
            rand_action = random_action(env, state)
            return np.array([rand_action]) if np.ndim(rand_action) == 0 else rand_action

    def _parse_hint_output(self, output):
        """Parse Edax's hint output to extract best move.

        Hint output looks like:
         depth|score|       time   |  nodes (N)  |   N/s    | principal variation
        ------+-----+--------------+-------------+----------+----------------------
            6   +00        0:00.001          3280    3280000 D6 c4 G5 c6

        The first move in "principal variation" is the best move.
        """
        lines = output.split('\n')
        for line in lines:
            # Look for line with move notation (letters/numbers, not dashes or plus signs)
            # Skip the header and separator lines
            if '------' in line or 'depth|score' in line:
                continue

            # Look for lines with actual search results
            # Format: "    6   +00        0:00.001          3280    3280000 D6 c4 G5 c6"
            parts = line.split()
            if len(parts) >= 6:
                # The principal variation starts after the N/s column
                # Find the first move notation (letter+number)
                for part in parts:
                    if len(part) >= 2 and part[0].upper() in 'ABCDEFGH' and part[1].isdigit():
                        return part.upper()

        return None

    def _parse_edax_move(self, output):
        """Parse Edax's move from output text.

        Looks for patterns like:
        - "Edax plays D6"
        - "plays D6"
        - Move notation followed by number (e.g., "D6 1")
        """
        # Try to find "plays XX" or "Edax plays XX" pattern
        match = re.search(r'(?:plays|play)\s+([A-H][1-8])', output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Look for board display with numbered move (most recent move marked with 1)
        # Pattern: "4 - - - O * 1 - -" on the left side or "|  |  |()|##| 1|  |  |  |" on right
        lines = output.split('\n')
        for line in lines:
            # Check right side board (between pipes)
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 10:  # Has both left and right displays
                    # Look at right board display (parts[1] through parts[8])
                    for col_idx, cell in enumerate(parts[1:9]):
                        cell = cell.strip()
                        # Check if this cell contains a number (move marker)
                        if cell.isdigit() and cell != '0':
                            # Extract row number from start of line
                            row_match = re.search(r'^\s*(\d+)', line)
                            if row_match:
                                row = row_match.group(1)
                                col = chr(ord('A') + col_idx)
                                return f"{col}{row}"

        return None

    def __del__(self):
        """Clean up Edax subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self._send_command("quit")
                self.process.wait(timeout=1)
            except:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except:
                    self.process.kill()

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
