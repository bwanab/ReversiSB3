"""
Named Othello openings (moves.txt) as board positions.

moves.txt lines look like "C4c3D3c5B2|X-square Opening (t3)  bad": a move sequence in
standard notation (column a-h, row 1-8; upper/lower case just alternates for
readability, colors alternate) and a name. Lines are replayed on ReversiEnvCNN's board.
"""

import numpy as np

from util.reversi import ReversiEnvCNN
from util.util import BLACK
from util.board_features import PERMS


def parse_moves(seq):
    """'C4c3D3' -> flat action indices (row * 8 + col)."""
    seq = seq.strip()
    return [(int(seq[i + 1]) - 1) * 8 + "abcdefgh".index(seq[i].lower()) for i in range(0, len(seq), 2)]


def load_openings(path="moves.txt"):
    """Returns a list of (moves, name) with moves as flat action indices."""
    openings = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            if "|" not in line:
                continue
            seq, name = line.rstrip("\n").split("|", 1)
            openings.append((parse_moves(seq), name.strip()))
    return openings


def replay(moves):
    """Yield (position, move) along a line: position is the (1, 8, 8) board from the
    side-to-move's perspective before `move`, then the final position with move None."""
    env = ReversiEnvCNN(opponent="Random")
    board, _ = env.reset()
    board = board.copy()
    player = BLACK
    for move in moves:
        assert env.is_valid(board, player, np.array([0, move // 8, move % 8])), f"illegal move {move} in {moves}"
        yield board * player, move
        env.player = player
        env.get_next_state(board, (0, move // 8, move % 8))
        player = -player
    yield board * player, None


def canonical(flat):
    """Symmetry-canonical form of a flat (64,) board: the smallest of its 8 rotations/reflections."""
    return min(flat[p].tobytes() for p in PERMS)


def book_positions(openings, min_plies=4):
    """Map each distinct position (after >= min_plies moves, side-to-move perspective) to the
    set of book moves played from it. Keyed by the flat board's bytes (no symmetry merging,
    since book moves are given in the position's own orientation)."""
    book = {}
    for moves, name in openings:
        for ply, (pos, move) in enumerate(replay(moves)):
            if ply < min_plies:
                continue
            key = pos.reshape(64).astype(np.int8).tobytes()
            entry = book.setdefault(key, set())
            if move is not None:
                entry.add(move)
    return book
