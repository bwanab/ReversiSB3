"""
Python wrapper for the exact endgame solver (solver/endgame.c; build with solver/build.sh).

Boards are flat (64,) int8 arrays from the side-to-move's view (1 = side to move), as in
util/search.py. Scores are final disc differences for the side to move with perfect play, empty
squares going to the winner (official rule; Edax's convention).
"""

import ctypes
import os
import sys

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solver")
_LIB_PATH = os.path.join(_DIR, "libendgame.dylib" if sys.platform == "darwin" else "libendgame.so")
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise RuntimeError(f"{_LIB_PATH} not found; build it with solver/build.sh")
        lib = ctypes.CDLL(_LIB_PATH)
        u64 = ctypes.c_uint64
        lib.eg_moves.argtypes, lib.eg_moves.restype = [u64, u64], u64
        lib.eg_flips.argtypes, lib.eg_flips.restype = [u64, u64, ctypes.c_int], u64
        lib.eg_solve.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong)]
        lib.eg_solve.restype = ctypes.c_int
        _lib = lib
    return _lib


_WEIGHTS = np.array([1 << i for i in range(64)], dtype=np.uint64)


def to_bits(board):
    """(P, O) bitboards (Python ints) for a (64,) side-to-move board."""
    board = np.asarray(board).reshape(64)
    return int(_WEIGHTS[board == 1].sum()), int(_WEIGHTS[board == -1].sum())


def _squares(mask):
    return [i for i in range(64) if mask >> i & 1]


def moves(board):
    """Legal moves (sorted square indices) for the side to move."""
    return _squares(_load().eg_moves(*to_bits(board)))


def flips(board, square):
    """Squares flipped when the side to move plays `square`."""
    P, O = to_bits(board)
    return _squares(_load().eg_flips(P, O, square))


def solve(board, alpha=-64, beta=64):
    """(score, best_move, nodes): exact final disc difference for the side to move with perfect play,
    its best move (-1 if it must pass or the game is over), and the number of positions searched.
    With a window narrower than (-64, 64) the score is exact inside it and a bound outside."""
    P, O = to_bits(board)
    best, nodes = ctypes.c_int(), ctypes.c_longlong()
    score = _load().eg_solve(P, O, alpha, beta, ctypes.byref(best), ctypes.byref(nodes))
    return score, best.value, nodes.value


_pool = None


def solve_async(boards):
    """Start solving several boards on a shared pool of threads (one per CPU; the C solver releases the
    GIL, and each thread keeps its transposition table between calls). Returns a function that waits
    and returns the list of (score, best_move, nodes), so other work (e.g. a network call) can run
    in the meantime."""
    global _pool
    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="solver")
    futures = [_pool.submit(solve, b) for b in boards]
    return lambda: [f.result() for f in futures]


def solve_many(boards):
    """solve() for several boards in parallel threads."""
    return solve_async(boards)()


def empties(board):
    return int((np.asarray(board).reshape(64) == 0).sum())
