"""
Play many games at once in lockstep, so the model's moves for all of them go through one batched
choose_many() call per step (see util.search.SearchPlayer). Per-call GPU latency dominates single
positions, so this raises throughput roughly in proportion to the number of games.

Boards are flat (64,) int8 arrays from the side-to-move's perspective (1 = side to move).
"""

import numpy as np

from util.search import legal_moves, play_move


def sb_play_starts(n, random_plies=0, seed=None, start_positions=None):
    """The starting boards sb-play.py would use for n games (model to move): its random-opening
    sequence for `seed`, or start_positions in order (cycling). Lets results be compared exactly."""
    if start_positions is not None:
        positions = np.asarray(start_positions, dtype=np.int8).reshape(-1, 64)
        return [positions[i % len(positions)].copy() for i in range(n)]
    from util.reversi import ReversiEnvCNN
    from util.play import random_opening
    env = ReversiEnvCNN(opponent="Random")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n):
        env.reset()
        if random_plies:
            while not random_opening(env, random_plies, rng):
                env.reset()
        starts.append(env.board.reshape(64).astype(np.int8).copy())   # BLACK (the model) to move
    return starts


def play_vs_edax(choose_many, starts, edax_depth, client=None):
    """Play one game per start (model to move first) against Edax at `edax_depth`, all in lockstep.

    choose_many(list of boards) -> list of moves. Returns per-game results from the model's view:
    an array of final disc differences (model - Edax); > 0 win, 0 draw, < 0 loss.
    """
    if client is None:
        from util.edax_client import EdaxClient
        client = EdaxClient()
    boards = [np.asarray(b, dtype=np.int8).reshape(64).copy() for b in starts]
    model_to_move = [True] * len(boards)
    result = [None] * len(boards)
    while True:
        # resolve forced passes and finished games
        for i, b in enumerate(boards):
            if result[i] is not None:
                continue
            if len(legal_moves(b)) == 0:
                if len(legal_moves(-b)) == 0:
                    diff = int((b == 1).sum() - (b == -1).sum())       # side to move's view
                    result[i] = diff if model_to_move[i] else -diff
                    continue
                boards[i] = -b
                model_to_move[i] = not model_to_move[i]
        active = [i for i in range(len(boards)) if result[i] is None]
        if not active:
            return np.array(result)
        mine = [i for i in active if model_to_move[i]]
        if mine:
            for i, m in zip(mine, choose_many([boards[i] for i in mine])):
                boards[i] = play_move(boards[i], m)
                model_to_move[i] = False
        else:
            for i in active:
                move = client.analyze(boards[i].reshape(8, 8), depth=edax_depth)["move"]
                boards[i] = play_move(boards[i], move)
                model_to_move[i] = True
