"""
Move scores: the expected final disc margin of every legal move, for the side to move, like the move
scores of Edax, Egaroucid or Kifubox. Used by web_play.py's score hints (/api/scores) and by
ScorePlayer, which always plays the best-scored move (match.py "hint").

Exact (endgame solver) with at most solve_empties empty squares; otherwise one MCTS search of the
position (no early stop, so every move gets visits): each move's average value converted back to
discs with 16 * atanh (the inverse of the value head's tanh(margin / 16) scale), or the value head's
estimate of the position after the move if the search never tried it.
"""

import numpy as np

from util.search import legal_moves, play_move, policy_value, value_evaluator, LEAF_SCORE_SCALE


class MoveScorer:
    def __init__(self, model, sims=400, solve_empties=18, leaf_solve_empties=16):
        from util.mcts import MCTS
        self.sims, self.solve_empties = sims, solve_empties
        self.mcts = MCTS(lambda bs: policy_value(model, bs), sims=sims, early_stop=False,
                         leaf_solve_empties=leaf_solve_empties)
        self.value = value_evaluator(model)

    def margins(self, boards):
        """For each board (side to move = 1, with a legal move): ({move: margin in discs}, exact)."""
        boards = [np.asarray(b, dtype=np.int8).reshape(64) for b in boards]
        out = [None] * len(boards)
        late = [i for i, b in enumerate(boards) if (b == 0).sum() <= self.solve_empties]
        early = [i for i in range(len(boards)) if i not in set(late)]
        if late:
            from util.endgame import solve_many
            jobs = [(i, int(m)) for i in late for m in legal_moves(boards[i])]
            solved = solve_many([play_move(boards[i], m) for i, m in jobs])     # children: opponent's view
            for i in late:
                out[i] = ({}, True)
            for (i, m), (s, _, _) in zip(jobs, solved):
                out[i][0][m] = float(-s)
        if early:
            self.mcts.run([boards[i] for i in early])
            jobs = [(i, int(m)) for i in early for m in legal_moves(boards[i])]
            child_values = self.value(np.array([play_move(boards[i], m) for i, m in jobs]))
            roots = dict(zip(early, self.mcts.last_roots))
            for i in early:
                out[i] = ({}, False)
            for (i, m), v in zip(jobs, child_values):
                root = roots[i]
                j = int(np.flatnonzero(root.moves == m)[0])
                q = root.w[j] / root.n[j] if root.n[j] > 0 else -float(v)
                out[i][0][m] = float(LEAF_SCORE_SCALE * np.arctanh(np.clip(q, -0.999, 0.999)))
        return out


class ScorePlayer:
    """Always plays the best-scored move (what a person following the score hints plays). Same
    interface as SearchPlayer for match.py (choose_many)."""

    solve_empties, depth = 0, 0
    mcts = None

    def __init__(self, model, sims=400, solve_empties=18):
        self.scorer = MoveScorer(model, sims, solve_empties)

    def choose_many(self, boards):
        return [max(m, key=m.get) for m, _ in self.scorer.margins(boards)]

    def __call__(self, board):
        return self.choose_many([board])[0]
