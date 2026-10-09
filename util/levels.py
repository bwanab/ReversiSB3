"""
Playing strength levels 1-10, like the levels of other Othello programs (web_play.py --level,
eval_batch.py --level). Level 10 is the strongest player (web_play.py --strong); below it the MCTS
budget roughly halves per level and the exact endgame solver starts later; levels 1-3 use the policy
network alone. STRENGTH: the Edax search depth each level plays about even with, measured with
r256x12_mid1 on the 200 balanced openings (CLAUDE.md, step 31).
"""

import numpy as np

from util.search import SearchPlayer, legal_moves, policy_logits

LEVELS = {
    1: dict(temperature=1.5),
    2: dict(temperature=0.7),
    3: dict(),
    4: dict(mcts_sims=16),
    5: dict(mcts_sims=32, solve_empties=10),
    6: dict(mcts_sims=64, solve_empties=12),
    7: dict(mcts_sims=100, solve_empties=14),
    8: dict(mcts_sims=200, solve_empties=16, leaf_solve_empties=14),
    9: dict(mcts_sims=400, solve_empties=18, leaf_solve_empties=16),
    10: dict(mcts_sims=800, solve_empties=18, leaf_solve_empties=16),
}
STRONGEST = max(LEVELS)

STRENGTH = {
    1: "well below Edax depth 1 (wins 14.5% vs Edax-1)",
    2: "about Edax depth 1",
    3: "about Edax depth 2",
    4: "about Edax depth 4",
    5: "about Edax depth 5",
    6: "about Edax depth 7",
    7: "about Edax depth 8-9",
    8: "about Edax depth 10-11",
    9: "about Edax depth 13",
    10: "about Edax depth 15 (wins 55.5% vs Edax-14, even with Egaroucid level 12)",
}


def describe(level):
    cfg = LEVELS[level]
    if "temperature" in cfg:
        return f"policy network, sampled (temperature {cfg['temperature']})"
    if "mcts_sims" not in cfg:
        return "policy network, top move"
    parts = [f"MCTS {cfg['mcts_sims']} simulations"]
    if cfg.get("leaf_solve_empties"):
        parts.append(f"leaf solves <= {cfg['leaf_solve_empties']}")
    if cfg.get("solve_empties"):
        parts.append(f"solver <= {cfg['solve_empties']} empties")
    return ", ".join(parts)


class PolicySampler:
    """Samples moves from the policy network's softmax at a temperature (weak levels). Same interface
    as SearchPlayer for the callers that use it (choose_many, solve_empties, mcts, depth)."""

    solve_empties, mcts, depth = 0, None, 0

    def __init__(self, model, temperature, seed=None):
        self.model, self.temperature = model, temperature
        self.rng = np.random.default_rng(seed)

    def choose_many(self, boards):
        boards = [np.asarray(b, dtype=np.int8).reshape(64) for b in boards]
        moves = []
        for b, lg in zip(boards, policy_logits(self.model, boards)):
            legal = legal_moves(b)
            x = lg[legal].astype(np.float64) / self.temperature
            p = np.exp(x - x.max())
            moves.append(int(self.rng.choice(legal, p=p / p.sum())))
        return moves

    def __call__(self, board):
        return self.choose_many([board])[0]


def make_player(model, level, seed=None):
    cfg = LEVELS[level]
    if "temperature" in cfg:
        return PolicySampler(model, cfg["temperature"], seed)
    return SearchPlayer(model, depth=0, **cfg)
