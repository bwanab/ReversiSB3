"""
Shallow minimax (negamax) search over Reversi positions, scored by a network's value head.

Boards are flat (64,) int8 arrays from the side-to-move's perspective (1 = side to move,
-1 = opponent, 0 = empty), the same convention the env hands the model. After a move the
board is negated so the next side to move is again 1; each node's score is from its own
side-to-move's perspective and a parent's score is the negation of its children's.

Search to `depth` plies (a forced pass doesn't use up depth), then score the leaves with
`evaluate`, all in one batched call. Positions where neither side can move are scored
exactly by disc count (+1 win, -1 loss, 0 draw), never by the evaluator.
"""

import numpy as np
import torch

from util.board_features import _SRC, DIRECTIONS

_DIRS = [dr * 8 + dc for dr, dc in DIRECTIONS]


def legal_moves(board):
    """Flat indices of the side to move's legal moves on a (64,) board."""
    own = np.append(board == 1, False)
    opp = np.append(board == -1, False)
    empty = board == 0
    x = own[_SRC] & opp[:-1]                      # (8, 64): opponent stones next to one of ours
    for _ in range(5):
        x |= np.append(x, np.zeros((8, 1), bool), axis=1)[np.arange(8)[:, None], _SRC] & opp[:-1]
    reach = np.append(x, np.zeros((8, 1), bool), axis=1)[np.arange(8)[:, None], _SRC]
    return np.flatnonzero((reach & empty).any(axis=0))


def play_move(board, move):
    """Board after the side to move plays `move`, from the new side to move's perspective."""
    b = board.copy()
    r, c = divmod(move, 8)
    b[move] = 1
    for dr, dc in DIRECTIONS:
        rr, cc, run = r + dr, c + dc, []
        while 0 <= rr < 8 and 0 <= cc < 8 and b[rr * 8 + cc] == -1:
            run.append(rr * 8 + cc)
            rr, cc = rr + dr, cc + dc
        if run and 0 <= rr < 8 and 0 <= cc < 8 and b[rr * 8 + cc] == 1:
            b[run] = 1
    return -b


def _expand(board, depth, leaves):
    """Build the search tree. Nodes: ('X', exact score) | ('L', leaf index) |
    ('P', child) for a forced pass | ('N', [(move, child), ...])."""
    moves = legal_moves(board)
    if len(moves) == 0:
        if len(legal_moves(-board)) == 0:
            diff = int((board == 1).sum() - (board == -1).sum())
            return ('X', float(np.sign(diff)))
        return ('P', _expand(-board, depth, leaves))
    if depth == 0:
        leaves.append(board)
        return ('L', len(leaves) - 1)
    return ('N', [(m, _expand(play_move(board, m), depth - 1, leaves)) for m in moves])


def _score(node, values):
    kind, payload = node
    if kind == 'X':
        return payload
    if kind == 'L':
        return float(values[payload])
    if kind == 'P':
        return -_score(payload, values)
    return max(-_score(child, values) for _, child in payload)


def search(board, depth, evaluate, root_moves=None):
    """Negamax to `depth` plies from `board` (side to move = 1).

    evaluate: (N, 64) boards -> (N,) scores, each from that board's side-to-move perspective.
    root_moves: optional subset of legal moves to consider at the root (e.g. policy top-k).
    Returns (best_move, {move: score}) with scores from the root side's perspective.
    """
    board = np.asarray(board, dtype=np.int8).reshape(64)
    moves = legal_moves(board) if root_moves is None else np.asarray(root_moves)
    assert len(moves) > 0, "search called on a position with no legal move"
    leaves = []
    children = [(int(m), _expand(play_move(board, m), depth - 1, leaves)) for m in moves]
    values = evaluate(np.array(leaves)) if leaves else np.zeros(0)
    scores = {m: -_score(child, values) for m, child in children}
    best = max(scores, key=scores.get)
    return best, scores


def value_evaluator(model):
    """evaluate() backed by an SB3 policy's value head (side-to-move perspective, like the env)."""
    policy = model.policy

    def evaluate(boards):
        obs = torch.as_tensor(boards.reshape(-1, 1, 8, 8), dtype=torch.float32, device=policy.device)
        policy.set_training_mode(False)
        with torch.no_grad():
            return policy.predict_values(obs).reshape(-1).cpu().numpy()
    return evaluate


class SearchPlayer:
    """Chooses moves for the model by searching `depth` plies with its value head.

    top_k: if set, only the policy head's k most likely legal moves are searched at the root.
    """

    def __init__(self, model, depth=2, top_k=None):
        self.model, self.depth, self.top_k = model, depth, top_k
        self.evaluate = value_evaluator(model)

    def root_moves(self, board):
        moves = legal_moves(board)
        if self.top_k is None or len(moves) <= self.top_k:
            return moves
        policy = self.model.policy
        obs = torch.as_tensor(board.reshape(1, 1, 8, 8), dtype=torch.float32, device=policy.device)
        mask = np.zeros((1, 64), dtype=bool)
        mask[0, moves] = True
        with torch.no_grad():
            probs = policy.get_distribution(obs, action_masks=mask).distribution.probs[0].cpu().numpy()
        return moves[np.argsort(-probs[moves])[:self.top_k]]

    def __call__(self, board):
        board = np.asarray(board, dtype=np.int8).reshape(64)
        best, _ = search(board, self.depth, self.evaluate, self.root_moves(board))
        return best
