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

import os

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


def search_many(boards, depth, evaluate, root_moves=None):
    """Negamax to `depth` plies from each of several boards (side to move = 1), scoring the
    leaves of all their trees in one evaluate() call.

    root_moves: optional list (one per board) of root move subsets, e.g. the policy's top k.
    Returns a list of (best_move, {move: score}) with scores from each root side's perspective.
    """
    leaves, trees = [], []
    for i, board in enumerate(boards):
        board = np.asarray(board, dtype=np.int8).reshape(64)
        moves = legal_moves(board) if root_moves is None or root_moves[i] is None else np.asarray(root_moves[i])
        assert len(moves) > 0, "search called on a position with no legal move"
        trees.append([(int(m), _expand(play_move(board, m), depth - 1, leaves)) for m in moves])
    values = evaluate(np.array(leaves)) if leaves else np.zeros(0)
    results = []
    for children in trees:
        scores = {m: -_score(child, values) for m, child in children}
        results.append((max(scores, key=scores.get), scores))
    return results


def search_many_pruned(boards, depth, evaluate, top_moves, k):
    """Like search_many, but every node searches only its k best moves, chosen by
    top_moves(list of boards, k) -> list of move arrays (e.g. the policy's top k), called once per
    tree level for all nodes at that level. The tree grows level by level; leaves are scored in one
    evaluate() call. Same rules as search_many (passes don't use depth, finished games exact);
    with k >= the number of legal moves the scores are identical to search_many's.
    Returns a list of (best_move, {move: score}) from each root side's perspective.
    """
    leaves = []

    def resolve(node, board, remaining):
        """Fill node in place unless it needs expanding; returns (node, board, remaining) if it does."""
        if len(legal_moves(board)) == 0:
            if len(legal_moves(-board)) == 0:
                diff = int((board == 1).sum() - (board == -1).sum())
                node[:] = ['X', float(np.sign(diff))]
                return None
            child = [None, None]
            node[:] = ['P', child]
            return resolve(child, -board, remaining)
        if remaining == 0:
            leaves.append(board)
            node[:] = ['L', len(leaves) - 1]
            return None
        return node, board, remaining

    roots = [[None, None] for _ in boards]
    frontier = [(root, np.asarray(b, dtype=np.int8).reshape(64), depth) for root, b in zip(roots, boards)]
    for _, b, _ in frontier:
        assert len(legal_moves(b)) > 0, "search called on a position with no legal move"
    while frontier:
        chosen = top_moves([b for _, b, _ in frontier], k)
        next_frontier = []
        for (node, board, remaining), moves in zip(frontier, chosen):
            children = []
            for m in moves:
                child = [None, None]
                children.append((int(m), child))
                pending = resolve(child, play_move(board, int(m)), remaining - 1)
                if pending is not None:
                    next_frontier.append(pending)
            node[:] = ['N', children]
        frontier = next_frontier
    values = evaluate(np.array(leaves)) if leaves else np.zeros(0)
    results = []
    for root in roots:
        scores = {m: -_score(child, values) for m, child in root[1]}
        results.append((max(scores, key=scores.get), scores))
    return results


def search(board, depth, evaluate, root_moves=None):
    """Negamax to `depth` plies from `board` (side to move = 1); see search_many.
    Returns (best_move, {move: score}) with scores from the root side's perspective."""
    return search_many([board], depth, evaluate, None if root_moves is None else [root_moves])[0]


# Largest batch per network call. On Apple MPS, calls with more than ~65,535 positions silently
# return wrong values (measured 2026-10-04: exact up to 50k, ~2/3 wrong at 100k, independent of
# network width), so big batches are split; at these sizes chunking costs no speed.
MAX_BATCH = 16384


def _chunked(fn, boards, max_batch=None):
    """fn applied to boards in chunks of at most max_batch (default MAX_BATCH), concatenated."""
    max_batch = max_batch or MAX_BATCH
    boards = np.asarray(boards)
    if len(boards) <= max_batch:
        return fn(boards)
    return np.concatenate([fn(boards[i:i + max_batch]) for i in range(0, len(boards), max_batch)])


def policy_logits(model, boards, max_batch=None):
    """Policy logits (N, 64) for (N, 64) side-to-move boards (batched network calls)."""
    policy = model.policy

    def logits(chunk):
        obs = torch.as_tensor(chunk.reshape(-1, 1, 8, 8), dtype=torch.float32, device=policy.device)
        policy.set_training_mode(False)
        with torch.no_grad():
            features = policy.extract_features(obs)
            latent_pi, _ = policy.mlp_extractor(features)
            return policy.action_net(latent_pi).cpu().numpy()
    return _chunked(logits, boards, max_batch)


def policy_top_moves(model, boards, k):
    """The policy's k most likely legal moves for each board (best first), in one network call."""
    logits = policy_logits(model, boards)
    out = []
    for b, lg in zip(boards, logits):
        moves = legal_moves(np.asarray(b).reshape(64))
        out.append(moves[np.argsort(-lg[moves], kind="stable")[:k]])
    return out


def value_evaluator(model, max_batch=None):
    """evaluate() backed by an SB3 policy's value head (side-to-move perspective, like the env)."""
    policy = model.policy

    def values(chunk):
        obs = torch.as_tensor(chunk.reshape(-1, 1, 8, 8), dtype=torch.float32, device=policy.device)
        policy.set_training_mode(False)
        with torch.no_grad():
            return policy.predict_values(obs).reshape(-1).cpu().numpy()
    return lambda boards: _chunked(values, boards, max_batch)


class SearchPlayer:
    """Chooses moves for the model by searching `depth` plies with its value head (depth 0:
    the policy's top move).

    top_k: if set, only the policy head's k most likely legal moves are searched at the root
    (with prune_all, at every node of the tree).
    solve_empties: positions with at most this many empty squares are played by the exact endgame
    solver (util/endgame.py) instead; 0 = off.
    """

    def __init__(self, model, depth=2, top_k=None, prune_all=False, solve_empties=0):
        self.model, self.depth, self.top_k, self.prune_all = model, depth, top_k, prune_all
        self.solve_empties = solve_empties
        self.evaluate = value_evaluator(model)

    def choose_many(self, boards):
        """Best move for each of several boards, with one policy call (if top_k or depth 0)
        and one value call for all of them. depth 0 = the policy's top move. Boards with at most
        solve_empties empty squares get the exact endgame solver's move instead (solved in
        parallel threads; the C solver releases the GIL)."""
        boards = [np.asarray(b, dtype=np.int8).reshape(64) for b in boards]
        if self.solve_empties:
            late = [i for i, b in enumerate(boards) if (b == 0).sum() <= self.solve_empties]
            if late:
                from concurrent.futures import ThreadPoolExecutor
                from util.endgame import solve
                rest = [i for i in range(len(boards)) if i not in set(late)]
                moves = [None] * len(boards)
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                    for i, (_, m, _) in zip(late, pool.map(solve, [boards[i] for i in late])):
                        moves[i] = m
                if rest:
                    for i, m in zip(rest, self._choose([boards[i] for i in rest])):
                        moves[i] = m
                return moves
        return self._choose(boards)

    def _choose(self, boards):
        if self.depth == 0:
            return [int(m[0]) for m in policy_top_moves(self.model, boards, 1)]
        if self.prune_all and self.top_k is not None:
            top_moves = lambda bs, k: policy_top_moves(self.model, bs, k)
            return [best for best, _ in search_many_pruned(boards, self.depth, self.evaluate, top_moves, self.top_k)]
        roots = policy_top_moves(self.model, boards, self.top_k) if self.top_k is not None else None
        return [best for best, _ in search_many(boards, self.depth, self.evaluate, roots)]

    def __call__(self, board):
        return self.choose_many([board])[0]
