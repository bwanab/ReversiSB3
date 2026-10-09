"""
Monte Carlo Tree Search (AlphaZero-style PUCT) over Reversi positions, batched across many games.

Same board convention as util/search.py: flat (64,) boards from the side to move's perspective,
negated after every move. Each node holds, per legal move, a prior P (the policy's softmax over legal
moves), a visit count N and a total value W from that node's side to move's perspective. A simulation
walks down from the root choosing argmax Q + U, with
    Q = W / N   (unvisited moves: the node's own value minus fpu_reduction, "first play urgency")
    U = c_puct * P * sqrt(1 + sum N) / (1 + N),
stops at the first unexpanded move, evaluates that position with the network (policy and value in one
call), and backs the value up the path, negated at every ply. After `sims` simulations the move
played is the most visited one.

Batching: each round, every root collects up to `parallel` leaves; a "virtual loss" (one visit
counted as a loss while a leaf is pending) steers the root's later selections in the same round to
other lines. All leaves of all roots are evaluated in one network call.

Values are on the value head's scale, tanh(score / 16) with score in discs. Finished games are
scored the same way from the official final score (empties to the winner), and positions with at
most leaf_solve_empties empty squares are solved exactly (util/endgame.py) and become fixed leaves.
A side with no legal move passes (a node with the single move PASS); passes are part of the tree.
"""

import numpy as np

from util.search import legal_moves, play_move, LEAF_SCORE_SCALE

PASS = -1


class Node:
    __slots__ = ("board", "value", "exact", "moves", "prior", "n", "w", "children")

    def __init__(self, board, value, moves=None, prior=None, exact=False):
        self.board, self.value, self.exact = board, float(value), exact
        self.moves, self.prior = moves, prior
        if moves is not None:
            self.n = np.zeros(len(moves))
            self.w = np.zeros(len(moves))
            self.children = [None] * len(moves)


def final_value(board, scale=LEAF_SCORE_SCALE):
    """tanh(official final score / scale) of a finished game, side to move's perspective."""
    own, opp = int((board == 1).sum()), int((board == -1).sum())
    diff = own - opp
    diff += int(np.sign(diff)) * (64 - own - opp)
    return float(np.tanh(diff / scale))


class MCTS:
    """net(boards) -> (logits (N, 64), values (N,)) for (N, 64) side-to-move boards.

    sims: simulations per move; c_puct: exploration weight; parallel: leaves per root per round
    (more = bigger network batches, slightly less exact search); fpu_reduction: unvisited moves are
    valued at the node's value minus this; leaf_solve_empties: solve positions with at most this many
    empty squares exactly (0 = off).

    Adaptive budget (step 30): early_stop (default) ends a root's search once the most visited move
    leads the runner-up by more than the simulations left, so the choice can no longer change: the same
    moves as the full budget, ~30% fewer simulations on average. max_sims > sims (measured: rarely
    triggers, no gain)
    extends a root past `sims`, in steps of sims / 4 up to max_sims, while it is unsettled: the
    runner-up has more than extend_ratio of the leader's visits, or the most visited move isn't the
    best valued one (among moves with at least 10% of the leader's visits)."""

    def __init__(self, net, sims=400, c_puct=1.0, parallel=4, fpu_reduction=0.3, leaf_solve_empties=0,
                 scale=LEAF_SCORE_SCALE, max_sims=None, early_stop=True, extend_ratio=0.6):
        self.max_sims, self.early_stop, self.extend_ratio = max(max_sims or sims, sims), early_stop, extend_ratio
        self.net, self.sims, self.c_puct, self.parallel = net, sims, c_puct, parallel
        self.fpu_reduction, self.leaf_solve_empties, self.scale = fpu_reduction, leaf_solve_empties, scale
        self.evaluations = 0                     # network positions evaluated (cost measure)

    def run(self, boards):
        """Search each board (side to move must have a legal move). Returns a list of
        (best_move, {move: visits}) per board."""
        boards = [np.asarray(b, dtype=np.int8).reshape(64) for b in boards]
        for b in boards:
            assert len(legal_moves(b)) > 0, "MCTS called on a position with no legal move"
        roots = self._make_nodes(boards, solve=False)
        done = [0] * len(roots)
        limit = [self.sims] * len(roots)
        active = [len(r.moves) > 1 for r in roots]
        while any(active):
            pending, requests = set(), []                    # requests: (path, board)
            for r, root in enumerate(roots):
                if not active[r]:
                    continue
                for _ in range(min(self.parallel, limit[r] - done[r])):
                    done[r] += 1
                    req = self._select(root, pending)
                    if req is not None:
                        requests.append(req)
            for (path, _), node in zip(requests, self._make_nodes([b for _, b in requests])):
                parent, i = path[-1]
                parent.children[i] = node
                self._backup(path, node.value)
            for r, root in enumerate(roots):
                if active[r]:
                    active[r] = self._continue(root, done[r], limit, r)
        self.last_roots = roots                              # for inspection
        self.last_sims = done
        results = []
        for root in roots:
            q = np.where(root.n > 0, root.w / np.maximum(root.n, 1), -np.inf)
            best = max(range(len(root.moves)), key=lambda i: (root.n[i], q[i], root.prior[i]))
            results.append((int(root.moves[best]), {int(m): int(n) for m, n in zip(root.moves, root.n)}))
        return results

    def _continue(self, root, done, limit, r):
        """Whether root r keeps searching (may raise limit[r] to extend it)."""
        n = np.sort(root.n)[::-1]
        if done < limit[r]:
            return not (self.early_stop and n[0] - n[1] > limit[r] - done)
        if limit[r] < self.max_sims:
            q = np.where(root.n >= 0.1 * n[0], root.w / np.maximum(root.n, 1), -np.inf)
            if n[1] > self.extend_ratio * n[0] or np.argmax(q) != np.argmax(root.n):
                limit[r] = min(self.max_sims, limit[r] + max(1, self.sims // 4))
                return True
        return False

    def _pick(self, node):
        n = node.n
        q = np.where(n > 0, node.w / np.maximum(n, 1), node.value - self.fpu_reduction)
        u = self.c_puct * node.prior * np.sqrt(1.0 + n.sum()) / (1.0 + n)
        return int(np.argmax(q + u))

    def _select(self, root, pending):
        """Walk down with virtual loss. Returns (path, leaf board) for a leaf to evaluate, or None
        if the walk ended on an exact node (backed up here) or collided with a pending leaf."""
        node, path = root, []
        while True:
            if node.exact:
                self._backup(path, node.value)
                return None
            i = self._pick(node)
            path.append((node, i))
            node.n[i] += 1
            node.w[i] -= 1                                    # virtual loss (node's perspective)
            child = node.children[i]
            if child is None:
                key = (id(node), i)
                if key in pending:                            # same leaf already requested this round
                    for nd, j in path:
                        nd.n[j] -= 1
                        nd.w[j] += 1
                    return None
                pending.add(key)
                m = node.moves[i]
                return path, (-node.board if m == PASS else play_move(node.board, int(m)))
            node = child

    @staticmethod
    def _backup(path, value):
        """value: from the leaf's side to move's perspective. Removes the virtual loss."""
        for node, i in reversed(path):
            value = -value
            node.w[i] += value + 1

    def _make_nodes(self, boards, solve=True):
        kinds, net_boards, solve_boards = [], [], []
        for b in boards:
            empties = int((b == 0).sum())
            if len(legal_moves(b)):
                kind = "net"
            elif len(legal_moves(-b)):
                kind = "pass"
            else:
                kinds.append(("final", None))
                continue
            if solve and self.leaf_solve_empties and empties <= self.leaf_solve_empties:
                kinds.append(("solve", len(solve_boards)))
                solve_boards.append(b)
            else:
                kinds.append((kind, len(net_boards)))
                net_boards.append(b if kind == "net" else -b)
        if solve_boards:                                  # solves run during the network call
            from util.endgame import solve_async
            solving = solve_async(solve_boards)
        if net_boards:
            logits, values = self.net(np.array(net_boards))
            self.evaluations += len(net_boards)
        if solve_boards:
            solved = [s for s, _, _ in solving()]
        nodes = []
        for b, (kind, j) in zip(boards, kinds):
            if kind == "final":
                nodes.append(Node(b, final_value(b, self.scale), exact=True))
            elif kind == "solve":
                nodes.append(Node(b, np.tanh(solved[j] / self.scale), exact=True))
            else:
                nb = net_boards[j]
                moves = legal_moves(nb)
                x = logits[j][moves].astype(np.float64)
                prior = np.exp(x - x.max())
                node = Node(nb, values[j], moves, prior / prior.sum())
                if kind == "pass":
                    wrapper = Node(b, -node.value, np.array([PASS]), np.ones(1))
                    wrapper.children[0] = node
                    node = wrapper
                nodes.append(node)
        return nodes
