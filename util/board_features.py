"""
Board encodings and symmetries shared by the network and BC training.

Boards are from the side-to-move's perspective (own stones = 1, opponent = -1,
empty = 0), which is what the env hands the model (always BLACK) and what the
BC datasets store (WHITE's positions are flipped).
"""

import numpy as np
import torch

N_PLANES = 5  # own, opponent, empty, own legal moves, opponent legal moves

# The 8 compass directions as (row, col) steps
DIRECTIONS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _neighbor_table():
    """SRC[d, s] = the square whose value moves onto square s when shifting one step in
    direction d (i.e. the square at s - d), or 64 (an always-zero pad column) if that is
    off the board."""
    src = np.full((len(DIRECTIONS), 64), 64, dtype=np.int64)
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for r in range(8):
            for c in range(8):
                rr, cc = r - dr, c - dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    src[d, r * 8 + c] = rr * 8 + cc
    return src


_SRC = _neighbor_table()
_SRC_ON = {}  # per-device copies


def _src(device):
    if device not in _SRC_ON:
        _SRC_ON[device] = torch.as_tensor(_SRC, device=device)
    return _SRC_ON[device]


def legal_moves(own, opp):
    """Legal-move mask for the side owning `own`.

    own, opp: (N, 8, 8) float tensors of 0/1. Returns an (N, 8, 8) 0/1 tensor.
    For each direction, x marks opponent stones in a contiguous run starting next to
    one of our stones; the empty square just past the end of a run is a legal move.
    All 8 directions are handled together: each step shifts the (N, 8 directions, 64)
    tensor by indexing through the precomputed neighbor table, so a call is a couple of
    dozen tensor ops however many directions there are (cheap for single positions).
    """
    n = own.shape[0]
    src = _src(own.device)
    dirs = torch.arange(len(DIRECTIONS), device=own.device)[:, None]
    own_f, opp_f = own.reshape(n, 64), opp.reshape(n, 64)
    empty = 1 - own_f - opp_f
    zero = own_f.new_zeros(n, 1)

    def shift(x):
        # x: (N, 8, 64) -> each direction's plane shifted one step along that direction
        padded = torch.cat([x, zero[:, None, :].expand(n, x.shape[1], 1)], dim=2)
        return padded[:, dirs, src]

    x = torch.cat([own_f, zero], dim=1)[:, src] * opp_f[:, None, :]  # (N, 8, 64)
    for _ in range(5):  # a run can be at most 6 opponent stones long
        x = torch.maximum(x, shift(x) * opp_f[:, None, :])
    moves = (shift(x) * empty[:, None, :]).amax(dim=1)
    return moves.reshape(n, 8, 8)


def board_to_planes(board):
    """Convert (N, 1, 8, 8) boards with values -1/0/1 into (N, 5, 8, 8) 0/1 planes."""
    b = board[:, 0]
    own = (b > 0.5).float()
    opp = (b < -0.5).float()
    empty = 1 - own - opp
    return torch.stack([own, opp, empty, legal_moves(own, opp), legal_moves(opp, own)], dim=1)


def _symmetry_tables():
    """Index tables for the 8 symmetries of the square (4 rotations x optional mirror).

    PERMS[k][j] is the square whose content lands on square j under symmetry k, so a
    transformed flat board is board_flat[PERMS[k]]. INV_PERMS[k][a] is where square a
    moves to, which is how an action is remapped.
    """
    grid = np.arange(64).reshape(8, 8)
    perms = []
    for k in range(8):
        t = np.rot90(grid, k % 4)
        if k >= 4:
            t = np.fliplr(t)
        perms.append(t.flatten())
    perms = np.array(perms)
    inv = np.argsort(perms, axis=1)
    return perms, inv


PERMS, INV_PERMS = _symmetry_tables()


def random_symmetry(states, actions):
    """Apply an independent random symmetry to each (state, action) pair in a batch.

    states: (N, 1, 8, 8) tensor, actions: (N,) long tensor of flat indices (row * 8 + col).
    """
    device = states.device
    perms = torch.as_tensor(PERMS, device=device)
    inv = torch.as_tensor(INV_PERMS, device=device)
    k = torch.randint(0, 8, (states.shape[0],), device=device)
    flat = states.reshape(states.shape[0], 64)
    new_states = flat.gather(1, perms[k]).reshape(states.shape)
    new_actions = inv[k, actions]
    return new_states, new_actions
