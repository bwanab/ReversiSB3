"""
Board encodings and symmetries shared by the network and BC training.

Boards are from the side-to-move's perspective (own stones = 1, opponent = -1,
empty = 0), which is what the env hands the model (always BLACK) and what the
BC datasets store (WHITE's positions are flipped).
"""

import numpy as np
import torch
import torch.nn.functional as F

N_PLANES = 5  # own, opponent, empty, own legal moves, opponent legal moves

# The 8 compass directions as (row, col) steps
DIRECTIONS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _shift(t, dr, dc):
    """Shift an (N, 8, 8) tensor by (dr, dc): out[:, r, c] = t[:, r - dr, c - dc], zero-filled."""
    p = F.pad(t, (1, 1, 1, 1))
    return p[:, 1 - dr:9 - dr, 1 - dc:9 - dc]


def legal_moves(own, opp):
    """Legal-move mask for the side owning `own`.

    own, opp: (N, 8, 8) float tensors of 0/1. Returns an (N, 8, 8) 0/1 tensor.
    For each direction, x marks opponent stones in a contiguous run starting next to
    one of our stones; the empty square just past the end of a run is a legal move.
    """
    empty = 1 - own - opp
    moves = torch.zeros_like(own)
    for dr, dc in DIRECTIONS:
        x = _shift(own, dr, dc) * opp
        for _ in range(5):  # a run can be at most 6 opponent stones long
            x = torch.maximum(x, _shift(x, dr, dc) * opp)
        moves = torch.maximum(moves, _shift(x, dr, dc) * empty)
    return moves


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
