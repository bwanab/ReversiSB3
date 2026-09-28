#!/usr/bin/env python3
"""
How closely do models follow opening theory (moves.txt)?

For every book position (after --min-plies moves) that has a known book reply:
  - top-1 in book: the model's most likely move is one of the book continuations
  - book mass:     total probability the model puts on book continuations
  - chance:        the same mass for a uniform choice among legal moves (baseline)
Also reports the probability of the X-square move in the "X-square Opening" lines,
which moves.txt marks as bad.

Usage:
    python opening_agreement.py -m resnet128x8_bc10_bconly_CNN_test planes_aug_bc10_rl2m_CNN_test
"""

import argparse
import numpy as np
import torch
from sb3_contrib import MaskablePPO

from util.openings import load_openings, replay, book_positions
from util.reversi import build_reversi, ReversiEnvCNN
from util.util import BLACK


def move_probs(model, positions, env):
    """Masked policy probabilities for a list of (1, 8, 8) side-to-move boards."""
    obs = torch.as_tensor(np.stack(positions)).float().to(model.device)
    masks = np.stack([env._get_valid(p, BLACK) == 1 for p in positions])
    model.policy.set_training_mode(False)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs, action_masks=masks)
        return dist.distribution.probs.cpu().numpy(), masks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--models', nargs='+', required=True,
                        help='model names under models/ (without .zip)')
    parser.add_argument('-i', '--input', default='moves.txt')
    parser.add_argument('--min-plies', type=int, default=4)
    args = parser.parse_args()

    openings = load_openings(args.input)
    book = {k: v for k, v in book_positions(openings, args.min_plies).items() if v}
    positions = [np.frombuffer(k, dtype=np.int8).reshape(1, 8, 8) for k in book]
    book_moves = list(book.values())

    # Positions where the X-square move was played in the "X-square Opening" lines
    # (position, move) just before the line's final move, which is the X-square move
    xsq = [list(replay(moves))[-2] for moves, name in openings if name.startswith("X-square")]

    env = ReversiEnvCNN(opponent="Random")
    print(f"{len(positions)} book positions with a book reply (after >= {args.min_plies} plies), "
          f"{len(xsq)} X-square positions\n")
    header = f"{'model':45s} {'top-1 in book':>13s} {'book mass':>10s} {'chance':>7s} {'P(X-square)':>12s}"
    print(header)
    print("-" * len(header))
    for name in args.models:
        model = MaskablePPO.load(f"models/{name}", env=build_reversi("Random"))
        probs, masks = move_probs(model, positions, env)
        top1 = np.mean([probs[i].argmax() in book_moves[i] for i in range(len(positions))])
        mass = np.mean([probs[i, list(book_moves[i])].sum() for i in range(len(positions))])
        chance = np.mean([len(book_moves[i]) / masks[i].sum() for i in range(len(positions))])
        xp, _ = move_probs(model, [p for p, _ in xsq], env)
        p_x = np.mean([xp[i, m] for i, (_, m) in enumerate(xsq)])
        print(f"{name:45s} {100 * top1:12.1f}% {100 * mass:9.1f}% {100 * chance:6.1f}% {100 * p_x:11.2f}%")


if __name__ == '__main__':
    main()
