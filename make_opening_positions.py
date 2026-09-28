#!/usr/bin/env python3
"""
Build an evaluation set of named-opening positions from moves.txt.

Every position along every book line after --min-plies moves, from the side-to-move's
perspective (so the model, which always plays BLACK, plays whichever side is to move),
with rotations/reflections merged. These are sound, balanced positions that strong
players actually reach, a cleaner test of middlegame strength than random openings.

Usage:
    python make_opening_positions.py -o opening_positions.npy
    python sb-play.py -m MODEL -e 400 -o Edax -p 2 -d --start-positions opening_positions.npy --seed 42
"""

import argparse
import numpy as np

from util.openings import load_openings, replay, canonical
from util.reversi import ReversiEnvCNN
from util.util import BLACK


def opening_positions(openings, min_plies=4):
    env = ReversiEnvCNN(opponent="Random")
    seen, positions = set(), []
    for moves, _ in openings:
        for ply, (pos, _move) in enumerate(replay(moves)):
            if ply < min_plies:
                continue
            flat = pos.reshape(64).astype(np.int8)
            key = canonical(flat)
            if key in seen or not env.has_valid(pos, BLACK):
                continue
            seen.add(key)
            positions.append(flat)
    return np.array(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default='moves.txt', help='named openings file')
    parser.add_argument('--min-plies', type=int, default=4, help='skip positions this early in a line')
    parser.add_argument('-o', '--output', default='opening_positions.npy', help='output .npy file')
    args = parser.parse_args()

    positions = opening_positions(load_openings(args.input), args.min_plies)
    np.save(args.output, positions)
    stones = (positions != 0).sum(axis=1)
    print(f"Saved {len(positions):,} distinct opening positions ({stones.min()}-{stones.max()} stones) "
          f"to {args.output}")


if __name__ == '__main__':
    main()
