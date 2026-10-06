#!/usr/bin/env python3
"""
Pick N distinct positions in an empties range from existing label files (e.g. to relabel them with
another teacher). Output: (N, 64) int8 .npy, side to move = 1, for label_positions.py --positions.

Usage:
  python select_positions.py --labels labels_d*_all.npz --min-empties 21 --max-empties 40 \\
      -n 1000000 -o positions_mid_relabel.npy
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--min-empties", type=int, default=0)
    parser.add_argument("--max-empties", type=int, default=64)
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    boards = np.unique(np.concatenate([np.load(f)["boards"] for f in args.labels]).astype(np.int8), axis=0)
    empties = (boards == 0).sum(axis=1)
    pool = boards[(empties >= args.min_empties) & (empties <= args.max_empties)]
    rng = np.random.default_rng(args.seed)
    pick = pool[rng.choice(len(pool), size=min(args.num, len(pool)), replace=False)]
    np.save(args.output, pick)
    print(f"Saved {len(pick):,} of {len(pool):,} distinct positions with {args.min_empties}-{args.max_empties} "
          f"empties to {args.output}")


if __name__ == "__main__":
    main()
