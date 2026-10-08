#!/usr/bin/env python3
"""
Solver benchmark: time util/endgame.solve on real-game positions at several empty counts and check
the scores against a reference file (made by the first run with --make-reference).

  uv run python solver/bench.py --make-reference   # once, with a trusted solver build
  uv run python solver/bench.py                    # after changes: times, nodes, and mismatches
Positions come from the Edax label files (labels_d*_all.npz), side to move with a legal move.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.endgame import solve
from util.search import legal_moves

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_reference.npz")
COUNTS = {14: 200, 16: 100, 18: 40, 20: 12}


def positions(seed=0):
    files = sorted(glob.glob("labels_d1[24]_all.npz"))[:1] or sorted(glob.glob("labels_d*_all.npz"))[:1]
    boards = np.load(files[0])["boards"].reshape(-1, 64)
    empties = (boards == 0).sum(axis=1)
    rng = np.random.default_rng(seed)
    out = {}
    for e, n in COUNTS.items():
        idx = rng.permutation(np.flatnonzero(empties == e))
        out[e] = [boards[i] for i in idx if len(legal_moves(boards[i]))][:n]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-reference", action="store_true")
    args = parser.parse_args()
    pos = positions()
    ref = None if args.make_reference else np.load(REF)
    saved = {}
    for e, boards in pos.items():
        t = time.time()
        res = [solve(b) for b in boards]
        secs = time.time() - t
        scores = np.array([r[0] for r in res])
        nodes = np.array([r[2] for r in res])
        saved[f"scores_{e}"] = scores
        line = (f"{e} empties: {len(boards):3d} positions, {1000 * secs / len(boards):8.2f} ms/position, "
                f"{nodes.mean() / 1000:9.1f}k nodes/position")
        if ref is not None:
            bad = int((ref[f"scores_{e}"] != scores).sum())
            line += f", {'OK' if bad == 0 else f'{bad} SCORE MISMATCHES'}"
        print(line, flush=True)
    if args.make_reference:
        np.savez(REF, **saved)
        print(f"saved {REF}")


if __name__ == "__main__":
    main()
