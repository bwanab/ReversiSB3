#!/usr/bin/env python3
"""
Balanced random openings (like the XOT sets used to test Othello engines): random N-ply openings,
generated exactly as sb-play/eval_batch do (--random-opening N --seed S), kept only if Edax at
--depth scores them within +-max_score discs for the side to move. Many plain random openings are
already decided (seed 42: only 19% within +-2, ~46% decided by 10+ discs), which dilutes what a
benchmark measures.

Usage:
  python make_balanced_openings.py -n 200 -o balanced_openings.npy
  python eval_batch.py -m MODEL -p 2,4,6,8 -e 200 --start-positions balanced_openings.npy
"""

import argparse

import numpy as np

from util.lockstep import sb_play_starts
from label_positions import label_boards


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--num", type=int, default=200, help="balanced openings to keep")
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-score", type=int, default=2, help="keep |Edax score| <= this (discs)")
    parser.add_argument("--depth", type=int, default=14, help="Edax depth for scoring")
    parser.add_argument("--candidates", type=int, default=2000, help="random openings to generate and score")
    parser.add_argument("-w", "--workers", type=int, default=12)
    parser.add_argument("-o", "--output", default="balanced_openings.npy")
    args = parser.parse_args()

    starts = np.array(sb_play_starts(args.candidates, args.plies, args.seed))
    scores = np.array([r[1] for r in label_boards(starts, args.depth, False, args.workers)])
    keep = np.flatnonzero(np.abs(scores) <= args.max_score)
    _, first = np.unique(starts[keep], axis=0, return_index=True)   # drop repeated openings
    keep = keep[np.sort(first)][:args.num]
    if len(keep) < args.num:
        raise SystemExit(f"only {len(keep)} balanced openings among {args.candidates}; raise --candidates")
    np.save(args.output, starts[keep])
    print(f"Saved {len(keep)} openings with |Edax depth-{args.depth} score| <= {args.max_score} "
          f"(from the first {keep[-1] + 1} of seed {args.seed}'s {args.plies}-ply random openings; "
          f"{np.mean(np.abs(scores) <= args.max_score):.0%} of candidates qualify) to {args.output}")


if __name__ == "__main__":
    main()
