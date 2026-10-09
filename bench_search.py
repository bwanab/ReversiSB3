#!/usr/bin/env python3
"""
Quick search benchmark: choose a move with each search configuration on labeled positions (every
move scored, e.g. by label_positions.py --every-move) and report regret (discs lost per move vs the
labeler's best move), best-move rate, time, and network positions evaluated per move.

Much faster and less noisy than games for tuning a search, but graded by the labeler (Edax depth 14
here), which a deep search can occasionally out-think; confirm with games (eval_batch.py).

Usage:
  python bench_search.py -m r256x12_mid1_CNN_test --labels bench/heldout_d14.npz \\
      --config "top" --config "negamax:5:6" --config "mcts:400:1.0"
Configs: "top" (policy top move); "negamax:DEPTH[:EARLY_DEPTH]" (top 3 at every level, leaf solves
<= 16, early depth above 30 empties); "mcts:SIMS:C[:FPU[:PARALLEL]]" (leaf solves <= 16);
"amcts:SIMS:MAX_SIMS[:EARLY_STOP]" (adaptive budget, util/mcts.py; early stop on by default).
"""

import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.search import SearchPlayer
from util.util import get_device


def make_player(model, spec):
    kind, *a = spec.split(":")
    if kind == "top":
        return SearchPlayer(model, depth=0)
    if kind == "negamax":
        return SearchPlayer(model, depth=int(a[0]), top_k=3, prune_all=True, leaf_solve_empties=16,
                            early_depth=int(a[1]) if len(a) > 1 else 0, early_above=30)
    if kind == "amcts":                     # amcts:SIMS:MAX_SIMS[:EARLY_STOP 0/1]
        return SearchPlayer(model, mcts_sims=int(a[0]), mcts_max_sims=int(a[1]),
                            mcts_early_stop=bool(int(a[2])) if len(a) > 2 else True, leaf_solve_empties=16)
    if kind == "mcts":
        return SearchPlayer(model, mcts_sims=int(a[0]), mcts_c=float(a[1]),
                            mcts_fpu=float(a[2]) if len(a) > 2 else 0.3,
                            mcts_parallel=int(a[3]) if len(a) > 3 else 4, leaf_solve_empties=16)
    raise ValueError(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("-n", type=int, default=None, help="use the first n positions")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    d = np.load(args.labels)
    boards, scores = d["boards"].reshape(-1, 64), d["move_scores"].astype(np.float64)
    if args.n:
        boards, scores = boards[:args.n], scores[:args.n]
    best = np.nanmax(scores, axis=1)
    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=device)
    print(f"{args.model}, {len(boards)} positions from {args.labels}")
    for spec in args.config:
        player = make_player(model, spec)
        t = time.time()
        moves = np.array(player.choose_many(list(boards)))
        secs = time.time() - t
        regret = best - scores[np.arange(len(boards)), moves]
        evals = (f", {player.mcts.evaluations / len(boards):.0f} evals/move, "
                 f"{np.mean(player.mcts.last_sims):.0f} sims/move" if player.mcts else "")
        print(f"{spec:22s} regret {regret.mean():.3f} discs, best move {100 * (regret == 0).mean():.1f}%, "
              f"blunders>=4 {100 * (regret >= 4).mean():.1f}%, {1000 * secs / len(boards):.0f} ms/move{evals}",
              flush=True)


if __name__ == "__main__":
    main()
