#!/usr/bin/env python3
"""
Evaluate a model against Edax with all games in lockstep (batched network calls), much faster than
sb-play.py for top-move and search play. Uses the same starting positions as sb-play.py, so
results match it game for game.

The model plays its top move (--search-depth 0, the default) or searches with its value head.
Sampled play (-d) and Model-vs-Model games: use sb-play.py.

Usage:
  python eval_batch.py -m r192x10_dagger -p 1,2,3 -e 200 --random-opening 8 --seed 42 \\
      --search-depth 2 --search-top-k 3
  python eval_batch.py -m r192x10_dagger -p 2 -e 150 --start-positions opening_positions.npy
  python eval_batch.py -m r192x10_bc_epoch{05,10,15,20}_CNN_test -p 2 -e 200 --random-opening 8 --seed 42
"""

import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.search import SearchPlayer
from util.lockstep import sb_play_starts, play_vs_edax
from util.util import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, nargs="+",
                        help="model name(s) as for sb-play.py (no models/ or .zip); several = a learning curve")
    parser.add_argument("-p", "--depths", default="2", help="comma-separated Edax depths")
    parser.add_argument("-e", "--episodes", type=int, default=200)
    parser.add_argument("--random-opening", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-positions", default=None)
    parser.add_argument("--search-depth", type=int, default=0, help="0 = policy top move")
    parser.add_argument("--search-top-k", type=int, default=None)
    parser.add_argument("--search-depth-early", type=int, default=0,
                        help="search this deep instead while more than --early-above squares are empty")
    parser.add_argument("--early-above", type=int, default=30)
    parser.add_argument("--leaf-solve-empties", type=int, default=0,
                        help="score search leaves with at most this many empty squares exactly (solver)")
    parser.add_argument("--solve-empties", type=int, default=0,
                        help="play positions with at most this many empty squares with the exact endgame solver")
    parser.add_argument("--search-prune-all", action="store_true",
                        help="apply --search-top-k at every node of the search tree, not only the root")
    parser.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu")
    args = parser.parse_args()
    if not (args.random_opening or args.start_positions):
        parser.error("deterministic play from the standard opening replays one game; "
                     "use --random-opening N or --start-positions FILE")

    device = get_device() if args.device == "auto" else args.device
    positions = np.load(args.start_positions) if args.start_positions else None
    starts = sb_play_starts(args.episodes, args.random_opening, args.seed, positions)
    mode = "top move" if args.search_depth == 0 else \
        f"search depth {args.search_depth}" + (f" top {args.search_top_k}" if args.search_top_k else "") + \
        (" every level" if args.search_prune_all else "") + \
        (f" + solver <= {args.solve_empties} empties" if args.solve_empties else "") + \
        (f" + leaf solves <= {args.leaf_solve_empties}" if args.leaf_solve_empties else "") + \
        (f" + depth {args.search_depth_early} above {args.early_above} empties" if args.search_depth_early else "")
    for name in args.model:
        model = MaskablePPO.load(f"models/{name}", env=build_reversi("Random"), device=device)
        player = SearchPlayer(model, depth=args.search_depth, top_k=args.search_top_k, prune_all=args.search_prune_all,
                              solve_empties=args.solve_empties,
                              leaf_solve_empties=args.leaf_solve_empties,
                              early_depth=args.search_depth_early, early_above=args.early_above)
        for depth in (int(d) for d in args.depths.split(",")):
            t = time.time()
            diffs = play_vs_edax(player.choose_many, starts, depth)
            wins, draws = int((diffs > 0).sum()), int((diffs == 0).sum())
            print(f"{name} | {mode} | edax-{depth}: Black wins: {100 * wins / len(diffs)}% "
                  f"({wins} wins, {draws} draws, {len(diffs) - wins - draws} losses, "
                  f"{time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
