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
"""

import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.search import SearchPlayer
from util.lockstep import sb_play_starts, play_vs_edax


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="model name (as for sb-play.py, no models/ or .zip)")
    parser.add_argument("-p", "--depths", default="2", help="comma-separated Edax depths")
    parser.add_argument("-e", "--episodes", type=int, default=200)
    parser.add_argument("--random-opening", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-positions", default=None)
    parser.add_argument("--search-depth", type=int, default=0, help="0 = policy top move")
    parser.add_argument("--search-top-k", type=int, default=None)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if not (args.random_opening or args.start_positions):
        parser.error("deterministic play from the standard opening replays one game; "
                     "use --random-opening N or --start-positions FILE")

    model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=args.device)
    player = SearchPlayer(model, depth=args.search_depth, top_k=args.search_top_k)
    positions = np.load(args.start_positions) if args.start_positions else None
    starts = sb_play_starts(args.episodes, args.random_opening, args.seed, positions)
    mode = "top move" if args.search_depth == 0 else \
        f"search depth {args.search_depth}" + (f" top {args.search_top_k}" if args.search_top_k else "")
    for depth in (int(d) for d in args.depths.split(",")):
        t = time.time()
        diffs = play_vs_edax(player.choose_many, starts, depth)
        wins, draws = int((diffs > 0).sum()), int((diffs == 0).sum())
        print(f"{args.model} | {mode} | edax-{depth}: Black wins: {100 * wins / len(diffs)}% "
              f"({wins} wins, {draws} draws, {len(diffs) - wins - draws} losses, "
              f"{time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
