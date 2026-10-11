#!/usr/bin/env python3
"""
Play two of our players against each other: every start position twice, colors swapped, all games in
lockstep (util/lockstep.play_match). Opening luck cancels out; a score of 50% means even.

Players:
  level:N          strength level N (util/levels.py, as in web_play.py)
  hint[:SIMS]      always plays the best move by the score hints (util/hints.py: exact with <= 18
                   empties, else MCTS SIMS (default 400) and the highest average value)

Usage:
  python match.py -a hint -b level:9 --start-positions balanced_openings.npy -e 200
  python match.py -a level:7 -b level:8 -e 100       # 100 starts = 200 games
"""

import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO

from util.levels import LEVELS, describe, make_player
from util.lockstep import play_match, sb_play_starts
from util.reversi import build_reversi
from util.util import get_device


def make(model, spec, seed):
    kind, _, arg = spec.partition(":")
    if kind == "level" and arg.isdigit() and int(arg) in LEVELS:
        return make_player(model, int(arg), seed=seed), f"level {arg} ({describe(int(arg))})"
    if kind == "hint":
        from util.hints import ScorePlayer
        sims = int(arg) if arg else 400
        return ScorePlayer(model, sims=sims), f"hint (best score; MCTS {sims}, exact <= 18 empties)"
    raise SystemExit(f"unknown player {spec!r}: use level:N or hint[:SIMS]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", default="r256x12_mid1_CNN_test")
    parser.add_argument("-a", required=True, help="player A (level:N or hint[:SIMS])")
    parser.add_argument("-b", required=True, help="player B")
    parser.add_argument("-e", "--starts", type=int, default=200, help="start positions (each played twice)")
    parser.add_argument("--start-positions", default="balanced_openings.npy")
    parser.add_argument("--random-opening", type=int, default=0, help="instead: N random plies from the start")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=device)
    player_a, desc_a = make(model, args.a, args.seed)
    player_b, desc_b = make(model, args.b, args.seed + 1)
    positions = None if args.random_opening else np.load(args.start_positions)
    starts = sb_play_starts(args.starts, args.random_opening, args.seed, positions)
    t = time.time()
    diffs = play_match(player_a.choose_many, player_b.choose_many, starts)
    wins, draws = int((diffs > 0).sum()), int((diffs == 0).sum())
    losses = len(diffs) - wins - draws
    n = len(starts)
    as_first, as_second = diffs[:n], diffs[n:]
    print(f"A: {desc_a}\nB: {desc_b}")
    print(f"A vs B: {wins} wins, {draws} draws, {losses} losses in {len(diffs)} games -> "
          f"A scores {100 * (wins + draws / 2) / len(diffs):.1f}% (draws 1/2; 50% = even); "
          f"A moving first {100 * ((as_first > 0).sum() + (as_first == 0).sum() / 2) / n:.1f}%, "
          f"second {100 * ((as_second > 0).sum() + (as_second == 0).sum() / 2) / n:.1f}%; "
          f"mean disc margin {diffs.mean():+.1f}; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
