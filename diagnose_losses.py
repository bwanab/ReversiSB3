#!/usr/bin/env python3
"""
Where are games lost? Play a model (top move or search) against Edax from random/named openings,
record every model move, score each recorded position with Edax at --teacher-depth (every move),
and break the model's move errors down by game phase (empty squares).

Per model move: value V = Edax's best score (discs, model's view), regret = V - score of the move
played. Between two model moves the value changes only through Edax's move, so Edax's errors are
measured too. Near the end (empties <= teacher depth) the scores are exact.

For each lost game, the "turning move" is the model's last move made from a position that was not
losing (V >= 0) after which the position never recovered (V < 0 at every later model move).

Usage:
  python diagnose_losses.py -m r256x12_sdag1_CNN_test -p 8 --random-opening 8 --seed 42 \\
      --search-depth 5 --search-top-k 3 --search-prune-all
"""

import argparse
import collections

import numpy as np
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.search import SearchPlayer
from util.lockstep import sb_play_starts, play_vs_edax
from util.util import get_device
from label_positions import label_boards

PHASES = [(41, 64, "opening/early (41+ empty)"), (31, 40, "middlegame (31-40)"), (21, 30, "middlegame (21-30)"),
          (13, 20, "late (13-20)"), (0, 12, "endgame (0-12, exact)")]


def phase_of(empties):
    for lo, hi, name in PHASES:
        if lo <= empties <= hi:
            return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--edax-depth", type=int, default=8)
    parser.add_argument("-e", "--episodes", type=int, default=200)
    parser.add_argument("--random-opening", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-positions", default=None)
    parser.add_argument("--search-depth", type=int, default=0)
    parser.add_argument("--search-top-k", type=int, default=None)
    parser.add_argument("--search-prune-all", action="store_true")
    parser.add_argument("--leaf-solve-empties", type=int, default=0,
                        help="score search leaves with at most this many empty squares exactly (solver)")
    parser.add_argument("--solve-empties", type=int, default=0, help="exact endgame solver at <= N empties")
    parser.add_argument("--teacher-depth", type=int, default=14)
    parser.add_argument("-w", "--workers", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not (args.random_opening or args.start_positions):
        parser.error("use --random-opening N or --start-positions FILE")

    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=device)
    player = SearchPlayer(model, depth=args.search_depth, top_k=args.search_top_k, prune_all=args.search_prune_all,
                          solve_empties=args.solve_empties,
                              leaf_solve_empties=args.leaf_solve_empties)
    starts = sb_play_starts(args.episodes, args.random_opening, args.seed,
                            np.load(args.start_positions) if args.start_positions else None)
    moves = []                                    # (game, board, move)
    diffs = play_vs_edax(player.choose_many, starts, args.edax_depth,
                         on_model_move=lambda g, b, m: moves.append((g, b, m)))
    wins, draws = int((diffs > 0).sum()), int((diffs == 0).sum())
    print(f"{args.model} vs Edax-{args.edax_depth}: {100 * wins / len(diffs):.1f}% wins, {draws} draws, "
          f"{len(diffs) - wins - draws} losses; {len(moves):,} model moves to grade")

    results = label_boards(np.array([b for _, b, _ in moves]), args.teacher_depth, True, args.workers)
    games = collections.defaultdict(list)          # game -> [(empties, V, value after move, regret)]
    for (g, b, m), (_, _, ms) in zip(moves, results):
        v = float(np.nanmax(ms))
        games[g].append(((b == 0).sum(), v, float(ms[m]), v - float(ms[m])))

    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    edax_err = collections.defaultdict(lambda: collections.defaultdict(list))
    turning = collections.Counter()
    lost_from_start = 0
    for g, seq in games.items():
        outcome = "lost" if diffs[g] < 0 else ("won" if diffs[g] > 0 else "drawn")
        for i, (e, v, after, reg) in enumerate(seq):
            rows[outcome][phase_of(e)].append(reg)
            if i + 1 < len(seq):                   # Edax's move in between (model's view)
                edax_err[outcome][phase_of(e)].append(max(0.0, seq[i + 1][1] - after))
        if outcome == "lost":
            ok = [i for i, (_, v, _, _) in enumerate(seq) if v >= 0]
            if not ok:
                lost_from_start += 1
            else:
                turning[phase_of(seq[ok[-1]][0])] += 1

    for outcome in ("lost", "won"):
        n = sum(1 for g in games if (diffs[g] < 0 if outcome == "lost" else diffs[g] > 0))
        print(f"\n{outcome} games ({n}): model regret per phase (discs, Edax depth-{args.teacher_depth} scores)")
        total = sum(sum(r) for r in rows[outcome].values()) or 1
        print(f"  {'phase':26s} {'moves':>6s} {'mean regret':>11s} {'share':>6s} {'blunders>=4':>11s} "
              f"{'Edax gift/move':>14s}")
        for _, _, name in PHASES:
            r = rows[outcome][name]
            if not r:
                continue
            gift = edax_err[outcome][name]
            print(f"  {name:26s} {len(r):6d} {np.mean(r):11.2f} {100 * sum(r) / total:5.1f}% "
                  f"{100 * np.mean(np.array(r) >= 4):10.1f}% {np.mean(gift) if gift else 0:14.2f}")
    print(f"\nlost games: turning move by phase (last move from a non-losing position): "
          + ", ".join(f"{name}: {turning[name]}" for _, _, name in PHASES)
          + f"; already losing at the first move: {lost_from_start}")


if __name__ == "__main__":
    main()
