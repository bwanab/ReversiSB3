#!/usr/bin/env python3
"""
Grade the opponent's moves in games recorded by web_play.py (games/*.json), e.g. games against the
Piccolo app: for each opponent move, the probability and rank our policy gave it, and its cost in
discs by Egaroucid's score of every legal move (level --level). Answers whether the moves the policy
thought unlikely were good (a blind spot: the search, guided by the policy, may not look there) or
bad (the opponent's mistakes).

Usage:
  python analyze_games.py games/*.json [-m r256x12_mid1_CNN_test] [--level 14] [-v]
"""

import argparse
import json

import numpy as np
from sb3_contrib import MaskablePPO

from util.egaroucid_client import EgaroucidClient
from util.reversi import build_reversi
from util.search import legal_moves, play_move, policy_logits
from util.util import get_device

PHASES = [(41, 64, "opening (41+ empty)"), (21, 40, "middlegame (21-40)"), (0, 20, "late (0-20)")]


def phase_of(empties):
    return next(name for lo, hi, name in PHASES if lo <= empties <= hi)


def check_replay(moves):
    """True if every recorded board follows from the previous one by its move (passes allowed)."""
    for prev, nxt in zip(moves, moves[1:]):
        sign = 1 if prev["color"] == "black" else -1
        after = play_move(np.array(prev["board_before"], dtype=np.int8) * sign, prev["action"])  # mover's view, negated
        expected = -after * sign
        if not np.array_equal(expected, np.array(nxt["board_before"], dtype=np.int8)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", nargs="+")
    parser.add_argument("-m", "--model", default="r256x12_mid1_CNN_test")
    parser.add_argument("--level", type=int, default=14, help="Egaroucid level for grading")
    parser.add_argument("--low", type=float, default=0.05, help="policy probability counted as 'unlikely'")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every opponent move")
    args = parser.parse_args()

    model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=get_device())
    engine = EgaroucidClient(threads=8)
    rows = []                                   # (phase, prob, rank, n_legal, regret, top_regret)
    for path in args.games:
        g = json.load(open(path))
        if not check_replay(g["moves"]):
            print(f"{path}: moves don't replay consistently; skipped")
            continue
        opp = [m for m in g["moves"] if m["by"] == "human"]
        print(f"{path}: {g['player']}, model {g['model_color']}, final black {g['black']} white {g['white']}"
              f"{'' if g['finished'] else ' (unfinished)'}, {len(opp)} opponent moves")
        for m in opp:
            board = np.array(m["board_before"], dtype=np.int8) * (1 if m["color"] == "black" else -1)
            legal = legal_moves(board)
            if len(legal) < 2:
                continue
            lg = policy_logits(model, board[None])[0][legal]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            i = int(np.flatnonzero(legal == m["action"])[0])
            rank = int((p > p[i]).sum()) + 1
            scores = engine.analyze_all(board, args.level, len(legal))
            best = max(scores.values())
            regret = best - scores[m["action"]]
            top_regret = best - scores[int(legal[int(np.argmax(p))])]
            empties = int((board == 0).sum())
            rows.append((phase_of(empties), p[i], rank, len(legal), regret, top_regret))
            if args.verbose:
                print(f"  {m['square']:>3} ({empties} empty): policy {100 * p[i]:5.1f}% rank {rank}/{len(legal)}, "
                      f"costs {regret:+.0f} discs (policy's top move would cost {top_regret:+.0f})")
    engine.close()
    if not rows:
        return

    print(f"\nOpponent moves graded by Egaroucid level {args.level} (cost = discs lost vs its best move)")
    print(f"  {'phase':20s} {'moves':>5s} {'unlikely':>8s} {'cost if unlikely':>16s} {'cost otherwise':>14s} "
          f"{'unlikely & good (<=1)':>21s} {'policy top-move cost':>20s}")
    for _, _, name in PHASES:
        r = [x for x in rows if x[0] == name]
        if not r:
            continue
        low = [x for x in r if x[1] < args.low]
        rest = [x for x in r if x[1] >= args.low]
        good_low = sum(1 for x in low if x[4] <= 1)
        mean = lambda xs: f"{np.mean([x[4] for x in xs]):.2f}" if xs else "-"
        print(f"  {name:20s} {len(r):5d} {len(low):8d} {mean(low):>16s} {mean(rest):>14s} "
              f"{good_low:21d} {np.mean([x[5] for x in r]):20.2f}")
    print(f"  (unlikely = policy probability < {100 * args.low:.0f}%)")


if __name__ == "__main__":
    main()
