#!/usr/bin/env python3
"""
MCTS self-play for the self-teacher round (CLAUDE.md step 28): the model plays itself, choosing
every move with MCTS (util/mcts.py), and each position it searched is saved with what the search
concluded, as training targets for the network:
  visits      the root's visit distribution (policy target, AlphaZero's)
  root_value  the root's average value, side to move's view, tanh scale (what the search thinks)
  outcome     the game's final score in discs from that position's side to move (what happened)

Games start from random real-game positions (--start-positions, 8-20 stones) and the first
--temp-moves moves of each game are sampled in proportion to visit counts, for variety; after that
the most visited move is played. Positions with at most --solve-empties empty squares are played by
the exact solver and not recorded, nor are positions with a single legal move.

Workers (-w) are separate processes sharing the GPU; each plays --games games in lockstep, so all its
games' MCTS leaves go through one network call. Output: one .npz with boards (N, 64) int8 (side to
move = 1), visits (N, 64) float32, root_value (N,), outcome (N,), empties (N,), and the settings.

Usage:
  python selfplay_mcts.py -m r256x12_mid1 -n 800000 -o selfplay_r256x12_st1.npz -w 3
"""

import argparse
import os
import time

import numpy as np


def worker(wid, args, target, out_path):
    import torch
    from sb3_contrib import MaskablePPO
    from util.reversi import build_reversi
    from util.search import legal_moves, play_move, policy_value
    from util.mcts import MCTS, final_value
    from util.endgame import solve_async
    from util.util import get_device

    rng = np.random.default_rng(args.seed + 1000 * wid)
    torch.manual_seed(args.seed + wid)
    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.model}_CNN_test", env=build_reversi("Random"), device=device)
    mcts = MCTS(lambda bs: policy_value(model, bs), sims=args.sims, c_puct=args.c, parallel=args.parallel,
                fpu_reduction=args.fpu, leaf_solve_empties=args.leaf_solve_empties)
    starts = np.load(args.start_positions).reshape(-1, 64).astype(np.int8)

    def new_game():
        return {"board": starts[rng.integers(len(starts))].copy(), "color": 1, "plies": 0, "records": []}

    def settle(g):
        """Pass while the side to move has no move; returns False when the game is over."""
        while len(legal_moves(g["board"])) == 0:
            if len(legal_moves(-g["board"])) == 0:
                return False
            g["board"], g["color"] = -g["board"], -g["color"]
        return True

    out = {k: [] for k in ("boards", "visits", "root_value", "outcome")}
    games = [new_game() for _ in range(args.games)]
    done_positions, finished_games, t0 = 0, 0, time.time()
    starting = True
    while games:
        # finish games that are over; start new ones while more positions are needed
        live = []
        for g in games:
            if settle(g):
                live.append(g)
                continue
            final = final_value(g["board"])                       # tanh scale, side to move's view
            own, opp = int((g["board"] == 1).sum()), int((g["board"] == -1).sum())
            score = own - opp + int(np.sign(own - opp)) * (64 - own - opp)
            for board, visits, q, color in g["records"]:
                out["boards"].append(board)
                out["visits"].append(visits)
                out["root_value"].append(q)
                out["outcome"].append(score * color * g["color"])
            done_positions += len(g["records"])
            finished_games += 1
            if starting and done_positions + sum(len(x["records"]) for x in games) < target:
                ng = new_game()
                if settle(ng):
                    live.append(ng)
        games = live
        starting = done_positions + sum(len(x["records"]) for x in games) < target
        if not games:
            break
        late = [g for g in games if (g["board"] == 0).sum() <= args.solve_empties]
        early = [g for g in games if (g["board"] == 0).sum() > args.solve_empties]
        solving = solve_async([g["board"] for g in late]) if late else None
        if early:
            results = mcts.run([g["board"] for g in early])
            for g, (best, visits), root in zip(early, results, mcts.last_roots):
                vec = np.zeros(64, dtype=np.float32)
                for m, n in visits.items():
                    vec[m] = n
                if len(root.moves) > 1 and vec.sum() > 0:
                    q = float(root.w.sum() / root.n.sum())
                    g["records"].append((g["board"].copy(), vec / vec.sum(), q, g["color"]))
                    if g["plies"] < args.temp_moves:
                        best = int(rng.choice(64, p=vec / vec.sum()))
                g["board"], g["color"], g["plies"] = play_move(g["board"], best), -g["color"], g["plies"] + 1
        if late:
            for g, (_, m, _) in zip(late, solving()):
                g["board"], g["color"], g["plies"] = play_move(g["board"], int(m)), -g["color"], g["plies"] + 1
        if wid == 0 and finished_games and finished_games % 50 == 0:
            rate = done_positions / (time.time() - t0)
            print(f"[worker 0] {done_positions:,} positions from {finished_games} games, {rate:.1f}/s", flush=True)

    np.savez(out_path, boards=np.array(out["boards"], dtype=np.int8).reshape(-1, 64),
             visits=np.array(out["visits"], dtype=np.float32).reshape(-1, 64),
             root_value=np.array(out["root_value"], dtype=np.float32),
             outcome=np.array(out["outcome"], dtype=np.float32))
    print(f"[worker {wid}] {done_positions:,} positions from {finished_games} games "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="model name under models/ (no _CNN_test.zip)")
    parser.add_argument("-n", "--num-positions", type=int, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--workers", type=int, default=3)
    parser.add_argument("--games", type=int, default=128, help="games in lockstep per worker")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--fpu", type=float, default=0.3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--leaf-solve-empties", type=int, default=16)
    parser.add_argument("--solve-empties", type=int, default=18)
    parser.add_argument("--temp-moves", type=int, default=8, help="sample this many first moves by visits")
    parser.add_argument("--start-positions", default="start_positions.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    base = os.path.splitext(args.output)[0]
    parts = [f"{base}.part{w}.npz" for w in range(args.workers)]
    per = -(-args.num_positions // args.workers)
    t = time.time()
    procs = [ctx.Process(target=worker, args=(w, args, per, parts[w])) for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode for p in procs):
        raise SystemExit("a worker failed; parts kept: " + ", ".join(x for x in parts if os.path.exists(x)))
    data = [np.load(x) for x in parts]
    merged = {k: np.concatenate([d[k] for d in data]) for k in ("boards", "visits", "root_value", "outcome")}
    np.savez(args.output, **merged, empties=(merged["boards"] == 0).sum(axis=1).astype(np.int8),
             model=args.model, sims=args.sims, c=args.c, fpu=args.fpu, temp_moves=args.temp_moves)
    for x in parts:
        os.remove(x)
    n = len(merged["boards"])
    print(f"Saved {n:,} positions to {args.output} in {(time.time() - t) / 60:.1f} min "
          f"({n / (time.time() - t):.1f}/s)")


if __name__ == "__main__":
    main()
