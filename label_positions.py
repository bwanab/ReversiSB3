#!/usr/bin/env python3
"""
Label positions with Edax: its best move and score, and optionally a score for every legal move.

Positions are sampled (deduplicated) from a BC dataset, where they are stored from the side to
move's perspective (1 = side to move). The script starts its own Edax servers (one per worker,
on separate sockets) so labeling runs in parallel, and stops them when done.

Output (.npz):
  boards      (N, 64) int8   side-to-move perspective
  best_move   (N,)    int16  Edax's move at --depth
  score       (N,)    int16  Edax's score at --depth, discs from the side to move's view
  move_scores (N, 64) float32, only with --every-move: score of each legal move from the side to
              move's view (NaN for illegal moves) = -(Edax score of the resulting position at
              depth - 1), or the exact result if the game ends / the continuation after a pass
  depth, source metadata

Usage:
  python label_positions.py -n 100000 --depth 12 --every-move -o labels_d12_100k.npz
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from util.edax_client import EdaxClient
from util.search import legal_moves, play_move

_client = None


def _init_worker(socket_queue):
    global _client
    _client = EdaxClient(socket_path=socket_queue.get())


def _edax(board, depth):
    """(move, score) for a (64,) board with a legal move for the side to move."""
    r = _client.analyze(board.reshape(8, 8), depth=depth)
    return r["move"], r["score"]


def _exact(board):
    """Final disc difference from the side to move's view (for finished games)."""
    return float((board == 1).sum() - (board == -1).sum())


def _position_score(board, depth):
    """Edax score of `board` from its side to move's view, handling passes and finished games."""
    if len(legal_moves(board)) > 0:
        return float(_edax(board, depth)[1])
    if len(legal_moves(-board)) > 0:          # side to move must pass
        return -float(_edax(-board, depth)[1])
    return _exact(board)


def label_one(args):
    board, depth, every_move = args
    move, score = _edax(board, depth)
    scores = None
    if every_move:
        scores = np.full(64, np.nan, dtype=np.float32)
        for m in legal_moves(board):
            child = play_move(board, m)              # opponent to move
            scores[m] = -_position_score(child, max(depth - 1, 1))
    return move, score, scores


def sample_positions(dataset_file, n, min_stones, max_stones, seed, exclude=()):
    with open(dataset_file, "rb") as f:
        package = pickle.load(f)
    data = package["dataset"] if "dataset" in package else package["moves"]
    states = np.stack([item["state"] for item in data]).astype(np.int8).reshape(len(data), 64)
    stones = (states != 0).sum(axis=1)
    states = np.unique(states[(stones >= min_stones) & (stones <= max_stones)], axis=0)
    if exclude:
        done = set()
        for f in exclude:
            done.update(b.tobytes() for b in np.load(f)["boards"])
        states = states[np.array([s.tobytes() not in done for s in states], dtype=bool)]
        print(f"{len(states):,} positions left after excluding {len(done):,} already labeled")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(states), size=min(n, len(states)), replace=False)
    boards = states[pick]
    has_move = np.array([len(legal_moves(b)) > 0 for b in boards])
    return boards[has_move]


def start_servers(k):
    paths = [f"/tmp/edax_label_{os.getpid()}_{i}.sock" for i in range(k)]
    procs = [subprocess.Popen([sys.executable, "edax_server.py", "--socket", p],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for p in paths]
    deadline = time.time() + 60
    while not all(os.path.exists(p) for p in paths):
        if time.time() > deadline:
            raise RuntimeError("Edax servers did not start")
        time.sleep(0.2)
    return procs, paths


_egaroucid = None


def _init_egaroucid_worker():
    global _egaroucid
    from util.egaroucid_client import EgaroucidClient
    _egaroucid = EgaroucidClient(threads=1)


def label_one_egaroucid(args):
    """Egaroucid labels at `level`: every legal move scored (`hint n`), so best move and score are
    the best of them. Same format as label_one with every_move."""
    board, level, _ = args
    scores = np.full(64, np.nan, dtype=np.float32)
    for m, v in _egaroucid.analyze_all(board, level, len(legal_moves(board))).items():
        scores[m] = v
    best = int(np.nanargmax(scores))
    return best, int(scores[best]), scores


def _run_pool(workers, initializer, initargs, fn, jobs, n, progress):
    t = time.time()
    results = []
    with Pool(workers, initializer=initializer, initargs=initargs) as pool:
        for i, r in enumerate(pool.imap(fn, jobs, chunksize=64)):
            results.append(r)
            if progress and (i + 1) % 10_000 == 0:
                rate = (i + 1) / (time.time() - t)
                print(f"  {i + 1:,} labeled, {rate:.0f}/s, ~{(n - i - 1) / rate / 60:.0f} min left", flush=True)
    return results


def label_boards(boards, depth, every_move, workers, progress=False, teacher="edax"):
    """Label (N, 64) boards (side to move = 1, each with a legal move). teacher "edax": Edax at
    `depth`, using `workers` Edax servers started (and stopped) here; "egaroucid": Egaroucid at level
    `depth`, one process per worker, always scoring every move. Returns a list of
    (best_move, score, move_scores or None), as label_one."""
    jobs = ((b, depth, every_move) for b in boards)
    if teacher == "egaroucid":
        return _run_pool(workers, _init_egaroucid_worker, (), label_one_egaroucid, jobs, len(boards), progress)
    procs, paths = start_servers(workers)
    try:
        from multiprocessing import Manager
        with Manager() as manager:
            queue = manager.Queue()
            for p in paths:
                queue.put(p)
            results = _run_pool(workers, _init_worker, (queue,), label_one, jobs, len(boards), progress)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset", default="combined_bc_dataset.pkl")
    parser.add_argument("-n", "--num-positions", type=int, default=100_000)
    parser.add_argument("--depth", type=int, default=12, help="Edax depth, or Egaroucid level")
    parser.add_argument("--teacher", choices=["edax", "egaroucid"], default="edax",
                        help="egaroucid: Egaroucid for Console (always scores every move; implies --every-move)")
    parser.add_argument("--every-move", action="store_true", help="also score every legal move")
    parser.add_argument("--min-stones", type=int, default=5)
    parser.add_argument("--max-stones", type=int, default=60)
    parser.add_argument("-w", "--workers", type=int, default=10, help="parallel Edax servers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", nargs="*", default=[], help="label files whose positions to skip")
    parser.add_argument("--positions", help="label the (N, 64) boards in this .npy (e.g. from collect_positions.py) "
                                            "instead of sampling --dataset")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.positions:
        boards = np.load(args.positions).astype(np.int8)[:args.num_positions]
        boards = boards[np.array([len(legal_moves(b)) > 0 for b in boards], dtype=bool)]
    else:
        boards = sample_positions(args.dataset, args.num_positions, args.min_stones, args.max_stones, args.seed,
                                  args.exclude)
    if args.teacher == "egaroucid":
        args.every_move = True
    print(f"Labeling {len(boards):,} positions with {args.teacher} at {'level' if args.teacher == 'egaroucid' else 'depth'} "
          f"{args.depth}{' (every move)' if args.every_move else ''}, {args.workers} workers")

    t = time.time()
    results = label_boards(boards, args.depth, args.every_move, args.workers, progress=True, teacher=args.teacher)
    elapsed = time.time() - t

    out = dict(boards=boards,
               best_move=np.array([r[0] for r in results], dtype=np.int16),
               score=np.array([r[1] for r in results], dtype=np.int16),
               depth=args.depth, teacher=args.teacher, source=args.positions or args.dataset)
    if args.every_move:
        out["move_scores"] = np.stack([r[2] for r in results])
    np.savez_compressed(args.output, **out)
    print(f"Saved {len(boards):,} labeled positions to {args.output} "
          f"({elapsed / 60:.1f} min, {len(boards) / elapsed:.0f}/s)")


if __name__ == "__main__":
    main()
