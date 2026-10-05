#!/usr/bin/env python3
"""
Collect the positions a model reaches in its own games (for DAgger: label them with Edax and
add them to the training data). The model either samples its moves from its policy or, with
--search-depth, plays like the searching player (pruned search, with --explore randomness).

Games run in lockstep so the model's moves for all of them go through one batched network call.
The model samples its moves from its policy (for variety). Opponents per game: the model itself
(self-play; both sides' positions are recorded) or Edax at a given depth. Games start from N-ply
random openings or from positions in an .npy (e.g. opening_positions.npy).

Every position where the model is to move is recorded from its side-to-move perspective
(1 = side to move); the output is the deduplicated (N, 64) int8 array, minus any positions in
--exclude label files.

Usage:
  python collect_positions.py -m edax2m_graded -n 1000000 -o dagger_positions.npy \\
      --exclude labels_d12_*.npz
"""

import argparse
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.search import legal_moves, play_move, SearchPlayer, policy_top_moves
from util.edax_client import EdaxClient
from util.util import get_device


def standard_board():
    b = np.zeros(64, dtype=np.int8)
    b[27] = b[36] = -1          # d4, e5 WHITE
    b[28] = b[35] = 1           # e4, d5 BLACK (to move)
    return b


def random_opening(plies, rng):
    """Board (side to move = 1) after `plies` random moves from the standard start, or None if
    the game ended."""
    b = standard_board()
    played = 0
    while played < plies:
        moves = legal_moves(b)
        if len(moves) == 0:
            if len(legal_moves(-b)) == 0:
                return None
            b = -b
            continue
        b = play_move(b, int(rng.choice(moves)))
        played += 1
    return b


class Game:
    def __init__(self, board, opponent, model_to_move):
        self.board = board              # side to move = 1
        self.opponent = opponent        # "self" or an Edax depth
        self.model_to_move = model_to_move


def search_moves(player, model, boards, k, explore, rng):
    """The search's move for each board; with probability `explore` a random one of the policy's
    top k instead (variety, since the search is deterministic)."""
    moves = player.choose_many(boards)
    explorers = [i for i in range(len(boards)) if rng.random() < explore]
    if explorers:
        for i, top in zip(explorers, policy_top_moves(model, [boards[i] for i in explorers], k)):
            moves[i] = int(rng.choice(top))
    return moves


def sample_moves(policy, boards, rng):
    """Sample one move per board from the policy (masked to legal moves)."""
    obs = torch.as_tensor(np.stack(boards).reshape(-1, 1, 8, 8), dtype=torch.float32, device=policy.device)
    masks = np.zeros((len(boards), 64), dtype=bool)
    for i, b in enumerate(boards):
        masks[i, legal_moves(b)] = True
    with torch.no_grad():
        probs = policy.get_distribution(obs, action_masks=masks).distribution.probs.cpu().numpy()
    return [int(rng.choice(64, p=p / p.sum())) for p in probs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="model name under models/ (no _CNN_test.zip)")
    parser.add_argument("-n", "--num-positions", type=int, default=1_000_000, help="distinct positions to collect")
    parser.add_argument("--opponents", default="self:0.4,1:0.2,2:0.2,3:0.2",
                        help="opponent mix: 'self' or an Edax depth, with weights")
    parser.add_argument("--random-plies", type=int, default=8)
    parser.add_argument("--start-positions", default="opening_positions.npy",
                        help="half the games start here (the other half from random openings); '' for none")
    parser.add_argument("--parallel-games", type=int, default=256)
    parser.add_argument("--exclude", nargs="*", default=[], help="label files whose positions to skip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu")
    parser.add_argument("--search-depth", type=int, default=0,
                        help="choose the model's moves by search (pruned, --search-top-k at every level) "
                             "instead of sampling from the policy; 0 = sample")
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--explore", type=float, default=0.2,
                        help="with --search-depth: chance per move of a random policy top-k move instead")
    parser.add_argument("--min-empties", type=int, default=0, help="only record positions with at least this many empties")
    parser.add_argument("--max-empties", type=int, default=64, help="only record positions with at most this many empties")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = MaskablePPO.load(f"models/{args.model}_CNN_test", env=build_reversi("Random"),
                             device=get_device() if args.device == "auto" else args.device)
    policy = model.policy
    policy.set_training_mode(False)
    player = SearchPlayer(model, depth=args.search_depth, top_k=args.search_top_k, prune_all=True) \
        if args.search_depth > 0 else None
    edax = EdaxClient()
    opp_names, opp_weights = zip(*[(o.split(":")[0], float(o.split(":")[1])) for o in args.opponents.split(",")])
    opp_weights = np.array(opp_weights) / sum(opp_weights)
    named = np.load(args.start_positions) if args.start_positions else None

    excluded = set()
    for f in args.exclude:
        excluded.update(b.tobytes() for b in np.load(f)["boards"])
    seen = set()

    def new_game():
        while True:
            if named is not None and rng.random() < 0.5:
                board = named[rng.integers(len(named))].copy()
            else:
                board = random_opening(args.random_plies, rng)
            if board is not None and (len(legal_moves(board)) or len(legal_moves(-board))):
                break
        opp = opp_names[rng.choice(len(opp_names), p=opp_weights)]
        opponent = "self" if opp == "self" else int(opp)
        # against Edax the model takes either side with equal probability
        return Game(board, opponent, model_to_move=(opponent == "self" or rng.random() < 0.5))

    games = [new_game() for _ in range(args.parallel_games)]
    t, n_games = time.time(), 0
    while len(seen) < args.num_positions:
        # resolve passes / finished games
        for i, g in enumerate(games):
            while True:
                if len(legal_moves(g.board)):
                    break
                if len(legal_moves(-g.board)) == 0:      # game over
                    games[i] = g = new_game()
                    n_games += 1
                    continue
                g.board = -g.board                       # forced pass
                if g.opponent != "self":
                    g.model_to_move = not g.model_to_move
        model_games = [g for g in games if g.model_to_move or g.opponent == "self"]
        for g in model_games:
            key = g.board.tobytes()
            if key not in excluded and args.min_empties <= (g.board == 0).sum() <= args.max_empties:
                seen.add(key)
        boards = [g.board for g in model_games]
        chosen = search_moves(player, model, boards, args.search_top_k, args.explore, rng) if player \
            else sample_moves(policy, boards, rng)
        for g, m in zip(model_games, chosen):
            g.board = play_move(g.board, m)
            if g.opponent != "self":
                g.model_to_move = False
        for g in games:
            if g.opponent != "self" and not g.model_to_move and len(legal_moves(g.board)):
                move = edax.analyze(g.board.reshape(8, 8), depth=g.opponent)["move"]
                g.board = play_move(g.board, move)
                g.model_to_move = True
        if n_games and n_games % 2000 < args.parallel_games // 8:
            rate = len(seen) / (time.time() - t)
            print(f"  {len(seen):,} positions from ~{n_games:,} games, {rate:.0f}/s", flush=True)

    out = np.frombuffer(b"".join(list(seen)[:args.num_positions]), dtype=np.int8).reshape(-1, 64)
    np.save(args.output, out)
    stones = (out != 0).sum(axis=1)
    print(f"Saved {len(out):,} positions ({stones.min()}-{stones.max()} stones, mean {stones.mean():.0f}) "
          f"from ~{n_games:,} games to {args.output} in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
