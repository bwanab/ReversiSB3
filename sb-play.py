
import gymnasium as gym
import numpy as np
from util.reversi import build_reversi
from util.play import play
from util.search import SearchPlayer
from util.opponents import get_opponent

import torch
torch.device("cpu") # torch.device("mps")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def play_games(file, 
               num_games=100, 
               verbose=False, 
               opponentName="Random", 
               deterministic=False, 
               opponent_model = None,
               depth=2,
               random_opening=0,
               seed=None,
               start_positions=None,
               search_depth=0,
               search_top_k=None):
    env = build_reversi(opponent=opponentName, verbose=verbose, opponent_model=opponent_model, depth=depth)
    #opponent = get_opponent(opponentName, file=file, env=env)

    model = MaskablePPO.load(file, env=env)
    # model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    chooser = SearchPlayer(model, depth=search_depth, top_k=search_top_k) if search_depth > 0 else None
    start_sequence = np.load(start_positions) if start_positions else None
    return play(model, num_games, None, deterministic, verbose,
                random_opening_plies=random_opening, seed=seed, return_draws=True, chooser=chooser,
                start_sequence=start_sequence)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-e", "--episodes", default=10)
    parser.add_argument("-m", "--model", default = "dorkS_CNN_test")
    parser.add_argument("-r", "--opp_model", default = "models/current_best")
    parser.add_argument("-o", "--opponent", default="Model")
    parser.add_argument("-d", "--non_deterministic", action='store_true')
    parser.add_argument("-p", "--depth", default="2", help="depth of min/max search when RAI is opponent")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--random-opening", type=int, default=0,
                        help="start each game with this many random plies (both sides) to vary positions")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for --random-opening / --start-positions, so different models face the same openings")
    parser.add_argument("--start-positions", type=str, default=None,
                        help="start game i from position i of this .npy (e.g. opening_positions.npy from "
                             "make_opening_positions.py), in order, cycling if -e exceeds the count; model to move")
    parser.add_argument("--search-depth", type=int, default=0,
                        help="choose the model's moves by minimax this many plies deep, scored by its value "
                             "head (0 = policy head as usual). Not used with -o Model")
    parser.add_argument("--search-top-k", type=int, default=None,
                        help="with --search-depth, search only the policy's top k moves at the root")
    args = parser.parse_args()

    import time
    n_games = int(args.episodes)
    depth = int(args.depth)
    # black_wins = play_games("models/" + args.model, n_games, verbose=True, opponentName=args.opponent,deterministic=args.deterministic)
    start = time.time()
    deterministic = not bool(args.non_deterministic)
    if args.opponent == "Model":
        # Each half plays the same sequence of starting positions (same seed) with the colors
        # swapped, so opening luck largely cancels out.
        half = n_games // 2
        opening = dict(random_opening=args.random_opening, seed=args.seed, start_positions=args.start_positions)
        wins_as_black, draws1 = play_games("models/" + args.model,
                                half,
                                verbose=bool(args.verbose),
                                opponentName=args.opponent,
                                deterministic=deterministic,
                                opponent_model=args.opp_model,
                                depth=depth, **opening)
        opp_wins_as_black, draws2 = play_games(args.opp_model,
                                half,
                                verbose=bool(args.verbose),
                                opponentName=args.opponent,
                                deterministic=deterministic,
                                opponent_model="models/" + args.model,
                                depth=depth, **opening)
        wins_as_white = half - opp_wins_as_black - draws2
        draws = draws1 + draws2
        wins = wins_as_black + wins_as_white
        score = (wins + 0.5 * draws) / (2 * half)
        print(f"Model: {wins} wins ({wins_as_black} as BLACK, {wins_as_white} as WHITE), "
              f"{2 * half - wins - draws} losses, {draws} draws; score {100 * score:.1f}% "
              f"(draws count 1/2; 50% = even)")

    else:
        black_wins, draws = play_games("models/" + args.model,
                                n_games, 
                                verbose=bool(args.verbose), 
                                opponentName=args.opponent,
                                deterministic=deterministic, 
                                opponent_model=args.opp_model,
                                depth=depth,
                                random_opening=args.random_opening,
                                seed=args.seed,
                                start_positions=args.start_positions,
                                search_depth=args.search_depth,
                                search_top_k=args.search_top_k)
        print(f"Black wins: {100 * black_wins / n_games}%")

    end = time.time()
    print(f"time elapsed: {end - start}")

