
import gymnasium as gym
from util.reversi import build_reversi
from util.play import play
from util.opponents import get_opponent

import torch
torch.device("cpu") # torch.device("mps")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def play_games(file, num_games=100, verbose=False, opponentName="Random", deterministic=False, opponent_model = None):
    env = build_reversi(opponent=opponentName, verbose=verbose, opponent_model=opponent_model)
    #opponent = get_opponent(opponentName, file=file, env=env)

    model = MaskablePPO.load(file, env=env)
    # model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    black_wins = play(model, num_games, None, deterministic, verbose)
    return black_wins

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-e", "--episodes", default=10)
    parser.add_argument("-m", "--model", default = "dorkF_CNN_test")
    parser.add_argument("-r", "--opp_model", default = "dorkE_CNN_test")
    parser.add_argument("-o", "--opponent", default="Model")
    parser.add_argument("-d", "--non_deterministic", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    import time
    n_games = int(args.episodes)
    # black_wins = play_games("models/" + args.model, n_games, verbose=True, opponentName=args.opponent,deterministic=args.deterministic)
    start = time.time()
    deterministic = not bool(args.non_deterministic)
    black_wins = play_games("models/" + args.model,
                            n_games, 
                            verbose=bool(args.verbose), 
                            opponentName=args.opponent,
                            deterministic=deterministic, 
                            opponent_model=args.opp_model)
    end = time.time()
    print(f"time elapsed: {end - start}")
    print(f"Black wins: {100 * black_wins / n_games}%")
