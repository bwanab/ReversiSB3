
import gymnasium as gym
import boardgame2
from util.util import get_opponent, play

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def play_games(file, num_games=100, verbose=False, opponentName="Random", deterministic=False):
    env = gym.make("Reversi-v0")
    opponent = get_opponent(opponentName, file=file, env=env)

    model = MaskablePPO.load(file, env=env)
    model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    black_wins = play(model, num_games, opponent, deterministic, verbose)
    return black_wins

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-e", "--episodes", default=100)
    parser.add_argument("-m", "--model", default = "reversi_ppo_5layer_03LR_512")
    parser.add_argument("-o", "--opponent", default="Human")
    parser.add_argument("-d", "--deterministic", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    n_games = int(args.episodes)
    # black_wins = play_games("models/" + args.model, n_games, verbose=True, opponentName=args.opponent,deterministic=args.deterministic)
    black_wins = play_games("models/" + args.model, n_games, verbose=bool(args.verbose), opponentName=args.opponent,deterministic=args.deterministic)
    print(f"Black wins: {100 * black_wins / n_games}%")
