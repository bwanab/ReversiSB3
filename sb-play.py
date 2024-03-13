
import gymnasium as gym
import boardgame2
from util.util import get_opponent, play

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def play_games(file, num_games=100, verbose=False, opponentName="Random"):
    env = gym.make("Reversi-v0")
    opponent = get_opponent(opponentName, file, env)

    model = MaskablePPO.load(file, env=env)
    model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')
    black_wins = play(model, env, num_games, opponent, verbose)
    return black_wins

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'train',
                    description = 'meant to train a reversi ml, current just doing tests',
                    epilog = 'Text at the bottom of help')

    parser.add_argument("-e", "--episodes", default=100)
    parser.add_argument("-m", "--model", default = "ppo_reversi_256")
    parser.add_argument("-o", "--opponent", default="Model")
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    n_games = int(args.episodes)
    black_wins = play_games(args.model, n_games, verbose=bool(args.verbose), opponentName=args.opponent)
    print(f"Black wins: {100 * black_wins / n_games}%")
