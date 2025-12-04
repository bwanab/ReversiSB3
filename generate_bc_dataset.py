#!/usr/bin/env python3
"""
Generate Behavioral Cloning (BC) dataset from RAI games.

Creates a dataset of (state, action) pairs from RAI playing against
various opponents (Model, Random). RAI plays both Black and White to
ensure balanced perspective coverage.

Usage:
    python generate_bc_dataset.py --games 10000 --model best_model --output bc_dataset.pkl
"""

import argparse
import pickle
import numpy as np
from tqdm import tqdm
from util.reversi import ReversiEnvCNN
from util.opponents import get_opponent
from util.util import BLACK, WHITE, get_device

def generate_bc_dataset(
    num_games=10000,
    model_file=None,
    model_ratio=0.7,
    rai_depth=2,
    net_width=512,
    output_file="bc_dataset.pkl",
    verbose=False
):
    """
    Generate BC training dataset from RAI games.

    Parameters
    ----------
    num_games : int
        Number of games to generate
    model_file : str
        Model file to use as opponent (without .zip extension)
    model_ratio : float
        Ratio of games against Model vs Random (0.7 = 70% Model, 30% Random)
    rai_depth : int
        RAI search depth (default: 2)
    net_width : int
        Neural network width for loading model opponent
    output_file : str
        Output pickle file path
    verbose : bool
        Print game details

    Returns
    -------
    dataset : list
        List of dicts with keys: 'state', 'action', 'color', 'game_num'
    """

    print(f"Generating BC dataset:")
    print(f"  Games: {num_games:,}")
    print(f"  RAI depth: {rai_depth}")
    print(f"  Model opponent: {model_file or 'None'}")
    print(f"  Model ratio: {model_ratio:.1%} (Random ratio: {1-model_ratio:.1%})")
    print(f"  Output: {output_file}")
    print()

    # Create environment
    env = ReversiEnvCNN()
    device = get_device()

    # Create RAI opponent
    rai = get_opponent("RAI", depth=rai_depth)

    # Statistics tracking
    dataset = []
    stats = {
        'games_completed': 0,
        'rai_black_wins': 0,
        'rai_white_wins': 0,
        'rai_black_games': 0,
        'rai_white_games': 0,
        'moves_collected': 0,
        'model_opponent_games': 0,
        'random_opponent_games': 0
    }

    # Generate games
    for game_num in tqdm(range(num_games), desc="Generating games"):
        # Alternate RAI color
        rai_plays_black = (game_num % 2 == 0)
        rai_color = BLACK if rai_plays_black else WHITE

        # Choose opponent based on model_ratio
        use_model = (model_file is not None) and (np.random.random() < model_ratio)

        if use_model:
            opponent = get_opponent(
                "Model",
                opponent_model=model_file,
                env=env,
                net_width=net_width,
                verbose=False
            )
            stats['model_opponent_games'] += 1
        else:
            opponent = get_opponent("Random")
            stats['random_opponent_games'] += 1

        # Track which color RAI plays
        if rai_plays_black:
            stats['rai_black_games'] += 1
        else:
            stats['rai_white_games'] += 1

        # Play game using low-level environment methods (bypass env.step())
        # This allows opponent-vs-opponent play without RL training constraints

        # Set opponent player attributes (needed for their get_action methods)
        rai.player = rai_color
        opponent.player = -rai_color

        # Initialize game
        state, info = env.reset()
        done = False
        game_moves = []
        current_player = BLACK  # BLACK always goes first in Reversi

        while not done:
            # Check if current player has valid moves
            if not env.has_valid(state, current_player):
                # No valid moves - pass to other player
                current_player = -current_player

                # Check if other player also has no valid moves
                if not env.has_valid(state, current_player):
                    # Neither player has moves - game over
                    done = True
                    break
                continue

            # Set env.player BEFORE calling get_action (opponents need this)
            env.player = current_player

            # Determine which opponent makes the move
            if current_player == rai_color:
                # RAI's turn - collect this move for training
                action = rai.get_action(env, state)

                # Save (state, action) pair
                # If RAI is WHITE, flip board state to represent WHITE's perspective
                if rai_color == WHITE:
                    saved_state = state.copy() * -1  # Flip perspective
                else:
                    saved_state = state.copy()

                game_moves.append({
                    'state': saved_state,
                    'action': int(action) if isinstance(action, np.ndarray) else action,
                    'color': rai_color,
                    'game_num': game_num
                })
            else:
                # Opponent's turn - don't collect
                action = opponent.get_action(env, state)

            # Execute move using environment's low-level method
            # Convert action to tuple format (channel, row, col)
            action_int = int(action) if isinstance(action, np.ndarray) else action
            action_tuple = (0, action_int // 8, action_int % 8)

            # Update board state directly (bypass step())
            # env.player already set above for get_action()
            state = env.get_next_state(state, action_tuple)
            env.board = state  # Update environment's internal board

            # Check if game is over
            winner = env.get_winner(state)
            if winner is not None:
                done = True
            else:
                # Switch to next player
                current_player = -current_player

        # Record game result (from RAI's perspective)
        # reward is from current player's perspective, but we need to check who won
        # Get final scores
        black_score = np.sum(state[0] == 1)
        white_score = np.sum(state[0] == -1)

        if rai_plays_black and black_score > white_score:
            stats['rai_black_wins'] += 1
        elif not rai_plays_black and white_score > black_score:
            stats['rai_white_wins'] += 1

        # Add game moves to dataset
        dataset.extend(game_moves)
        stats['moves_collected'] += len(game_moves)
        stats['games_completed'] += 1

        if verbose and game_num % 100 == 0:
            print(f"\nGame {game_num}: {'Black' if rai_plays_black else 'White'} "
                  f"vs {'Model' if use_model else 'Random'}, "
                  f"{len(game_moves)} moves collected")

    # Print statistics
    print("\n" + "="*60)
    print("BC Dataset Generation Complete")
    print("="*60)
    print(f"Total games: {stats['games_completed']:,}")
    print(f"Total moves collected: {stats['moves_collected']:,}")
    print(f"Avg moves per game: {stats['moves_collected'] / stats['games_completed']:.1f}")
    print()
    print(f"RAI as Black: {stats['rai_black_games']:,} games, "
          f"{stats['rai_black_wins']:,} wins "
          f"({100 * stats['rai_black_wins'] / stats['rai_black_games']:.1f}%)")
    print(f"RAI as White: {stats['rai_white_games']:,} games, "
          f"{stats['rai_white_wins']:,} wins "
          f"({100 * stats['rai_white_wins'] / stats['rai_white_games']:.1f}%)")
    print()
    print(f"Opponent breakdown:")
    print(f"  Model: {stats['model_opponent_games']:,} games "
          f"({100 * stats['model_opponent_games'] / stats['games_completed']:.1f}%)")
    print(f"  Random: {stats['random_opponent_games']:,} games "
          f"({100 * stats['random_opponent_games'] / stats['games_completed']:.1f}%)")
    print("="*60)

    # Save dataset with metadata
    dataset_package = {
        'dataset': dataset,
        'metadata': {
            'num_games': num_games,
            'rai_depth': rai_depth,
            'model_file': model_file,
            'model_ratio': model_ratio,
            'stats': stats
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(dataset_package, f)

    print(f"\nDataset saved to: {output_file}")
    print(f"File size: {len(pickle.dumps(dataset_package)) / 1024 / 1024:.1f} MB")

    return dataset, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate BC dataset from RAI games',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-g', '--games', type=int, default=10000,
                        help='Number of games to generate')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model file to use as opponent (without .zip extension)')
    parser.add_argument('-r', '--model-ratio', type=float, default=0.7,
                        help='Ratio of games against Model (0.7 = 70%% Model, 30%% Random)')
    parser.add_argument('-d', '--rai-depth', type=int, default=2,
                        help='RAI minimax search depth')
    parser.add_argument('-w', '--net-width', type=int, default=512,
                        help='Neural network width for model opponent')
    parser.add_argument('-o', '--output', type=str, default='bc_dataset.pkl',
                        help='Output pickle file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    # Validate arguments
    if args.model_ratio < 0 or args.model_ratio > 1:
        parser.error("--model-ratio must be between 0 and 1")

    if args.model is None and args.model_ratio > 0:
        print("WARNING: --model not specified but --model-ratio > 0")
        print("         All games will use Random opponent")
        args.model_ratio = 0

    # Generate dataset
    generate_bc_dataset(
        num_games=args.games,
        model_file=args.model,
        model_ratio=args.model_ratio,
        rai_depth=args.rai_depth,
        net_width=args.net_width,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
