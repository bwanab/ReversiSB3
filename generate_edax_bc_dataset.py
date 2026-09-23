#!/usr/bin/env python3
"""
Generate Behavioral Cloning (BC) dataset from Edax games.

Creates a dataset of (state, action) pairs from Edax playing against
various opponents (Model, Random). Edax plays both Black and White to
ensure balanced perspective coverage.

Can generate at multiple depths and merge with existing datasets (like wthor).

Usage:
    # Generate Edax-7 dataset
    python generate_edax_bc_dataset.py --games 5000 --depths 7 --model current_model --output edax7_dataset.pkl

    # Generate mixed depth dataset (Edax-7, 8, 9)
    python generate_edax_bc_dataset.py --games 6000 --depths 7,8,9 --model current_model --output edax_mixed.pkl

    # Merge with wthor dataset
    python generate_edax_bc_dataset.py --games 5000 --depths 7,8 --model current_model --merge wthor_bc_dataset.pkl --output combined_dataset.pkl
"""

import argparse
import pickle
import numpy as np
from tqdm import tqdm
from util.reversi import ReversiEnvCNN
from util.opponents import get_opponent
from util.util import BLACK, WHITE, get_device

def generate_edax_bc_dataset(
    num_games=10000,
    model_file=None,
    model_ratio=0.7,
    edax_depths=[7],
    net_width=512,
    output_file="edax_bc_dataset.pkl",
    merge_dataset=None,
    verbose=False
):
    """
    Generate BC training dataset from Edax games.

    Parameters
    ----------
    num_games : int
        Number of NEW games to generate
    model_file : str
        Model file to use as opponent (without .zip extension)
    model_ratio : float
        Ratio of games against Model vs Random (0.7 = 70% Model, 30% Random)
    edax_depths : list of int
        List of Edax depths to use (games distributed evenly)
    net_width : int
        Neural network width for loading model opponent
    output_file : str
        Output pickle file path
    merge_dataset : str
        Optional existing dataset to merge with (e.g., wthor_bc_dataset.pkl)
    verbose : bool
        Print game details

    Returns
    -------
    dataset : list
        List of dicts with keys: 'state', 'action', 'color', 'game_num', 'expert_depth'
    """

    print(f"Generating Edax BC dataset:")
    print(f"  New games: {num_games:,}")
    print(f"  Edax depths: {edax_depths}")
    print(f"  Model opponent: {model_file or 'None'}")
    print(f"  Model ratio: {model_ratio:.1%} (Random ratio: {1-model_ratio:.1%})")
    if merge_dataset:
        print(f"  Merging with: {merge_dataset}")
    print(f"  Output: {output_file}")
    print()

    # Load existing dataset if merging
    merged_data = []
    merged_stats = {}
    merged_source = None
    if merge_dataset:
        print(f"Loading existing dataset: {merge_dataset}")
        with open(merge_dataset, 'rb') as f:
            existing = pickle.load(f)

            # Handle different dataset formats
            if 'dataset' in existing:
                # Edax/RAI BC format
                merged_data = existing['dataset']
                merged_stats = existing.get('metadata', {}).get('stats', {})
                merged_source = "BC dataset"
            elif 'moves' in existing:
                # wthor format
                merged_data = existing['moves']
                merged_stats = existing.get('metadata', {})
                merged_source = "wthor dataset"
            else:
                print(f"  WARNING: Unknown dataset format, keys: {existing.keys()}")
                merged_data = []

        print(f"  Loaded {len(merged_data):,} moves from {merged_source}")

        # Show schema of first entry
        if merged_data:
            sample_keys = merged_data[0].keys()
            print(f"  Existing schema: {list(sample_keys)}")
        print()

    # Create environment
    env = ReversiEnvCNN()
    device = get_device()

    # Statistics tracking
    dataset = []
    stats = {
        'games_completed': 0,
        'edax_black_wins': 0,
        'edax_white_wins': 0,
        'edax_black_games': 0,
        'edax_white_games': 0,
        'moves_collected': 0,
        'model_opponent_games': 0,
        'random_opponent_games': 0,
        'depth_distribution': {depth: 0 for depth in edax_depths}
    }

    # Distribute games across depths
    games_per_depth = num_games // len(edax_depths)
    remaining_games = num_games % len(edax_depths)

    # Generate games
    game_num = 0
    for depth_idx, edax_depth in enumerate(edax_depths):
        # Calculate games for this depth
        depth_games = games_per_depth + (1 if depth_idx < remaining_games else 0)
        stats['depth_distribution'][edax_depth] = depth_games

        print(f"\nGenerating {depth_games:,} games at Edax depth {edax_depth}...")

        # Create Edax opponent for this depth
        edax = get_opponent("Edax", depth=edax_depth)

        for i in tqdm(range(depth_games), desc=f"Edax-{edax_depth}"):
            # Alternate Edax color
            edax_plays_black = (game_num % 2 == 0)
            edax_color = BLACK if edax_plays_black else WHITE

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

            # Track which color Edax plays
            if edax_plays_black:
                stats['edax_black_games'] += 1
            else:
                stats['edax_white_games'] += 1

            # Set opponent player attributes
            edax.player = edax_color
            opponent.player = -edax_color

            # Initialize game
            state, info = env.reset()
            done = False
            game_moves = []
            current_player = BLACK  # BLACK always goes first

            while not done:
                # Check if current player has valid moves
                if not env.has_valid(state, current_player):
                    current_player = -current_player
                    if not env.has_valid(state, current_player):
                        done = True
                        break
                    continue

                # Set env.player for get_action
                env.player = current_player

                # Determine which opponent makes the move
                if current_player == edax_color:
                    # Edax's turn - collect this move for training
                    action = edax.get_action(env, state)

                    # Get Edax's evaluation of this position (after search)
                    # Score is in centipawns (disk difference * 100)
                    try:
                        evaluation = edax.edax.get_score() / 100.0  # Convert to disk difference
                    except:
                        evaluation = None  # Fallback if score unavailable

                    # Save (state, action) pair with perspective flipping
                    if edax_color == WHITE:
                        saved_state = state.copy() * -1  # Flip perspective
                    else:
                        saved_state = state.copy()

                    game_moves.append({
                        'state': saved_state,
                        'action': int(action) if isinstance(action, np.ndarray) else action,
                        'color': edax_color,
                        'game_num': game_num,
                        'expert': 'Edax',
                        'expert_depth': edax_depth,
                        'value': evaluation  # Edax's evaluation of the position
                    })
                else:
                    # Opponent's turn - don't collect
                    action = opponent.get_action(env, state)

                # Execute move
                action_int = int(action) if isinstance(action, np.ndarray) else action
                action_tuple = (0, action_int // 8, action_int % 8)

                state = env.get_next_state(state, action_tuple)
                env.board = state

                # Check if game is over
                winner = env.get_winner(state)
                if winner is not None:
                    done = True
                else:
                    current_player = -current_player

            # Record game result
            black_score = np.sum(state[0] == 1)
            white_score = np.sum(state[0] == -1)

            if edax_plays_black and black_score > white_score:
                stats['edax_black_wins'] += 1
            elif not edax_plays_black and white_score > black_score:
                stats['edax_white_wins'] += 1

            # Add game moves to dataset
            dataset.extend(game_moves)
            stats['moves_collected'] += len(game_moves)
            stats['games_completed'] += 1

            if verbose and game_num % 100 == 0:
                print(f"\nGame {game_num}: {'Black' if edax_plays_black else 'White'} "
                      f"vs {'Model' if use_model else 'Random'}, "
                      f"{len(game_moves)} moves collected")

            game_num += 1

    # Combine with merged data if applicable
    total_dataset = merged_data + dataset
    total_moves = len(total_dataset)

    # Print statistics
    print("\n" + "="*60)
    print("Edax BC Dataset Generation Complete")
    print("="*60)
    print(f"NEW games generated: {stats['games_completed']:,}")
    print(f"NEW moves collected: {stats['moves_collected']:,}")
    print(f"Avg moves per game: {stats['moves_collected'] / stats['games_completed']:.1f}")
    print()
    print(f"Edax as Black: {stats['edax_black_games']:,} games, "
          f"{stats['edax_black_wins']:,} wins "
          f"({100 * stats['edax_black_wins'] / stats['edax_black_games']:.1f}%)")
    print(f"Edax as White: {stats['edax_white_games']:,} games, "
          f"{stats['edax_white_wins']:,} wins "
          f"({100 * stats['edax_white_wins'] / stats['edax_white_games']:.1f}%)")
    print()
    print(f"Opponent breakdown:")
    print(f"  Model: {stats['model_opponent_games']:,} games "
          f"({100 * stats['model_opponent_games'] / stats['games_completed']:.1f}%)")
    print(f"  Random: {stats['random_opponent_games']:,} games "
          f"({100 * stats['random_opponent_games'] / stats['games_completed']:.1f}%)")
    print()
    print(f"Depth distribution:")
    for depth in sorted(edax_depths):
        print(f"  Edax-{depth}: {stats['depth_distribution'][depth]:,} games")

    if merge_dataset:
        print()
        print(f"MERGED DATASET:")
        print(f"  Previous moves: {len(merged_data):,}")
        print(f"  New moves: {stats['moves_collected']:,}")
        print(f"  TOTAL moves: {total_moves:,}")

    print("="*60)

    # Save dataset with metadata
    dataset_package = {
        'dataset': total_dataset,  # Always use 'dataset' key for consistency
        'metadata': {
            'num_games': num_games,
            'edax_depths': edax_depths,
            'model_file': model_file,
            'model_ratio': model_ratio,
            'merged_from': merge_dataset,
            'merged_source': merged_source,
            'new_games_stats': stats,
            'previous_stats': merged_stats,
            'total_moves': total_moves,
            'format_note': 'Combined dataset may have mixed schemas (wthor + Edax). BC training uses state/action only.'
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(dataset_package, f)

    print(f"\nDataset saved to: {output_file}")
    print(f"File size: {len(pickle.dumps(dataset_package)) / 1024 / 1024:.1f} MB")

    return total_dataset, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate BC dataset from Edax games',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-g', '--games', type=int, default=5000,
                        help='Number of NEW games to generate')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model file to use as opponent (without .zip extension)')
    parser.add_argument('-r', '--model-ratio', type=float, default=0.7,
                        help='Ratio of games against Model (0.7 = 70%% Model, 30%% Random)')
    parser.add_argument('-d', '--depths', type=str, default='7',
                        help='Edax depths (comma-separated, e.g., "7,8,9")')
    parser.add_argument('-w', '--net-width', type=int, default=512,
                        help='Neural network width for model opponent')
    parser.add_argument('-o', '--output', type=str, default='edax_bc_dataset.pkl',
                        help='Output pickle file')
    parser.add_argument('--merge', type=str, default=None,
                        help='Existing dataset to merge with (e.g., wthor_bc_dataset.pkl)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    # Parse depths
    try:
        edax_depths = [int(d.strip()) for d in args.depths.split(',')]
    except ValueError:
        parser.error("--depths must be comma-separated integers (e.g., '7,8,9')")

    # Validate arguments
    if args.model_ratio < 0 or args.model_ratio > 1:
        parser.error("--model-ratio must be between 0 and 1")

    if args.model is None and args.model_ratio > 0:
        print("WARNING: --model not specified but --model-ratio > 0")
        print("         All games will use Random opponent")
        args.model_ratio = 0

    # Generate dataset
    generate_edax_bc_dataset(
        num_games=args.games,
        model_file=args.model,
        model_ratio=args.model_ratio,
        edax_depths=edax_depths,
        net_width=args.net_width,
        output_file=args.output,
        merge_dataset=args.merge,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
