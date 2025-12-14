#!/usr/bin/env python3
"""
Parse WThor PGN files into BC training dataset.

Reads Othello/Reversi games in PGN format from WThor database and converts
them into (state, action) pairs for behavioral cloning training.
"""

import os
import re
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from util.reversi import ReversiEnvCNN
from util.util import BLACK, WHITE


def notation_to_action(move_str):
    """
    Convert algebraic notation to action number.

    Examples:
        F5 → column F (5), row 5 → (row=4, col=5) → action=4*8+5=37
        A1 → column A (0), row 1 → (row=0, col=0) → action=0
        H8 → column H (7), row 8 → (row=7, col=7) → action=63

    Parameters
    ----------
    move_str : str
        Move in algebraic notation (e.g., "F5", "A1")

    Returns
    -------
    int
        Action number (0-63)
    """
    move_str = move_str.strip().upper()

    # Column letter A-H → 0-7
    col = ord(move_str[0]) - ord('A')

    # Row number 1-8 → 0-7
    row = int(move_str[1]) - 1

    # Convert to action number
    action = row * 8 + col

    return action


def parse_pgn_file(pgn_path):
    """
    Parse a single PGN file and extract games.

    Parameters
    ----------
    pgn_path : str
        Path to PGN file

    Returns
    -------
    list
        List of games, where each game is a dict with metadata and moves
    """
    games = []
    current_game = None

    with open(pgn_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                if current_game and current_game.get('moves'):
                    games.append(current_game)
                    current_game = None
                continue

            # Parse metadata
            if line.startswith('['):
                if current_game is None:
                    current_game = {'metadata': {}, 'moves': []}

                # Extract key-value from [Key "Value"]
                match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
                if match:
                    key, value = match.groups()
                    current_game['metadata'][key] = value

            # Parse move line
            elif current_game is not None:
                # Format: "move_number. BLACK_MOVE WHITE_MOVE"
                # Example: "1. F5 D6"
                match = re.match(r'(\d+)\.\s+(\w+)(?:\s+(\w+))?', line)
                if match:
                    move_num, black_move, white_move = match.groups()
                    current_game['moves'].append({
                        'number': int(move_num),
                        'black': black_move,
                        'white': white_move  # May be None if game ends
                    })

        # Don't forget last game
        if current_game and current_game.get('moves'):
            games.append(current_game)

    return games


def parse_game_outcome(result_str):
    """
    Parse game result string to get scores.

    Parameters
    ----------
    result_str : str
        Result in format "black_score-white_score" (e.g., "22-42")

    Returns
    -------
    tuple
        (black_score, white_score) or (None, None) if invalid
    """
    match = re.match(r'(\d+)-(\d+)', result_str)
    if match:
        black_score = int(match.group(1))
        white_score = int(match.group(2))
        return black_score, white_score
    return None, None


def compute_outcome_value(color, black_score, white_score):
    """
    Compute outcome value for a given player color.

    Returns value from the perspective of the given color:
    - Positive if that color won
    - Negative if that color lost
    - Zero for draw

    Using score difference normalized to [-1, 1] range.

    Parameters
    ----------
    color : int
        BLACK (1) or WHITE (-1)
    black_score : int
        Final score for Black
    white_score : int
        Final score for White

    Returns
    -------
    float
        Outcome value in range [-1, 1]
    """
    # Score difference from this player's perspective
    if color == BLACK:
        diff = black_score - white_score
    else:  # WHITE
        diff = white_score - black_score

    # Normalize to [-1, 1] range
    # Maximum difference is 64 (one player has all pieces)
    outcome = diff / 64.0

    # Clamp to [-1, 1] just in case
    outcome = max(-1.0, min(1.0, outcome))

    return outcome


def game_to_training_data(game, verbose=False):
    """
    Convert a game into (state, action, outcome) training pairs.

    Parameters
    ----------
    game : dict
        Game dict with 'metadata' and 'moves'
    verbose : bool
        Print warnings for invalid moves

    Returns
    -------
    list
        List of dicts with 'state', 'action', 'color', 'game_metadata', 'outcome'
    """
    env = ReversiEnvCNN()
    state, _ = env.reset()

    training_data = []
    game_id = f"{game['metadata'].get('Event', 'Unknown')} - {game['metadata'].get('Black', '?')} vs {game['metadata'].get('White', '?')}"

    # Parse game outcome
    result_str = game['metadata'].get('Result', '32-32')  # Default to draw if missing
    black_score, white_score = parse_game_outcome(result_str)
    if black_score is None:
        # Invalid result, skip game
        if verbose:
            print(f"Warning: Invalid result '{result_str}' in game: {game_id}")
        return []

    for move_data in game['moves']:
        # Black's move
        if move_data['black']:
            try:
                action = notation_to_action(move_data['black'])

                # Verify this is a valid move
                valid_moves = env.get_valid(state).flatten()
                action_tuple = (0, action // 8, action % 8)

                if valid_moves[action] == 0:
                    if verbose:
                        print(f"Warning: Invalid BLACK move {move_data['black']} in game: {game_id}")
                    return []  # Skip this game if invalid move found

                # Save (state, action, outcome) from BLACK's perspective
                outcome_value = compute_outcome_value(BLACK, black_score, white_score)
                training_data.append({
                    'state': state.copy(),
                    'action': action,
                    'color': BLACK,
                    'game_metadata': game_id,
                    'outcome': outcome_value
                })

                # Execute Black's move
                state = env.get_next_state(state, action_tuple)
                env.board = state

            except Exception as e:
                if verbose:
                    print(f"Error processing BLACK move {move_data['black']}: {e}")
                return []

        # White's move (may be None if game ended after Black's move)
        if move_data['white']:
            try:
                action = notation_to_action(move_data['white'])

                # Verify this is a valid move
                valid_moves = env.get_valid(state).flatten()
                action_tuple = (0, action // 8, action % 8)

                if valid_moves[action] == 0:
                    if verbose:
                        print(f"Warning: Invalid WHITE move {move_data['white']} in game: {game_id}")
                    return []  # Skip this game if invalid move found

                # Save (state, action, outcome) from WHITE's perspective
                # Flip board so WHITE sees itself as positive
                outcome_value = compute_outcome_value(WHITE, black_score, white_score)
                training_data.append({
                    'state': state.copy() * -1,  # Flip perspective
                    'action': action,
                    'color': WHITE,
                    'game_metadata': game_id,
                    'outcome': outcome_value
                })

                # Execute White's move
                state = env.get_next_state(state, action_tuple)
                env.board = state

            except Exception as e:
                if verbose:
                    print(f"Error processing WHITE move {move_data['white']}: {e}")
                return []

    return training_data


def parse_wthor_dataset(pgn_dir, max_games=None, year_range=None, verbose=False):
    """
    Parse all WThor PGN files into BC training dataset.

    Parameters
    ----------
    pgn_dir : str
        Directory containing WTHOR-*.pgn files
    max_games : int, optional
        Maximum number of games to process (for testing)
    year_range : tuple, optional
        (min_year, max_year) to filter games by year
    verbose : bool
        Print detailed progress

    Returns
    -------
    dict
        Dataset dict with 'moves' and 'metadata'
    """
    # Find all PGN files
    pgn_files = sorted([
        os.path.join(pgn_dir, f)
        for f in os.listdir(pgn_dir)
        if f.startswith('WTHOR-') and f.endswith('.pgn')
    ])

    # Filter by year if specified
    if year_range:
        min_year, max_year = year_range
        pgn_files = [
            f for f in pgn_files
            if min_year <= int(f.split('WTHOR-')[1].split('.')[0]) <= max_year
        ]

    print(f"Found {len(pgn_files)} PGN files to process")

    all_training_data = []
    total_games = 0
    skipped_games = 0

    for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
        year = os.path.basename(pgn_path).split('-')[1].split('.')[0]
        games = parse_pgn_file(pgn_path)

        for game in games:
            if max_games and total_games >= max_games:
                break

            training_data = game_to_training_data(game, verbose=verbose)

            if training_data:
                all_training_data.extend(training_data)
                total_games += 1
            else:
                skipped_games += 1

        if max_games and total_games >= max_games:
            break

    # Count by color
    black_moves = sum(1 for m in all_training_data if m['color'] == BLACK)
    white_moves = sum(1 for m in all_training_data if m['color'] == WHITE)

    dataset = {
        'moves': all_training_data,
        'metadata': {
            'source': 'WThor PGN database',
            'total_games': total_games,
            'skipped_games': skipped_games,
            'total_moves': len(all_training_data),
            'black_moves': black_moves,
            'white_moves': white_moves,
            'years': year_range if year_range else 'all',
            'pgn_files_processed': len(pgn_files)
        }
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Parse WThor PGN files into BC dataset')
    parser.add_argument('-d', '--pgn-dir', type=str, default='../othello-games/pgn',
                        help='Directory containing WTHOR-*.pgn files')
    parser.add_argument('-o', '--output', type=str, default='wthor_bc_dataset.pkl',
                        help='Output pickle file for BC dataset')
    parser.add_argument('-g', '--max-games', type=int, default=None,
                        help='Maximum number of games to process (for testing)')
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum year to include')
    parser.add_argument('--max-year', type=int, default=None,
                        help='Maximum year to include')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress and warnings')

    args = parser.parse_args()

    # Determine year range
    year_range = None
    if args.min_year or args.max_year:
        year_range = (
            args.min_year if args.min_year else 1977,
            args.max_year if args.max_year else 2019
        )

    print(f"{'='*60}")
    print(f"WThor PGN → BC Dataset Parser")
    print(f"{'='*60}")
    print(f"PGN directory: {args.pgn_dir}")
    if year_range:
        print(f"Year range: {year_range[0]}-{year_range[1]}")
    if args.max_games:
        print(f"Max games: {args.max_games}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Parse dataset
    dataset = parse_wthor_dataset(
        args.pgn_dir,
        max_games=args.max_games,
        year_range=year_range,
        verbose=args.verbose
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    for key, value in dataset['metadata'].items():
        print(f"{key}: {value}")
    print(f"{'='*60}\n")

    # Save dataset
    print(f"Saving dataset to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"✓ Dataset saved!")
    print(f"\nTo train with this dataset:")
    print(f"  python bc_train.py -d {args.output} -m wthor_pretrained -e 20 -b 256")


if __name__ == '__main__':
    main()
