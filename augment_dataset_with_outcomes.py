#!/usr/bin/env python3
"""
Augment existing WThor BC dataset with game outcomes.

Reads PGN files to extract game outcomes, then adds outcome values
to each move in the dataset without replaying games.
"""

import os
import re
import pickle
import argparse
from tqdm import tqdm
from util.util import BLACK, WHITE


def parse_game_outcome(result_str):
    """
    Parse WThor result string to get outcome.

    Result format: "black_score-white_score" (e.g., "22-42")

    Returns
    -------
    tuple
        (black_score, white_score)
    """
    match = re.match(r'(\d+)-(\d+)', result_str)
    if match:
        black_score = int(match.group(1))
        white_score = int(match.group(2))
        return black_score, white_score
    return None, None


def extract_game_outcomes(pgn_dir):
    """
    Extract game outcomes from all PGN files.

    Returns
    -------
    dict
        Mapping from game_id to (black_score, white_score)
    """
    pgn_files = sorted([
        os.path.join(pgn_dir, f)
        for f in os.listdir(pgn_dir)
        if f.startswith('WTHOR-') and f.endswith('.pgn')
    ])

    print(f"Found {len(pgn_files)} PGN files")

    outcomes = {}

    for pgn_path in tqdm(pgn_files, desc="Extracting outcomes"):
        with open(pgn_path, 'r') as f:
            current_metadata = {}

            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    # End of game - store outcome if we have all metadata
                    if current_metadata:
                        event = current_metadata.get('Event', 'Unknown')
                        black = current_metadata.get('Black', '?')
                        white = current_metadata.get('White', '?')
                        result = current_metadata.get('Result')

                        # Create game ID matching what parser uses
                        game_id = f"{event} - {black} vs {white}"

                        if result:
                            black_score, white_score = parse_game_outcome(result)
                            if black_score is not None:
                                outcomes[game_id] = (black_score, white_score)

                        current_metadata = {}
                    continue

                # Parse metadata
                if line.startswith('['):
                    match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
                    if match:
                        key, value = match.groups()
                        current_metadata[key] = value

            # Don't forget last game
            if current_metadata:
                event = current_metadata.get('Event', 'Unknown')
                black = current_metadata.get('Black', '?')
                white = current_metadata.get('White', '?')
                result = current_metadata.get('Result')

                game_id = f"{event} - {black} vs {white}"

                if result:
                    black_score, white_score = parse_game_outcome(result)
                    if black_score is not None:
                        outcomes[game_id] = (black_score, white_score)

    return outcomes


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


def augment_dataset(dataset_file, pgn_dir, output_file):
    """
    Augment dataset with game outcomes.

    Parameters
    ----------
    dataset_file : str
        Path to existing dataset pickle file
    pgn_dir : str
        Directory containing PGN files
    output_file : str
        Path for augmented dataset
    """
    print("="*60)
    print("Augmenting Dataset with Game Outcomes")
    print("="*60)
    print(f"Input dataset: {dataset_file}")
    print(f"PGN directory: {pgn_dir}")
    print(f"Output: {output_file}")
    print("="*60)
    print()

    # Load existing dataset
    print("Loading existing dataset...")
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

    moves = dataset['moves']
    metadata = dataset['metadata']

    print(f"Loaded {len(moves):,} moves from {metadata.get('total_games', '?'):,} games")
    print()

    # Extract game outcomes from PGN files
    print("Extracting game outcomes from PGN files...")
    outcomes = extract_game_outcomes(pgn_dir)
    print(f"Extracted outcomes for {len(outcomes):,} games")
    print()

    # Augment moves with outcomes
    print("Augmenting moves with outcome values...")
    matched = 0
    unmatched = 0

    for move in tqdm(moves, desc="Processing moves"):
        game_id = move['game_metadata']

        if game_id in outcomes:
            black_score, white_score = outcomes[game_id]
            outcome_value = compute_outcome_value(move['color'], black_score, white_score)
            move['outcome'] = outcome_value
            matched += 1
        else:
            # Shouldn't happen, but handle gracefully
            move['outcome'] = 0.0  # Neutral if we can't find outcome
            unmatched += 1

    print(f"\nMatched: {matched:,} moves")
    print(f"Unmatched: {unmatched:,} moves")
    print()

    # Update metadata
    metadata['has_outcomes'] = True
    metadata['outcome_format'] = 'normalized_score_diff'
    metadata['outcome_range'] = '[-1, 1]'

    # Save augmented dataset
    augmented_dataset = {
        'moves': moves,
        'metadata': metadata
    }

    print(f"Saving augmented dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(augmented_dataset, f)

    print("✓ Done!")
    print()
    print("Dataset now includes 'outcome' field for each move:")
    print("  - Range: [-1, 1]")
    print("  - Positive = player won")
    print("  - Negative = player lost")
    print("  - Zero = draw")
    print()
    print("Ready for BC training with value network!")


def main():
    parser = argparse.ArgumentParser(
        description='Augment WThor dataset with game outcomes'
    )
    parser.add_argument('-d', '--dataset', type=str, default='wthor_bc_dataset.pkl',
                        help='Input dataset pickle file')
    parser.add_argument('--pgn-dir', type=str, default='../othello-games/pgn',
                        help='Directory containing PGN files')
    parser.add_argument('-o', '--output', type=str, default='wthor_bc_dataset_with_values.pkl',
                        help='Output dataset pickle file')

    args = parser.parse_args()

    augment_dataset(args.dataset, args.pgn_dir, args.output)


if __name__ == '__main__':
    main()
