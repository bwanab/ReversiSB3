#!/usr/bin/env python3
"""
Extract distinct early-game positions from a BC dataset for varied-start RL training.

Positions are stored from the side-to-move's perspective (own stones = 1), so each one is
a valid "BLACK to move" start for ReversiEnvCNN, and the mover has a legal move (one was
played from it). The output is an int8 array of shape (N, 64), deduplicated so popular
opening lines aren't over-represented.

Usage:
    python make_start_positions.py --dataset combined_bc_dataset.pkl \
        --min-stones 8 --max-stones 20 --output start_positions.npy
Then:
    python sb-train.py ... --start-positions start_positions.npy
"""

import argparse
import pickle
import numpy as np


def extract_start_positions(dataset, min_stones, max_stones):
    states = np.stack([item['state'] for item in dataset]).astype(np.int8).reshape(len(dataset), 64)
    stones = (states != 0).sum(axis=1)
    selected = states[(stones >= min_stones) & (stones <= max_stones)]
    return np.unique(selected, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dataset', default='combined_bc_dataset.pkl', help='BC dataset pickle')
    parser.add_argument('--min-stones', type=int, default=8, help='fewest stones on the board (4 = start)')
    parser.add_argument('--max-stones', type=int, default=20, help='most stones on the board')
    parser.add_argument('-o', '--output', default='start_positions.npy', help='output .npy file')
    args = parser.parse_args()

    with open(args.dataset, 'rb') as f:
        package = pickle.load(f)
    dataset = package['dataset'] if 'dataset' in package else package['moves']

    positions = extract_start_positions(dataset, args.min_stones, args.max_stones)
    np.save(args.output, positions)
    print(f"Saved {len(positions):,} distinct positions with {args.min_stones}-{args.max_stones} stones "
          f"to {args.output}")


if __name__ == '__main__':
    main()
