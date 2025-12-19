#!/usr/bin/env python3
"""
Test script for EdaxOpponent integration.
"""

import numpy as np
from util.reversi import ReversiEnvCNN
from util.opponents import get_opponent
from util.util import render

def test_edax_opponent():
    """Test that EdaxOpponent can play moves."""
    print("Testing EdaxOpponent integration...")
    print("=" * 60)

    # Create environment
    env = ReversiEnvCNN()
    board, _ = env.reset()

    # Create Edax opponent at depth 6
    print("\nInitializing Edax opponent (depth 6)...")
    edax = get_opponent("Edax", depth=6)
    print("✓ Edax opponent created successfully")

    # Play 5 moves
    print("\nPlaying 5 moves with Edax...")
    for i in range(5):
        print(f"\n--- Move {i+1} ---")
        print(f"Current player: {'Black' if env.player == 1 else 'White'}")

        # Display board
        render(board)

        # Get Edax's move
        action = edax.get_action(env, board)
        action_num = action[0]

        # Convert action to notation
        row = action_num // 8
        col = action_num % 8
        notation = f"{'ABCDEFGH'[col]}{row + 1}"

        print(f"Edax plays: {notation} (action {action_num})")

        # Make the move
        action_tuple = (0, row, col)
        board = env.get_next_state(board, action_tuple)

        # Check if game ended
        winner = env.get_winner(board)
        if winner is not None:
            print("\nGame ended!")
            break

    print("\n" + "=" * 60)
    print("✓ EdaxOpponent test completed successfully!")
    print("\nFinal board:")
    render(board)

    # Test different depths
    print("\n" + "=" * 60)
    print("Testing different depth levels...")

    for depth in [4, 6, 8]:
        print(f"\nTesting Edax at depth {depth}...")
        env2 = ReversiEnvCNN()
        board2, _ = env2.reset()

        edax_test = get_opponent("Edax", depth=depth)
        action = edax_test.get_action(env2, board2)

        row = action[0] // 8
        col = action[0] % 8
        notation = f"{'ABCDEFGH'[col]}{row + 1}"

        print(f"  ✓ Depth {depth}: {notation}")

        # Clean up
        del edax_test

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nEdaxOpponent is ready to use in training:")
    print("  python sb-train.py -o Edax --edax-depth 6 ...")
    print("  python sb-play.py -o Edax --edax-depth 6 ...")

if __name__ == "__main__":
    test_edax_opponent()
