#!/usr/bin/env python3
"""Test EdaxOpponent integration with ReversiEnvCNN."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from util.reversi import ReversiEnvCNN
from util.opponents import EdaxOpponent, RandomOpponent
from util.util import BLACK, WHITE

def test_edax_vs_random():
    """Test Edax opponent integrated with environment."""
    print("Testing Edax integrated with ReversiEnvCNN...")
    print("=" * 60)

    # Create environment with Edax as opponent
    import gymnasium as gym
    env = gym.make("ReversiCNN-v0", opponent="Edax", depth=4)

    # Play a few moves to test
    print("\nPlaying game with Edax opponent (depth 4)...\n")

    state, _ = env.reset()
    done = False
    move_count = 0

    while not done and move_count < 10:  # Play 10 moves to test
        move_count += 1

        # Get all valid moves
        valid_actions = env.all_valid_actions(state)

        if len(valid_actions) == 0:
            print(f"Move {move_count}: Black has no valid moves, passing")
            state, reward, done, truncated, info = env.step(None)
        else:
            # Play a random move for the agent (Black)
            import random
            action = random.choice(valid_actions)
            row = action // 8
            col = action % 8
            move_str = f"{'abcdefgh'[col]}{row + 1}"
            print(f"Move {move_count}: Black plays {move_str}")
            state, reward, done, truncated, info = env.step(action)

        if done:
            break

    # Count pieces
    board = state[0]
    black_count = int((board == 1).sum())
    white_count = int((board == -1).sum())

    print(f"\nCurrent score after {move_count} moves: Black {black_count} - White {white_count}")
    print("=" * 60)
    print("Test completed successfully! ✓\n")


def test_edax_opening():
    """Test that Edax makes reasonable opening moves."""
    print("\nTesting Edax opening moves...")
    print("=" * 60)

    env = ReversiEnvCNN()
    edax = EdaxOpponent(depth=6)
    edax.player = BLACK

    # Get opening move
    state, _ = env.reset()
    action = edax.get_action(env, state)

    if len(action) == 0:
        print("ERROR: Edax returned no move for opening position!")
        return False

    action_num = action[0]
    row = action_num // 8
    col = action_num % 8
    move_str = f"{'abcdefgh'[col]}{row + 1}"

    # Standard opening moves in Reversi
    standard_openings = ['d3', 'c4', 'f5', 'e6']

    print(f"Edax opening move: {move_str}")

    if move_str in standard_openings:
        print(f"✓ Valid standard opening move")
    else:
        print(f"⚠ Unusual opening (expected one of: {', '.join(standard_openings)})")

    print("=" * 60)
    return True


if __name__ == "__main__":
    test_edax_opening()
    print()
    test_edax_vs_random()
