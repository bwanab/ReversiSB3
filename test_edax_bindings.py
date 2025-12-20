#!/usr/bin/env python3
"""Test Edax Python bindings.

This script tests the direct Python bindings to Edax engine via ctypes,
which provides a stateless interface without GTP protocol overhead.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from util.edax_engine import EdaxEngine


def test_engine_creation():
    """Test basic engine creation and destruction."""
    print("Test 1: Engine creation")
    print("=" * 60)

    try:
        edax = EdaxEngine(depth=6)
        print(f"✓ Engine created: {edax}")
        print(f"  Depth: {edax.depth}")
        del edax
        print("✓ Engine destroyed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    print()
    return True


def test_opening_position():
    """Test Edax on initial board position."""
    print("Test 2: Opening position")
    print("=" * 60)

    # Create initial Reversi position
    # Standard starting position:
    #   a b c d e f g h
    # 1 . . . . . . . .
    # 2 . . . . . . . .
    # 3 . . . . . . . .
    # 4 . . . W B . . .
    # 5 . . . B W . . .
    # 6 . . . . . . . .
    # 7 . . . . . . . .
    # 8 . . . . . . . .
    #
    # Assuming Black (1) to move
    initial = np.zeros((1, 8, 8), dtype=np.int8)
    initial[0, 3, 3] = -1  # d4 = White
    initial[0, 3, 4] = 1   # e4 = Black
    initial[0, 4, 3] = 1   # d5 = Black
    initial[0, 4, 4] = -1  # e5 = White

    try:
        edax = EdaxEngine(depth=6)
        move = edax.get_move(initial)

        if move is None:
            print("✗ Edax returned no move for opening position!")
            return False

        # Convert to chess notation
        row = move // 8
        col = move % 8
        notation = f"{'abcdefgh'[col]}{row + 1}"

        print(f"✓ Edax opening move: {move} ({notation})")

        # Check if it's a standard opening move
        standard_openings = ['d3', 'c4', 'f5', 'e6']
        if notation in standard_openings:
            print(f"✓ Standard opening move")
        else:
            print(f"⚠ Unusual opening (expected one of: {', '.join(standard_openings)})")

        # Get score and nodes
        score = edax.get_score()
        nodes = edax.get_nodes()
        print(f"  Score: {score}")
        print(f"  Nodes: {nodes:,}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_multiple_depths():
    """Test different search depths."""
    print("Test 3: Multiple depths")
    print("=" * 60)

    initial = np.zeros((1, 8, 8), dtype=np.int8)
    initial[0, 3, 3] = -1
    initial[0, 3, 4] = 1
    initial[0, 4, 3] = 1
    initial[0, 4, 4] = -1

    try:
        for depth in [4, 6, 8]:
            edax = EdaxEngine(depth=depth)
            move = edax.get_move(initial)
            row = move // 8
            col = move % 8
            notation = f"{'abcdefgh'[col]}{row + 1}"
            nodes = edax.get_nodes()
            print(f"  Depth {depth}: {notation} ({nodes:,} nodes)")
            del edax

        print("✓ Multiple depths tested successfully")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_mid_game_position():
    """Test Edax on a mid-game position."""
    print("Test 4: Mid-game position")
    print("=" * 60)

    # Create a mid-game position (arbitrary but valid)
    mid_game = np.zeros((1, 8, 8), dtype=np.int8)

    # Set up a position with more pieces
    # Example: After a few moves
    mid_game[0, 2, 3] = 1   # d3 = Black
    mid_game[0, 3, 2] = 1   # c4 = Black
    mid_game[0, 3, 3] = 1   # d4 = Black
    mid_game[0, 3, 4] = 1   # e4 = Black
    mid_game[0, 4, 3] = 1   # d5 = Black
    mid_game[0, 4, 4] = -1  # e5 = White
    mid_game[0, 4, 5] = -1  # f5 = White
    mid_game[0, 5, 4] = -1  # e6 = White

    try:
        edax = EdaxEngine(depth=6)
        move = edax.get_move(mid_game)

        if move is None:
            print("  Edax returned no move (pass)")
        else:
            row = move // 8
            col = move % 8
            notation = f"{'abcdefgh'[col]}{row + 1}"
            score = edax.get_score()
            nodes = edax.get_nodes()
            print(f"✓ Edax move: {notation}")
            print(f"  Score: {score}")
            print(f"  Nodes: {nodes:,}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_no_moves_position():
    """Test Edax on position with no legal moves."""
    print("Test 5: No legal moves (pass)")
    print("=" * 60)

    # Create a position where current player has no moves
    # This is artificial but tests the edge case
    no_moves = np.zeros((1, 8, 8), dtype=np.int8)

    # Fill board in a way that current player (1) has no moves
    # Just Black pieces on one side, White on another with no valid moves
    # (This may not be a legal game state, but tests the function)
    for i in range(8):
        no_moves[0, 0, i] = 1  # Top row all Black
        no_moves[0, 7, i] = -1 # Bottom row all White

    try:
        edax = EdaxEngine(depth=4)
        move = edax.get_move(no_moves)

        if move is None:
            print("✓ Correctly returned None (no legal moves)")
        else:
            print(f"⚠ Expected None, got move {move}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EDAX PYTHON BINDINGS TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        test_engine_creation,
        test_opening_position,
        test_multiple_depths,
        test_mid_game_position,
        test_no_moves_position,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
