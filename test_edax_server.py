#!/usr/bin/env python3
"""Test Edax client/server architecture."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.edax_client import EdaxClient


def test_server_connection():
    """Test basic connection to Edax server."""
    print("Test 1: Server Connection")
    print("=" * 60)

    try:
        client = EdaxClient(depth=6)
        print("✓ Client created")

        # Test opening position
        initial = np.zeros((1, 8, 8), dtype=np.int8)
        initial[0, 3, 3] = -1  # d4 = White
        initial[0, 3, 4] = 1   # e4 = Black
        initial[0, 4, 3] = 1   # d5 = Black
        initial[0, 4, 4] = -1  # e5 = White

        print("Requesting move from server...")
        move = client.get_move(initial)

        if move is None:
            print("✗ Server returned None")
            return False

        row = move // 8
        col = move % 8
        notation = f"{'abcdefgh'[col]}{row + 1}"

        print(f"✓ Server returned move: {notation} (index {move})")

        # Verify it's a standard opening
        standard_openings = ['d3', 'c4', 'f5', 'e6']
        if notation in standard_openings:
            print("✓ Valid opening move")
        else:
            print(f"⚠ Unusual opening (expected one of: {', '.join(standard_openings)})")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure Edax server is running:")
        print("  python edax_server.py")
        return False

    print()
    return True


def test_multiple_depths():
    """Test different depths."""
    print("Test 2: Multiple Depths")
    print("=" * 60)

    initial = np.zeros((1, 8, 8), dtype=np.int8)
    initial[0, 3, 3] = -1
    initial[0, 3, 4] = 1
    initial[0, 4, 3] = 1
    initial[0, 4, 4] = -1

    try:
        for depth in [1, 4, 6]:
            client = EdaxClient(depth=depth)
            start = time.time()
            move = client.get_move(initial)
            elapsed = time.time() - start

            row = move // 8
            col = move % 8
            notation = f"{'abcdefgh'[col]}{row + 1}"

            print(f"  Depth {depth}: {notation} ({elapsed*1000:.1f}ms)")

        print("✓ Multiple depths working")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print()
    return True


def test_with_environment():
    """Test with actual environment."""
    print("Test 3: Integration with Environment")
    print("=" * 60)

    try:
        from util.reversi import ReversiEnvCNN
        from util.opponents import EdaxOpponent

        # Create environment with Edax opponent
        env = ReversiEnvCNN(opponent='Edax', depth=4)
        state, _ = env.reset()

        print("Playing one move against Edax...")
        valid_actions = env.all_valid_actions(state)
        action = valid_actions[0]

        state, reward, done, truncated, info = env.step(action)

        print("✓ Environment integration working")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EDAX CLIENT/SERVER TEST SUITE")
    print("=" * 60)
    print("\nNote: Make sure Edax server is running:")
    print("  python edax_server.py")
    print()

    tests = [
        test_server_connection,
        test_multiple_depths,
        test_with_environment,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        sys.exit(1)
