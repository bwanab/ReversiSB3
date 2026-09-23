#!/usr/bin/env python3
"""Compare RAI vs Edax performance."""

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.edax_client import EdaxClient


def create_test_position():
    """Create a test position."""
    state = np.zeros((1, 8, 8), dtype=np.int8)
    state[0, 3, 3] = -1
    state[0, 3, 4] = 1
    state[0, 4, 3] = 1
    state[0, 4, 4] = -1
    return state


def benchmark_rai(depth=2, num_moves=20):
    """Benchmark RAI opponent."""
    print(f"\n{'='*60}")
    print(f"RAI Depth {depth}")
    print(f"{'='*60}")

    try:
        from util.opponents import RAIOpponent
        from util.reversi import build_reversi

        env = build_reversi(opponent = "RAI")
        rai = RAIOpponent(depth=depth)
        state = create_test_position()

        # Warmup
        rai.get_action(env, state)

        # Benchmark
        times = []
        for i in range(num_moves):
            start = time.time()
            action = rai.get_action(env, state)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"Moves tested: {num_moves}")
        print(f"Avg time: {avg_time*1000:.2f}ms")
        print(f"Min time: {min_time*1000:.2f}ms")
        print(f"Max time: {max_time*1000:.2f}ms")
        print(f"Moves/sec: {1/avg_time:.1f}")

        return avg_time

    except Exception as e:
        print(f"Error: {e}")
        return None


def benchmark_edax(depth=1, num_moves=20):
    """Benchmark Edax opponent."""
    print(f"\n{'='*60}")
    print(f"Edax Depth {depth}")
    print(f"{'='*60}")

    try:
        client = EdaxClient(depth=depth)
        state = create_test_position()

        # Warmup
        client.get_move(state)

        # Benchmark
        times = []
        for i in range(num_moves):
            start = time.time()
            move = client.get_move(state)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"Moves tested: {num_moves}")
        print(f"Avg time: {avg_time*1000:.2f}ms")
        print(f"Min time: {min_time*1000:.2f}ms")
        print(f"Max time: {max_time*1000:.2f}ms")
        print(f"Moves/sec: {1/avg_time:.1f}")

        return avg_time

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Edax server is running: python edax_server.py")
        return None


def estimate_training_time(opponent_name, time_per_move, timesteps=1_000_000):
    """Estimate total training time."""
    # Assumptions:
    # - 4 parallel environments
    # - n_steps = 2048
    # - ~60 moves per game (30 from each player)
    # - Opponent makes ~half the moves

    n_envs = 4
    n_steps = 2048
    moves_per_game = 30  # Opponent moves only

    steps_per_iteration = n_steps * n_envs
    iterations = timesteps / steps_per_iteration

    # Each step might trigger an opponent move
    # On average, opponent makes a move every 2 steps (alternating turns)
    opponent_moves_per_iteration = n_steps  # Rough estimate

    time_per_iteration = opponent_moves_per_iteration * time_per_move
    total_time = iterations * time_per_iteration

    print(f"\n{opponent_name} Training Estimate ({timesteps:,} timesteps):")
    print(f"  Time per move: {time_per_move*1000:.2f}ms")
    print(f"  Opponent moves per iteration: ~{opponent_moves_per_iteration}")
    print(f"  Time per iteration: {time_per_iteration:.2f}s")
    print(f"  Iterations needed: {iterations:.0f}")
    print(f"  Total opponent time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Estimated training time: {total_time*1.5/60:.1f} min (with overhead)")

    return total_time


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAI vs EDAX PERFORMANCE COMPARISON")
    print("="*60)

    results = {}

    # Benchmark RAI
    print("\n" + "="*60)
    print("BENCHMARKING RAI")
    print("="*60)

    rai_2 = benchmark_rai(depth=2, num_moves=10)
    if rai_2:
        results['RAI-2'] = rai_2

    # Only test RAI depth 4 if user wants (it's slow)
    import sys
    if '--include-rai-4' in sys.argv:
        rai_4 = benchmark_rai(depth=4, num_moves=5)
        if rai_4:
            results['RAI-4'] = rai_4

    # Benchmark Edax
    print("\n" + "="*60)
    print("BENCHMARKING EDAX (via server)")
    print("="*60)

    edax_1 = benchmark_edax(depth=1, num_moves=20)
    if edax_1:
        results['Edax-1'] = edax_1

    edax_4 = benchmark_edax(depth=4, num_moves=20)
    if edax_4:
        results['Edax-4'] = edax_4

    edax_6 = benchmark_edax(depth=6, num_moves=20)
    if edax_6:
        results['Edax-6'] = edax_6

    edax_10 = benchmark_edax(depth=10, num_moves=10)
    if edax_10:
        results['Edax-10'] = edax_10

    # Summary comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)

    if results:
        print("\nTime per move:")
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:12s}: {time_val*1000:8.2f}ms  ({1/time_val:8.1f} moves/sec)")

        # Speed comparisons
        if 'RAI-2' in results and 'Edax-1' in results:
            speedup = results['RAI-2'] / results['Edax-1']
            print(f"\nEdax-1 is {speedup:.0f}x faster than RAI-2")

        if 'RAI-2' in results and 'Edax-6' in results:
            speedup = results['RAI-2'] / results['Edax-6']
            print(f"Edax-6 is {speedup:.1f}x faster than RAI-2 (similar strength, much faster)")

        if 'RAI-4' in results and 'Edax-10' in results:
            speedup = results['RAI-4'] / results['Edax-10']
            print(f"Edax-10 is {speedup:.0f}x faster than RAI-4 (much stronger too)")

        # Training time estimates
        print("\n" + "="*60)
        print("TRAINING TIME ESTIMATES (1M timesteps)")
        print("="*60)

        for name, time_val in results.items():
            estimate_training_time(name, time_val, 1_000_000)

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
For training purposes:
- Edax-1: Very fast, good for initial training (weak opponent)
- Edax-4: Good balance of speed and strength
- Edax-6: Similar speed to RAI-2, but MUCH stronger
- Edax-10: Strong opponent, slower but still faster than RAI-4

RAI is Python-based, Edax is optimized C.
Edax is faster at ALL comparable depths.
    """)
