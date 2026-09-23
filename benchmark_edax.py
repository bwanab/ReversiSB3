#!/usr/bin/env python3
"""Benchmark Edax client/server performance."""

import time
import sys
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.edax_client import EdaxClient


def create_test_position():
    """Create a simple test position."""
    state = np.zeros((1, 8, 8), dtype=np.int8)
    state[0, 3, 3] = -1
    state[0, 3, 4] = 1
    state[0, 4, 3] = 1
    state[0, 4, 4] = -1
    return state


def benchmark_single_request(depth=1, num_requests=100):
    """Benchmark single-threaded request performance."""
    print(f"\n{'='*60}")
    print(f"Benchmark 1: Single-threaded (depth={depth})")
    print(f"{'='*60}")

    client = EdaxClient(depth=depth)
    state = create_test_position()

    # Warmup
    client.get_move(state)

    # Benchmark
    start = time.time()
    for i in range(num_requests):
        move = client.get_move(state)
    elapsed = time.time() - start

    avg_time = elapsed / num_requests
    requests_per_sec = num_requests / elapsed

    print(f"Total time: {elapsed:.3f}s")
    print(f"Requests: {num_requests}")
    print(f"Avg time per request: {avg_time*1000:.2f}ms")
    print(f"Requests/sec: {requests_per_sec:.1f}")

    return avg_time


def benchmark_parallel_requests(depth=1, num_workers=4, requests_per_worker=25):
    """Benchmark parallel request performance (simulates SB3 parallel envs)."""
    print(f"\n{'='*60}")
    print(f"Benchmark 2: Parallel ({num_workers} workers, depth={depth})")
    print(f"{'='*60}")

    state = create_test_position()
    total_requests = num_workers * requests_per_worker

    def worker_task(worker_id):
        client = EdaxClient(depth=depth)
        worker_start = time.time()

        for i in range(requests_per_worker):
            move = client.get_move(state)

        worker_elapsed = time.time() - worker_start
        return worker_elapsed

    # Run parallel workers
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        worker_times = list(executor.map(worker_task, range(num_workers)))
    total_elapsed = time.time() - start

    avg_worker_time = sum(worker_times) / len(worker_times)
    requests_per_sec = total_requests / total_elapsed

    print(f"Total time: {total_elapsed:.3f}s")
    print(f"Total requests: {total_requests}")
    print(f"Avg worker time: {avg_worker_time:.3f}s")
    print(f"Requests/sec (overall): {requests_per_sec:.1f}")
    print(f"Avg time per request: {total_elapsed/total_requests*1000:.2f}ms")

    # Check for serialization
    theoretical_parallel = avg_worker_time
    actual_total = total_elapsed
    efficiency = (theoretical_parallel / actual_total) * 100

    print(f"\nParallel efficiency: {efficiency:.1f}%")
    if efficiency < 80:
        print("⚠️  LOW EFFICIENCY - Server is likely serializing requests!")
    else:
        print("✓ Good parallel efficiency")

    return total_elapsed / total_requests


def benchmark_direct_vs_client(depth=1, num_requests=50):
    """Compare direct library calls vs client/server."""
    print(f"\n{'='*60}")
    print(f"Benchmark 3: Direct Library vs Client/Server (depth={depth})")
    print(f"{'='*60}")

    from util.edax_engine import EdaxEngine

    state = create_test_position()

    # Benchmark direct library
    engine = EdaxEngine(depth=depth)
    engine.get_move(state)  # Warmup

    start = time.time()
    for i in range(num_requests):
        move = engine.get_move(state)
    direct_elapsed = time.time() - start
    direct_avg = direct_elapsed / num_requests

    # Benchmark client/server
    client = EdaxClient(depth=depth)
    client.get_move(state)  # Warmup

    start = time.time()
    for i in range(num_requests):
        move = client.get_move(state)
    client_elapsed = time.time() - start
    client_avg = client_elapsed / num_requests

    overhead = client_avg - direct_avg
    overhead_pct = (overhead / direct_avg) * 100

    print(f"Direct library: {direct_avg*1000:.2f}ms per request ({num_requests/direct_elapsed:.1f} req/s)")
    print(f"Client/server:  {client_avg*1000:.2f}ms per request ({num_requests/client_elapsed:.1f} req/s)")
    print(f"Overhead:       {overhead*1000:.2f}ms ({overhead_pct:.1f}%)")

    return overhead


def estimate_training_performance(depth=1):
    """Estimate training iterations/sec based on benchmarks."""
    print(f"\n{'='*60}")
    print(f"Benchmark 4: Training Performance Estimate (depth={depth})")
    print(f"{'='*60}")

    # Typical SB3 setup
    num_envs = 4  # Default for DummyVecEnv
    moves_per_game = 30  # Average

    # Measure parallel performance (simulates training)
    state = create_test_position()

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        def single_game():
            client = EdaxClient(depth=depth)
            for _ in range(moves_per_game):
                client.get_move(state)

        list(executor.map(lambda _: single_game(), range(num_envs)))

    elapsed = time.time() - start
    games_per_sec = num_envs / elapsed

    # SB3 typically reports iterations (1 iteration = n_steps across all envs)
    # With n_steps=2048 and 4 envs, that's 2048*4 = 8192 steps
    n_steps = 2048
    total_steps = n_steps * num_envs
    steps_per_game = moves_per_game * 2  # Both players
    games_needed = total_steps / steps_per_game
    time_per_iteration = games_needed / games_per_sec
    iterations_per_sec = 1 / time_per_iteration

    print(f"Setup: {num_envs} parallel envs, {moves_per_game} moves/game")
    print(f"Time to play {num_envs} parallel games: {elapsed:.3f}s")
    print(f"Games/sec: {games_per_sec:.2f}")
    print(f"\nSB3 Training (n_steps={n_steps}):")
    print(f"Steps per iteration: {total_steps}")
    print(f"Games needed per iteration: {games_needed:.1f}")
    print(f"Estimated time per iteration: {time_per_iteration:.2f}s")
    print(f"Estimated iterations/sec: {iterations_per_sec:.2f}")

    return iterations_per_sec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Edax client/server")
    parser.add_argument("--depth", type=int, default=1, help="Edax search depth")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("EDAX CLIENT/SERVER BENCHMARK SUITE")
    print("="*60)
    print(f"\nMake sure Edax server is running: python edax_server.py")
    print(f"Testing with depth={args.depth}, {args.workers} parallel workers\n")

    try:
        # Run benchmarks
        single_time = benchmark_single_request(depth=args.depth, num_requests=100)
        parallel_time = benchmark_parallel_requests(
            depth=args.depth,
            num_workers=args.workers,
            requests_per_worker=25
        )
        overhead = benchmark_direct_vs_client(depth=args.depth, num_requests=50)
        estimated_perf = estimate_training_performance(depth=args.depth)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Single-threaded:     {1/single_time:.1f} req/s ({single_time*1000:.2f}ms/req)")
        print(f"Parallel ({args.workers} workers): {1/parallel_time:.1f} req/s ({parallel_time*1000:.2f}ms/req)")
        print(f"Socket overhead:     {overhead*1000:.2f}ms")
        print(f"Estimated training:  {estimated_perf:.2f} iterations/sec")
        print()

        if parallel_time > single_time * 2:
            print("⚠️  WARNING: Parallel performance is poor!")
            print("    The server is likely processing requests sequentially.")
            print("    Solution: Implement multi-threaded server or connection pooling.")
        else:
            print("✓ Parallel performance looks good!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Edax server is running:")
        print("  python edax_server.py")
        sys.exit(1)
