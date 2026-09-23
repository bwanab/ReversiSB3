#!/usr/bin/env python3
"""Comprehensive depth test with multiple positions."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.edax_engine import EdaxEngine


def test_depth_scaling():
    """Test that deeper searches take more time and nodes."""
    
    # Simple midgame position
    state = np.zeros((1, 8, 8), dtype=np.int8)
    state[0, 2, 3] = 1
    state[0, 3, 2] = 1
    state[0, 3, 3] = 1
    state[0, 3, 4] = 1
    state[0, 4, 3] = 1
    state[0, 4, 4] = -1
    state[0, 4, 5] = -1
    state[0, 5, 4] = -1
    
    print("Depth  | Time (ms) | Nodes     | Nodes/sec | Score")
    print("-------|-----------|-----------|-----------|------")
    
    for depth in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
        engine = EdaxEngine(depth=depth)
        
        # Warmup
        engine.get_move(state)
        
        # Time it
        start = time.time()
        move = engine.get_move(state)
        elapsed = time.time() - start
        
        nodes = engine.get_nodes()
        score = engine.get_score()
        nps = nodes / elapsed if elapsed > 0 else 0
        
        print(f"{depth:6d} | {elapsed*1000:9.2f} | {nodes:9,d} | {nps:9,.0f} | {score:+5d}")

if __name__ == "__main__":
    test_depth_scaling()
