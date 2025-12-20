# Edax Python Bindings Implementation Plan

## Problem Summary

**Goal**: Use Edax (world-class C-based Reversi engine) as an opponent for training our RL agent.

**Challenge**: The GTP protocol approach had board synchronization issues:
- We need to tell Edax what moves the agent made
- Detecting moves by comparing board states is unreliable after Edax's move is applied
- The environment doesn't track move history
- Without game simulation logic, perfect synchronization is complex

**Solution**: Create Python bindings to Edax's C library for direct function calls.

## Why Python Bindings?

1. **Direct board position setting** - Call `edax_set_board(bitboards)` without move history
2. **No synchronization issues** - Just pass current state, get move back
3. **Faster** - No subprocess/pipe overhead (~1000x faster than GTP)
4. **Cleaner API** - Simple: `move = edax.get_move(board_state)`
5. **Full access** - Can also get eval scores, multiple move candidates, etc.

## Edax Code Structure (Already Analyzed)

### Key Files in `../edax-reversi/src/`:
- **board.h/c**: Board representation (bitboards)
- **play.h/c**: High-level play interface
- **search.h/c**: Search engine
- **eval.h/c**: Position evaluation

### Key Structures:

```c
// board.h
typedef struct Board {
    uint64_t player;    // Bitboard of current player's pieces
    uint64_t opponent;  // Bitboard of opponent's pieces
} Board;

// play.h
typedef struct Play {
    Board board;
    Search search;
    Result result;
    Book *book;
    int player;
    int level;
    // ... (simplified)
} Play;

// search.h
typedef struct Result {
    int move;      // Best move found (0-63)
    int score;     // Position evaluation
    Line pv;       // Principal variation
    // ... (simplified)
} Result;
```

### Key Functions:

```c
// Initialization
void play_init(Play*, Book*);
void play_free(Play*);

// Set position
void play_set_board(Play*, const char*);  // Takes string format
int board_set(Board*, const char*);       // Lower-level

// Search for best move
void play_go(Play*, const bool);  // Result stored in play->result.move

// Board utilities
uint64_t board_get_moves(const Board*);  // Get legal moves bitboard
```

## Implementation Steps

### 1. Create C Wrapper (`edax_wrapper.c`)

Create a simple C file that exposes a clean API:

```c
// edax_wrapper.c
#include "play.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include <stdlib.h>

typedef struct {
    Play play;
    Book book;
} EdaxEngine;

// Initialize engine at given search depth
EdaxEngine* edax_create(int level) {
    EdaxEngine *engine = malloc(sizeof(EdaxEngine));

    // Initialize book (can be NULL if not using opening book)
    book_init(&engine->book);

    // Initialize play structure
    play_init(&engine->play, &engine->book);

    // Set search level (depth)
    engine->play.level = level;
    search_set_level(&engine->play.search, level, 0);

    return engine;
}

// Get best move for given position
// player_bits: 64-bit bitboard of current player's pieces
// opponent_bits: 64-bit bitboard of opponent's pieces
// Returns: move index 0-63, or -1 if no legal moves
int edax_get_move(EdaxEngine *engine, uint64_t player_bits, uint64_t opponent_bits) {
    // Set board position
    engine->play.board.player = player_bits;
    engine->play.board.opponent = opponent_bits;

    // Update search board
    Board board = {player_bits, opponent_bits};
    search_set_board(&engine->play.search, &board, BLACK);

    // Search for best move
    play_go(&engine->play, false);  // false = don't ponder

    // Return best move
    return engine->play.result.move;
}

// Get evaluation score for position
int edax_get_score(EdaxEngine *engine) {
    return engine->play.result.score;
}

// Clean up
void edax_destroy(EdaxEngine *engine) {
    play_free(&engine->play);
    book_free(&engine->book);
    free(engine);
}
```

### 2. Build Shared Library

Create a Makefile or build script:

```bash
# Build edax as shared library on macOS
cd /Users/bill/src/edax-reversi/src

# Compile wrapper with edax objects
gcc -O3 -march=native -DUSE_GAS_MMX -DUSE_MSVC_X86 \
    -c edax_wrapper.c -o edax_wrapper.o

# Link into shared library (.dylib on macOS, .so on Linux)
gcc -dynamiclib -o libedax.dylib \
    edax_wrapper.o \
    *.o \
    -lpthread

# Copy to project directory
cp libedax.dylib /Users/bill/src/ReversiSB3/
```

**Note**: May need to compile all of Edax's .c files first if .o files don't exist:

```bash
# Compile all edax sources
make build OS=osx ARCH=native

# Then build wrapper and link
gcc -O3 -march=native -c edax_wrapper.c
gcc -dynamiclib -o libedax.dylib edax_wrapper.o [all other .o files] -lpthread
```

### 3. Python Bindings (`util/edax_engine.py`)

```python
"""
Python bindings to Edax engine via ctypes.
"""

import ctypes
import numpy as np
import os

class EdaxEngine:
    """Python interface to Edax Reversi engine."""

    def __init__(self, depth=6, lib_path=None):
        """Initialize Edax engine.

        Args:
            depth: Search depth (4-20 reasonable range)
            lib_path: Path to libedax.dylib (auto-detect if None)
        """
        # Find library
        if lib_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(project_root, "libedax.dylib")

        # Load shared library
        self.lib = ctypes.CDLL(lib_path)

        # Define function signatures
        self.lib.edax_create.argtypes = [ctypes.c_int]
        self.lib.edax_create.restype = ctypes.c_void_p

        self.lib.edax_get_move.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.edax_get_move.restype = ctypes.c_int

        self.lib.edax_get_score.argtypes = [ctypes.c_void_p]
        self.lib.edax_get_score.restype = ctypes.c_int

        self.lib.edax_destroy.argtypes = [ctypes.c_void_p]
        self.lib.edax_destroy.restype = None

        # Create engine instance
        self.engine = self.lib.edax_create(depth)
        self.depth = depth

    def _state_to_bitboards(self, state):
        """Convert numpy board state to bitboards.

        Args:
            state: Numpy array shape (1, 8, 8) or (3, 8, 8)
                   Values: 1=current player, -1=opponent, 0=empty

        Returns:
            (player_bits, opponent_bits): Two uint64 bitboards
        """
        board = state[0] if state.shape[0] >= 1 else state

        player_bits = 0
        opponent_bits = 0

        for row in range(8):
            for col in range(8):
                bit_pos = row * 8 + col
                cell = board[row, col]

                if cell == 1:
                    player_bits |= (1 << bit_pos)
                elif cell == -1:
                    opponent_bits |= (1 << bit_pos)

        return player_bits, opponent_bits

    def get_move(self, state):
        """Get best move for given board state.

        Args:
            state: Numpy array representing board position
                   Current player's pieces = 1
                   Opponent's pieces = -1
                   Empty squares = 0

        Returns:
            int: Best move index (0-63), or None if no legal moves
        """
        player_bits, opponent_bits = self._state_to_bitboards(state)

        move = self.lib.edax_get_move(
            self.engine,
            ctypes.c_uint64(player_bits),
            ctypes.c_uint64(opponent_bits)
        )

        return move if move >= 0 else None

    def get_score(self):
        """Get evaluation score for last position searched.

        Returns:
            int: Score in centipawns (or Edax's score units)
        """
        return self.lib.edax_get_score(self.engine)

    def __del__(self):
        """Clean up engine on deletion."""
        if hasattr(self, 'engine') and self.engine:
            self.lib.edax_destroy(self.engine)
```

### 4. Update EdaxOpponent (`util/opponents.py`)

```python
class EdaxOpponent(Opponent):
    """Opponent using Edax engine via Python bindings."""

    def __init__(self, **kwargs):
        super().__init__()
        from util.edax_engine import EdaxEngine

        self.depth = kwargs.get("depth", 6)
        self.edax = EdaxEngine(depth=self.depth)

    def get_action(self, env, state):
        """Get Edax's move for current position.

        Args:
            env: Game environment
            state: Current board state

        Returns:
            np.array: Action as single-element array
        """
        try:
            # State already has current player as 1, opponent as -1
            move = self.edax.get_move(state)

            if move is None:
                # No legal moves (shouldn't happen if env is correct)
                return random_action(env, state)

            return np.array([move])

        except Exception as e:
            print(f"WARNING: Edax failed: {e}, using random")
            return random_action(env, state)

    def __del__(self):
        """Cleanup."""
        # EdaxEngine handles its own cleanup
        pass
```

## Testing Plan

### 1. Unit Test (`test_edax_bindings.py`)

```python
#!/usr/bin/env python3
"""Test Edax Python bindings."""

from util.edax_engine import EdaxEngine
import numpy as np

def test_edax_engine():
    """Test basic Edax functionality."""
    print("Testing Edax Python bindings...")

    # Create engine
    edax = EdaxEngine(depth=6)
    print("✓ Engine created")

    # Test opening position
    initial = np.zeros((1, 8, 8), dtype=np.int8)
    initial[0, 3, 3] = -1  # White
    initial[0, 3, 4] = 1   # Black
    initial[0, 4, 3] = 1   # Black
    initial[0, 4, 4] = -1  # White

    move = edax.get_move(initial)
    print(f"✓ Opening move: {move}")

    # Convert to notation
    row = move // 8
    col = move % 8
    notation = f"{'abcdefgh'[col]}{row + 1}"
    print(f"  = {notation}")

    # Should be one of the standard openings
    assert notation in ['d3', 'c4', 'f5', 'e6'], f"Unexpected opening: {notation}"
    print("✓ Valid opening move")

    # Test score
    score = edax.get_score()
    print(f"✓ Position score: {score}")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_edax_engine()
```

### 2. Integration Test

```python
# Test with actual environment
from util.reversi import ReversiEnvCNN
import gymnasium as gym

env = gym.make('ReversiCNN-v0', opponent='Edax', depth=6)
state, _ = env.reset()

# Play 10 moves
for i in range(10):
    valid = env.all_valid_actions(state)
    action = valid[0]  # Agent plays first valid move

    state, reward, done, truncated, info = env.step(action)
    print(f"Move {i+1}: Agent played, Edax responded")

    if done:
        break

print("Integration test complete!")
```

## File Structure

```
ReversiSB3/
├── libedax.dylib                    # Compiled shared library (created)
├── util/
│   ├── edax_engine.py              # Python bindings (create)
│   └── opponents.py                # Update EdaxOpponent class
├── test_edax_bindings.py           # Unit tests (create)
├── EDAX_BINDINGS_PLAN.md          # This document
└── docs/
    └── edax_integration.md         # Additional notes (optional)

edax-reversi/
└── src/
    ├── edax_wrapper.c              # C wrapper (create)
    ├── edax_wrapper.h              # Header (create)
    └── [existing Edax sources]
```

## Potential Issues & Solutions

### Issue 1: Edax needs eval.dat file

**Solution**: Edax looks for `data/eval.dat` relative to working directory.
- Option A: Set working directory when calling
- Option B: Modify wrapper to specify data path
- Option C: Copy eval.dat to project directory

### Issue 2: Thread safety

Edax uses global state in some places.
**Solution**: Create separate engine instances for each thread/game.

### Issue 3: Memory leaks

**Solution**: Ensure `edax_destroy()` is always called (use Python `__del__` or context manager).

### Issue 4: Bitboard bit ordering

Edax uses specific bit ordering for board positions.
**Solution**: Test with known positions and verify move correspondence.

## Performance Expectations

- **Depth 4**: ~0.01s per move (100 moves/sec)
- **Depth 6**: ~0.1s per move (10 moves/sec)
- **Depth 10**: ~5-10s per move

Compare to:
- RAI depth 2: ~0.1s per move
- RAI depth 4: ~10s per move (Python overhead)

**Expected speedup**: 100-1000x faster than equivalent Python implementation.

## Next Steps (When Resuming)

1. **Create edax_wrapper.c** in `../edax-reversi/src/`
2. **Build shared library** using Edax's existing Makefile structure
3. **Create util/edax_engine.py** with ctypes bindings
4. **Write test_edax_bindings.py** and verify it works
5. **Update EdaxOpponent** in util/opponents.py
6. **Test integration** with training system
7. **Benchmark** performance vs RAI

## References

- Edax source: `../edax-reversi/`
- Current GTP implementation: `util/edax_gtp.py` (can be removed after bindings work)
- Python ctypes docs: https://docs.python.org/3/library/ctypes.html

## Session Context

This plan was created because GTP-based EdaxOpponent had board synchronization issues. The fundamental problem was detecting opponent moves by comparing states - after Edax's move is applied by the environment, the next state comparison includes both Edax's move and the agent's new move, making reliable detection impossible without game simulation logic.

Python bindings solve this by allowing us to simply call `move = edax.get_move(current_board_state)` without any move history tracking.
