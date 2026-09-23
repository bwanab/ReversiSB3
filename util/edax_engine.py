"""
Python bindings to Edax engine via ctypes.

Provides a clean Python interface to the Edax Reversi engine compiled
as a shared library. Used by edax_server.py for the client/server architecture.
"""

import ctypes
import numpy as np
import os


class EdaxEngine:
    """Python interface to Edax Reversi engine.

    This class provides a simple stateless interface: pass a board state,
    get back the best move. No synchronization or move history tracking needed.

    Example:
        >>> edax = EdaxEngine(depth=6)
        >>> # state shape (1, 8, 8) or (3, 8, 8), values: 1=current player, -1=opponent, 0=empty
        >>> move = edax.get_move(state)  # Returns 0-63, or None if no legal moves
        >>> score = edax.get_score()     # Get evaluation score
    """

    def __init__(self, depth=6, lib_path=None):
        """Initialize Edax engine.

        Args:
            depth: Search depth (1-60, typical range 4-20)
                  - Depth 4: ~0.01s per move
                  - Depth 6: ~0.1s per move
                  - Depth 10: ~5-10s per move
            lib_path: Path to libedax.dylib (auto-detected if None)
        """
        # Find library
        if lib_path is None:
            # Try adjacent to edax-reversi directory
            edax_bin = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "../edax-reversi/bin/libedax.dylib"
            )
            lib_path = os.path.normpath(edax_bin)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Edax library not found at: {lib_path}\n"
                f"Expected location: ../edax-reversi/bin/libedax.dylib"
            )

        # Load shared library
        self.lib = ctypes.CDLL(lib_path)

        # Define function signatures
        self.lib.edax_create.argtypes = [ctypes.c_int]
        self.lib.edax_create.restype = ctypes.c_void_p

        self.lib.edax_get_move.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64
        ]
        self.lib.edax_get_move.restype = ctypes.c_int

        self.lib.edax_get_score.argtypes = [ctypes.c_void_p]
        self.lib.edax_get_score.restype = ctypes.c_int

        self.lib.edax_get_nodes.argtypes = [ctypes.c_void_p]
        self.lib.edax_get_nodes.restype = ctypes.c_uint64

        self.lib.edax_destroy.argtypes = [ctypes.c_void_p]
        self.lib.edax_destroy.restype = None

        # Create engine instance
        self.engine = self.lib.edax_create(depth)
        if not self.engine:
            raise RuntimeError(f"Failed to create Edax engine at depth {depth}")

        self.depth = depth

    def _state_to_bitboards(self, state):
        """Convert numpy board state to bitboards.

        Args:
            state: Numpy array shape (1, 8, 8) or (3, 8, 8)
                   Values: 1=current player, -1=opponent, 0=empty

        Returns:
            (player_bits, opponent_bits): Two uint64 bitboards
        """
        # Extract board (first channel if multi-channel)
        if state.ndim == 3 and state.shape[0] >= 1:
            board = state[0]
        elif state.ndim == 2:
            board = state
        else:
            raise ValueError(f"Invalid state shape: {state.shape}")

        if board.shape != (8, 8):
            raise ValueError(f"Board must be 8x8, got {board.shape}")

        player_bits = 0
        opponent_bits = 0

        for row in range(8):
            for col in range(8):
                # Bit position: row * 8 + col (0-63)
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
                 Index mapping: row * 8 + col
                 Example: a1=0, h1=7, a8=56, h8=63
        """
        player_bits, opponent_bits = self._state_to_bitboards(state)

        move = self.lib.edax_get_move(
            self.engine,
            ctypes.c_uint64(player_bits),
            ctypes.c_uint64(opponent_bits)
        )

        # -1 indicates no legal moves (pass)
        return move if move >= 0 else None

    def get_score(self):
        """Get evaluation score for last position searched.

        Returns:
            int: Evaluation score in centidisks (1/100th of a disk)
                 Positive = favorable for current player
                 Negative = favorable for opponent
        """
        return self.lib.edax_get_score(self.engine)

    def get_nodes(self):
        """Get number of nodes searched in last search.

        Returns:
            int: Node count
        """
        return self.lib.edax_get_nodes(self.engine)

    def __del__(self):
        """Clean up engine on deletion."""
        if hasattr(self, 'engine') and self.engine:
            self.lib.edax_destroy(self.engine)
            self.engine = None

    def __repr__(self):
        return f"EdaxEngine(depth={self.depth})"
