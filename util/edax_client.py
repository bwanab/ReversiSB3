"""
Edax Client - Connects to Edax server for move requests.

This client is multiprocessing-safe because it only uses sockets,
no shared C library state.
"""

import socket
import json
import numpy as np


class EdaxClient:
    """Client for communicating with Edax server."""

    def __init__(self, socket_path="/tmp/edax_server.sock", depth=6):
        """Initialize client.

        Args:
            socket_path: Path to Unix socket where server is listening
            depth: Default search depth for this client
        """
        self.socket_path = socket_path
        self.depth = depth

    def get_move(self, state, depth=None):
        """Get best move from Edax server.

        Args:
            state: Numpy array shape (1, 8, 8) or (3, 8, 8)
                   Values: 1=current player, -1=opponent, 0=empty
            depth: Search depth (uses self.depth if None)

        Returns:
            int: Best move (0-63), or None if no legal moves
        """
        if depth is None:
            depth = self.depth

        # Extract board if multi-channel
        if state.ndim == 3 and state.shape[0] >= 1:
            board = state[0]
        elif state.ndim == 2:
            board = state
        else:
            raise ValueError(f"Invalid state shape: {state.shape}")

        # Prepare request
        request = {
            "state": board.tolist(),
            "depth": depth
        }

        # Send request to server
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)

            # Send request (newline-delimited JSON)
            sock.sendall(json.dumps(request).encode('utf-8') + b"\n")

            # Receive response
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            sock.close()

            # Parse response
            response = json.loads(data.decode('utf-8'))

            if "error" in response:
                raise RuntimeError(f"Edax server error: {response['error']}")

            return response["move"]

        except ConnectionRefusedError:
            raise RuntimeError(
                f"Cannot connect to Edax server at {self.socket_path}\n"
                f"Start the server with: python edax_server.py"
            )
        except Exception as e:
            raise RuntimeError(f"Error communicating with Edax server: {e}")

    def get_score(self):
        """Get score from last move.

        Note: Not implemented in client/server model.
        Would require storing last response.
        """
        return 0

    def get_nodes(self):
        """Get nodes from last move.

        Note: Not implemented in client/server model.
        Would require storing last response.
        """
        return 0

    def __repr__(self):
        return f"EdaxClient(depth={self.depth}, socket={self.socket_path})"
