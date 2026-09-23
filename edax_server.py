#!/usr/bin/env python3
"""
Edax Server - Standalone process for Edax engine.

Runs Edax in a dedicated process and accepts requests via Unix socket.
This avoids all multiprocessing/fork issues with SB3 training.

Protocol:
  Request:  {"state": [[...8x8 board...]], "depth": 6}
  Response: {"move": 37, "score": -5, "nodes": 2277351}
"""

import socket
import json
import sys
import os
import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.edax_engine import EdaxEngine
import numpy as np


class EdaxServer:
    """Server that runs Edax engine and responds to move requests."""

    def __init__(self, socket_path="/tmp/edax_server.sock"):
        self.socket_path = socket_path
        self.engines = {}  # Cache engines by depth

    def get_engine(self, depth):
        """Get or create engine at specified depth."""
        if depth not in self.engines:
            print(f"Creating Edax engine at depth {depth}", file=sys.stderr)
            self.engines[depth] = EdaxEngine(depth=depth)
        return self.engines[depth]

    def handle_request(self, request):
        """Process a move request and return response."""
        try:
            # Parse request
            state = np.array(request["state"], dtype=np.int8)
            depth = request.get("depth", 6)

            # Get move from engine
            engine = self.get_engine(depth)
            move = engine.get_move(state)

            # Get additional info
            score = engine.get_score() if move is not None else 0
            nodes = engine.get_nodes() if move is not None else 0

            return {
                "move": int(move) if move is not None else None,
                "score": int(score),
                "nodes": int(nodes)
            }

        except Exception as e:
            return {
                "error": str(e),
                "move": None
            }

    def start(self):
        """Start the server."""
        # Remove old socket if exists
        try:
            os.unlink(self.socket_path)
        except OSError:
            if os.path.exists(self.socket_path):
                raise

        # Create Unix socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(5)

        print(f"Edax server listening on {self.socket_path}", file=sys.stderr)

        # Handle shutdown gracefully
        def shutdown_handler(signum, frame):
            print("\nShutting down Edax server...", file=sys.stderr)
            server.close()
            os.unlink(self.socket_path)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        # Accept connections
        while True:
            try:
                conn, _ = server.accept()

                # Read request (newline-delimited JSON)
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break

                if not data:
                    conn.close()
                    continue

                # Process request
                request = json.loads(data.decode('utf-8'))
                response = self.handle_request(request)

                # Send response
                conn.sendall(json.dumps(response).encode('utf-8') + b"\n")
                conn.close()

            except Exception as e:
                print(f"Error handling request: {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Edax engine server")
    parser.add_argument("--socket", default="/tmp/edax_server.sock",
                       help="Unix socket path (default: /tmp/edax_server.sock)")
    args = parser.parse_args()

    server = EdaxServer(socket_path=args.socket)
    print(f"Starting Edax server...", file=sys.stderr)
    server.start()
