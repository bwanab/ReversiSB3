"""
Egaroucid (https://www.egaroucid.nyanyan.dev/) as an opponent, through its console mode.

Egaroucid for Console is built from source (see CLAUDE.md, "Egaroucid"); its path comes from the
EGAROUCID environment variable or defaults to ~/src/Egaroucid/bin/Egaroucid_for_Console.out. It runs
from its bin/ directory so it finds resources/ (evaluation weights).

Console mode rather than GTP: our evaluations start from arbitrary stored positions, which
`setboard` loads directly; GTP can only reach a position by replaying moves from the start.
The side to move is always presented as black (X), so the same code serves either color.

Levels 1-10 are full-width searches of that many moves (no selectivity), with exact endgame search
near the end; higher levels prune selectively and search deeper (`-levelinfo`).
"""

import os
import subprocess

import numpy as np

DEFAULT_PATH = os.path.expanduser("~/src/Egaroucid/bin/Egaroucid_for_Console.out")


class EgaroucidClient:
    """One Egaroucid process. analyze(board, depth) mirrors EdaxClient: board is (8, 8) or (64,)
    from the side to move's view (1 = side to move); depth is Egaroucid's level."""

    def __init__(self, path=None, threads=4, book=False):
        path = path or os.environ.get("EGAROUCID", DEFAULT_PATH)
        if not os.path.exists(path):
            raise RuntimeError(f"Egaroucid not found at {path}; build it (CLAUDE.md, 'Egaroucid') "
                               f"or set EGAROUCID")
        args = [path, "-q", "-t", str(threads)] + ([] if book else ["-nobook"])
        self.proc = subprocess.Popen(args, cwd=os.path.dirname(path), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.level = None

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def analyze(self, board, depth):
        board = np.asarray(board).reshape(64)
        if depth != self.level:
            self._send(f"level {depth}")
            self.level = depth
        squares = "".join("X" if v == 1 else ("O" if v == -1 else "-") for v in board)
        self._send(f"setboard {squares} X")
        self._send("hint 1")
        # output: a header row then one data row "| level | depth | move | score | ..."
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Egaroucid exited unexpectedly")
            fields = [f.strip() for f in line.split("|")]
            if len(fields) > 5 and fields[1].isdigit():
                move, score = fields[3], fields[4]
                col, row = "abcdefgh".index(move[0].lower()), int(move[1]) - 1
                return {"move": row * 8 + col, "score": int(score)}

    def analyze_all(self, board, depth, n_moves):
        """Scores (discs, side to move's view) for the n_moves best moves: {square: score}.
        Pass n_moves = the number of legal moves to score every move (Egaroucid's `hint n`)."""
        board = np.asarray(board).reshape(64)
        if depth != self.level:
            self._send(f"level {depth}")
            self.level = depth
        squares = "".join("X" if v == 1 else ("O" if v == -1 else "-") for v in board)
        self._send(f"setboard {squares} X")
        self._send(f"hint {n_moves}")
        scores = {}
        while len(scores) < n_moves:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Egaroucid exited unexpectedly")
            fields = [f.strip() for f in line.split("|")]
            if len(fields) > 5 and fields[1].isdigit():
                move = fields[3]
                scores["abcdefgh".index(move[0].lower()) + (int(move[1]) - 1) * 8] = int(fields[4])
        return scores

    def get_move(self, state, depth=None):
        return self.analyze(state, depth or self.level or 8)["move"]

    def close(self):
        if self.proc.poll() is None:
            try:
                self._send("exit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
