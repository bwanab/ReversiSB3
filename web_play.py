#!/usr/bin/env python3
"""
Web interface for playing Reversi against a trained model: a page (templates/, static/) and a JSON
API (web_app_spec.md). Several people can play at once; each game has an unguessable id that the
page sends with every request.

Hardened for running in public (2026-10-09 audit): games by id with per-game locks; inputs validated
(moves must be legal and it must be the person's turn); generic error messages; limits on games, idle
time, request size and request rate; model moves computed one at a time (shared engine lock);
security headers; the debugger only on localhost; game records named by server-made ids only.
For a public site run it behind a reverse proxy with TLS (nginx, Caddy) and --trust-proxy.

Games live in a store (web_store.py): in memory by default, or in Redis (--redis-url) so several
instances can share them (serverless hosting).
"""

import argparse
import contextlib
import json
import logging
import os
import re
import secrets
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

from util.levels import LEVELS, STRENGTH, STRONGEST, describe, make_player
from util.search import SearchPlayer, legal_moves, play_move, policy_logits
from web_store import MemoryStore, RedisStore, StoreBusy

log = logging.getLogger("web_play")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096          # requests are tiny JSON bodies

BLACK, WHITE = 1, -1
COLOR_NAME = {BLACK: "black", WHITE: "white"}
GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
GAME_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class Config:
    """Server settings (set from the command line in main())."""
    model = None
    model_name = ""
    default_level = STRONGEST
    custom_player = None          # from explicit search options: level "custom"
    custom_desc = ""
    record = True
    max_games = 200
    idle_timeout = 2 * 3600       # seconds without a request before a game is dropped
    rate_per_minute = 120         # API requests per client address per minute
    new_games_per_minute = 10


ENGINE_LOCK = threading.Lock()    # model moves and probabilities are computed one at a time (per instance)
PLAYERS = {}                      # level -> player, created on first use (under ENGINE_LOCK)
STORE = MemoryStore()             # games and rate-limit counters (web_store.py); main() may switch to Redis


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def notation(action):
    """Action 0-63 -> 'A1'..'H8' (column letter, row number; row 1 is the top row)."""
    return f"{'ABCDEFGH'[action % 8]}{action // 8 + 1}"


def player_for(level):
    if level == "custom":
        return Config.custom_player
    if level not in PLAYERS:
        PLAYERS[level] = make_player(Config.model, level)
    return PLAYERS[level]


def level_desc(level):
    return Config.custom_desc if level == "custom" else f"level {level}: {describe(level)}"


def parse_level(value):
    if value == "custom" and Config.custom_player is not None:
        return "custom"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ApiError(f"level must be 1-{STRONGEST}")
    try:
        level = int(value)
    except ValueError:
        raise ApiError(f"level must be 1-{STRONGEST}")
    if level not in LEVELS:
        raise ApiError(f"level must be 1-{STRONGEST}")
    return level


class Game:
    """One game: the board as 64 values (1 = black, -1 = white, row-major from the top left)."""

    def __init__(self, model_color, level, game_id=None):
        self.id = game_id or secrets.token_urlsafe(18)
        self.model_color, self.level = model_color, level
        self.board = np.zeros(64, dtype=np.int8)
        self.board[[27, 36]], self.board[[28, 35]] = WHITE, BLACK
        self.to_move = BLACK
        self.history = []             # {action, color, by, board_before, method}
        self.redo = []                # actions of undone human moves
        self.over, self.winner = False, None
        self.last_move = None         # info on the model's latest move
        self.model_passed = False
        self.started = time.strftime("%Y%m%d-%H%M%S")

    def to_dict(self):
        """What the store keeps: the moves, not the boards (from_dict replays them by the rules)."""
        return {"id": self.id, "model_color": self.model_color, "level": self.level, "started": self.started,
                "moves": [[m["action"], 1 if m["color"] == "black" else -1, m["by"], m.get("method")]
                          for m in self.history],
                "redo": self.redo, "last_move": self.last_move, "model_passed": self.model_passed}

    @classmethod
    def from_dict(cls, d):
        game = cls(d["model_color"], d["level"], d["id"])
        for action, color, by, method in d["moves"]:
            if game.over or color != game.to_move or action not in set(int(m) for m in game.legal()):
                raise ValueError(f"stored game {d['id']} doesn't replay")
            game.apply(action, by, method)
        game.started, game.redo = d["started"], list(d["redo"])
        game.last_move, game.model_passed = d["last_move"], d["model_passed"]
        return game

    @property
    def human_color(self):
        return -self.model_color

    def legal(self, color=None):
        color = self.to_move if color is None else color
        return legal_moves(self.board * color)

    def apply(self, action, by, method=None):
        before = self.board.copy()
        color = self.to_move
        self.board = (-play_move(self.board * color, action) * color).astype(np.int8)
        self.history.append({"action": int(action), "color": COLOR_NAME[color], "by": by,
                             "board_before": before, **({"method": method} if method else {})})
        # next to move: the opponent, unless it has to pass; game over if neither can move
        if len(self.legal(-color)):
            self.to_move = -color
        elif len(self.legal(color)):
            self.to_move = color
        else:
            self.over = True
            diff = int((self.board == BLACK).sum() - (self.board == WHITE).sum())
            self.winner = "black" if diff > 0 else "white" if diff < 0 else "draw"

    def model_moves(self):
        """Let the model move while it is its turn (it moves again when the person must pass)."""
        while not self.over and self.to_move == self.model_color:
            action, info = model_move(self)
            self.apply(action, "model", info["method"])
            self.last_move = {"action": action, "player": "model", **info}

    def human_move(self, action, from_redo=False):
        if self.over:
            raise ApiError("the game is over")
        if self.to_move != self.human_color:
            raise ApiError("it is not your turn")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 64:
            raise ApiError("action must be an integer 0-63")
        if action not in set(int(m) for m in self.legal()):
            raise ApiError("illegal move")
        if not from_redo:
            self.redo = []
        self.apply(action, "human")
        self.last_move = None
        self.model_passed = not self.over and self.to_move == self.human_color
        self.model_moves()

    def undo(self):
        """Take back the person's last move and the model's replies after it."""
        human = [i for i, m in enumerate(self.history) if m["by"] == "human"]
        if not human:
            raise ApiError("no moves to undo")
        entry = self.history[human[-1]]
        self.redo.append(entry["action"])
        self.board = entry["board_before"].copy()
        del self.history[human[-1]:]
        self.to_move, self.over, self.winner, self.model_passed = self.human_color, False, None, False
        model = [m for m in self.history if m["by"] == "model"]
        self.last_move = {"action": model[-1]["action"], "player": "model",
                          "method": model[-1].get("method", "")} if model else None

    def redo_move(self):
        if not self.redo:
            raise ApiError("no moves to redo")
        self.human_move(self.redo.pop(), from_redo=True)

    def move_probabilities(self):
        """The policy network's probability for each legal move of the side to move."""
        legal = self.legal()
        with ENGINE_LOCK:
            lg = policy_logits(Config.model, (self.board * self.to_move)[None])[0][legal].astype(np.float64)
        p = np.exp(lg - lg.max())
        return dict(zip((int(m) for m in legal), (float(x) for x in p / p.sum())))

    def state(self):
        """The JSON state sent after every request."""
        human_turn = not self.over and self.to_move == self.human_color
        probs = self.move_probabilities() if human_turn else {}
        valid = [{"row": a // 8, "col": a % 8, "action": a, **({"probability": probs[a]} if a in probs else {})}
                 for a in (int(m) for m in self.legal())] if human_turn else []
        out = {"game_id": self.id, "board": self.board.reshape(8, 8).tolist(),
               "current_player": COLOR_NAME[self.to_move], "valid_moves": valid,
               "piece_count": {"black": int((self.board == BLACK).sum()), "white": int((self.board == WHITE).sum())},
               "game_over": self.over, "winner": self.winner, "model_color": COLOR_NAME[self.model_color],
               "level": self.level, "can_undo": any(m["by"] == "human" for m in self.history),
               "can_redo": bool(self.redo) and not self.over}
        if self.last_move:
            out["last_move"] = self.last_move
        if self.model_passed:
            out["model_passed"] = True
        return out

    def save(self):
        """Write the game as it stands to games/<start time>_<id prefix>.json (see web_app_spec.md)."""
        if not Config.record or not self.history:
            return
        moves = [{"square": notation(m["action"]), "action": m["action"], "color": m["color"], "by": m["by"],
                  "board_before": m["board_before"].astype(int).tolist(),
                  **({"method": m["method"]} if m.get("method") else {})} for m in self.history]
        record = {"model": Config.model_name, "player": level_desc(self.level), "level": self.level,
                  "model_color": COLOR_NAME[self.model_color], "moves": moves,
                  "final_board": self.board.astype(int).tolist(), "black": int((self.board == BLACK).sum()),
                  "white": int((self.board == WHITE).sum()), "finished": self.over}
        os.makedirs(GAMES_DIR, exist_ok=True)
        path = os.path.join(GAMES_DIR, f"{self.started}_{self.id[:8]}.json")   # server-made name only
        try:
            with open(path, "w") as f:
                json.dump(record, f, indent=1)
        except OSError:
            log.exception("could not save game record %s", path)


def model_move(game):
    """The model's move in game (side to move = the model) and how it was chosen."""
    flat = game.board * game.to_move
    empties = int((flat == 0).sum())
    legal = legal_moves(flat)
    with ENGINE_LOCK:
        player = player_for(game.level)
        action = int(player.choose_many([flat])[0])
        lg = policy_logits(Config.model, flat[None])[0][legal].astype(np.float64)
        exact = None
        if player.solve_empties and empties <= player.solve_empties:
            from util.endgame import solve
            exact = int(solve(flat)[0])
    p = np.exp(lg - lg.max())
    p /= p.sum()
    order = np.argsort(-p)[:5]
    analysis = [{"notation": notation(int(legal[i])), "probability": float(p[i]), "action": int(legal[i])}
                for i in order]
    if all(a["action"] != action for a in analysis):
        i = int(np.flatnonzero(legal == action)[0])
        analysis.append({"notation": notation(action), "probability": float(p[i]), "action": action})
    if exact is not None:
        method = f"solver ({empties} empty)"
    elif player.mcts is not None:
        method = f"MCTS {player.mcts.sims} simulations"
    elif getattr(player, "temperature", None):
        method = f"policy sample (temperature {player.temperature})"
    elif player.depth > 0:
        depth = player.early_depth if player.early_depth and empties > player.early_above else player.depth
        method = f"search depth {depth}"
    else:
        method = "network"
    if game.level != "custom":
        method = f"level {game.level}: {method}"
    info = {"method": method, "analysis": analysis}
    if exact is not None:
        info["exact_score"] = exact
    return action, info


# ---- request handling ----

def rate_limit(kind, per_minute):
    if not STORE.hit(f"{kind}:{request.remote_addr or '?'}", per_minute):
        raise ApiError("too many requests; slow down", 429)


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("expected a JSON object")
    return data


def game_id_of(value):
    if not isinstance(value, str) or not GAME_ID_RE.match(value):
        raise ApiError("missing or malformed game_id")
    return value


@contextlib.contextmanager
def open_game(game_id, write=True):
    """The game, locked for this request (across instances with Redis); saved back afterwards (which
    also restarts its idle timeout)."""
    try:
        with STORE.locked(game_id):
            data = STORE.load(game_id)
            if data is None:
                raise ApiError("unknown or expired game; start a new game", 404)
            game = Game.from_dict(data)
            yield game
            if write:
                STORE.save(game_id, game.to_dict(), Config.idle_timeout)
                game.save()
            else:
                STORE.save(game_id, data, Config.idle_timeout)
    except StoreBusy:
        raise ApiError("the game is busy with another request; try again", 409)


@app.before_request
def limit_api_rate():
    if request.path.startswith("/api/"):
        rate_limit("api", Config.rate_per_minute)


@app.errorhandler(ApiError)
def api_error(e):
    return jsonify({"error": e.message}), e.status


@app.errorhandler(404)
def not_found(e):
    return (jsonify({"error": "not found"}), 404) if request.path.startswith("/api/") else (e.get_response(), 404)


@app.errorhandler(405)
def bad_method(e):
    return jsonify({"error": "method not allowed"}), 405


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "request too large"}), 413


@app.errorhandler(Exception)
def internal_error(e):
    log.exception("unhandled error on %s", request.path)
    return jsonify({"error": "internal error"}), 500


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = body()
    rate_limit("new_game", Config.new_games_per_minute)
    color = data.get("model_color", "black")
    if color not in ("black", "white"):
        raise ApiError("model_color must be 'black' or 'white'")
    level = Config.default_level if data.get("level") is None else parse_level(data["level"])
    game = Game(BLACK if color == "black" else WHITE, level)
    if not STORE.create(game.id, game.to_dict(), Config.idle_timeout):
        raise ApiError("the server is busy; try again later", 503)
    with open_game(game.id) as game:
        game.model_moves()
        return jsonify(game.state())


@app.route("/api/make_move", methods=["POST"])
def make_move():
    data = body()
    with open_game(game_id_of(data.get("game_id"))) as game:
        game.human_move(data.get("action"))
        return jsonify(game.state())


@app.route("/api/undo", methods=["POST"])
def undo():
    with open_game(game_id_of(body().get("game_id"))) as game:
        game.undo()
        return jsonify(game.state())


@app.route("/api/redo", methods=["POST"])
def redo():
    with open_game(game_id_of(body().get("game_id"))) as game:
        game.redo_move()
        return jsonify(game.state())


@app.route("/api/game_state", methods=["GET"])
def game_state():
    with open_game(game_id_of(request.args.get("game_id")), write=False) as game:
        return jsonify(game.state())


@app.route("/api/set_level", methods=["POST"])
def set_level():
    data = body()
    gid, level = game_id_of(data.get("game_id")), parse_level(data.get("level"))
    with open_game(gid) as game:
        game.level = level
        return jsonify({"game_id": game.id, "level": level, "description": level_desc(level)})


@app.route("/api/levels", methods=["GET"])
def levels():
    out = [{"level": n, "description": describe(n), "strength": STRENGTH[n]} for n in sorted(LEVELS)]
    if Config.custom_player is not None:
        out.append({"level": "custom", "description": Config.custom_desc})
    return jsonify({"levels": out, "default": Config.default_level})


@app.route("/api/get_moves", methods=["GET"])
def get_moves():
    with open_game(game_id_of(request.args.get("game_id")), write=False) as game:
        moves = [{"number": i + 1, "player": m["color"], "by": m["by"], "action": m["action"],
                  "notation": notation(m["action"])} for i, m in enumerate(game.history)]
    return jsonify({"moves": moves, "total": len(moves)})


def main():
    parser = argparse.ArgumentParser(description="Web interface for Reversi (API: web_app_spec.md)")
    parser.add_argument("-m", "--model", required=True, help="model name under models/ (without .zip)")
    parser.add_argument("-p", "--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (127.0.0.1: this machine only; for a public site keep it local "
                             "and put a reverse proxy with TLS in front, or use 0.0.0.0 inside a container)")
    parser.add_argument("--device", default="auto", help="auto (CUDA, else MPS, else CPU), or cpu / mps / cuda")
    parser.add_argument("--level", type=int, default=STRONGEST, choices=sorted(LEVELS),
                        help=f"default strength level for new games (util/levels.py); default {STRONGEST}")
    parser.add_argument("--strong", action="store_true", help=f"default level {STRONGEST} (the strongest)")
    parser.add_argument("--search-depth", type=int, default=0,
                        help="custom player: negamax search this deep (pruned at every level)")
    parser.add_argument("--search-top-k", type=int, default=3, help="custom player: moves considered per level")
    parser.add_argument("--search-depth-early", type=int, default=0,
                        help="custom player: search this deep while more than --early-above squares are empty")
    parser.add_argument("--early-above", type=int, default=30)
    parser.add_argument("--solve-empties", type=int, default=0,
                        help="custom player: exact endgame solver at <= this many empty squares")
    parser.add_argument("--leaf-solve-empties", type=int, default=0,
                        help="custom player: with search/MCTS, score positions with <= this many empties exactly")
    parser.add_argument("--mcts-sims", type=int, default=0, help="custom player: MCTS with this many simulations")
    parser.add_argument("--no-record", action="store_true", help="don't save games to games/")
    parser.add_argument("--max-games", type=int, default=Config.max_games,
                        help="concurrent games kept in memory (in-memory store only)")
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL"),
                        help="keep games, locks and rate limits in Redis (e.g. redis://host:6379/0; default: the "
                             "REDIS_URL environment variable), so several instances can serve the same games")
    parser.add_argument("--idle-minutes", type=float, default=Config.idle_timeout / 60,
                        help="drop a game after this long without requests")
    parser.add_argument("--rate", type=int, default=Config.rate_per_minute,
                        help="API requests per client address per minute")
    parser.add_argument("--trust-proxy", action="store_true",
                        help="behind one reverse proxy: take the client address from X-Forwarded-For")
    parser.add_argument("--debug", action="store_true",
                        help="Flask debug mode (interactive debugger: only allowed on 127.0.0.1/localhost)")
    parser.add_argument("-w", "--net-width", type=int, default=512, help=argparse.SUPPRESS)   # legacy, unused
    args = parser.parse_args()
    if args.debug and args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--debug runs an interactive debugger that executes code; only use it on localhost")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from sb3_contrib import MaskablePPO
    from util.reversi import build_reversi
    from util.util import get_device
    device = get_device() if args.device == "auto" else args.device
    print(f"Loading model: {args.model}")
    Config.model = MaskablePPO.load(f"models/{args.model}", env=build_reversi("Random"), device=device)
    Config.model_name = args.model
    Config.default_level = STRONGEST if args.strong else args.level
    Config.record, Config.max_games = not args.no_record, args.max_games
    Config.idle_timeout, Config.rate_per_minute = args.idle_minutes * 60, args.rate
    global STORE
    STORE = RedisStore(args.redis_url) if args.redis_url else MemoryStore(args.max_games)
    STORE_DESC = "Redis" if args.redis_url else f"memory (max {args.max_games} games)"

    searching = args.search_depth or args.mcts_sims
    if searching or args.solve_empties:
        Config.custom_player = SearchPlayer(
            Config.model, depth=args.search_depth, top_k=args.search_top_k, prune_all=True,
            solve_empties=args.solve_empties, leaf_solve_empties=args.leaf_solve_empties if searching else 0,
            early_depth=args.search_depth_early, early_above=args.early_above, mcts_sims=args.mcts_sims)
        parts = [f"MCTS {args.mcts_sims} simulations" if args.mcts_sims else
                 f"search depth {args.search_depth}" if args.search_depth else "network top move"]
        if args.search_depth_early:
            parts.append(f"depth {args.search_depth_early} above {args.early_above} empties")
        if args.leaf_solve_empties and searching:
            parts.append(f"leaf solves <= {args.leaf_solve_empties}")
        if args.solve_empties:
            parts.append(f"solver <= {args.solve_empties} empties")
        Config.custom_desc = "custom: " + ", ".join(parts)
        Config.default_level = "custom"

    if args.trust_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    print(f"\n{'=' * 60}\nReversi Web Interface\n{'=' * 60}")
    print(f"Model: {args.model} (device {device})")
    print(f"Default player: {level_desc(Config.default_level)}")
    print(f"Games kept in {STORE_DESC}; " + (f"recorded to {GAMES_DIR}" if Config.record else "not recorded"))
    print(f"\nStarting server at http://{args.host}:{args.port}\n{'=' * 60}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
