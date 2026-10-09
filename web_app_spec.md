# web_play.py: web interface and HTTP API

`web_play.py` is a Flask app for playing against a trained model in the browser. It serves one page
(`templates/index.html`, `static/script.js`, `static/style.css`) and a JSON API that the page calls.
Several people can play at once: each game has an id. This document describes the API as implemented
(2026-10-09).

It replaces `web_app_spec.txt`, the spec of an earlier Julia web service (`/start/...`, `/play/...`,
`/get_action/...`) that guided the first version of this app. The Python app departed from it (POST
requests with JSON bodies, undo/redo) and at first kept one global game; game ids came back with the
2026-10-09 security audit (see "Security").

## Running

```bash
uv run python web_play.py -m r256x12_mid1_CNN_test               # then open http://127.0.0.1:5000
```

| Option | Meaning |
| --- | --- |
| `-m/--model` (required) | model under `models/`, without `.zip` |
| `--level N` | default strength level for new games, 1 (weakest) to 10 (strongest, the default) |
| `--strong` | default level 10 |
| `--mcts-sims N`, `--search-depth N`, `--search-top-k K`, `--search-depth-early N`, `--early-above E`, `--solve-empties N`, `--leaf-solve-empties N` | define a **custom** player, offered as level `"custom"` and made the default |
| `--device` | `auto` (CUDA, else MPS, else CPU), `cpu`, `mps`, `cuda` |
| `-p/--port`, `--host` | default 5000 on 127.0.0.1 (this machine only) |
| `--no-record` | don't save games to `games/` |
| `--max-games N` | in-memory store: games kept at once (default 200); new games get HTTP 503 when full |
| `--redis-url URL` | keep games, locks and rate limits in Redis (default: the `REDIS_URL` environment variable); see "Deployment" |
| `--idle-minutes M` | a game with no requests for this long is dropped (default 120) |
| `--rate N` | API requests per client address per minute (default 120); new games: 10 per minute |
| `--trust-proxy` | behind one reverse proxy: take the client address from `X-Forwarded-For` |
| `--debug` | Flask debug mode; refused unless `--host` is local (its debugger can run code) |

## Strength levels

Defined in `util/levels.py` (also `eval_batch.py --level N`); level 10 is the strongest player.

| Level | Player | Strength (measured, step 31) |
| --- | --- | --- |
| 1 | policy network, sampled at temperature 1.5 | well below Edax depth 1 |
| 2 | policy network, sampled at temperature 0.7 | about Edax depth 1 |
| 3 | policy network, top move | about Edax depth 2 |
| 4 | MCTS 16 simulations | about Edax depth 4 |
| 5 | MCTS 32, solver <= 10 empties | about Edax depth 5 |
| 6 | MCTS 64, solver <= 12 | about Edax depth 7 |
| 7 | MCTS 100, solver <= 14 | about Edax depth 8-9 |
| 8 | MCTS 200, solver <= 16, leaf solves <= 14 | about Edax depth 10-11 |
| 9 | MCTS 400, solver <= 18, leaf solves <= 16 | about Edax depth 13 |
| 10 | MCTS 800, solver <= 18, leaf solves <= 16 | about Edax depth 15; even with Egaroucid level 12 |

"About Edax depth N": wins roughly half its games against Edax searching N moves deep, from 200
balanced openings, with `r256x12_mid1`. MCTS levels stop early when the move is decided (step 30).

A game's level can change at any time (`/api/set_level`); it applies from the model's next move. Each
model move's `method` starts with its level (e.g. `level 7: MCTS 100 simulations`).

## Page options

- **Model plays as**: black or white, for the next new game.
- **Strength**: the level (1-10, or Custom); sent with "New Game", and changes during a game apply to
  that game from the model's next move.
- **Save game**: downloads the game as a small JSON file (`reversi-<date>-<time>.json`, see
  `/api/export`), e.g. to continue a long game against a person elsewhere after a wait.
- **Load game**: continues a saved file (with its own model color and level), or a plain move list
  such as `f5d6c3d3` in a `.txt` file (the model takes the color and level chosen in the controls).
  Loading starts a new game (new id) at that position; if it is the model's turn, it moves.
- **Show hints**: shows or hides the model's move probabilities on your candidate moves, its list of
  top policy moves, and the solver's verdict ("model wins by N with perfect play"). Display only: the
  server always sends them. Remembered per browser (`localStorage`).

## Games and ids

- `POST /api/new_game` creates a game and returns its `game_id`: 24 random URL-safe characters
  (`secrets.token_urlsafe(18)`). Every other request about the game must carry it: in the JSON body
  for POSTs, as `?game_id=` for GETs.
- The id is the game's only credential: whoever has it can play the game. It is never listed by the
  server and isn't written in full to disk.
- The person plays one color, the model the other. The server answers each of the person's moves with
  the model's reply in the same response, including any passes (the model moves again when the person
  must pass). It is therefore always the person's turn after a request, unless the game is over.
- Games live in a store (`web_store.py`): in memory by default (lost when the server restarts), or in
  Redis. Either way a game is dropped after `--idle-minutes` without requests (then requests get HTTP
  404, "unknown or expired game").
- A game is stored as its list of moves (plus redo list, level and last-move info) and rebuilt by
  replaying the moves with the rules on every request, so a stored game is always consistent.
- Requests on the same game are handled one at a time (a per-game lock, held across instances with
  Redis); a request that can't get the lock within 60 s gets HTTP 409.

## Data conventions

| Item | Format |
| --- | --- |
| Board | 8x8 list of rows; `board[row][col]` is `1` (black), `-1` (white) or `0` (empty). Row 0 is the top row |
| Action | integer `row * 8 + col` (0-63) |
| Notation | column letter + row number: action 0 = `A1` (top left), 19 = `D3`, 63 = `H8` |
| Color | `"black"` / `"white"` (black moves first) |
| Winner | `"black"`, `"white"`, `"draw"`, or `null` while the game goes on |

**State** (the response of `new_game`, `make_move`, `undo`, `redo`, `game_state`):

```json
{
  "game_id": "E6eKEOm3foU-1hmBX0Y1FTSY",
  "board": [[0, 0, ...], ...],
  "current_player": "black",
  "valid_moves": [{"row": 2, "col": 3, "action": 19, "probability": 0.41}, ...],
  "piece_count": {"black": 4, "white": 1},
  "game_over": false,
  "winner": null,
  "model_color": "white",
  "level": 10,
  "can_undo": true,
  "can_redo": false,
  "model_moves": [37],
  "last_move": { ... },
  "model_passed": true
}
```

- `valid_moves`: the person's legal moves (empty when the game is over). `probability` is the model's
  *policy network* probability for the move: the network's guess at what the person will play, not an
  evaluation.
- `last_move` (present once the model has moved): the model's latest move.

  ```json
  {"action": 37, "player": "model", "method": "level 10: MCTS 800 simulations",
   "analysis": [{"notation": "F5", "probability": 0.62, "action": 37}, ...], "exact_score": 4}
  ```

  `analysis`: the policy network's top 5 moves (plus the move played if it wasn't among them), i.e.
  what the network alone would play. `method`: `network`, `policy sample (temperature T)`,
  `MCTS N simulations`, `search depth N` or `solver (N empty)`, prefixed with the level unless the
  custom player is used. `exact_score`: with the solver, the final disc difference for the model with
  perfect play. After an undo, `last_move` is the model's latest remaining move, without `analysis`.
- `model_moves`: the model's moves since the person's last move, in the order played. Usually one; two
  or more when the person had to pass in between (the page then colors the earlier ones amber and
  numbers all of them 1, 2, ...). Empty after an undo.
- `model_passed: true`: the model had no legal move after the person's move.

**Errors**: a JSON object `{"error": "<message>"}` with HTTP status:

- 400: malformed request (not a JSON object, missing or malformed `game_id`, `action` not an integer
  0-63, illegal move, not your turn, game over, bad `model_color` or `level`, nothing to undo/redo);
- 404: unknown or expired game, or unknown API path;
- 405: wrong method; 409: the game is busy with another request; 413: request body over 4 KB;
  429: too many requests; 503: game limit reached;
- 500: internal error (always just `"internal error"`; details go to the server log).

## Endpoints

### `GET /`

The game page.

### `POST /api/new_game`

Request: `{"model_color": "black" | "white", "level": 1-10 | "custom"}`, both optional (defaults:
`"black"` and the server's default level). Response: the state of the new game, after the model's
first move if the model plays black.

### `POST /api/make_move`

Request: `{"game_id": "...", "action": 0-63}`. The move must be legal and it must be the person's turn.
Response: the state after the move and the model's reply.

### `POST /api/undo`

Request: `{"game_id": "..."}`. Takes back the person's last move together with the model's replies
after it, so it is the person's turn again in the position before that move. Response: the state.

### `POST /api/redo`

Request: `{"game_id": "..."}`. Replays the most recently undone move of the person; the model then
replies again (at higher levels and with sampling it may choose differently). A new move clears the
redo list. Response: the state.

### `POST /api/load_game`

Request: `{"moves": "f5d6c3..." | ["f5", "d6", ...] | [37, 43, ...], "model_color": "black" | "white",
"level": 1-10 | "custom"}` (`model_color` and `level` optional, as in `new_game`). A new game replaying
the moves from the standard start: each move must be legal for the side to move; passes are inferred
(when the side to move has no legal move, the other side moves). Square names are case-insensitive;
spaces, commas, dashes and dots between them are ignored; at most 60 moves. If the model is then to
move, it moves. Response: the state of the new game. Errors (400) name the first illegal move, e.g.
`"move 2 (f5) is illegal for white"`.

### `GET /api/export?game_id=...`

What the page saves:

```json
{"format": "reversisb3-game", "version": 1, "moves": "f5d6c3d3c4", "model_color": "black",
 "level": 10, "model": "r256x12_mid1_CNN_test", "saved": "2026-10-09 15:30:12",
 "black": 7, "white": 2, "finished": false}
```

`moves`, `model_color` and `level` are what `load_game` needs; the rest is for reading the file.

### `GET /api/game_state?game_id=...`

The current state, unchanged.

### `POST /api/set_level`

Request: `{"game_id": "...", "level": 1-10 | "custom"}`. Response:
`{"game_id": "...", "level": 7, "description": "level 7: MCTS 100 simulations, solver <= 14 empties"}`.

### `GET /api/levels`

The levels and the server's default (not tied to a game):

```json
{"levels": [{"level": 1, "description": "policy network, sampled (temperature 1.5)",
             "strength": "well below Edax depth 1 (wins 14.5% vs Edax-1)"}, ...,
            {"level": "custom", "description": "custom: MCTS 400 simulations, ..."}],
 "default": 10}
```

The `custom` entry appears only when the server was started with search options.

### `GET /api/get_moves?game_id=...`

The game's moves:

```json
{"moves": [{"number": 1, "player": "black", "by": "human", "action": 19, "notation": "D3"}, ...],
 "total": 12}
```

`player` is the color that moved, `by` is `human` or `model`.

## Game records

Unless `--no-record`, every game is saved to `games/<YYYYmmdd-HHMMSS>_<first 8 characters of the id>.json`,
rewritten after each move, undo and redo so the file matches the game as it stands (lines abandoned by
undo are not kept):

```json
{
  "model": "r256x12_mid1_CNN_test",
  "player": "level 10: MCTS 800 simulations, leaf solves <= 16, solver <= 18 empties",
  "level": 10,
  "model_color": "black",
  "moves": [
    {"square": "F5", "action": 37, "color": "black", "by": "model",
     "board_before": [0, 0, ... 64 values, 1 = black, -1 = white ...],
     "method": "level 10: MCTS 800 simulations"},
    {"square": "F6", "action": 45, "color": "white", "by": "human", "board_before": [...]}
  ],
  "final_board": [...64 values...],
  "black": 33, "white": 31,
  "finished": true
}
```

`player` and `level` are the level when the file was last written; if the level changed mid-game,
each model move's `method` shows the level that chose it. `analyze_games.py games/*.json` grades the
person's (or an opponent app's) moves with Egaroucid next to the policy probabilities the model gave
them (see CLAUDE.md, "Recording and grading games").

## Security

Audit of 2026-10-09 (before it: one global game, debug mode always on); what the server does now:

| Risk | Handling |
| --- | --- |
| Werkzeug's interactive debugger executes code for anyone who can trigger an error | debug off by default; `--debug` refused unless the host is local |
| Path traversal: a request field (`model_color`) went into the record's file name | file names are built only from the time and the server-made id; `model_color` must be `black`/`white` |
| Anyone could play or reset anyone's game (one global game) | games by unguessable id; per-game locks serialize requests to the same game |
| Unvalidated moves (any value, illegal moves, moves out of turn) could corrupt a game | `action` must be an integer 0-63, legal, on the person's turn; the rules are `util/search.py`'s |
| Exception text returned to clients | generic `"internal error"`; details logged server-side |
| Non-JSON bodies caused unhandled errors | bodies must be JSON objects (else 400) |
| Resource exhaustion (MCTS costs ~1 s per move) | max games, idle expiry, per-address request and new-game rate limits, 4 KB request limit; model moves computed one at a time (shared lock), which also keeps the network and MCTS objects single-threaded |
| Clickjacking, content sniffing | `Content-Security-Policy` (`default-src 'self'`, `frame-ancestors 'none'`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`; API responses `Cache-Control: no-store` |
| XSS | the page writes server data with `textContent` only |
| CSRF | no cookies or logins, so nothing for another site to ride on |

## Deployment

**One always-on server** (simplest): keep `--host 127.0.0.1` and put a reverse proxy with TLS (nginx,
Caddy) in front, started with `--trust-proxy` so rate limits see client addresses. Flask's built-in
server is used (threaded). With the in-memory store, run exactly one process.

**Serverless / several instances** (e.g. Cloud Run, Azure Container Apps): start every instance with
`--redis-url` (or set `REDIS_URL`), e.g. a managed Redis or Upstash. Games, per-game locks (`SET NX`
with a 120 s expiry, so a crashed instance can't hold a game) and rate-limit counters (per minute) are
then shared, any instance can serve any request, and instances can scale to zero. Keys:
`reversi:game:<id>` (JSON, expiring after the idle timeout), `reversi:lock:<id>`,
`reversi:rate:<kind>:<client>:<minute>`. Notes:

- Each instance has its own engine lock, so each computes one model move at a time.
- Cold start (measured on the M4 Max, CPU): importing PyTorch, loading the network and the first
  level-10 move take ~2.5 s; on a serverless platform expect ~5-10 s with container start. GPU
  instances start much slower (larger images, GPU initialization).
- `--max-games` applies only to the in-memory store; with Redis the idle timeout and Redis's memory
  bound the number of games.
- Instance disks are temporary: use `--no-record`, or collect `games/` from an always-on host.

Speed without a GPU (measured, one game, M4 Max CPU, 12 threads): level 7 ~0.5 s per move, level 9
~1.3 s, level 10 ~1.8 s; MPS (GPU) is ~2-2.7x faster. Typical cloud CPUs are slower per core. Many
simultaneous players at high levels queue for each instance's engine.
