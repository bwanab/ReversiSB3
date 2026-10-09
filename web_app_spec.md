# web_play.py: web interface and HTTP API

`web_play.py` is a Flask app for playing a person against a trained model in the browser. It serves
one page (`templates/index.html`, `static/script.js`, `static/style.css`) and a small JSON API that
the page calls. This document describes the API as implemented (2026-10-09).

It replaces `web_app_spec.txt`, the spec of an earlier Julia web service (`/start/...`, `/play/...`,
`/get_action/...` with game UUIDs) that guided the first version of this app. The Python app departed
from it: one game per server, POST requests with JSON bodies, and undo/redo.

## Running

```bash
uv run python web_play.py -m r256x12_mid1_CNN_test --strong     # then open http://127.0.0.1:5000
```

| Option | Meaning |
|---|---|
| `-m/--model` (required) | model under `models/`, without `.zip` |
| `--strong` | the strongest player: `--mcts-sims 800 --solve-empties 18 --leaf-solve-empties 16` |
| `--mcts-sims N` | choose the model's moves by MCTS with N simulations |
| `--search-depth N`, `--search-top-k K`, `--search-depth-early N`, `--early-above E` | negamax search instead of MCTS (pruned at every level) |
| `--solve-empties N` | play exactly with the endgame solver at <= N empty squares (with or without search) |
| `--leaf-solve-empties N` | with search/MCTS: score positions with <= N empties exactly |
| `--device` | `auto` (CUDA, else MPS, else CPU), `cpu`, `mps`, `cuda` |
| `-p/--port`, `--host` | default 5000 on 127.0.0.1 |
| `-w/--net-width` | legacy CNN option, ignored by saved models |

With none of the search options the model plays the policy network's top move.

## Model

- **One game per server process**, held in a global `game_state`. There are no game IDs or sessions;
  every request acts on the current game, and two browsers share it.
- The model plays one color (chosen at `new_game`), the person the other. The server always answers
  a person's move with the model's reply in the same response, including any passes.
- Every game is saved to `games/<YYYYmmdd-HHMMSS>_<model>_model-<color>.json`, rewritten after each
  move, undo and redo (see "Game records").

## Data conventions

| Item | Format |
|---|---|
| Board | 8x8 list of rows; `board[row][col]` is `1` (black), `-1` (white) or `0` (empty). Row 0 is the top row |
| Action | integer `row * 8 + col` (0-63) |
| Notation | column letter + row number: action 0 = `A1` (top left), 19 = `D3`, 63 = `H8` |
| Color | `"black"` / `"white"` (black moves first) |
| Winner | `"black"`, `"white"` or `"draw"` |

**Valid move** object:

```json
{"row": 2, "col": 3, "action": 19, "probability": 0.41}
```

`probability` is present only when it is the person's turn. It is the model's *policy network*
probability for that move, i.e. the network's prediction of the person's move, not an evaluation.

**`last_move`** (after the model has moved):

```json
{
  "action": 37, "player": "model",
  "analysis": [{"notation": "F5", "probability": 0.62, "action": 37}, ...],
  "method": "MCTS 800 simulations",
  "exact_score": 4
}
```

- `analysis`: the policy network's top 5 moves with their probabilities (plus the move played, if it
  wasn't among them). It shows what the network alone would play; the move actually played comes
  from the configured player and is marked in the UI.
- `method`: how the move was chosen: `network`, `MCTS N simulations`, `search depth N`, or
  `solver (N empty)`.
- `exact_score`: only with the solver: the final disc difference for the model with perfect play.

After a person's move, the server sets `last_move` to `{"action": ..., "player": "human"}` internally,
but responses carry the model's `last_move` (the person's move is already known to the page).

**Standard state response** (fields vary by endpoint, see below):

```json
{
  "board": [[0, 0, ...], ...],
  "current_player": "white",
  "valid_moves": [ ...valid move objects... ],
  "piece_count": {"black": 4, "white": 1},
  "game_over": false,
  "last_move": { ... },
  "model_passed": true,
  "can_undo": true,
  "can_redo": false
}
```

- `model_passed: true` appears when the model had no legal move and the turn came back to the person.
- When the person has no legal move after the model's move, the model simply moves again (the
  response shows the position after the model's last move; no flag is set).
- Game over: `game_over: true`, `winner`, `piece_count`, `valid_moves: []`, `board`; `current_player`
  is omitted.

**Errors**: HTTP 400 with `{"error": "<message>"}` (no game in progress, game over, no action given,
nothing to undo/redo, or an exception while applying a move).

## Endpoints

### `GET /`
The game page.

### `POST /api/new_game`
Start a new game (discarding the current one; its record file stays).

Request: `{"model_color": "black" | "white"}` (default `"black"`).

Response: the standard state for the starting position, plus `"model_color"`. If the model plays
black, the response is instead the state after the model's first move (with `last_move`).

### `POST /api/make_move`
Play the person's move, then the model's reply.

Request: `{"action": <0-63>}`. The move is not validated against the legal moves beyond what the
game logic does; the page only sends moves from `valid_moves`.

Response, one of:
- the person's move ended the game: game-over state;
- the model has no move: standard state with `model_passed: true` (or game over if neither side can
  move);
- otherwise: the state after the model's reply (with `last_move`), or game over if it ended the game.

### `GET /api/game_state`
The current state without changing it: standard fields plus `winner` and `model_name`. Probabilities
are included when it is the person's turn. Not used by the page; useful for scripts and debugging.

### `POST /api/undo`
Take back the last move (person's or model's) and put it on the redo stack. The side who made it is
to move again. The page undoes once per click, so taking back your own move after the model's reply
takes two clicks.

Request body: none needed (`{}`).

Response: standard state (`game_over: false`), with probabilities if it is now the person's turn. If it
is the model's turn after an undo, the model does not move automatically: undo again to take back
your own move, or redo. (Clicking a square in that state plays it as the model's color.)

### `POST /api/redo`
Replay the most recently undone move. A new move clears the redo stack.

Response: standard state; if it is then the model's turn, the model moves (as in `make_move`) and the
response is the state after its move.

### `GET /api/get_moves`
The move history of the current game:

```json
{"moves": [{"number": 1, "player": "black", "action": 37, "notation": "F5"}, ...], "total": 12}
```

`player` is the color that moved (not person/model).

## Game records

`games/<YYYYmmdd-HHMMSS>_<model>_model-<color>.json`, rewritten after every move, undo and redo so the
file always matches the game as it stands (lines abandoned by undo are not kept):

```json
{
  "model": "r256x12_mid1_CNN_test",
  "player": "MCTS 800 simulations, leaf solves <= 16, solver <= 18 empties",
  "model_color": "black",
  "moves": [
    {"square": "F5", "action": 37, "color": "black", "by": "model",
     "board_before": [0, 0, ... 64 values, 1 = black, -1 = white ...],
     "method": "MCTS 800 simulations"},
    {"square": "F6", "action": 45, "color": "white", "by": "human", "board_before": [...]}
  ],
  "final_board": [...64 values...],
  "black": 33, "white": 31,
  "finished": true
}
```

`analyze_games.py games/*.json` grades the person's (or opponent app's) moves with Egaroucid next to
the policy probabilities the model gave them (see CLAUDE.md, "Recording and grading games").

## Known quirks

- Some game-over responses omit `can_undo`/`can_redo`; the page then leaves the buttons as they were.
- After an undo that leaves the model to move, the page shows the model's legal moves and a click
  plays one for the model's color (see `/api/undo`).
- The `probability` shown on the person's moves and the `analysis` of the model's move are always the
  bare policy network's, even when the model plays with MCTS or the solver.
- One shared game per server: opening the page in a second browser continues the same game.
