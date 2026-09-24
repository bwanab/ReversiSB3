# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReversiSB3 is a Reversi (Othello) AI training project using Stable Baselines 3 with PyTorch. The project implements a CNN-based reinforcement learning agent that learns to play Reversi through self-play and opponent training.

## Training Goals

The original milestones were measured against the ReversiAI (RAI) engine: beat RAI with 2-move
lookahead once in 100 games, then in a majority, then beat RAI-4 in a majority. Those were
exceeded long ago; Edax is much stronger than RAI at every depth and is now the benchmark.

**Current goal:** win against progressively deeper Edax searches. The previous best model beats
Edax up to depth 6 almost every game but hits a wall at depth 8 (see Training History below).

**Current direction (2026-09):** retrain from scratch on the new machine/stack, possibly with
changes to the CNN, reusing the BC dataset. The previous best model is kept as a benchmark.

## Key Architecture Components

### Core Environment (`util/reversi.py`)
- `ReversiEnvCNN`: Custom Gymnasium environment for Reversi with CNN observation space
- Implements 8x8 board with standard Reversi rules
- Uses action masking to prevent invalid moves
- Supports multiple opponent types during training

### Neural Network (`util/reversi_cnn.py`)
- `ReversiCNN`: SB3 features extractor; input is the (1, 8, 8) board with values -1/0/1
- Five 3x3 conv layers (64, 64, 128, 128, 256 channels, padding 1, ReLU), flattened, then a
  linear layer to 256 features
- Used with MaskablePPO's `CnnPolicy` and `normalize_images=False`

### Training System (`sb-train.py`)
- MaskablePPO (sb3-contrib) with action masking, TensorBoard logging, and checkpoints
- `--mode` selects the opponent mix: `random`, `selfplay` (vs a periodically refreshed copy of
  itself, plus `--random-ratio` random games), `sequential` (random phase then self-play),
  `mixed` (self-play / random / RAI by `--selfplay-ratio` and `--rai-ratio`), and
  `edax-curriculum` (Edax at `--edax-depths` weighted by `--edax-ratios`, plus random)

### Opponent System (`util/opponents.py`)
- Opponents: `Random`, `Human`, `Model` (a saved model playing WHITE), `RAI` (reversi-python-ai
  minimax), `Edax` (via `util/edax_client.py` talking to `edax_server.py`)
- `get_opponent()` factory; `get_action(env, state)` returns a 0-d array action
- Used for both training and evaluation

### Utility Functions (`util/util.py`)
- Model loading/saving with `get_model()`
- Action masking with `mask_fn()`
- Learning rate scheduling functions
- Device detection (CPU/CUDA/MPS)

## Common Commands

### Training a Model
```bash
# Edax curriculum (needs the Edax server running, see Development Setup)
uv run python sb-train.py --mode edax-curriculum -m MODEL_NAME \
    --edax-depths 4,5,6 --edax-ratios 0.3,0.3,0.3 --random-ratio 0.1 \
    --timesteps 1000000 --refresh-interval 100000 -lr 3e-6

# Self-play (plus 10% random games)
uv run python sb-train.py --mode selfplay -m MODEL_NAME --random-ratio 0.1 \
    --timesteps 200000 --refresh-interval 50000 -lr 3e-6

# Mixed: self-play / RAI-1 / remainder random
uv run python sb-train.py --mode mixed -m MODEL_NAME --timesteps 1000000 \
    --selfplay-ratio 0.85 --rai-ratio 0.10 --rai-depth 1 -lr 5e-7
```
Common options: `-m/--model` (name; saved as `models/{name}_CNN_test.zip`), `--timesteps`,
`-lr/--learning-rate` (or `--start-lr`/`--end-lr` for decay), `--refresh-interval` (self-play
opponent refresh / checkpoint cadence), `-t/--test-opponent`. `-e`, `-o`, `-p`, `-r` are legacy
flags; the step log uses `-e`/`-p` with `--mode mixed`, but prefer `--timesteps`.

### Playing/Evaluating Models
```bash
uv run python sb-play.py -m MODEL_NAME_CNN_test -e 100 -o Edax -p 6 -d
```
Options:
- `-m/--model`: model name without `models/` or `.zip` (the script prepends `models/`)
- `-e/--episodes`: number of games
- `-o/--opponent`: `Random`, `RAI`, `Edax`, `Model` (with `-r models/<opponent model>`)
- `-p/--depth`: RAI/Edax search depth
- `-d/--non_deterministic`: sample moves from the policy. **Without `-d` play is deterministic**,
  which replays the same game every time and gives misleading win rates (0% or 100%); always use
  `-d` for win-rate measurements
- `-v/--verbose`: display game details

The model always plays BLACK. Model-vs-Model (`-o Model -r ...`) is color-balanced.

### Behavioral Cloning (BC) Training

The previous best model was BC-pretrained on WThor human games plus Edax games:
```bash
# 1. Parse the WThor PGN database into a BC dataset
uv run python parse_wthor_pgn.py -d ../othello-games/pgn -o wthor_bc_dataset.pkl
# (value targets were added with augment_dataset_with_outcomes.py -> wthor_bc_dataset_with_values.pkl)

# 2. Add Edax games (Edax depths 7,8,9 vs a model/random) and merge
uv run python generate_edax_bc_dataset.py --games 9000 --depths 7,8,9 \
    --model models/wthor_rl_CNN_test --model-ratio 0.8 \
    --merge wthor_bc_dataset_with_values.pkl --output combined_bc_dataset.pkl

# 3. BC-train a fresh model (per-epoch checkpoints; pick the best epoch)
uv run python bc_train.py --dataset combined_bc_dataset.pkl --model edax_bc_pretrained \
    --epochs 20 --batch-size 256 --learning-rate 1e-4
```
`combined_bc_dataset.pkl` (~3.45M samples) is in the project root (git-ignored; original copy in
`reversisb3_oob/datasets/`), so steps 1-2 only need rerunning to change the data.
`bc_train.py` also takes `--value-coef` (value-head loss weight, default 0.5), `-v/--val-split`, and
`--device` (default `auto`: MPS/CUDA if available, else CPU). On the M4 Max, MPS is ~13x faster than
CPU for BC (a full-dataset epoch is ~2.3 min vs ~29 min); CPU was only faster on the old M1 with
torch 2.0, which is why it used to be hard-coded. 2-epoch baseline on the full dataset: val
accuracy ~46-47% after epoch 1 and ~52-53% after epoch 2 (matches the old machine's run).
`generate_bc_dataset.py` is the older RAI-based generator.

### Installing Dependencies
The project uses [uv](https://docs.astral.sh/uv/); dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
```bash
uv sync                      # create/update .venv from uv.lock
uv run python sb-train.py …  # run a script inside .venv
uv add <package>             # add a dependency
```

### Development Setup
uv manages the virtual environment in `.venv/` (Python 3.14). Key dependencies:
- `stable-baselines3>=2.9.0`
- `sb3-contrib>=2.9.0`
- `torch>=2.14.0`
- `gymnasium>=1.3.0`
- `reversi-python-ai` installed from git

**gymnasium >= 1.0 note:** wrappers no longer forward custom attributes (`board`, `player`,
`get_valid`, `set_opponent`, ...) to the underlying env. `build_reversi()` therefore returns the
unwrapped `ReversiEnvCNN`, and code holding a wrapped env (e.g. `ActionMasker`, SB3's `Monitor`)
must go through `env.unwrapped`. Assigning attributes on a wrapper (`env.board = ...`) silently
sets them on the wrapper, not the real env.

### Edax
Edax runs in a separate process (`edax_server.py`, Unix socket `/tmp/edax_server.sock`) because
SB3's forked envs corrupted the shared C library state; see `EDAX_SERVER_USAGE.md`. Start it
before any `-o Edax` play or `edax-curriculum` training: `uv run python edax_server.py`.
- Library: `~/src/edax-reversi/bin/libedax.dylib`, built from the fork `bwanab/edax-reversi`
  (adds the C API in `src/edax_wrapper.c`). Weights: `~/src/edax-reversi/bin/data/eval.dat`
  and `book.dat`.
- Build (from `~/src/edax-reversi/src`):
  `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk clang -std=c17 -O3 -march=native -mdynamic-no-pic -D_GNU_SOURCE=1 -DNDEBUG -dynamiclib -o ../bin/libedax.dylib all_lib.c -lm`
  The `SDKROOT` override is needed while the Command Line Tools linker (26.x) is older than the
  default macOS 27 SDK; drop it once the CLT is upgraded to match.

## Project Structure

### Model Storage
- `models/`: Trained model files (.zip format)
- `checkpoints/`: Training checkpoints saved every 50k steps
- `models/*/log/`: TensorBoard logs for training metrics

### Training Data
- Models are saved with naming convention: `{MODEL_NAME}_CNN_test.zip`
- Checkpoints include step count in filename
- TensorBoard logs track win rates and training progress

### Test Files
- `sanity*.py`: Development test scripts
- Various Jupyter notebooks for analysis and plotting

## Training History & Status

Authoritative logs: `exax_pretrain_steps.doc` (25 numbered steps: exact commands and Edax win
rates after each), `edax_train_results.csv`, `edax_bc_pretrained_training.csv` (BC curves),
`status_summary_2026-09-23.md` (analysis and suggested next steps), `session_notes.md` (earlier
RAI-era context, including the Piccolo iPhone app comparison).

### Earlier phases (RAI era)
Pure RL from scratch (random -> self-play -> mixed with RAI) reached ~5-7M timesteps and plateaued
at ~6% vs RAI-2 (non-deterministic). Early runs against Random became unstable after ~250k steps
(explained_variance collapse), which led to the lower LR / larger batch / fewer epochs below.
This motivated BC pretraining on RAI games: `bc_pretrained` (BC + RL, ~6M timesteps) reached
87% vs RAI-1 and 86% vs RAI-2, but was RAI-specialized: 0% against the Piccolo iPhone app at
levels 3-4. That led to BC on WThor human expert games plus Edax games instead.

### Previous best model: `edax_bc_pretrained_CNN_test`
- BC on `combined_bc_dataset.pkl` (WThor + Edax-7/8/9), 20 epochs; epoch 18 selected
- ~11M+ RL timesteps alternating `edax-curriculum` blocks (500k-2M steps, depths moving from
  1-3 up to 4-10, `--random-ratio` 0.1 then 0) with `selfplay`/`mixed` blocks (self-play +
  random + RAI-1)
- LR decayed across blocks: 1e-5 -> 5e-6 -> 3e-6 -> 2e-6 -> 1e-6 -> 5e-7 -> 3e-7

Final results (step 25):

| Edax depth | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| Win % | 89 | 94 | 97 | 97 | 69 | **0** | 23 | **0** |

Observations from the log:
- Self-play blocks right after curriculum blocks often dropped Edax win rates (e.g. step 4 -> 5:
  Edax-3 75% -> 1%); gains held better once LR was below ~1e-6.
- Win rates at a depth often collapsed to ~0% when that depth entered the curriculum, then recovered
  over later blocks (e.g. Edax-6 was 0% for steps 9-13 before reaching 97%).
- Edax-8 stayed at 0-1% through every block, while Edax-9 reached ~23%: an even-depth or specific
  tactical gap rather than a smooth strength limit. Undiagnosed; `status_summary_2026-09-23.md`
  lists hypotheses and a diagnosis plan (verbose non-deterministic games vs Edax-8).

### Hyperparameters (`get_model()` in util/util.py)
```python
learning_rate = 1e-5   # default; overridden per run by -lr
batch_size = 256
n_steps = 2048
ent_coef = 0.03
n_epochs = 5
gae_lambda = 0.90
gamma = 0.98
clip_range = 0.1
features_dim = 256     # ReversiCNN output
```

**Metrics to monitor:** `explained_variance` (> 0.5, ideally 0.7+), `approx_kl` (< 0.05),
`clip_fraction` (consistently > 0.3 means the clip range is binding), and win rate vs Edax at
several depths (always non-deterministic, `-d`).

**Loading models saved on the old machine** (Python 3.10, SB3 2.0) works for evaluation, but
their pickled `clip_range`/`lr_schedule` can't be deserialized on Python 3.14; SB3 warns
(`code() argument 13 must be str, not int`) and falls back to clip_range 0.2. Resuming training
from such a model needs `custom_objects={"clip_range": 0.1, ...}` in the load.

## Device Support

The training system automatically detects and uses:
- Apple Silicon GPU (MPS) if available
- CUDA GPU if available
- CPU as fallback

Device selection is handled in `sb-train.py` and passed to model creation.

## Testing

Run the suite with `./run_tests.sh` (all tests) or `./run_tests.sh <name>` for one group
(`environment`, `scenarios`, `edge_cases`, `training`, `integration`, `bc`, `focused`,
`training_issues`, `step`, `lr`, ...). The script sets `PYTHONPATH` to the project root and runs
through `uv run`. All 83 tests in `tests/` are part of the runner and pass. `test_edax_opponent.py`
in the project root is a separate script that needs the Edax server running.

Env behaviors worth knowing when writing tests:
- Game over is `env.get_winner(board) is not None` (there is no `is_game_over`).
- To play a move directly, set `env.player` then call `env.get_next_state(board, (0, row, col))`;
  it mutates `board` in place and flips `env.player`.
- `step()` handles passes internally, so BLACK is only handed positions with a legal move.
- On game end `step()` resets the board, so the returned `obs` is the new starting position; the
  final position is in `info['terminal_observation']`.
- Flat action index = `row * 8 + col`; BLACK's opening moves are 19, 26, 37, 44.

An earlier version of this file listed "critical" env bugs (missing `is_game_over`/`_place`,
bad masking, wrong initial moves). Those were bugs in the tests, not the env, and have been fixed.

## Deferred Work

### Opponent-vs-opponent play (not currently needed)

There is no tool for pitting two arbitrary opponents against each other (e.g. Random vs RAI,
RAI vs Edax, or a model playing WHITE). `sb-play.py` only covers *model (as BLACK) vs opponent*.

**History:** `util/play.py` had an `alt_play(env, num_games, black_player, white_player)` for this,
written in April 2024 (`e13dae2`) against boardgame2's `Reversi-v0`, whose `step()` played one move
for whichever side was to move. It broke when the project moved to `ReversiEnvCNN`, whose `step()`
is a two-ply SB3 training function (BLACK's move + the env's built-in opponent), so `white_player`
was never consulted and results were counted on the already-reset board. It was removed as dead code.

**To implement:** write a helper, e.g. `play_opponents(black, white, num_games) -> (black_wins,
white_wins, draws)` in `util/play.py`, modeled on the game loop in `generate_bc_dataset.py`
(see commit `a02f34e`):
- Don't use `env.step()`. Drive the game with `env.has_valid(state, player)` (pass handling:
  if the side to move has no move, switch; if neither does, the game is over),
  `env.get_next_state(state, (0, a // 8, a % 8))`, and `env.get_winner(state)`.
- Set `env.player = current_player` before each `get_action()` call: `RandomOpponent` and
  `RAIOpponent` read `env.player`, not their own `player`.
- Set each opponent's `.player` to its color (`BLACK` = 1 / `WHITE` = -1). `Opponent.player`
  defaults to -1, and `ModelOpponent` and `EdaxOpponent` flip the board by `self.player`, so a
  BLACK-side model/Edax left at the default would see an inverted board.
- `get_opponent()` caches `ModelOpponent`s per model file (`opponent_map`), so the same model on
  both sides would share one object and one `.player`; construct `ModelOpponent` directly for
  self-play matchups. A model under test is wrapped the same way:
  `ModelOpponent(opponent_model="models/<name>", env=env)`.
- `get_action()` returns a 0-d array; convert with `int(action)`.
- Alternate colors across games (as the BC generators do) so results aren't biased toward BLACK.
