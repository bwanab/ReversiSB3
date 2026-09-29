# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReversiSB3 is a Reversi (Othello) AI training project using Stable Baselines 3 with PyTorch. The project implements a CNN-based reinforcement learning agent that learns to play Reversi through self-play and opponent training.

## Training Goals

The original milestones were measured against the ReversiAI (RAI) engine: beat RAI with 2-move
lookahead once in 100 games, then in a majority, then beat RAI-4 in a majority. Those were
exceeded long ago; Edax is much stronger than RAI at every depth and is now the benchmark.

**Current goal:** win against progressively deeper Edax searches. The previous best model beats
Edax up to depth 6 almost every game from the standard opening but hits a wall at depth 8, and
**from randomized openings it wins ~0%**: its Edax results were memorized lines, not playing
strength (see "Memorization finding" below). Measure progress from random openings.

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
- `input_planes=True` (`get_model(..., input_planes=True)`, `bc_train.py --input-planes`) expands
  the board inside the network into 5 planes: own, opponent, empty, own legal moves, opponent legal
  moves (`util/board_features.py`). The observation stays (1, 8, 8), so envs/opponents are
  unchanged, and old models load as before (the flag defaults to False and is saved with the model)

### Residual policy (`util/reversi_resnet.py`)
- `ReversiResNetPolicy` (a `MaskableActorCriticPolicy`): 5 input planes → conv stem → `blocks`
  residual blocks of `channels` (3x3 conv + BatchNorm) → spatial heads. The policy head is 1x1
  convs giving one logit per square (`action_net` is `Identity`); the value head is a 1x1 conv →
  Linear 64→64 → `value_net`. The trunk is shared by both heads
- Create with `get_model(..., model_type="resnet", channels=64, blocks=6)` or
  `bc_train.py --arch resnet --channels 64 --blocks 6` (64x6 ≈ 0.45M params, 128x8 ≈ 2.4M;
  the CNN is 4.8M). The policy class is saved in the zip, so `sb-train.py`/`sb-play.py` load it as-is
- Used with MaskablePPO's `CnnPolicy` and `normalize_images=False`

### Training System (`sb-train.py`)
- MaskablePPO (sb3-contrib) with action masking, TensorBoard logging, and checkpoints
- `--mode` selects the opponent mix: `random`, `selfplay` (vs a periodically refreshed copy of
  itself, plus `--random-ratio` random games), `sequential` (random phase then self-play),
  `mixed` (self-play / random / RAI by `--selfplay-ratio` and `--rai-ratio`),
  `edax-curriculum` (Edax at `--edax-depths` weighted by `--edax-ratios`, plus random), and
  `selfplay-edax` (self-play at `--selfplay-ratio` + Edax at `--edax-depths`/`--edax-ratios` +
  `--random-ratio`; ratios are normalized, each `--refresh-interval` block trains the opponents in
  that order, split by `util/training.py:mixed_block_plan`)

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

# Self-play + Edax + Random (60/30/10), varied starts
uv run python sb-train.py --mode selfplay-edax -m MODEL_NAME --timesteps 1000000 \
    --selfplay-ratio 0.6 --edax-depths 1,2,3 --edax-ratios 0.1,0.1,0.1 --random-ratio 0.1 \
    --refresh-interval 100000 -lr 1e-5 --start-positions start_positions.npy

# Mixed: self-play / RAI-1 / remainder random
uv run python sb-train.py --mode mixed -m MODEL_NAME --timesteps 1000000 \
    --selfplay-ratio 0.85 --rai-ratio 0.10 --rai-depth 1 -lr 5e-7
```
**Varied starts (use for all Edax training):** `--start-positions start_positions.npy` starts each
game from a random one of ~119k distinct real-game positions (8-20 stones, BLACK to move) instead
of the standard opening, so RL can't just memorize lines against Edax. **Leave
`--standard-start-ratio` at 0 whenever Edax is an opponent**: even 5% standard-opening games
(~1,500 per 1M steps) let RL relearn lines (a curriculum run reached 40% vs Edax-3 from the standard
opening but 3.5% from random openings). Self-play doesn't leak this way. Build the file (git-ignored) with
`uv run python make_start_positions.py` (from `combined_bc_dataset.pkl`; `--min-stones`/`--max-stones`).
The env option is `ReversiEnvCNN(start_positions=..., standard_start_prob=...)`.

Common options: `-m/--model` (name; saved as `models/{name}_CNN_test.zip`), `--timesteps`,
`-lr/--learning-rate` (or `--start-lr`/`--end-lr` for decay), `--refresh-interval` (self-play
opponent refresh / checkpoint cadence), `-t/--test-opponent`. `-e`, `-o`, `-p`, `-r` are legacy
flags; the step log uses `-e`/`-p` with `--mode mixed`, but prefer `--timesteps`.

**`--n-envs N`** (default 1) runs N games in lockstep in one process (`DummyVecEnv`), so the
model's moves are evaluated in one batched call; `n_steps` is set to `2048 // N` to keep 2048 steps
per PPO update, and checkpoints still land every 50k steps. Measured (128x8, MPS): vs Edax-2 126 ->
341 (8 envs) -> 395 steps/s (16); self-play 113 -> 192 -> 207 (the opponent network's moves are
still one at a time inside each game). Opponents are set on every env through
`util/training.py:set_opponent`; a refreshed self-play opponent is loaded once and shared.
**`--eval-games N`** (default 0) plays N games every 10k steps on a separate env vs
`--test-opponent` and logs `black_wins`; the model is saved every 10k steps either way. (Before
2026-09-29 this callback always played 100 deterministic games on the training env itself, costing
~25% of run time and leaving the env out of sync with SB3's last observation.)

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
- `--random-opening N`: start each game with N random plies (both sides), then the model plays
  BLACK. `--seed S` makes the openings reproducible so different models face the same positions.
  **Use this for headline win rates** (e.g. `--random-opening 8 --seed 42`); standard-opening
  results mostly measure memorized lines against Edax (see Training History)
- `--start-positions FILE`: start each game from a random position in a .npy (model to move),
  seeded by `--seed`. `opening_positions.npy` (from `make_opening_positions.py`: the 150 distinct
  positions along the named openings in `moves.txt`, symmetry-merged, 8-22 stones) gives a balanced,
  realistic second evaluation next to `--random-opening`

`opening_agreement.py -m MODEL ...` reports how often a model's top move is a `moves.txt` book
continuation (and its probability mass on book moves) over the 412 book positions after 4+ plies.

The model always plays BLACK. Model-vs-Model (`-o Model -r models/<other>`) is color-balanced and
takes `--random-opening`/`--start-positions`/`--seed` too: both halves see the same starting
positions with colors swapped, so opening luck cancels out. It prints wins as BLACK/WHITE, draws,
and a score (draws = 1/2; 50% = even). Head-to-heads are sensitive for close models but can be
biased when one model trained against the other (e.g. self-play vs its starting model).

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
`bc_train.py` also takes `--lr-schedule constant|cosine` (cosine decays to 0 over all epochs),
`--eval-games N` (per-epoch games vs Random and RAI-1; default 0 = skipped, since both saturate
near 100%; the CSV columns are left empty),
`--arch cnn|resnet` with `--channels`/`--blocks` (see Residual policy above), `--input-planes` (see ReversiCNN above), `--augment` (a random one of
the 8 board rotations/reflections per training example; validation is unaugmented),
`--value-coef` (value-head loss weight, default 0.5), `-v/--val-split`, and
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
  Edax-3 75% -> 1%); gains held better once LR was below ~1e-6. (Those Edax numbers were memorized
  lines, see below. Also, until 2026-09-28 self-play never refreshed its opponent within a run:
  `get_opponent()` caches model opponents per file path and the temp opponent file is overwritten in
  place, so every self-play block played a frozen copy from the start of the run. Fixed with
  `set_opponent(..., reload=True)`.)
- Win rates at a depth often collapsed to ~0% when that depth entered the curriculum, then recovered
  over later blocks (e.g. Edax-6 was 0% for steps 9-13 before reaching 97%).
- Edax-8 stayed at 0-1% through every block, while Edax-9 reached ~23%: an even-depth or specific
  tactical gap rather than a smooth strength limit. Undiagnosed; `status_summary_2026-09-23.md`
  lists hypotheses and a diagnosis plan (verbose non-deterministic games vs Edax-8).
  Superseded by the memorization finding below.

### Memorization finding (2026-09-27)
Edax at a fixed depth plays (nearly) deterministically from the standard opening, and RL against it
learned to replay winning lines rather than to play well. From randomized openings
(`sb-play.py --random-opening N --seed 42`, 100-200 games, `-d`):

| Model | Edax | Standard opening | 2-ply random | 8-ply random |
|---|---|---|---|---|
| `edax_bc_pretrained` (previous best, ~11M RL) | 6 | 94% | - | 0% |
| `edax_bc_pretrained` | 4 / 2 | 94% (log) / - | - | 0% / 5% |
| CNN + planes, 2M RL (`planes_aug_bc10_rl2m`) | 4 | 96% | 7% | 0.5% |
| ResNet 64x6, 2M RL (`resnet64x6_bc10_rl2m`) | 4 | 73% | 9% | 0.5% |
| CNN + planes, BC only (no RL) | 2 | 6% | - | 4% |

One random move per side already breaks it, and a BC-only model (nothing memorized) scores about the
same from random openings as from the standard one, so the random positions are fair. This explains
the 0% "cliff" at every depth not yet in the curriculum and the Edax-8 wall. Consequences: RL
against Edax needs varied starting positions, and evaluation should use random openings.

### Retrain (2026-09, branches `cnn-input-planes`, `resnet-policy`)
BC on `combined_bc_dataset.pkl`, 10 epochs, lr 1e-4, batch 256 (CSVs `*_bc10_training.csv`):

| Model | Params | Val loss (ep 10) | Val acc (best) | vs Edax-1 / Edax-2, BC only, 8-ply random openings, 200 games |
|---|---|---|---|---|
| Old CNN, raw board (old run) | 4.8M | 1.319 (min 1.270 @ ep 6, overfits) | 56.9% | - |
| CNN + planes, no augmentation | 4.8M | 1.278 (min 1.202 @ ep 5, overfits) | 58.0% | - |
| CNN + planes + augmentation | 4.8M | 1.155 | 58.1% | 16.5% / 3.5% |
| ResNet 64x6 + augmentation | 0.45M | 1.117 | 58.7% | 17.5% / 1.5% |
| **ResNet 128x8 + augmentation** | 2.4M | **1.043** (not overfitting) | **60.7%** | **29.5% / 7.0%** |

ResNet 128x8 is the best architecture. The RL runs on the CNN and 64x6 (Edax 1/2/3 then 2/3/4, 1M
steps each, lr 1e-5) reached 96%/73% vs Edax-4 from the standard opening, but that was memorization
(table above).

**BC recipe (decision A, 2026-09-28):** `resnet128x8_bc20cos_bconly` (20 epochs, `--lr-schedule cosine`,
val loss 1.003, acc 61.7%) vs the 10-epoch model: tie vs Edax 1/2 (random and named openings), but
~55.7% in 800 paired head-to-head games and book agreement 87% vs 85%. It is the base for future RL.

**RL recipe (decision B, 2026-09-28), all with varied starts, win % vs Edax-1/2/3/4 from 8-ply random
openings (200 games) and Edax-1/2 from named openings (300 games):**

| Model (ResNet 128x8, 10-epoch BC) | Random 1 / 2 / 3 / 4 | Named 1 / 2 | Standard 2 / 3 |
|---|---|---|---|
| BC only | 29.5 / 7.0 / - / - | 31.3 / 7.7 | - |
| + 1M Edax 1/2/3 curriculum (`_vs1m`) | 40.5 / 8.0 / 3.5 / 1.5 | 44.7 / 13.0 | 21 / 4 |
| `_vs1m` + 1M self-play (`_vs1m_sp1m`) | 34.0 / 10.5 / 3.0 / 1.0 | 41.3 / 12.7 | 14 / 4 |
| `_vs1m` + 1M more curriculum (`_vs1m_cur1m`) | 35.0 / 9.5 / 3.5 / 0.0 | 44.0 / 8.7 | **38 / 40** |

The first 1M RL steps gave real gains; the second million gave nothing with either recipe (ties within
noise), and the curriculum run relearned lines via its 5% standard-opening games (last column). Plan: a
mixed `selfplay-edax` recipe on `resnet128x8_bc20cos` with no standard starts, once rollouts are faster
(~30k games per 1M steps is too little experience).

**Rollout speed (profiled 2026-09-29, 128x8 on MPS):** vs Edax-2 111 steps/s, self-play 97 steps/s
after vectorizing the legal-move planes and disabling torch.distributions argument checks (were 101 and
72). PPO updates are only ~10% of the time. What remains is mostly fixed MPS latency per forward call
(~4.5 ms at batch 1, about the same at batch 32) plus Python game logic (~17%); running many games per
forward call (vectorized envs) is the next step. CPU inference is faster at batch 1 (2.2 ms) but PPO
updates are ~11x slower on CPU.

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
`training_issues`, `step`, `lr`, `features`, `resnet`, `play`, `starts`, `refresh`, `openings`, `mix`, ...).
The script sets `PYTHONPATH` to the project root and runs through `uv run`. All 129 tests in `tests/`
are part of the runner and pass. `test_edax_opponent.py`
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
