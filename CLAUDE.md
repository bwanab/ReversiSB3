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

**Current direction (2026-10-02):** the strongest model is `r192x10_dagger` (ResNet 192x10 from
`train_course.sh`: BC -> Edax depth-12 labels -> one DAgger round). Played with depth-2 search over
its top 3 moves it wins 71% vs Edax-2, 52.5% vs Edax-3 and 37.5% vs Edax-4 from random openings. PPO so far degraded best play. Next: more Edax labels,
including positions the model itself reaches; see "Edax labels" below.

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

#### Network size: channels x blocks (e.g. 192x10)
- **Channels** (128, 192, 256) = width: how many learned features describe each of the 64 squares
  after the input layer (e.g. "stable edge stone", "flippable along a diagonal").
- **Blocks** (8, 10, 12) = depth: residual blocks of two 3x3 convolutions. Each convolution lets a
  square's features absorb its 8 neighbours, so information travels 2 squares per block; 8 blocks
  already span the board. More blocks = more chained reasoning steps (this flip opens that diagonal,
  which gives up a corner).
- Parameters ~ 18 x blocks x channels^2 (compute per position scales the same way): 128x8 2.4M,
  192x10 6.7M, 256x8 9.4M, 256x12 14.2M. Doubling channels costs 4x, doubling blocks 2x. Training
  speed on the M4 Max (batch 256): 128x8 45 batches/s, 192x10 18 batches/s.
- Width vs depth: wide layers use a GPU more efficiently and are faster for single-position play
  (fewer sequential layers); depth suits chained tactics. Game networks usually grow both (AlphaZero:
  256 channels, 20-40 blocks for 19x19 Go). 128x8 -> 192x10 grew both and was the biggest single
  gain so far, so which one mattered is unknown; a same-compute comparison (e.g. 256x8 vs 192x14,
  both ~9.4M) would tell. We went straight to 256x12 (grow both).

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
- `-d/--non_deterministic`: sample moves from the policy; without it the model plays its top move.
  From the standard opening, top-move play vs a deterministic opponent replays one game (0% or
  100%), so there you need `-d`. **From varied starts (`--random-opening`/`--start-positions`),
  top-move play is the headline measure** (each start gives one distinct game): it measures the
  model's best play, and sampling roughly halves win rates (policy entropy ~0.87). With named
  openings use 150 games (each position once); more just repeats identical games.
- `-v/--verbose`: display game details
- `--random-opening N`: start each game with N random plies (both sides), then the model plays
  BLACK. `--seed S` makes the openings reproducible so different models face the same positions.
  **Use this for headline win rates** (e.g. `--random-opening 8 --seed 42`); standard-opening
  results mostly measure memorized lines against Edax (see Training History)
- `--start-positions FILE`: start game i from position i of a .npy (model to move), in order,
  cycling if `-e` exceeds the count (before 2026-10-01 it sampled with replacement, so 150 games
  covered only ~95 of the 150 positions and named-opening numbers were noisier). `opening_positions.npy` (from `make_opening_positions.py`: the 150 distinct
  positions along the named openings in `moves.txt`, symmetry-merged, 8-22 stones) gives a balanced,
  realistic second evaluation next to `--random-opening`

- `--search-depth N` (1-2 practical): choose the model's moves by negamax N plies deep, scoring
  leaf positions with the value head in one batched call (`util/search.py`; exact scores for finished
  games, forced passes don't use depth). `--search-top-k K` searches only the policy's top K root
  moves. Always plays the best-scoring move, so compare it with the policy's top move (no `-d`), not
  only with sampling (`-d`). Not used with `-o Model`. Depth 2 costs ~70 ms per move (Python tree).

**`eval_batch.py`** (2026-10-02) plays all games vs Edax in lockstep, batching the model's network
calls (`util/lockstep.py`, `SearchPlayer.choose_many`, `search_many`): same options as sb-play for
top move / `--search-depth`/`--search-top-k`, several Edax depths at once (`-p 1,2,3`), same starting
positions as sb-play. Top-move results match sb-play exactly; ~2 s per 200 top-move games and ~15 s
per 200 depth-2 search games (vs minutes). Search results can differ from older sb-play runs by a
few points: candidates are now ordered by policy logit, so exact ties (common late in games) go to
the policy's preferred move, and a single diverging move changes a whole game. The old SearchPlayer
also queried root candidates once in training mode right after loading. Use `eval_batch.py` for Edax
evaluations; sb-play for sampled play and Model-vs-Model.

`opening_agreement.py -m MODEL ...` reports how often a model's top move is a `moves.txt` book
continuation (and its probability mass on book moves) over the 412 book positions after 4+ plies.

`sb-play.py` refuses top-move play (no `-d`) or `--search-depth` against Edax, RAI or a Model
from the standard opening (every game would be identical); add `-d` or vary the starts. In
Model-vs-Model games the opponent model now plays the same way as `-m` (top move without `-d`,
sampled with it); before 2026-10-01 it always sampled, so top-move head-to-heads were really
top move vs sampled.

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

### Train of reasoning (2026-09-23 -> 2026-10-02)
Each step: what we saw -> what we concluded -> what we did. Details and numbers are in the sections below.
1. **Old CNN critique** (`cnn_critique_2026-09-26.md`): most parameters in one FC layer, raw +-1
   input, no symmetry use -> added 5 input planes + 8-fold augmentation (BC val loss 1.270 -> 1.155,
   no overfitting) -> ResNet with spatial heads (64x6 matched the CNN with 1/10 the parameters; 128x8
   clearly better).
2. **RL vs Edax looked great but generalized to nothing** (0% at every untrained depth) -> tested from
   random openings: every RL model, including the old best, collapsed to ~0% -> the "wins" were
   memorized lines against a near-deterministic Edax from the standard opening. -> Evaluate from
   random/named openings; train from varied real-game start positions.
3. **Varied-start PPO gained once, then plateaued** (1M -> 5M steps: no gain, even head-to-head vs
   itself) -> profiling and fixing rollout speed (planes, distribution checks, `--n-envs`) didn't change
   that; also found and fixed a self-play opponent that never refreshed.
4. **Top-move evaluation** (instead of sampling) showed PPO had actually *degraded* best play while
   the sampled numbers looked flat; the BC model was strongest. -> Stop PPO.
5. **Search with the PPO value head made play much worse**: the value head couldn't rank sibling
   positions (value regret 4.7 discs vs policy 1.9). -> Need better teaching targets.
6. **Edax labels** (its move and score for every legal move, depth 12, ~170 positions/s): supervised
   fine-tuning with graded targets steadily improved play (Edax-2 29.5% -> 44% from random openings
   with 2.2M labels) and the value head (regret 4.7 -> 2.8), and search turned from harmful to helpful.
7. **DAgger** (label positions the model itself reaches): little change to the bare policy, but a
   better value head -> stronger search (53.5% vs Edax-2).
8. **Bigger network (192x10)**: policy regret 1.70 -> 1.46, value regret 2.4 -> 1.8; with depth-2
   search 74% vs Edax-2, 55% vs Edax-3, 44% vs Edax-4 from random openings. -> Capacity was a limit;
   running 256x12.
9. **Engineering along the way:** batched lockstep evaluation (~40 min -> ~2.5 min per evaluation),
   fp16 tested and dropped (same moves, no speedup), portable device selection.
10. **Even bigger network (256x12, 14.2M)**: every validation metric ~5-10% better (policy regret
   1.40, value regret 1.64) and top-move play vs Edax-2 up ~8-12 points, but with search roughly a
   tie with 192x10 and only ~54% head-to-head. Per-epoch play curves show the Edax stage converging by
   epoch 3-5. -> Network size and stage length are no longer the main limit; the teacher (Edax depth
   12) and our own search depth are.
11. **Depth-3 search**: +20-25 points vs Edax-4 from named openings (`r256x12_edax` 69%), nothing from
   random openings. -> The value head on unfamiliar positions limits deeper search. Next: DAgger from
   random openings played with search, deeper Edax labels, pruning at every search level.
12. **Search pruned at every level** (policy's top 3 at each node): strength rises steadily with depth
   from random *and* named openings (Edax-4 ~40% -> ~70% at depth 5) and runs faster than root-only
   depth 3. -> The random-opening problem was searching implausible lines, not the value head. Search
   depth is now the strongest lever; DAgger-for-search (2) and a deeper teacher (3) wait. Next:
   measure vs Edax 5-8 with depths 5-6.

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

**Top-move finding (2026-09-30):** evaluated by top move instead of sampling, RL made the model
*worse*, steadily (win %, random openings 200 games / named openings 150 distinct):

| `resnet128x8_bc20cos` | Random Edax-1 | Named Edax-1 | Random Edax-2 | Named Edax-2 |
|---|---|---|---|---|
| BC only | **64.0** | **78.7** | **29.5** | **37.3** |
| + 1M mixed RL | 57.5 | 76.0 | 25.0 | 33.0 |
| + 4M | 55.5 | 67.3 | 24.5 | 28.3 |
| + 5M | 59.0 | 51.0 | 19.5 | 27.7 |

With sampling (`-d`) the same run looked like +10 points then flat. PPO optimizes sampled play and,
with `ent_coef=0.03`, keeps the policy spread (entropy ~0.87 throughout); the top choice drifted from
what BC learned. A 1-2 ply search scored by the PPO-trained value head (`--search-depth`) was far
worse than the policy's top move (1M checkpoint, random Edax-1: 11.5% / 17.5% at depths 1/2 vs 57.5%
top move): the value head (explained variance ~0.55) can't rank sibling positions. Head-to-heads
(sampled) also showed 4M = 1M (49%) while 1M beat BC-only 57%.

**Edax labels (2026-09-30):** `label_positions.py` labels deduplicated BC-dataset positions with
Edax's move and score (discs, side to move) at a fixed depth, optionally scoring every legal move
(`--every-move`: via the position after it at depth - 1, passes and finished games handled), running
one Edax server per worker (`-w`; ~170 positions/s every-move at depth 12 with 12 servers). Checked:
position score = best move score in 96% (99.4% within 2 discs). `edax_train.py` fine-tunes on the
labels (policy target `graded` = softmax(move_score / 2), `best` = Edax's move, or `score` = centered
logit regression; value target tanh(score / 16)) and reports policy regret and 1-ply value-search
regret in discs. The dataset has 2.2M distinct 5-60-stone positions. Results, fine-tuning
`resnet128x8_bc20cos_bconly` for 4 epochs at lr 5e-5 on 1M every-move labels (top move, random
openings 200 games):

| Model | Edax-1 | Edax-2 | Edax-3 | Policy regret | Value regret |
|---|---|---|---|---|---|
| BC only | 64.0 | 29.5 | 16.5 | 1.88 | 4.73 |
| `edax1m_best` | 68.5 | 30.0 | 14.0 | 1.84 | 3.57 |
| **`edax1m_graded`** | **68.5** | **38.5** | **20.0** | **1.82** | **3.29** |
| `edax1m_score` | 61.0 | 32.0 | 15.5 | 1.90 | 3.64 |
| **`edax2m_graded`** (all 2.2M dataset positions, 6 epochs) | 68.0 | **44.0** | **23.0** | **1.79** | **2.79** |

Named openings (all 150 once, Edax-1/2/3): BC 84.0/42.0/22.0, `edax1m_graded` 76.7/44.0/33.3,
`edax2m_graded` 82.0/49.3/20.0. Value regret keeps falling with more labels (4.7 -> 3.3 -> 2.8).

**DAgger (2026-10-01):** the dataset's positions are all labeled, so new positions come from the
model's own games: `collect_positions.py -m MODEL -n N -o positions.npy --exclude labels*.npz` plays
the model (sampled moves, lockstep batched) vs itself and Edax 1-3 from random and named openings and
saves the distinct positions it had to move in (~2,000/s); `label_positions.py --positions FILE`
labels them; `edax_train.py --labels` takes several files (repeats dropped).

A 100k-label pilot gained ~2-3 points. The value head predicts Edax scores well (MSE 0.25 -> 0.07)
but still ranks sibling moves far worse than the policy (3.3 vs 1.8 discs lost per move), so search
is still not expected to help.

**ResNet 192x10 course (2026-10-02, `./train_course.sh r192x10 192 10 edax_dagger1`, log
`course_r192x10.log`):** BC 20 epochs (val acc 62.5%, val loss 0.964 vs 1.003 for 128x8), then
graded Edax labels (3.2M, 6 epochs) -> `r192x10_edax`, then a DAgger round (+1M positions it reached)
-> `r192x10_dagger`. Validation: policy regret 1.46 discs, value regret 1.83 (128x8 `edax_dagger1`:
1.70 / 2.41). Win % vs Edax, random openings (200) / named (150):

| Model | Top move E-1 / E-2 / E-3 | Search (d2, top 3) E-1 / E-2 / E-3 / E-4 |
|---|---|---|
| `edax_dagger1` (128x8) | 70 / 37 / 26, named 84.7 / 56.7 / 28.7 | - / 53.5 / 36.5 / -, named - / 59.3 / 37.3 / - |
| `r192x10_edax` | 77.5 / 45 / 30.5, named 91.3 / 61.3 / 35.3 | 85.5 / 74 / 49 / 29, named 96.7 / 80 / 56.7 / 30.7 |
| **`r192x10_dagger`** | 82 / 43 / 30, named 86.7 / 55.3 / 40 | **92.5 / 71 / 52.5 / 37.5**, named **96 / 80.7 / 60 / 38** |

BC learning curve by play (`r192x10_bc_epochNN`, top move, `eval_batch.py -m` several models):
most strength arrives by epoch ~8-10 (random-opening Edax-2: 24% at epoch 2, 37.5 at 6, ~35-40 from
10 to 20); after that random openings stay flat while named openings keep improving (Edax-2 42 ->
59% from epoch 10 to 20), i.e. late BC epochs sharpen play in master-game-like positions. With the
cosine schedule the tail of the loss curve always flattens, so "still improving?" is judged by
evaluating per-epoch checkpoints (`edax_train.py` now saves `{model}_epochNN` too) or by comparing
runs of different lengths, not by the last epochs' loss.

Re-measured with `eval_batch.py` (the reference from now on), `r192x10_dagger` with search wins
88 / 74 / 55 / 44% (random) and 98 / 86.7 / 66.7 / 36% (named) vs Edax-1/2/3/4; top move unchanged.

Head-to-head (top move, paired) vs `edax_dagger1`: `r192x10_edax` 60.2% random / 55.3% named,
`r192x10_dagger` 66.2% / 58.3%. The bigger network raised both heads, and search now adds ~25-30
points at Edax-2/3: with search, `r192x10_dagger` beats Edax-3 more often than not.

**ResNet 256x12 course (2026-10-03, `./train_course.sh r256x12 256 12 r192x10_dagger`, log
`course_r256x12.log`):** BC val loss 0.958 / acc 63.2% (192x10: 0.964 / 62.5%). Validation policy /
value regret: `_edax` 1.45 / 1.77, `_dagger` 1.40 / 1.64 (192x10: 1.52 / 2.00 and 1.46 / 1.83).
Win % vs Edax (`eval_batch.py`), random / named openings:

| Model | Top move E-2 | Top move E-3 | Search E-2 | Search E-3 | Search E-4 |
|---|---|---|---|---|---|
| `r192x10_dagger` | 43 / 55.3 | 30 / 40 | 74 / 86.7 | 55 / 66.7 | 44 / 36 |
| `r256x12_edax` | 54.5 / 62.7 | 31.5 / 36.7 | 76 / 88 | 49.5 / 69.3 | 39.5 / 44.7 |
| `r256x12_dagger` | 50.5 / 67.3 | 27 / 46 | 73.5 / 84 | 56 / 67.3 | 41 / 38 |

Head-to-head vs `r192x10_dagger` (top move): `_edax` 51.7% / 43.5%, `_dagger` 51.9% / 57.5%.
Per-epoch curves (top move and search) rise through epoch 3-5 of each Edax stage, then level off.

**Depth-3 vs depth-2 search (top 3 root moves; win %, random E-2/3/4 | named E-2/3/4):**
`r192x10_dagger` d2 74 / 55 / 44 | 86.7 / 66.7 / 36, d3 65.5 / 47.5 / 28 | 92.7 / 66.7 / 56;
`r256x12_dagger` d2 73.5 / 56 / 41 | 84 / 67.3 / 38, d3 74.5 / 57 / 40.5 | 88 / 70.7 / 62;
`r256x12_edax` d2 76 / 49.5 / 39.5 | 88 / 69.3 / 44.7, d3 73 / 54.5 / 40 | 87.3 / 78 / 69.3.
From named (master-like) openings depth 3 gains +20-25 points vs Edax-4; from random openings it gains
nothing (192x10: clearly worse). Reading: where the value head is accurate (familiar positions),
deeper search converts it into strength; in unfamiliar positions deeper search amplifies its errors.
Depth-3 runs take ~25x longer than depth 2 (Python tree code, ~263 leaves/position).

**Search pruned at every level** (`--search-top-k 3 --search-prune-all`: each node searches only the
policy's top 3 moves; win %, random E-2/3/4 | named E-2/3/4; seconds per 200/150-game run):

| Model / search | Random | Named | s/run |
|---|---|---|---|
| `r256x12_edax` d2 root-only | 76 / 49.5 / 39.5 | 88 / 69.3 / 44.7 | 22 |
| `r256x12_edax` d3 root-only | 73 / 54.5 / 40 | 87.3 / 78 / 69.3 | 482 |
| `r256x12_edax` d3 every level | 80 / 66 / 49 | 88 / 72 / 68.7 | 33 |
| `r256x12_edax` d4 every level | 85.5 / 75 / 61.5 | 93.3 / 90.7 / 78.7 | 103 |
| `r256x12_edax` d5 every level | 88 / 79.5 / 70 | 98.7 / 92 / 82 | 302 |
| `r256x12_dagger` d3 / d4 / d5 every level | 80.5 / 69 / 57; 88.5 / 75.5 / 65.5; 92 / 81.5 / 72.5 | 88 / 74 / 66; 96 / 79.3 / 75.3; 97.3 / 84 / 74 | |

Strength keeps rising with depth from random openings too (Edax-4: ~40% at d2 -> ~70% at d5), and
pruned search is far faster than root-only. The random-opening failure of root-only depth 3 came from
searching implausible replies, which let the value head's occasional overrating of odd positions
decide; with the policy restricting each level to plausible moves, deeper search helps everywhere.

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
several depths (top move from random/named openings; see `-d` above).

**Loading models saved on the old machine** (Python 3.10, SB3 2.0) works for evaluation, but
their pickled `clip_range`/`lr_schedule` can't be deserialized on Python 3.14; SB3 warns
(`code() argument 13 must be str, not int`) and falls back to clip_range 0.2. Resuming training
from such a model needs `custom_objects={"clip_range": 0.1, ...}` in the load.

## Device Support and Portability

Goal: the code should move to other systems (e.g. Linux with an NVIDIA GPU/CUDA). Device selection
is automatic everywhere (`util/util.py:get_device()`, `--device auto` in `bc_train.py`,
`edax_train.py`, `eval_batch.py`, `collect_positions.py`): CUDA if available, else Apple MPS, else
CPU. Nothing in the Python code is macOS-specific. The Edax engine is: `libedax.dylib` is built for
macOS (with the `SDKROOT` override, see Edax above); a Linux machine needs a shared-library build
of the same fork (`bwanab/edax-reversi`, `src/edax_wrapper.c`) and the library path in
`util/edax_engine.py` (not yet done). Unix sockets (Edax server) work the same on Linux.

## Testing

Run the suite with `./run_tests.sh` (all tests) or `./run_tests.sh <name>` for one group
(`environment`, `scenarios`, `edge_cases`, `training`, `integration`, `bc`, `focused`,
`training_issues`, `step`, `lr`, `features`, `resnet`, `play`, `starts`, `refresh`, `openings`, `mix`, `search`, ...).
The script sets `PYTHONPATH` to the project root and runs through `uv run`. All 145 tests in `tests/`
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

### Bigger network, and search with the Edax-trained value head (deferred 2026-10-01)
- A larger trunk than 128x8 (it showed no overfitting on BC or the Edax labels), to absorb a
  stronger teacher.
- Search check with `edax1m_graded`'s value head: `sb-play.py --search-depth 2 --search-top-k 3`
  (value regret 3.3 vs policy 1.8 discs suggests it will still lose to the top move).

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
