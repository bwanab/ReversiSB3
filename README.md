# ReversiSB3

Reversi (Othello) AI training project using Stable Baselines 3 with PyTorch. Trains a CNN-based reinforcement learning agent through self-play and opponent training.

## Features

- **RL Training**: PPO algorithm with action masking for valid moves
- **Multiple Opponents**: Random, Self-play, Model-based, RAI (Python minimax), Edax (C++ engine)
- **CNN Architecture**: Custom feature extractor for 8x8 board processing
- **Training Modes**: Random, self-play, sequential curriculum, mixed 3-way training
- **Web Interface**: Play against trained models with move analysis
- **Behavioral Cloning**: Pre-train from expert games (WTHOR database)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model against random opponent
python sb-train.py --mode random --timesteps 100000 -m my_model

# Train with self-play
python sb-train.py --mode selfplay --timesteps 500000 -m my_model

# Play against your model
python sb-play.py -m my_model -e 100 -o Random
```

## Edax Engine Integration

The project includes Python bindings to the [Edax](https://github.com/abulmo/edax-reversi) Reversi engine, providing a fast, world-class opponent for training.

### Building the Edax Library

The Edax shared library provides direct Python bindings (~1000x faster than GTP protocol):

```bash
# Navigate to Edax source directory
cd /Users/bill/src/edax-reversi/src

# Build the shared library (macOS)
clang -std=c17 -O3 -march=native -mdynamic-no-pic \
      -D_GNU_SOURCE=1 -DNDEBUG -dynamiclib \
      -o ../bin/libedax.dylib all_lib.c -lm

# Verify the library was built
ls -lh ../bin/libedax.dylib
# Should show ~555KB file

# Verify exported symbols
nm -g ../bin/libedax.dylib | grep edax_
# Should show: edax_create, edax_destroy, edax_get_move, edax_get_score, edax_get_nodes
```

**For Linux**, replace `-dynamiclib` with `-shared` and change output to `.so`:
```bash
clang -std=c17 -O3 -march=native -D_GNU_SOURCE=1 -DNDEBUG \
      -shared -fPIC -o ../bin/libedax.so all_lib.c -lm
```

### Using Edax Opponent

```bash
# Train against Edax depth 1 (fast, ~100 moves/sec)
python sb-train.py -m my_model -o Edax --timesteps 1000000

# Train against Edax depth 4 (balanced, ~10 moves/sec)
python sb-train.py -m my_model -o Edax --depth 4 --timesteps 500000

# Evaluate against Edax depth 6
python sb-play.py -m my_model -e 100 -o Edax --depth 6
```

**Edax Strength Levels:**
- Depth 1: Fast tactical play (~0.01s/move, 100 moves/sec)
- Depth 4: Strong play (~0.1s/move, 10 moves/sec)
- Depth 6: Very strong (~1s/move, 1 move/sec)
- Depth 8+: Expert level (5-10s/move)

### Edax vs RAI Performance

| Opponent | Depth | Speed | Strength | Use Case |
|----------|-------|-------|----------|----------|
| RAI (Python) | 2 | ~0.1s/move | Moderate | Legacy |
| RAI (Python) | 4 | ~10s/move | Good | Too slow |
| **Edax (C)** | 4 | ~0.01s/move | Strong | Fast training |
| **Edax (C)** | 6 | ~0.1s/move | Very Strong | Balanced |
| **Edax (C)** | 8 | ~1s/move | Elite | Evaluation |

Edax depth 6 has the same speed as RAI depth 2 but is much stronger.

### Troubleshooting Edax

**Library not found:**
```bash
# Check library exists
ls /Users/bill/src/edax-reversi/bin/libedax.dylib

# Rebuild if missing (see build instructions above)
```

**Edax plays incorrectly (always loses):**
- Fixed in current version (board perspective flip bug resolved)
- Ensure you're using latest `util/opponents.py` with board flip: `alt_state = state * self.player`

**Verbose output during training:**
- Fixed in current version (output silenced via `options.noise = 100`)
- Ensure library was rebuilt after silencing changes

## Training Modes

### Mode 1: Random Only
```bash
python sb-train.py --mode random --timesteps 50000 -m test_model
```
Train exclusively against random opponent. Good for bootstrapping.

### Mode 2: Self-Play Mixed
```bash
python sb-train.py --mode selfplay --timesteps 200000 --random-ratio 0.2
```
Self-play with 20% random opponent. Model refreshed every 40k timesteps (default: timesteps/5).

### Mode 3: Sequential Curriculum
```bash
python sb-train.py --mode sequential \
  --random-timesteps 100000 \
  --selfplay-timesteps 300000 \
  --start-lr 3e-5 --end-lr 0
```
Phase 1: Random-only, then Phase 2: Self-play mixed with LR decay.

### Mode 4: Mixed Three-Way
```bash
python sb-train.py --mode mixed --timesteps 1000000 \
  --selfplay-ratio 0.75 --random-ratio 0.20 --rai-ratio 0.05 --rai-depth 2
```
Train against self (75%), random (20%), and RAI (5%) opponents.

### Legacy: Direct Opponent Training
```bash
# Train against specific opponent
python sb-train.py -m my_model -o Edax --timesteps 1000000

# Equivalent to 100% training against Edax
```

## Behavioral Cloning

Pre-train models from expert game databases (WTHOR format):

```bash
# Generate BC dataset from RAI games
python generate_bc_dataset.py -g 10000 -d 2 -o bc_dataset.pkl

# Train BC model
python bc_train.py -d bc_dataset.pkl -m bc_pretrained -e 20

# Fine-tune with RL
python sb-train.py --mode selfplay -m bc_pretrained --timesteps 500000
```

## Project Structure

```
ReversiSB3/
├── util/
│   ├── reversi.py          # Gymnasium environment
│   ├── reversi_cnn.py      # CNN architecture
│   ├── opponents.py        # Opponent implementations
│   ├── edax_engine.py      # Edax Python bindings
│   ├── play.py             # Game playing utilities
│   └── util.py             # Helper functions
├── models/                 # Trained models (.zip)
├── checkpoints/            # Training checkpoints
├── sb-train.py             # Training script
├── sb-play.py              # Evaluation script
├── generate_bc_dataset.py  # BC data generation
├── bc_train.py             # BC training
└── tests/                  # Test suite
```

## Web Interface

Play against trained models in your browser:

```bash
python web_app.py -m my_model
```

Features:
- Interactive board with click-to-move
- Move history with undo/redo
- Model move analysis (top-5 probabilities)
- Real-time board evaluation

## Training Goals

Progressive difficulty milestones:

1. **Basic**: Beat random opponent in >90% of games
2. **Intermediate**: Beat RAI depth 2 in >50% of games
3. **Advanced**: Beat Edax depth 4 in >25% of games
4. **Expert**: Beat Edax depth 6 in >10% of games

## Technical Details

**Environment**: Custom Gymnasium environment with action masking for valid moves

**Algorithm**: MaskablePPO (Proximal Policy Optimization with action masking)

**Neural Network**:
- Input: 8x8 board (3 channels: current player, opponent, valid moves)
- Architecture: CNN with 4 conv layers + 2 dense layers
- Output: 64 action logits (one per square)

**Hyperparameters** (see `util/util.py`):
- Learning rate: 1e-5 (configurable)
- Batch size: 256
- N-steps: 2048
- Entropy coefficient: 0.03
- Gamma: 0.98

## Development

Run tests:
```bash
PYTHONPATH=/Users/bill/src/ReversiSB3 python tests/run_tests.py

# Or specific test suites
python tests/run_tests.py bc          # Behavioral cloning tests
python test_edax_bindings.py          # Edax bindings tests
python test_edax_opponent.py          # Edax integration tests
```

## Documentation

- `CLAUDE.md` - Detailed project documentation and training observations
- `EDAX_BINDINGS_COMPLETE.md` - Edax integration implementation details
- `EDAX_BINDINGS_PLAN.md` - Original implementation plan

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Stable Baselines 3 2.0.0
- Gymnasium 0.28.1
- Edax 4.6 (for Edax opponent)

See `requirements.txt` for complete dependency list.

## License

This project uses the Edax engine which is licensed under GPL v3. See the [Edax repository](https://github.com/abulmo/edax-reversi) for details.
