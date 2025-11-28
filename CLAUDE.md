# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReversiSB3 is a Reversi (Othello) AI training project using Stable Baselines 3 with PyTorch. The project implements a CNN-based reinforcement learning agent that learns to play Reversi through self-play and opponent training.

## Training Goals

The project aims to develop an AI agent with progressively stronger play:

1. **Initial Target**: Beat RAI opponent with 2-move lookahead at least once in 100 games
2. **Intermediate Target**: Beat RAI opponent with 2-move lookahead in majority of games (>50%)  
3. **Advanced Target**: Beat RAI opponent with 4-move lookahead in majority of games (>50%)

These milestones provide measurable benchmarks against the ReversiAI (RAI) engine at different search depths, representing increasingly sophisticated strategic play.

## Key Architecture Components

### Core Environment (`util/reversi.py`)
- `ReversiEnvCNN`: Custom Gymnasium environment for Reversi with CNN observation space
- Implements 8x8 board with standard Reversi rules
- Uses action masking to prevent invalid moves
- Supports multiple opponent types during training

### Neural Network (`util/reversi_cnn.py`)
- Custom CNN feature extractor for board state processing
- Processes 8x8 board representations with multiple channels

### Training System (`sb-train.py`)
- Main training script using MaskablePPO from sb3-contrib
- Implements self-play with opponent switching
- Uses TensorBoard logging and checkpoint saving
- Supports training against different opponent types (Random, Model-based)

### Opponent System (`util/opponents.py`)
- Multiple opponent implementations: Random, Human, Model-based
- `get_opponent()` factory function for opponent selection
- Used for both training and evaluation

### Utility Functions (`util/util.py`)
- Model loading/saving with `get_model()`
- Action masking with `mask_fn()`
- Learning rate scheduling functions
- Device detection (CPU/CUDA/MPS)

## Common Commands

### Training a Model
```bash
python sb-train.py -m MODEL_NAME -e 200000 -o Model -t Random -w 512
```
Options:
- `-m/--model`: Model name (saved to `models/`)
- `-e/--episodes`: Training episodes per epoch
- `-o/--opponent`: Training opponent type
- `-t/--test_opponent`: Evaluation opponent type  
- `-w/--net_width`: Neural network width
- `-p/--epochs`: Training epochs

### Playing/Evaluating Models
```bash
python sb-play.py -m MODEL_NAME -e 100 -o Random -v
```
Options:
- `-m/--model`: Model file to load
- `-e/--episodes`: Number of games to play
- `-o/--opponent`: Opponent type
- `-v/--verbose`: Display game details

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Development Setup
The project uses a virtual environment in `venv-sb/`. Key dependencies:
- `stable-baselines3==2.0.0`
- `sb3-contrib==2.0.0` 
- `torch==2.0.1`
- `gymnasium==0.28.1`
- Custom reversi packages installed from git

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

## Training Flow

1. Environment creation with opponent configuration
2. Model initialization (load existing or create new CNN-based MaskablePPO)
3. Multi-epoch training with opponent switching:
   - Train against specified opponent (e.g., previous model)
   - Switch to random opponent for 25% of episodes
4. Periodic evaluation and checkpoint saving
5. TensorBoard logging of win rates and metrics

## Device Support

The training system automatically detects and uses:
- Apple Silicon GPU (MPS) if available
- CUDA GPU if available  
- CPU as fallback

Device selection is handled in `sb-train.py` and passed to model creation.

## Critical Issues Found

### Test Environment Setup
- Tests must be run with: `PYTHONPATH=/Users/bill/src/ReversiSB3 python tests/run_tests.py`
- Alternative: Use `./run_tests.sh` script

### ReversiEnvCNN Implementation Issues (Found via test suite)
The following critical methods are missing from `ReversiEnvCNN` in `util/reversi.py`:

1. **`is_game_over(board)` method** - Required for game termination detection
   - Used by training logic to determine episode end
   - Missing causes 7 test failures

2. **`_place(board, player, position)` method** - Required for piece placement
   - Used for move execution and piece capture mechanics
   - Missing causes piece capture test failures

3. **Action masking data type issue** - Returns wrong types instead of booleans
   - Should return boolean array for valid/invalid moves
   - Corrupts action space for RL training

4. **Valid move detection bug** - Incorrect initial board move validation
   - Move 27 should be valid on initial board but isn't detected
   - Indicates potential issues with move validation logic

### Impact on Training
These bugs likely explain training plateau because:
- Game termination detection broken → poor end-game learning
- Action masking corrupted → invalid action space feedback  
- Move validation errors → wrong legal move feedback
- Missing core methods → incomplete game mechanics

### Test Results Summary
- 34 tests run, 11 failures/errors (67.6% success rate)
- Primary issues in `ReversiEnvCNN` class implementation
- Tests identify specific methods and logic that need fixing

**Priority**: Fix these core environment issues before continuing training optimization.