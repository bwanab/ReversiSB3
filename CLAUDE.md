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

## Training Observations & Hyperparameter Experiments

### Observed Training Behavior (Random Opponent)
- **Initial Progress**: Models show good learning progress for ~250,000 timesteps
- **Collapse at 250k**: After 250k steps, training exhibits instability:
  - `explained_variance` drops sharply
  - `value_loss`, `policy_gradient_loss`, and `loss` metrics fall off
  - Indicates policy/value function instability

### Current Hyperparameters (util/util.py)
```python
learning_rate = 1e-5         # Configurable via -lr flag (was 2e-5)
batch_size = 256             # Increased from 128 for stability
n_steps = 2048
ent_coef = 0.03              # Increased from 0.02 for more exploration
n_epochs = 5                 # Reduced from 10 to prevent overfitting
gae_lambda = 0.90            # Reduced from 0.95 for less bias
gamma = 0.98                 # Discount factor
clip_range = 0.1             # PPO clip range
```

### Training Progress (Current Model Status)

**Total Training**: ~5M timesteps
- Initial 3M: Sequential (1M random + 2M selfplay) with LR decay (3e-5 → 0)
- Additional 2M: Self-play with low LR (5e-6) fine-tuning

**Performance vs RAI (2-move lookahead):**
- Deterministic: 100% (but replays same game)
- Non-deterministic: ~6% win rate
- vs RAI (1-move): ~20% win rate

**Training Metrics:**
- explained_variance: Improved to 0.45 (smoothed), peaks at 0.65
- Training stable with current hyperparameters
- Slow but steady progress

**Current Strategy**: Mixed mode training (Self 75% / Random 20% / RAI 5%)
- Introduces RAI exposure during training (not just evaluation)
- Cost-effective: ~1.7 hours RAI time per 1M timesteps
- Goal: Learn to counter RAI's minimax style

### Future Hyperparameter Experiments to Consider

**If learning rate reduction alone doesn't solve instability:**

1. **Batch Size** (current: 128)
   - Try: 256, 512
   - Larger batches = more stable gradients, less noise
   - Trade-off: Slower updates, more memory

2. **Number of Epochs** (current: 10)
   - Try: 5, 7
   - Fewer epochs = less overfitting on each batch
   - Risk: Slower learning from each experience batch

3. **GAE Lambda** (current: 0.95)
   - Try: 0.90, 0.92
   - Lower = less bias, more variance in advantage estimates
   - Affects credit assignment over time

4. **Entropy Coefficient** (current: 0.02)
   - Try: 0.03, 0.05
   - Higher = more exploration, less deterministic policy
   - Can prevent premature convergence

5. **Clip Range** (current: 0.1)
   - Try: 0.05, 0.08
   - Lower = more conservative policy updates
   - Prevents large destabilizing updates

**Metrics to Monitor:**
- `explained_variance`: Should stay > 0.5 (ideally 0.7+)
- `approx_kl`: Should stay < 0.05 (KL divergence between old/new policy)
- `clip_fraction`: If consistently > 0.3, clip range may be too restrictive
- Win rate stability over time

## Contingency Plan: BC Pre-training Approach

**Use Case**: If current model plateaus at middling performance (e.g., stuck at 6-15% vs RAI after 2-3M mixed mode timesteps)

### Behavioral Cloning (BC) → RL Fine-tuning Strategy

**Rationale:**
- Current model started with Random/Self-play (weak opponents)
- May be stuck in local minimum, unable to "unlearn" suboptimal strategies
- Fresh model with BC pre-training starts with expert (RAI) knowledge built-in
- Proven approach: AlphaGo, OpenAI Five used BC → RL curriculum

### Implementation Plan

#### Phase 1: Data Generation (One-Time, Offline)
```bash
# Generate RAI self-play games for BC dataset
# Run overnight: ~33 hours one-time cost
python generate_rai_games.py --games 5000 --depth 2 --output rai_dataset.pkl

# Produces:
# - 5000 games × ~30 moves = 150k (state, action) pairs
# - Saves to disk for reusable training data
```

#### Phase 2: BC Pre-training (Fast, ~1-2 hours)
```python
# Train fresh model using supervised learning on RAI dataset
# Model learns: "Given board state, what would RAI do?"

dataset = load_rai_games('rai_dataset.pkl')  # 150k examples
model = create_fresh_model(architecture='CNN', hyperparameters=current_tuned_params)

for epoch in range(20):
    for batch in dataset:
        # Supervised learning: predict RAI's action
        loss = cross_entropy(model.policy(states), rai_actions)
        optimizer.step()

# Result: Model that plays "like RAI" (strategic foundation)
```

#### Phase 3: RL Fine-tuning (Online, 2-3M timesteps)
```bash
# Continue BC model with mixed mode training
python sb-train.py --mode mixed \
  -m "bc_then_rl_model" \
  --timesteps 2000000 \
  --selfplay-ratio 0.75 \
  --rai-ratio 0.05 \
  --rai-depth 2 \
  -lr 3e-5

# Model now learns to BEAT RAI (not just imitate)
# Builds on strategic foundation from BC
```

#### Phase 4: Comparison
Test both approaches:
- **Current model** (5M+ pure RL): X% vs RAI
- **BC+RL model** (BC + 2M RL): Y% vs RAI
- If Y >> X: BC bootstrap approach validated

### When to Trigger BC Approach

**Decision Criteria (After 2-3M Mixed Mode Timesteps):**

**Plateau Indicators (Time for BC approach):**
- Win rate < 10% vs RAI after 2M mixed mode timesteps
- explained_variance stuck at 0.45 for 1M+ timesteps
- No improvement across 3+ consecutive training runs
- Learning curve shows sharp gains early, then completely flat

**Progress Indicators (Continue current approach):**
- Any win rate improvement trajectory (6% → 10% → 15%, etc.)
- explained_variance climbing (0.45 → 0.50 → 0.55+)
- Model behavior visibly improving (more strategic opening/endgame play)
- Steady or accelerating learning curve

### Technical Considerations

**Advantages:**
- Separates concerns: Learn basics (BC) → Learn to win (RL)
- One-time data generation cost vs. repeated online RAI calls
- Tests "local minimum" hypothesis for current model
- Curriculum learning: Expert imitation → Adversarial optimization

**Challenges:**
- Requires custom BC training code (not built into SB3)
- BC might teach "playing like RAI" vs. "exploiting RAI weaknesses"
- Throws away 5M timesteps of current model's learning
- Need careful handling of perspective (RAI plays both colors)

**Implementation Effort**: ~4-6 hours
- Data generation script
- BC training loop (PyTorch)
- Integration with existing model architecture
- Evaluation harness

### Priority

**Current Status**: Mixed mode training is active approach
**BC Approach**: Backup plan if plateau persists after thorough mixed mode evaluation

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