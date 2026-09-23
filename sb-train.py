import gymnasium as gym
from util.reversi import build_reversi, ReversiEnvCNN
from util.util import mask_fn, get_model, set_learning_rate
from util.play import play
from util.opponents import get_opponent, RandomOpponent

import numpy as np
import os.path
from time import time

import torch
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat, configure
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps, CheckpointCallback

# Setup device
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ MPS (Apple Silicon GPU) is available and will be used")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ CUDA GPU is available and will be used")
else:
    device = "cpu"
    print("✓ Using CPU")

def round_trip(training_env, opponent):
    env = training_env.envs[0]
    state = env.board
    player = env.player
    while player == -1:
        action = opponent.get_action(env, state)
        obs, rewards, term, info = training_env.step(action)
        state = obs[0]
        player = env.player
    return True

class FullRoundTripCallback(BaseCallback):
    def __init__(self, file, env, net_width, episodes = 100_000, verbose: int = 1, opponentName = "Random"):
        self.file = file
        self.env = env
        self.net_width = net_width
        self.threshold = 1
        self.r_factor = 1 / episodes
        self.opponentName = opponentName
        self.update_opponent()
        super(FullRoundTripCallback, self).__init__(verbose)

    def get_threshold(self):
        rval = self.threshold
        self.threshold -= self.r_factor
        if self.threshold < 0:
            self.threshold = 0.0
        return rval
    
    def update_opponent(self):
        self.opponent = get_opponent(self.opponentName, file=self.file, env=self.env, net_width=self.net_width)

    def get_action(self, env, state):
        return self.opponent.get_action(env, state)

    def _on_step(self) -> bool:
        return round_trip(self.env, self.opponent)


class PlayCallback(BaseCallback):
    def __init__(self, model, file, episodes = 100, opponent = RandomOpponent, verbose: bool = False):
        self.episodes = episodes
        self.verbose = verbose
        self.opponent = opponent
        self.model = model
        self.file = file
        self.logger = Logger("./" + file + ".log", TensorBoardOutputFormat("./" + file + ".log"))
        super(PlayCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        black_wins = play(self.model, self.episodes, self.opponent, True, self.verbose)
        self.logger.record(key="black_wins", value=black_wins)
        self.model.save(self.file)
        return True


import argparse

def learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, timesteps, lr=None):
    """Helper function to run training for a specified number of timesteps.

    Parameters
    ----------
    lr : float, optional
        If provided, sets the learning rate before training
    """
    # Set learning rate if provided
    if lr is not None:
        set_learning_rate(model, lr)
        print(f"📊 Learning rate: {lr:.2e}")

    test_opponent = get_opponent(args.test_opponent, file=file, env=env, net_width=net_width)
    playCB = PlayCallback(model, file, 100, test_opponent, False)
    everyNCB = EveryNTimesteps(n_steps=10000, callback=playCB)
    model.learn(total_timesteps=timesteps,
                progress_bar=True,
                callback=[everyNCB, checkpoint_cb],
                tb_log_name=args.model,
                reset_num_timesteps=False)

def train_random_only(model, env, args, file, checkpoint_cb, net_width, timesteps):
    """Mode 1: Train only against Random opponent."""
    print(f"🎯 Training Mode: Random-only for {timesteps:,} timesteps")
    env.set_opponent("Random", None)
    learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, timesteps)

def train_selfplay_mixed(model, env, args, file, checkpoint_cb, net_width, timesteps, random_ratio=0.2, refresh_interval=None):
    """Mode 2: Train with self-play mixed with random opponent."""
    if refresh_interval is None:
        refresh_interval = timesteps // 5
    
    random_timesteps = int(timesteps * random_ratio)
    selfplay_timesteps = timesteps - random_timesteps
    
    print(f"🤖 Training Mode: Self-play mixed for {timesteps:,} timesteps")
    print(f"   - Self-play: {selfplay_timesteps:,} timesteps ({100*(1-random_ratio):.0f}%)")
    print(f"   - Random: {random_timesteps:,} timesteps ({100*random_ratio:.0f}%)")
    print(f"   - Refresh interval: {refresh_interval:,} timesteps")
    
    # Create temp opponent model path
    temp_opponent = file + "_temp_opponent"
    
    timesteps_trained = 0
    while timesteps_trained < timesteps:
        # Determine timesteps for this block
        remaining = timesteps - timesteps_trained
        block_timesteps = min(refresh_interval, remaining)
        
        # Calculate ratio for this block
        block_random = int(block_timesteps * random_ratio)
        block_selfplay = block_timesteps - block_random
        
        print(f"\n--- Block {timesteps_trained//refresh_interval + 1}: {block_timesteps:,} timesteps ---")
        
        # Save current model as opponent
        print(f"💾 Saving current model as opponent: {temp_opponent}")
        model.save(temp_opponent)
        
        # Train against self (frozen opponent)
        if block_selfplay > 0:
            print(f"🤖 Self-play training: {block_selfplay:,} timesteps")
            env.set_opponent("Model", temp_opponent)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_selfplay)
        
        # Train against random
        if block_random > 0:
            print(f"🎲 Random training: {block_random:,} timesteps")
            env.set_opponent("Random", None)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_random)
        
        timesteps_trained += block_timesteps
    
    # Clean up temp opponent file
    try:
        if os.path.exists(temp_opponent + ".zip"):
            os.remove(temp_opponent + ".zip")
            print(f"🧹 Cleaned up temp opponent file: {temp_opponent}.zip")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up temp file: {e}")

def train_mixed_three_way(model, env, args, file, checkpoint_cb, net_width, timesteps,
                         selfplay_ratio=0.75, random_ratio=0.20, rai_ratio=0.05,
                         refresh_interval=None, rai_depth=2):
    """Mode 4: Train with three-way mix: Self-play, Random, and RAI opponents.

    Parameters
    ----------
    selfplay_ratio : float
        Fraction of timesteps against self (default: 0.75)
    random_ratio : float
        Fraction of timesteps against random (default: 0.20)
    rai_ratio : float
        Fraction of timesteps against RAI (default: 0.05)
    rai_depth : int
        RAI minimax search depth (default: 2)
    """
    if refresh_interval is None:
        refresh_interval = timesteps // 5

    # Normalize ratios to ensure they sum to 1.0
    total_ratio = selfplay_ratio + random_ratio + rai_ratio
    selfplay_ratio /= total_ratio
    random_ratio /= total_ratio
    rai_ratio /= total_ratio

    selfplay_timesteps = int(timesteps * selfplay_ratio)
    random_timesteps = int(timesteps * random_ratio)
    rai_timesteps = timesteps - selfplay_timesteps - random_timesteps  # Ensure exact total

    print(f"🎯 Training Mode: Mixed (Self/Random/RAI) for {timesteps:,} timesteps")
    print(f"   - Self-play: {selfplay_timesteps:,} timesteps ({100*selfplay_ratio:.1f}%)")
    print(f"   - Random: {random_timesteps:,} timesteps ({100*random_ratio:.1f}%)")
    print(f"   - RAI (depth={rai_depth}): {rai_timesteps:,} timesteps ({100*rai_ratio:.1f}%)")
    print(f"   - Refresh interval: {refresh_interval:,} timesteps")
    print(f"   ⏱️  Estimated RAI time: ~{rai_timesteps/30*4/60:.1f} minutes")

    # Create temp opponent model path
    temp_opponent = file + "_temp_opponent"

    timesteps_trained = 0
    while timesteps_trained < timesteps:
        # Determine timesteps for this block
        remaining = timesteps - timesteps_trained
        block_timesteps = min(refresh_interval, remaining)

        # Calculate timesteps for each opponent in this block
        block_selfplay = int(block_timesteps * selfplay_ratio)
        block_random = int(block_timesteps * random_ratio)
        block_rai = block_timesteps - block_selfplay - block_random

        print(f"\n--- Block {timesteps_trained//refresh_interval + 1}: {block_timesteps:,} timesteps ---")
        print(f"    Self: {block_selfplay:,} | Random: {block_random:,} | RAI: {block_rai:,}")

        # Save current model as opponent
        print(f"💾 Saving current model as opponent: {temp_opponent}")
        model.save(temp_opponent)

        # Train against self (frozen opponent)
        if block_selfplay > 0:
            print(f"🤖 Self-play training: {block_selfplay:,} timesteps")
            env.set_opponent("Model", temp_opponent)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_selfplay)

        # Train against random
        if block_random > 0:
            print(f"🎲 Random training: {block_random:,} timesteps")
            env.set_opponent("Random", None)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_random)

        # Train against RAI
        if block_rai > 0:
            print(f"🧠 RAI training (depth={rai_depth}): {block_rai:,} timesteps")
            env.set_opponent("RAI", None, depth=rai_depth)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_rai)

        timesteps_trained += block_timesteps

    # Clean up temp opponent file
    try:
        if os.path.exists(temp_opponent + ".zip"):
            os.remove(temp_opponent + ".zip")
            print(f"🧹 Cleaned up temp opponent file: {temp_opponent}.zip")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up temp file: {e}")

def train_edax_curriculum(model, env, args, file, checkpoint_cb, net_width, timesteps,
                          edax_depths, edax_ratios, random_ratio=0.0, refresh_interval=None):
    """Mode 5: Train with multiple Edax depths for curriculum learning.

    Parameters
    ----------
    edax_depths : list of int
        List of Edax depths to train against (e.g., [6, 7])
    edax_ratios : list of float
        Fraction of timesteps for each Edax depth (e.g., [0.7, 0.2])
    random_ratio : float
        Fraction of timesteps against random opponent (default: 0.0)
    refresh_interval : int
        How often to checkpoint and evaluate (default: timesteps // 5)
    """
    if refresh_interval is None:
        refresh_interval = timesteps // 5

    # Normalize ratios to ensure they sum to 1.0
    total_ratio = sum(edax_ratios) + random_ratio
    edax_ratios = [r / total_ratio for r in edax_ratios]
    random_ratio /= total_ratio

    # Calculate timesteps for each opponent
    edax_timesteps = [int(timesteps * r) for r in edax_ratios]
    random_timesteps = timesteps - sum(edax_timesteps)  # Ensure exact total

    print(f"🎯 Training Mode: Edax Curriculum for {timesteps:,} timesteps")
    for depth, ts, ratio in zip(edax_depths, edax_timesteps, edax_ratios):
        print(f"   - Edax-{depth}: {ts:,} timesteps ({100*ratio:.1f}%)")
    if random_timesteps > 0:
        print(f"   - Random: {random_timesteps:,} timesteps ({100*random_ratio:.1f}%)")
    print(f"   - Refresh interval: {refresh_interval:,} timesteps")

    timesteps_trained = 0
    while timesteps_trained < timesteps:
        # Determine timesteps for this block
        remaining = timesteps - timesteps_trained
        block_timesteps = min(refresh_interval, remaining)

        # Calculate timesteps for each opponent in this block
        block_edax = [int(block_timesteps * r) for r in edax_ratios]
        block_random = block_timesteps - sum(block_edax)

        print(f"\n--- Block {timesteps_trained//refresh_interval + 1}: {block_timesteps:,} timesteps ---")
        depth_str = " | ".join([f"Edax-{d}: {ts:,}" for d, ts in zip(edax_depths, block_edax)])
        if block_random > 0:
            depth_str += f" | Random: {block_random:,}"
        print(f"    {depth_str}")

        # Train against each Edax depth
        for depth, block_ts in zip(edax_depths, block_edax):
            if block_ts > 0:
                print(f"🧠 Edax-{depth} training: {block_ts:,} timesteps")
                env.set_opponent("Edax", None, depth=depth)
                learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_ts)

        # Train against random
        if block_random > 0:
            print(f"🎲 Random training: {block_random:,} timesteps")
            env.set_opponent("Random", None)
            learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_random)

        timesteps_trained += block_timesteps

    print(f"✅ Edax curriculum training complete!")

def train_sequential(model, env, args, file, checkpoint_cb, net_width, random_timesteps, selfplay_timesteps, random_ratio=0.2):
    """Mode 3: Train random-only, then self-play mixed with optional LR decay."""
    total_timesteps = random_timesteps + selfplay_timesteps
    use_lr_decay = hasattr(args, 'start_lr') and args.start_lr is not None and hasattr(args, 'end_lr') and args.end_lr is not None

    print(f"📈 Training Mode: Sequential")
    print(f"   Phase 1: Random-only for {random_timesteps:,} timesteps")
    print(f"   Phase 2: Self-play mixed for {selfplay_timesteps:,} timesteps")
    if use_lr_decay:
        print(f"   📉 LR Decay: {args.start_lr:.2e} → {args.end_lr:.2e}")

    timesteps_done = 0

    # Phase 1: Random-only
    print(f"\n🎯 Phase 1: Random-only training")
    if use_lr_decay:
        progress = timesteps_done / total_timesteps
        phase1_lr = args.start_lr - (args.start_lr - args.end_lr) * progress
        env.set_opponent("Random", None)
        learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, random_timesteps, lr=phase1_lr)
    else:
        train_random_only(model, env, args, file, checkpoint_cb, net_width, random_timesteps)

    timesteps_done += random_timesteps

    # Phase 2: Self-play mixed with LR decay
    print(f"\n🤖 Phase 2: Self-play mixed training")
    refresh_interval = selfplay_timesteps // 5  # Default refresh interval

    if use_lr_decay:
        # Custom self-play with LR decay
        random_ts = int(selfplay_timesteps * random_ratio)
        selfplay_ts = selfplay_timesteps - random_ts

        print(f"   - Self-play: {selfplay_ts:,} timesteps ({100*(1-random_ratio):.0f}%)")
        print(f"   - Random: {random_ts:,} timesteps ({100*random_ratio:.0f}%)")
        print(f"   - Refresh interval: {refresh_interval:,} timesteps")

        temp_opponent = file + "_temp_opponent"
        phase2_timesteps_done = 0

        while phase2_timesteps_done < selfplay_timesteps:
            remaining = selfplay_timesteps - phase2_timesteps_done
            block_timesteps = min(refresh_interval, remaining)
            block_random = int(block_timesteps * random_ratio)
            block_selfplay = block_timesteps - block_random

            print(f"\n--- Block {phase2_timesteps_done//refresh_interval + 1}: {block_timesteps:,} timesteps ---")

            # Save current model as opponent
            print(f"💾 Saving current model as opponent: {temp_opponent}")
            model.save(temp_opponent)

            # Train against self with LR decay
            if block_selfplay > 0:
                print(f"🤖 Self-play training: {block_selfplay:,} timesteps")
                progress = (timesteps_done + phase2_timesteps_done) / total_timesteps
                current_lr = args.start_lr - (args.start_lr - args.end_lr) * progress
                env.set_opponent("Model", temp_opponent)
                learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_selfplay, lr=current_lr)
                phase2_timesteps_done += block_selfplay

            # Train against random with LR decay
            if block_random > 0:
                print(f"🎲 Random training: {block_random:,} timesteps")
                progress = (timesteps_done + phase2_timesteps_done) / total_timesteps
                current_lr = args.start_lr - (args.start_lr - args.end_lr) * progress
                env.set_opponent("Random", None)
                learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, block_random, lr=current_lr)
                phase2_timesteps_done += block_random

        # Clean up temp opponent file
        try:
            if os.path.exists(temp_opponent + ".zip"):
                os.remove(temp_opponent + ".zip")
                print(f"🧹 Cleaned up temp opponent file: {temp_opponent}.zip")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp file: {e}")
    else:
        train_selfplay_mixed(model, env, args, file, checkpoint_cb, net_width, selfplay_timesteps, random_ratio, refresh_interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'sb-train',
                    description = 'Train Reversi RL agent with different training modes',
                    epilog = '''
Usage Examples:
  # Mode 1: Random-only training
  python sb-train.py --mode random --timesteps 50000 -m "test_model"

  # Mode 2: Self-play mixed with custom ratios
  python sb-train.py --mode selfplay --timesteps 200000 --random-ratio 0.3 --refresh-interval 40000

  # Mode 3: Sequential curriculum training
  python sb-train.py --mode sequential --random-timesteps 100000 --selfplay-timesteps 300000

  # Mode 3 with learning rate decay (3e-5 to 0 over 3M timesteps)
  python sb-train.py --mode sequential --random-timesteps 1000000 --selfplay-timesteps 2000000 \\
                     --start-lr 3e-5 --end-lr 0 -m "my_model"

  # Mode 4: Mixed training with Self/Random/RAI opponents (75/20/5 split)
  python sb-train.py --mode mixed --timesteps 1000000 -m "my_model" \\
                     --selfplay-ratio 0.75 --random-ratio 0.20 --rai-ratio 0.05 --rai-depth 2

  # Quick defaults
  python sb-train.py --mode selfplay --timesteps 100000  # Uses 20% random, refreshes every 20k steps
''',
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Training mode selection
    parser.add_argument("--mode", choices=["random", "selfplay", "sequential", "mixed", "edax-curriculum"], default="random",
                       help="Training mode: random, selfplay, sequential, mixed, or edax-curriculum")
    
    # Timesteps configuration
    parser.add_argument("--timesteps", type=int, default=200_000,
                       help="Number of timesteps for random/selfplay modes")
    parser.add_argument("--random-timesteps", type=int, default=100_000,
                       help="Number of timesteps for random phase in sequential mode")
    parser.add_argument("--selfplay-timesteps", type=int, default=200_000,
                       help="Number of timesteps for selfplay phase in sequential mode")
    
    # Self-play configuration
    parser.add_argument("--random-ratio", type=float, default=0.2,
                       help="Ratio of timesteps against random opponent (default: 0.2)")
    parser.add_argument("--refresh-interval", type=int, default=None,
                       help="Timesteps between self-model refreshes (default: timesteps//5)")

    # Mixed mode configuration (3-way: Self/Random/RAI)
    parser.add_argument("--selfplay-ratio", type=float, default=0.75,
                       help="Ratio of self-play in mixed mode (default: 0.75)")
    parser.add_argument("--rai-ratio", type=float, default=0.05,
                       help="Ratio of RAI training in mixed mode (default: 0.05)")
    parser.add_argument("-d", "--rai-depth", type=int, default=2,
                       help="RAI minimax search depth for mixed mode (default: 2)")

    parser.add_argument("-x", "--edax-depth", type=int, default=2,
                        help="search depth when opponent is RAI or Edax")

    # Edax curriculum mode arguments
    parser.add_argument("--edax-depths", type=str, default=None,
                       help="Comma-separated Edax depths for curriculum mode (e.g., '6,7')")
    parser.add_argument("--edax-ratios", type=str, default=None,
                       help="Comma-separated ratios for each Edax depth (e.g., '0.7,0.2')")

    # Model and environment configuration
    parser.add_argument("-m", "--model", default="dorkQ", help="Model name")
    parser.add_argument("-t", "--test-opponent", default="Random", help="Test opponent type")
    parser.add_argument("-w", "--net-width", type=int, default=512, help="Neural network width - Not for CNNs!")
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")

    # Learning rate decay (for sequential mode)
    parser.add_argument("--start-lr", type=float, default=None,
                       help="Starting learning rate for LR decay (sequential mode only)")
    parser.add_argument("--end-lr", type=float, default=None,
                       help="Ending learning rate for LR decay (sequential mode only)")

    # Legacy arguments (for backward compatibility)
    parser.add_argument("-p", "--epochs", type=int, default=None, help="Legacy: use --mode instead")
    parser.add_argument("-e", "--episodes", type=int, default=None, help="Legacy: use --timesteps instead")
    parser.add_argument("-o", "--opponent", default=None, help="Legacy: use --mode instead")
    parser.add_argument("-r", "--opp-model", default=None, help="Legacy opponent model")

    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.episodes is not None and args.timesteps == 200_000:
        args.timesteps = args.episodes
        print(f"⚠️  Using legacy --episodes argument. Consider using --timesteps instead.")
    
    if args.epochs is not None:
        print(f"⚠️  Legacy --epochs argument ignored. Use --mode instead for training configuration.")


    file = "models/" + args.model + "_CNN_test"
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=args.model + "_CNN_test",
    )

    # Initial environment setup (will be reconfigured based on mode)
    env: ReversiEnvCNN = ActionMasker(build_reversi(opponent="Random"), mask_fn)

    model = get_model(file, env, net_width=args.net_width, learning_rate=args.learning_rate, model_type="cnn", device=device)
    
    print(f"🚀 Starting training with model: {args.model}")
    print(f"💻 Using device: {device}")
    
        # Handle legacy opponent parameter (for backward compatibility)
    if args.opponent is not None:
        print(f"\n⚠️  Using legacy opponent parameter: {args.opponent}")
        print(f"   Consider using --mode for more flexible training options")
        print(f"🎯 Training against {args.opponent} opponent (depth={args.edax_depth})")

        # Set up environment with specified opponent
        print(f" opponent create with edax depth = {args.edax_depth}")
        env.set_opponent(args.opponent, args.opp_model, depth=args.edax_depth)

        # Train for specified timesteps
        learn_helper(PlayCallback, args, file, checkpoint_cb, env, args.net_width, model, args.timesteps)


    # Execute training based on mode
    elif args.mode == "random":
        train_random_only(model, env, args, file, checkpoint_cb, args.net_width, args.timesteps)

    elif args.mode == "selfplay":
        refresh_interval = args.refresh_interval or (args.timesteps // 5)
        train_selfplay_mixed(model, env, args, file, checkpoint_cb, args.net_width,
                           args.timesteps, args.random_ratio, refresh_interval)

    elif args.mode == "sequential":
        train_sequential(model, env, args, file, checkpoint_cb, args.net_width,
                        args.random_timesteps, args.selfplay_timesteps, args.random_ratio)

    elif args.mode == "mixed":
        refresh_interval = args.refresh_interval or (args.timesteps // 5)
        train_mixed_three_way(model, env, args, file, checkpoint_cb, args.net_width,
                            args.timesteps, args.selfplay_ratio, args.random_ratio,
                            args.rai_ratio, refresh_interval, args.rai_depth)

    elif args.mode == "edax-curriculum":
        # Parse comma-separated depths and ratios
        if args.edax_depths is None or args.edax_ratios is None:
            print("❌ Error: --edax-depths and --edax-ratios are required for edax-curriculum mode")
            print("   Example: --edax-depths 6,7 --edax-ratios 0.7,0.2 --random-ratio 0.1")
            exit(1)

        try:
            edax_depths = [int(d.strip()) for d in args.edax_depths.split(',')]
            edax_ratios = [float(r.strip()) for r in args.edax_ratios.split(',')]
        except ValueError as e:
            print(f"❌ Error parsing depths or ratios: {e}")
            print("   Example: --edax-depths 6,7 --edax-ratios 0.7,0.2")
            exit(1)

        if len(edax_depths) != len(edax_ratios):
            print(f"❌ Error: Number of depths ({len(edax_depths)}) must match number of ratios ({len(edax_ratios)})")
            exit(1)

        refresh_interval = args.refresh_interval or (args.timesteps // 5)
        train_edax_curriculum(model, env, args, file, checkpoint_cb, args.net_width,
                            args.timesteps, edax_depths, edax_ratios,
                            args.random_ratio, refresh_interval)

    # Save the final model
    model.save(file)
    print(f"✅ Training complete! Model saved to: {file}.zip")
