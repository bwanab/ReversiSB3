import gymnasium as gym
from util.reversi import build_reversi, ReversiEnvCNN
from util.util import mask_fn, get_model
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

def learn_helper(PlayCallback, args, file, checkpoint_cb, env, net_width, model, timesteps):
    """Helper function to run training for a specified number of timesteps."""
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

def train_sequential(model, env, args, file, checkpoint_cb, net_width, random_timesteps, selfplay_timesteps, random_ratio=0.2):
    """Mode 3: Train random-only, then self-play mixed."""
    print(f"📈 Training Mode: Sequential")
    print(f"   Phase 1: Random-only for {random_timesteps:,} timesteps")
    print(f"   Phase 2: Self-play mixed for {selfplay_timesteps:,} timesteps")
    
    # Phase 1: Random-only
    print(f"\n🎯 Phase 1: Random-only training")
    train_random_only(model, env, args, file, checkpoint_cb, net_width, random_timesteps)
    
    # Phase 2: Self-play mixed
    print(f"\n🤖 Phase 2: Self-play mixed training")
    refresh_interval = selfplay_timesteps // 5  # Default refresh interval
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
  
  # Quick defaults
  python sb-train.py --mode selfplay --timesteps 100000  # Uses 20% random, refreshes every 20k steps
''',
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Training mode selection
    parser.add_argument("--mode", choices=["random", "selfplay", "sequential"], default="random",
                       help="Training mode: random, selfplay, or sequential")
    
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
    
    # Model and environment configuration
    parser.add_argument("-m", "--model", default="dorkL", help="Model name")
    parser.add_argument("-t", "--test-opponent", default="Random", help="Test opponent type")
    parser.add_argument("-w", "--net-width", type=int, default=512, help="Neural network width")
    
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

    model = get_model(file, env, net_width=args.net_width, model_type="cnn", device=device)
    
    print(f"🚀 Starting training with model: {args.model}")
    print(f"💻 Using device: {device}")
    
    # Execute training based on mode
    if args.mode == "random":
        train_random_only(model, env, args, file, checkpoint_cb, args.net_width, args.timesteps)
    
    elif args.mode == "selfplay":
        refresh_interval = args.refresh_interval or (args.timesteps // 5)
        train_selfplay_mixed(model, env, args, file, checkpoint_cb, args.net_width, 
                           args.timesteps, args.random_ratio, refresh_interval)
    
    elif args.mode == "sequential":
        train_sequential(model, env, args, file, checkpoint_cb, args.net_width,
                        args.random_timesteps, args.selfplay_timesteps, args.random_ratio)

    # Save the final model
    model.save(file)
    print(f"✅ Training complete! Model saved to: {file}.zip")
