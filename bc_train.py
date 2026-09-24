#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Training Script.

Trains a fresh MaskablePPO model using supervised learning on a dataset
of RAI games. The model learns to imitate RAI's move selection through
classification (predicting RAI's action given board state).

This creates a BC-pretrained model that can then be fine-tuned with RL.

Usage:
    python bc_train.py --dataset bc_dataset.pkl --model bc_pretrained --epochs 20
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from util.reversi import ReversiEnvCNN
from util.util import get_model, get_device, mask_fn
from util.opponents import get_opponent
from sb3_contrib import MaskablePPO


class BCDataset(Dataset):
    """PyTorch Dataset for BC training."""

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : list
            List of dicts with keys: 'state', 'action', 'color', 'outcome'
        """
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        state = torch.FloatTensor(item['state'])
        action = torch.LongTensor([item['action']])
        outcome = torch.FloatTensor([item.get('outcome', 0.0)])  # Default to 0 if missing
        return state, action, outcome


def evaluate_model(model, env, opponent_type, num_games=100, rai_depth=1, verbose=False):
    """
    Evaluate model performance against an opponent.

    Parameters
    ----------
    model : MaskablePPO
        Model to evaluate
    env : ReversiEnvCNN
        Game environment
    opponent_type : str
        Type of opponent ('Random', 'RAI', etc.)
    num_games : int
        Number of games to play
    rai_depth : int
        RAI search depth (only used if opponent_type='RAI')
    verbose : bool
        Print game results

    Returns
    -------
    float
        Win rate (0.0 to 1.0)
    """
    # Get opponent with appropriate kwargs
    if opponent_type == 'RAI':
        opponent = get_opponent(opponent_type, depth=rai_depth)
    else:
        opponent = get_opponent(opponent_type)
    wins = 0

    for _ in range(num_games):
        state, _ = env.reset()
        done = False

        while not done:
            # Model plays as Black (player 1)
            if env.player == 1:
                # Model's turn
                action_masks = mask_fn(env)
                action, _ = model.predict(state, action_masks=action_masks, deterministic=False)
            else:
                # Opponent's turn
                action = opponent(env, state)

            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

        # Check if model (Black) won
        if reward > 0:
            wins += 1

    win_rate = wins / num_games
    return win_rate


def bc_train(
    dataset_file,
    model_name,
    epochs=20,
    batch_size=256,
    learning_rate=1e-4,
    net_width=512,
    val_split=0.1,
    value_coef=0.5,
    verbose=True,
    device="auto"
):
    """
    Train model using behavioral cloning on RAI dataset.

    Parameters
    ----------
    dataset_file : str
        Path to BC dataset pickle file
    model_name : str
        Name for saved BC model (without .zip extension)
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate for supervised learning
    net_width : int
        Neural network width
    val_split : float
        Validation set split (0.1 = 10% validation)
    verbose : bool
        Print training details

    Returns
    -------
    model : MaskablePPO
        BC-trained model
    history : dict
        Training history (loss, accuracy per epoch)
    """

    print("="*60)
    print("Behavioral Cloning Training")
    print("="*60)
    print(f"Dataset: {dataset_file}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.2e}")
    print(f"Validation split: {val_split:.1%}")
    print("="*60)
    print()

    # Load dataset
    print("Loading dataset...")
    with open(dataset_file, 'rb') as f:
        dataset_package = pickle.load(f)

    # Handle both WThor format ('moves' key) and RAI format ('dataset' key)
    if 'moves' in dataset_package:
        # WThor format
        dataset = dataset_package['moves']
        metadata = dataset_package['metadata']
        print(f"Dataset loaded: {len(dataset):,} examples")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Games: {metadata.get('total_games', '?'):,}")
    elif 'dataset' in dataset_package:
        # BC format (RAI, Edax, or combined)
        dataset = dataset_package['dataset']
        metadata = dataset_package['metadata']
        print(f"Dataset loaded: {len(dataset):,} examples")

        # Print metadata based on what's available
        if 'rai_depth' in metadata:
            # RAI BC dataset
            print(f"Metadata: {metadata['num_games']:,} games, "
                  f"RAI depth={metadata['rai_depth']}")
        elif 'edax_depths' in metadata:
            # Edax BC dataset (possibly combined with wthor)
            print(f"Metadata: {metadata.get('num_games', 'N/A')} new games")
            print(f"  Edax depths: {metadata['edax_depths']}")
            if metadata.get('merged_from'):
                print(f"  Merged from: {metadata['merged_source']}")
                print(f"  Total moves: {metadata['total_moves']:,}")
        else:
            # Generic metadata
            print(f"Metadata: {metadata.get('num_games', 'Unknown')} games")
    else:
        raise ValueError("Unknown dataset format: expected 'moves' or 'dataset' key")
    print()

    # Split train/val
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val

    # Shuffle dataset
    np.random.shuffle(dataset)

    train_dataset = BCDataset(dataset[:num_train])
    val_dataset = BCDataset(dataset[num_train:])

    print(f"Train examples: {num_train:,}")
    print(f"Val examples: {num_val:,}")
    print()

    # Create dataloaders
    # Note: num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create fresh model
    print("Creating fresh MaskablePPO model...")
    env = ReversiEnvCNN()
    if device == "auto":
        device = get_device()

    # Construct full model path (same format as sb-train.py)
    model_file = f"models/{model_name}_CNN_test"

    # Pass model_file to get_model() for TensorBoard logs
    # If model exists, it will be loaded; otherwise a new one is created
    model = get_model(
        model_file,  # Full model path for loading/TensorBoard
        env,
        net_width=net_width,
        learning_rate=learning_rate,  # Use BC learning rate
        device=device
    )

    print(f"Model created on device: {device}")
    print(f"Policy architecture: {model.policy}")
    print()

    # Get policy network and optimizer
    policy = model.policy
    policy.train()

    # Create optimizer for policy parameters only
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_value_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_value_loss': [],
        'random_wins': [],
        'rai1_wins': []
    }

    # Training loop
    print("Starting BC training...\n")

    for epoch in range(epochs):
        # Train phase
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_value_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for states, actions, outcomes in pbar:
            states = states.to(device)
            actions = actions.squeeze(1).to(device)  # Shape: [batch_size]
            outcomes = outcomes.squeeze(1).to(device)  # Shape: [batch_size]

            # Forward pass through policy and value networks
            features = policy.extract_features(states)
            latent_pi, latent_vf = policy.mlp_extractor(features)

            # Policy head: predict actions
            action_logits = policy.action_net(latent_pi)  # Shape: [batch_size, 64]

            # Value head: predict outcomes
            value_preds = policy.value_net(latent_vf).squeeze(-1)  # Shape: [batch_size]

            # Compute losses
            policy_loss = F.cross_entropy(action_logits, actions)
            value_loss = F.mse_loss(value_preds, outcomes)

            # Combined loss
            loss = policy_loss + value_coef * value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += policy_loss.item() * states.size(0)
            train_value_loss += value_loss.item() * states.size(0)
            predictions = torch.argmax(action_logits, dim=1)
            train_correct += (predictions == actions).sum().item()
            train_total += states.size(0)

            # Update progress bar
            pbar.set_postfix({
                'p_loss': f"{policy_loss.item():.4f}",
                'v_loss': f"{value_loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%"
            })

        # Average train metrics
        train_loss /= train_total
        train_value_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        policy.eval()
        val_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states, actions, outcomes in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                states = states.to(device)
                actions = actions.squeeze(1).to(device)
                outcomes = outcomes.squeeze(1).to(device)

                # Forward pass through policy and value networks
                features = policy.extract_features(states)
                latent_pi, latent_vf = policy.mlp_extractor(features)
                action_logits = policy.action_net(latent_pi)
                value_preds = policy.value_net(latent_vf).squeeze(-1)

                # Compute losses
                policy_loss = F.cross_entropy(action_logits, actions)
                value_loss = F.mse_loss(value_preds, outcomes)

                # Track metrics
                val_loss += policy_loss.item() * states.size(0)
                val_value_loss += value_loss.item() * states.size(0)
                predictions = torch.argmax(action_logits, dim=1)
                val_correct += (predictions == actions).sum().item()
                val_total += states.size(0)

        # Average val metrics
        val_loss /= val_total
        val_value_loss /= val_total
        val_acc = val_correct / val_total

        # Evaluate against Random and RAI-1
        print(f"\nEpoch {epoch+1}/{epochs} - Evaluating...")
        random_win_rate = evaluate_model(model, env, 'Random', num_games=100)
        rai1_win_rate = evaluate_model(model, env, 'RAI', num_games=100, rai_depth=1)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_value_loss'].append(train_value_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_value_loss'].append(val_value_loss)
        history['random_wins'].append(random_win_rate)
        history['rai1_wins'].append(rai1_win_rate)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Policy Loss: {train_loss:.4f}, Train Value Loss: {train_value_loss:.4f}, Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Policy Loss:   {val_loss:.4f}, Val Value Loss:   {val_value_loss:.4f}, Val Acc:   {100*val_acc:.2f}%")
        print(f"  Random Win Rate:   {100*random_win_rate:.1f}% ({random_win_rate*100:.0f}/100)")
        print(f"  RAI-1 Win Rate:    {100*rai1_win_rate:.1f}% ({rai1_win_rate*100:.0f}/100)")
        print()

        # Save checkpoint with epoch number (allows comparing different epochs)
        epoch_checkpoint = f"models/{model_name}_epoch{epoch+1:02d}_CNN_test.zip"
        model.save(epoch_checkpoint)
        print(f"  ✓ Epoch checkpoint: {epoch_checkpoint}")

        # Also save as "latest" (overwrites, convenient for resuming)
        latest_path = f"models/{model_name}_CNN_test.zip"
        model.save(latest_path)
        print(f"  ✓ Latest saved: {latest_path}")
        print()

    # Final save (redundant but ensures it's saved)
    model_path = f"models/{model_name}_CNN_test.zip"
    model.save(model_path)

    # Save training history to CSV
    import csv
    history_path = f"{model_name}_training.csv"
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_value_loss',
                        'val_loss', 'val_acc', 'val_value_loss',
                        'random_wins', 'rai1_wins'])
        for i in range(epochs):
            writer.writerow([
                f"{i+1}/{epochs}",
                f"{history['train_loss'][i]:.4f}",
                f"{history['train_acc'][i]:.4f}",
                f"{history['train_value_loss'][i]:.4f}",
                f"{history['val_loss'][i]:.4f}",
                f"{history['val_acc'][i]:.4f}",
                f"{history['val_value_loss'][i]:.4f}",
                f"{history['random_wins'][i]:.2f}",
                f"{history['rai1_wins'][i]:.2f}"
            ])

    print("="*60)
    print("BC Training Complete!")
    print("="*60)
    print(f"Final Train Accuracy: {100*history['train_acc'][-1]:.2f}%")
    print(f"Final Val Accuracy:   {100*history['val_acc'][-1]:.2f}%")
    print(f"Final Random Win Rate: {100*history['random_wins'][-1]:.1f}%")
    print(f"Final RAI-1 Win Rate:  {100*history['rai1_wins'][-1]:.1f}%")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print("="*60)

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Train model using behavioral cloning on RAI dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='BC dataset pickle file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Output model name (without .zip extension)')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='Learning rate for supervised learning')
    parser.add_argument('-w', '--net-width', type=int, default=512,
                        help='Neural network width')
    parser.add_argument('-v', '--val-split', type=float, default=0.1,
                        help='Validation set split fraction')
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training info')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Training device (default: auto = mps/cuda if available, else cpu)')

    args = parser.parse_args()

    # Validate arguments
    if args.val_split < 0 or args.val_split >= 1:
        parser.error("--val-split must be between 0 and 1")

    # Train model
    bc_train(
        dataset_file=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        net_width=args.net_width,
        val_split=args.val_split,
        value_coef=args.value_coef,
        verbose=args.verbose,
        device=args.device
    )


if __name__ == '__main__':
    main()
