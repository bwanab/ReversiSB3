"""
Test script to verify learning rate updates work correctly and consistently.
"""
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from util.reversi import build_reversi
from util.util import mask_fn, get_model
import os

def set_learning_rate(model, lr):
    """Update the learning rate of a model's optimizer.

    Updates three places to ensure LR persists across learn() calls:
    1. The optimizer's param_groups (actual LR used in training)
    2. model.learning_rate (the attribute)
    3. model.lr_schedule (the function SB3 calls during learn())
    """
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = lr
    model.learning_rate = lr
    # CRITICAL: Also update lr_schedule to prevent reset during learn()
    if hasattr(model, 'lr_schedule'):
        model.lr_schedule = lambda _: lr

def get_actual_learning_rate(model):
    """Get the actual learning rate from the optimizer (not just the attribute)."""
    # Check all parameter groups (there might be separate ones for policy/value)
    lrs = [param_group['lr'] for param_group in model.policy.optimizer.param_groups]
    return lrs

def verify_lr_update(model, new_lr, test_name):
    """Verify that setting LR actually works."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")

    # Get initial LR
    initial_lrs = get_actual_learning_rate(model)
    print(f"Initial optimizer LRs: {initial_lrs}")
    print(f"Initial model.learning_rate attribute: {model.learning_rate}")

    # Set new LR
    print(f"\nSetting new LR to: {new_lr:.2e}")
    set_learning_rate(model, new_lr)

    # Verify immediately after setting
    updated_lrs = get_actual_learning_rate(model)
    print(f"After update - optimizer LRs: {updated_lrs}")
    print(f"After update - model.learning_rate: {model.learning_rate}")

    # Check if all optimizer LRs match the target
    all_match = all(abs(lr - new_lr) < 1e-10 for lr in updated_lrs)
    attr_matches = abs(model.learning_rate - new_lr) < 1e-10

    if all_match and attr_matches:
        print(f"✓ PASS: LR successfully updated to {new_lr:.2e}")
        return True
    else:
        print(f"✗ FAIL: LR update inconsistent!")
        print(f"  Expected: {new_lr:.2e}")
        print(f"  Optimizer LRs: {updated_lrs}")
        print(f"  Attribute: {model.learning_rate}")
        return False

def test_lr_persistence_across_learn(model, env):
    """Test that LR persists across learn() calls."""
    print(f"\n{'='*60}")
    print(f"Test: LR Persistence Across learn() Calls")
    print(f"{'='*60}")

    test_lr = 1.5e-5
    set_learning_rate(model, test_lr)
    print(f"Set LR to: {test_lr:.2e}")

    # Get LR before learn
    lr_before = get_actual_learning_rate(model)
    print(f"LR before learn(): {lr_before}")

    # Run a short training
    print(f"\nRunning learn(total_timesteps=2048, reset_num_timesteps=False)...")
    model.learn(total_timesteps=2048, reset_num_timesteps=False, progress_bar=False)

    # Get LR after learn
    lr_after = get_actual_learning_rate(model)
    print(f"LR after learn(): {lr_after}")

    # Check if it stayed the same
    lrs_match = all(abs(lr_after[i] - lr_before[i]) < 1e-10 for i in range(len(lr_after)))

    if lrs_match:
        print(f"✓ PASS: LR remained stable across learn() call")
        return True
    else:
        print(f"✗ FAIL: LR changed during learn()!")
        print(f"  Before: {lr_before}")
        print(f"  After: {lr_after}")
        return False

def test_lr_with_schedule():
    """Test if LR updates work when model was created with a schedule."""
    print(f"\n{'='*60}")
    print(f"Test: LR Update on Model Created with Schedule")
    print(f"{'='*60}")

    # This tests if using linear_schedule at creation causes issues
    from util.util import linear_schedule

    env = ActionMasker(build_reversi(opponent="Random"), mask_fn)

    # Create model with a schedule (not a constant)
    from sb3_contrib import MaskablePPO
    from util.reversi_cnn import ReversiCNN

    policy_kwargs = dict(
        features_extractor_class=ReversiCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False
    )

    model = MaskablePPO("CnnPolicy",
                        env,
                        policy_kwargs=policy_kwargs,
                        learning_rate=linear_schedule(2e-5),  # Using a schedule!
                        device="cpu",
                        verbose=0)

    print(f"Model created with linear_schedule(2e-5)")
    print(f"Initial LR type: {type(model.learning_rate)}")
    initial_lrs = get_actual_learning_rate(model)
    print(f"Initial optimizer LRs: {initial_lrs}")

    # Try to override with a constant
    new_lr = 5e-6
    set_learning_rate(model, new_lr)

    updated_lrs = get_actual_learning_rate(model)
    print(f"After setting to {new_lr:.2e}, optimizer LRs: {updated_lrs}")

    all_match = all(abs(lr - new_lr) < 1e-10 for lr in updated_lrs)

    if all_match:
        print(f"✓ PASS: Can override schedule with constant LR")
        return True
    else:
        print(f"✗ FAIL: Schedule override didn't work")
        return False

def main():
    print("="*60)
    print("Learning Rate Update Verification Tests")
    print("="*60)

    # Setup
    device = "cpu"  # Use CPU for consistent testing
    env = ActionMasker(build_reversi(opponent="Random"), mask_fn)

    results = []

    # Test 1: Create new model and test basic LR update
    print("\n[Test 1: New Model - Basic LR Update]")
    model = get_model("models/test_lr_model", env, net_width=256, learning_rate=2e-5, device=device)
    results.append(verify_lr_update(model, 1e-5, "New Model - Set to 1e-5"))

    # Test 2: Update multiple times
    results.append(verify_lr_update(model, 5e-6, "Same Model - Set to 5e-6"))
    results.append(verify_lr_update(model, 3e-5, "Same Model - Set to 3e-5"))

    # Test 3: Save and reload model, then update LR
    print("\n[Test 2: Save/Reload Cycle]")
    model.save("models/test_lr_model_temp")
    model_reloaded = MaskablePPO.load("models/test_lr_model_temp", env=env)
    results.append(verify_lr_update(model_reloaded, 1.5e-5, "Reloaded Model - Set to 1.5e-5"))

    # Test 4: LR persistence across learn() calls
    results.append(test_lr_persistence_across_learn(model_reloaded, env))

    # Test 5: Model created with schedule
    results.append(test_lr_with_schedule())

    # Cleanup
    if os.path.exists("models/test_lr_model.zip"):
        os.remove("models/test_lr_model.zip")
    if os.path.exists("models/test_lr_model_temp.zip"):
        os.remove("models/test_lr_model_temp.zip")

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    total_tests = len(results)
    passed = sum(results)
    print(f"Passed: {passed}/{total_tests}")

    if passed == total_tests:
        print("✓ All tests passed! LR update mechanism is reliable.")
        return 0
    else:
        print(f"✗ {total_tests - passed} test(s) failed. LR update may be unreliable.")
        return 1

if __name__ == "__main__":
    exit(main())
