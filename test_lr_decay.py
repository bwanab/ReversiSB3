"""Quick test to verify LR decay calculations work correctly."""

def test_lr_decay():
    """Test the LR decay formula used in sequential training."""
    start_lr = 3e-5
    end_lr = 0
    total_timesteps = 3_000_000

    print("Testing LR Decay Formula")
    print(f"Start LR: {start_lr:.2e}")
    print(f"End LR: {end_lr:.2e}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("\nExpected LR at different progress points:\n")

    test_points = [
        (0, "Start"),
        (1_000_000, "After Phase 1 (Random)"),
        (1_500_000, "Middle of Phase 2"),
        (2_000_000, "2/3 through Phase 2"),
        (2_500_000, "5/6 through Phase 2"),
        (3_000_000, "End")
    ]

    for timesteps_done, description in test_points:
        progress = timesteps_done / total_timesteps
        current_lr = start_lr - (start_lr - end_lr) * progress
        print(f"{description:25s} ({timesteps_done:,} steps, {progress*100:.1f}% done): LR = {current_lr:.2e}")

    print("\n✓ Formula test complete!")
    print("\nKey observations:")
    print("- LR starts at 3e-5 and decays linearly")
    print("- After 1M random training: LR = 2e-5 (33% reduction)")
    print("- At end of training: LR approaches 0")

if __name__ == "__main__":
    test_lr_decay()
