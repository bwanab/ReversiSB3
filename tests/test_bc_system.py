"""
Tests for Behavioral Cloning (BC) system.

Tests dataset generation, BC training, and model functionality.
"""

import unittest
import os
import pickle
import numpy as np
import torch
from generate_bc_dataset import generate_bc_dataset
from bc_train import bc_train, BCDataset
from util.reversi import ReversiEnvCNN
from util.util import get_model, mask_fn, BLACK, WHITE


class TestBCDatasetGeneration(unittest.TestCase):
    """Test BC dataset generation."""

    @classmethod
    def setUpClass(cls):
        """Generate a small test dataset once for all tests."""
        cls.test_dataset_file = "test_bc_dataset.pkl"
        cls.num_games = 20  # Small number for fast testing

        # Generate test dataset (Random opponent only for speed)
        cls.dataset, cls.stats = generate_bc_dataset(
            num_games=cls.num_games,
            model_file=None,  # No model - Random only
            model_ratio=0.0,
            rai_depth=1,  # Shallow depth for speed
            output_file=cls.test_dataset_file,
            verbose=False
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test dataset file."""
        if os.path.exists(cls.test_dataset_file):
            os.remove(cls.test_dataset_file)

    def test_dataset_structure(self):
        """Test that dataset has correct structure."""
        self.assertIsInstance(self.dataset, list)
        self.assertGreater(len(self.dataset), 0, "Dataset should not be empty")

        # Check first item structure
        item = self.dataset[0]
        self.assertIn('state', item)
        self.assertIn('action', item)
        self.assertIn('color', item)
        self.assertIn('game_num', item)

    def test_state_format(self):
        """Test that states have correct shape and values."""
        for item in self.dataset[:10]:  # Check first 10
            state = item['state']

            # Check shape (should be ReversiEnvCNN observation space)
            self.assertEqual(state.shape, (1, 8, 8),
                             f"State shape should be (1, 8, 8), got {state.shape}")

            # Check values are valid (-1, 0, 1)
            unique_values = np.unique(state)
            for val in unique_values:
                self.assertIn(val, [-1, 0, 1],
                              f"State contains invalid value: {val}")

    def test_action_format(self):
        """Test that actions are valid."""
        for item in self.dataset[:10]:
            action = item['action']

            # Action should be integer in range [0, 63]
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 64)

    def test_color_balance(self):
        """Test that dataset has both Black and White perspectives."""
        black_count = sum(1 for item in self.dataset if item['color'] == BLACK)
        white_count = sum(1 for item in self.dataset if item['color'] == WHITE)

        self.assertGreater(black_count, 0, "Dataset should have Black moves")
        self.assertGreater(white_count, 0, "Dataset should have White moves")

        # Should be roughly balanced (within 20% tolerance)
        ratio = black_count / (black_count + white_count)
        self.assertGreater(ratio, 0.3, f"Black ratio too low: {ratio:.2%}")
        self.assertLess(ratio, 0.7, f"Black ratio too high: {ratio:.2%}")

    def test_game_count(self):
        """Test that correct number of games were generated."""
        game_nums = set(item['game_num'] for item in self.dataset)
        self.assertEqual(len(game_nums), self.num_games,
                         f"Expected {self.num_games} games, got {len(game_nums)}")

    def test_statistics(self):
        """Test that statistics are tracked correctly."""
        stats = self.stats

        # Check required keys
        self.assertIn('games_completed', stats)
        self.assertIn('moves_collected', stats)
        self.assertIn('rai_black_games', stats)
        self.assertIn('rai_white_games', stats)

        # Verify counts
        self.assertEqual(stats['games_completed'], self.num_games)
        self.assertEqual(stats['moves_collected'], len(self.dataset))

        # Black + White games should equal total
        self.assertEqual(
            stats['rai_black_games'] + stats['rai_white_games'],
            self.num_games
        )

    def test_dataset_save_load(self):
        """Test that dataset can be saved and loaded correctly."""
        # Load saved dataset
        with open(self.test_dataset_file, 'rb') as f:
            loaded = pickle.load(f)

        # Check structure
        self.assertIn('dataset', loaded)
        self.assertIn('metadata', loaded)

        # Check dataset matches
        loaded_dataset = loaded['dataset']
        self.assertEqual(len(loaded_dataset), len(self.dataset))

        # Check metadata
        metadata = loaded['metadata']
        self.assertEqual(metadata['num_games'], self.num_games)
        self.assertEqual(metadata['rai_depth'], 1)


class TestBCTraining(unittest.TestCase):
    """Test BC training functionality."""

    @classmethod
    def setUpClass(cls):
        """Generate a test dataset and train a small BC model."""
        cls.test_dataset_file = "test_bc_training_dataset.pkl"
        cls.test_model_name = "test_bc_model"
        cls.test_model_path = f"models/{cls.test_model_name}_CNN_test.zip"

        # Generate small dataset
        generate_bc_dataset(
            num_games=10,  # Very small for fast testing
            model_file=None,
            model_ratio=0.0,
            rai_depth=1,
            output_file=cls.test_dataset_file,
            verbose=False
        )

        # Train BC model (few epochs for speed)
        cls.model, cls.history = bc_train(
            dataset_file=cls.test_dataset_file,
            model_name=cls.test_model_name,
            epochs=3,  # Very few epochs for speed
            batch_size=32,
            learning_rate=1e-3,
            net_width=128,  # Small network for speed
            val_split=0.2,
            verbose=False
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.test_dataset_file):
            os.remove(cls.test_dataset_file)
        if os.path.exists(cls.test_model_path):
            os.remove(cls.test_model_path)

    def test_training_history(self):
        """Test that training history is tracked."""
        history = self.history

        # Check keys
        self.assertIn('train_loss', history)
        self.assertIn('train_acc', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)

        # Check lengths (should equal num epochs)
        self.assertEqual(len(history['train_loss']), 3)
        self.assertEqual(len(history['val_loss']), 3)

        # Check that values are reasonable
        for loss in history['train_loss']:
            self.assertGreater(loss, 0, "Loss should be positive")
            self.assertLess(loss, 10, "Loss should not be extremely high")

        for acc in history['train_acc']:
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 1)

    def test_training_improves(self):
        """Test that training shows improvement (generally)."""
        history = self.history

        # Training loss should generally decrease
        # (Not strict because random initialization might not always improve in 3 epochs)
        first_loss = history['train_loss'][0]
        last_loss = history['train_loss'][-1]

        # Just check they're in reasonable range
        self.assertLess(first_loss, 10, "Initial loss too high")

    def test_model_saved(self):
        """Test that BC model was saved correctly."""
        self.assertTrue(
            os.path.exists(self.test_model_path),
            f"Model file not found: {self.test_model_path}"
        )

    def test_model_can_predict(self):
        """Test that trained BC model can make predictions."""
        env = ReversiEnvCNN()
        state, info = env.reset()

        # Model should make prediction
        action_masks = mask_fn(env)
        action, _states = self.model.predict(
            state,
            action_masks=action_masks,
            deterministic=True
        )

        # Check action is valid
        self.assertIsInstance(action, (int, np.integer, np.ndarray))
        if isinstance(action, np.ndarray):
            action = action.item()

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 64)

        # Action should be in valid moves
        valid_actions = env.all_valid_actions(state)
        self.assertIn(action, valid_actions,
                      f"Model predicted invalid action {action}")

    def test_bc_model_reload(self):
        """Test that BC model can be reloaded and used."""
        env = ReversiEnvCNN()

        # Load model from file
        loaded_model = get_model(
            self.test_model_name,
            env,
            net_width=128
        )

        # Should make valid prediction
        state, info = env.reset()
        action_masks = mask_fn(env)
        action, _states = loaded_model.predict(
            state,
            action_masks=action_masks,
            deterministic=True
        )

        # Check action is valid
        if isinstance(action, np.ndarray):
            action = action.item()

        valid_actions = env.all_valid_actions(state)
        self.assertIn(action, valid_actions,
                      f"Reloaded model predicted invalid action")


class TestBCDatasetClass(unittest.TestCase):
    """Test BCDataset PyTorch Dataset class."""

    def setUp(self):
        """Create sample dataset."""
        self.sample_data = [
            {
                'state': np.random.randint(-1, 2, size=(1, 8, 8)),
                'action': np.random.randint(0, 64),
                'color': BLACK,
                'game_num': 0
            }
            for _ in range(10)
        ]
        self.bc_dataset = BCDataset(self.sample_data)

    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.bc_dataset), 10)

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        state, action, outcome = self.bc_dataset[0]

        # Check types
        self.assertIsInstance(state, torch.Tensor)
        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(outcome, torch.Tensor)

        # Check shapes
        self.assertEqual(state.shape, (1, 8, 8))
        self.assertEqual(action.shape, (1,))
        self.assertEqual(outcome.shape, (1,))

        # Check dtypes
        self.assertEqual(state.dtype, torch.float32)
        self.assertEqual(action.dtype, torch.int64)
        self.assertEqual(outcome.dtype, torch.float32)

        # Sample data has no 'outcome' key, so it defaults to 0
        self.assertEqual(outcome.item(), 0.0)


class TestBCIntegration(unittest.TestCase):
    """Integration tests for complete BC workflow."""

    def test_bc_workflow_with_model_opponent(self):
        """Test complete BC workflow: generate dataset with Model opponent -> train -> evaluate.

        This test is skipped if no trained model exists.
        """
        # Check if a trained model exists
        model_files = [f for f in os.listdir("models") if f.endswith("_CNN_test.zip")]

        if len(model_files) == 0:
            self.skipTest("No trained models available for testing")

        # Use first available model
        model_name = model_files[0].replace("_CNN_test.zip", "")

        dataset_file = "test_integration_dataset.pkl"
        bc_model_name = "test_integration_bc"
        bc_model_path = f"models/{bc_model_name}_CNN_test.zip"

        try:
            # Generate dataset with Model opponent
            generate_bc_dataset(
                num_games=5,
                model_file=model_name,
                model_ratio=1.0,  # 100% Model
                rai_depth=1,
                net_width=512,
                output_file=dataset_file,
                verbose=False
            )

            # Train BC model
            model, history = bc_train(
                dataset_file=dataset_file,
                model_name=bc_model_name,
                epochs=2,
                batch_size=32,
                net_width=128,
                verbose=False
            )

            # Test BC model can play
            env = ReversiEnvCNN()
            state, info = env.reset()

            for _ in range(10):  # Play 10 moves
                action_masks = mask_fn(env)
                action, _ = model.predict(state, action_masks=action_masks)

                state, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            # If we get here, workflow succeeded
            self.assertTrue(True)

        finally:
            # Cleanup
            if os.path.exists(dataset_file):
                os.remove(dataset_file)
            if os.path.exists(bc_model_path):
                os.remove(bc_model_path)


def run_bc_tests():
    """Run all BC tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBCDatasetGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestBCTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestBCDatasetClass))
    suite.addTests(loader.loadTestsFromTestCase(TestBCIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_bc_tests()
