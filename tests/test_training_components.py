import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN, build_reversi
from util.util import mask_fn, EMPTY, BLACK, WHITE
from util.opponents import RandomOpponent, get_opponent
from sb3_contrib.common.wrappers import ActionMasker


class TestTrainingComponents(unittest.TestCase):
    """Test components specifically related to ML training."""
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        self.masked_env = ActionMasker(build_reversi(opponent="Random"), mask_fn)
    
    def test_action_masking_correctness(self):
        """Test that action masking perfectly matches valid actions."""
        obs, _ = self.masked_env.reset()
        
        for _ in range(20):  # Test multiple states
            # Get valid actions from environment
            env = self.masked_env.envs[0]
            true_valid_actions = set(env.all_valid_actions(obs))
            
            # Get action mask
            action_mask = mask_fn(self.masked_env)
            masked_valid_actions = set(np.where(action_mask)[0])
            
            # They should be identical
            self.assertEqual(true_valid_actions, masked_valid_actions,
                           f"Action mask mismatch at step. True: {true_valid_actions}, Masked: {masked_valid_actions}")
            
            # Make a random valid move
            if len(true_valid_actions) > 0:
                action = np.random.choice(list(true_valid_actions))
                obs, reward, done, truncated, info = self.masked_env.step(action)
                if done:
                    break
            else:
                break
    
    def test_reward_signal_quality(self):
        """Test that reward signals are meaningful for training."""
        obs, _ = self.env.reset()
        
        game_rewards = []
        move_count = 0
        
        # Play a complete game and collect rewards
        while move_count < 100:  # Safety limit
            valid_actions = self.env.all_valid_actions(obs)
            
            if len(valid_actions) == 0:
                if self.env.is_game_over(obs):
                    break
                else:
                    # Pass situation
                    obs, reward, done, truncated, info = self.env.step(64)  # Pass action
                    game_rewards.append(reward)
                    if done:
                        break
            else:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.env.step(action)
                game_rewards.append(reward)
                if done:
                    break
            
            move_count += 1
        
        # Analyze reward structure
        non_zero_rewards = [r for r in game_rewards if r != 0]
        
        if len(non_zero_rewards) > 0:
            # Final reward should be meaningful
            final_reward = game_rewards[-1]
            self.assertIn(final_reward, [-1, 0, 1], 
                         "Final reward should be -1 (loss), 0 (draw), or 1 (win)")
        
        # Most intermediate rewards should be 0 (sparse rewards)
        intermediate_rewards = game_rewards[:-1] if len(game_rewards) > 1 else []
        zero_count = sum(1 for r in intermediate_rewards if r == 0)
        
        if len(intermediate_rewards) > 0:
            zero_ratio = zero_count / len(intermediate_rewards)
            self.assertGreater(zero_ratio, 0.8, 
                              "Most intermediate rewards should be zero (sparse)")
    
    def test_observation_consistency(self):
        """Test that observations are consistent and properly formatted."""
        obs, _ = self.env.reset()
        
        # Test observation properties that matter for CNN training
        self.assertEqual(obs.dtype, np.int8, "Observations should be int8 for memory efficiency")
        self.assertEqual(obs.shape, (1, 8, 8), "Observations should be (1, 8, 8) for CNN")
        
        # Test value range
        unique_values = np.unique(obs)
        valid_values = {EMPTY, BLACK, WHITE}
        for val in unique_values:
            self.assertIn(val, valid_values, f"Invalid observation value: {val}")
    
    def test_episode_termination_correctness(self):
        """Test that episodes terminate correctly."""
        obs, _ = self.env.reset()
        
        step_count = 0
        max_steps = 200  # Upper bound for Reversi game
        
        while step_count < max_steps:
            valid_actions = self.env.all_valid_actions(obs)
            
            if len(valid_actions) == 0:
                # Should either be game over or pass situation
                if self.env.is_game_over(obs):
                    break
                # Handle pass (this might vary based on implementation)
            else:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.env.step(action)
                
                if done:
                    # Episode ended - verify it's a valid end state
                    self.assertTrue(self.env.is_game_over(obs) or 
                                   len(valid_actions) == 0,
                                   "Episode should only end at valid terminal states")
                    break
            
            step_count += 1
        
        self.assertLess(step_count, max_steps, "Episode should terminate within reasonable time")
    
    def test_opponent_switching(self):
        """Test that opponent switching works correctly during training."""
        # Test Random opponent
        env_random = ReversiEnvCNN(opponent="Random", verbose=False)
        obs, _ = env_random.reset()
        
        self.assertIsNotNone(env_random.opponent, "Random opponent should be set")
        
        # Test opponent switching
        env_random.set_opponent("Random", None)
        self.assertIsNotNone(env_random.opponent, "Opponent should remain set after switching")
    
    def test_state_action_space_alignment(self):
        """Test that state and action spaces are properly aligned."""
        obs, _ = self.env.reset()
        
        # Action space should match board size
        expected_action_space_size = 8 * 8  # 64 for 8x8 board
        self.assertEqual(self.env.action_space.n, expected_action_space_size)
        
        # Test action to coordinate conversion
        for action in range(64):
            x, y = action // 8, action % 8
            self.assertLess(x, 8, f"X coordinate {x} out of bounds for action {action}")
            self.assertLess(y, 8, f"Y coordinate {y} out of bounds for action {action}")
            
            # Reverse conversion
            reconstructed_action = x * 8 + y
            self.assertEqual(action, reconstructed_action, 
                           "Action conversion should be reversible")
    
    def test_deterministic_reset(self):
        """Test that environment resets are deterministic for reproducible training."""
        # Multiple resets should produce identical initial states
        initial_states = []
        for _ in range(5):
            obs, _ = self.env.reset()
            initial_states.append(obs.copy())
        
        # All initial states should be identical
        reference_state = initial_states[0]
        for i, state in enumerate(initial_states[1:], 1):
            np.testing.assert_array_equal(reference_state, state, 
                                        f"Reset {i} produced different initial state")
    
    def test_training_stability_indicators(self):
        """Test for potential training instabilities."""
        obs, _ = self.env.reset()
        
        # Track various metrics that could indicate training issues
        action_distribution = np.zeros(64)
        reward_variance = []
        game_lengths = []
        
        num_games = 10
        
        for game in range(num_games):
            obs, _ = self.env.reset()
            game_rewards = []
            game_length = 0
            
            while game_length < 100:  # Game length limit
                valid_actions = self.env.all_valid_actions(obs)
                
                if len(valid_actions) == 0:
                    if self.env.is_game_over(obs):
                        break
                else:
                    action = np.random.choice(valid_actions)
                    action_distribution[action] += 1
                    
                    obs, reward, done, truncated, info = self.env.step(action)
                    game_rewards.append(reward)
                    
                    if done:
                        break
                
                game_length += 1
            
            game_lengths.append(game_length)
            if len(game_rewards) > 1:
                reward_variance.append(np.var(game_rewards))
        
        # Check for reasonable game length distribution
        avg_game_length = np.mean(game_lengths)
        self.assertGreater(avg_game_length, 10, "Games should last more than 10 moves on average")
        self.assertLess(avg_game_length, 80, "Games shouldn't be excessively long")
        
        # Check that actions are reasonably distributed (not all concentrated)
        # This is a heuristic - in real games some actions are more common
        non_zero_actions = np.sum(action_distribution > 0)
        self.assertGreater(non_zero_actions, 10, 
                          "Should use a reasonable variety of actions across games")


class TestEnvironmentIntegration(unittest.TestCase):
    """Test integration between environment components."""
    
    def test_masked_env_integration(self):
        """Test integration between base env and action masking."""
        masked_env = ActionMasker(build_reversi(opponent="Random"), mask_fn)
        obs, _ = masked_env.reset()
        
        # Test that masked environment behaves correctly
        for _ in range(10):
            action_mask = mask_fn(masked_env)
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = masked_env.step(action)
                
                if done:
                    break
    
    def test_build_reversi_function(self):
        """Test the build_reversi factory function."""
        # Test different opponent types
        for opponent_type in ["Random"]:
            env = build_reversi(opponent=opponent_type)
            self.assertIsNotNone(env, f"Failed to build environment with {opponent_type} opponent")
            
            # Test basic functionality
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (1, 8, 8), "Built environment should have correct shape")


if __name__ == '__main__':
    unittest.main(verbosity=2)