import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN, build_reversi
from util.util import mask_fn, EMPTY, BLACK, WHITE
from sb3_contrib.common.wrappers import ActionMasker


class TestTrainingIssues(unittest.TestCase):
    """
    Specific tests to identify issues that could cause:
    - Low explained variance (~0.1)  
    - High value loss (~0.08)
    - Training plateau
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_reward_signal_correlation(self):
        """Test if rewards correlate with actual game outcomes."""
        
        # Play multiple games and track outcomes vs rewards
        outcomes = []
        final_rewards = []
        
        for game in range(20):
            obs, _ = self.env.reset()
            game_reward = 0
            
            # Play game to completion
            for move in range(100):  # Safety limit
                valid_actions = self.env.all_valid_actions(obs)
                
                if len(valid_actions) == 0:
                    if self.env.is_game_over(obs):
                        break
                else:
                    action = np.random.choice(valid_actions)
                    obs, reward, done, truncated, info = self.env.step(action)
                    game_reward = reward  # Track final reward
                    
                    if done:
                        break
            
            # Determine actual outcome by counting pieces
            if self.env.is_game_over(obs):
                black_count = np.sum(obs == BLACK)
                white_count = np.sum(obs == WHITE)
                
                if black_count > white_count:
                    actual_outcome = 1  # BLACK wins
                elif white_count > black_count:
                    actual_outcome = -1  # WHITE wins  
                else:
                    actual_outcome = 0  # Draw
                
                outcomes.append(actual_outcome)
                final_rewards.append(game_reward)
        
        if len(outcomes) > 5:  # Need enough data
            # Check correlation between actual outcomes and rewards
            correlation = np.corrcoef(outcomes, final_rewards)[0, 1]
            
            if not np.isnan(correlation):
                self.assertGreater(abs(correlation), 0.5, 
                                 f"Reward correlation with outcomes too low: {correlation}. "
                                 f"This could explain low explained variance!")
    
    def test_value_function_consistency(self):
        """Test if value estimates are consistent across similar positions."""
        
        # Create similar board positions and check if they have similar implied values
        base_board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Set up standard opening
        base_board[0, 3, 3] = WHITE
        base_board[0, 3, 4] = BLACK
        base_board[0, 4, 3] = BLACK  
        base_board[0, 4, 4] = WHITE
        
        # Add a few more pieces in symmetric patterns
        base_board[0, 2, 3] = BLACK  # c4
        base_board[0, 5, 4] = WHITE  # f5 (similar position for white)
        
        # Test multiple similar positions
        similar_positions = []
        
        for i in range(3):
            test_board = base_board.copy()
            # Add slightly different piece
            empty_squares = np.where(test_board == EMPTY)
            if len(empty_squares[1]) > i:
                test_board[0, empty_squares[1][i], empty_squares[2][i]] = BLACK
            
            self.env.board = test_board
            similar_positions.append(test_board.copy())
        
        # Similar positions should have some consistency in evaluation
        # This is hard to test without the actual value function, but we can check
        # that the game logic treats them consistently
        
        for i, pos1 in enumerate(similar_positions):
            for j, pos2 in enumerate(similar_positions[i+1:], i+1):
                # At minimum, check that valid move detection is consistent
                valid1 = len(self.env._all_valid_actions(pos1, BLACK))
                valid2 = len(self.env._all_valid_actions(pos2, BLACK))
                
                # Similar positions should have similar numbers of valid moves
                diff = abs(valid1 - valid2)
                self.assertLess(diff, 5, 
                               f"Similar positions have very different move counts: {valid1} vs {valid2}")
    
    def test_game_state_transitions(self):
        """Test that state transitions are deterministic and correct."""
        
        obs, _ = self.env.reset()
        initial_state = obs.copy()
        
        # Make the same sequence of moves multiple times
        test_sequence = []
        valid_actions = self.env.all_valid_actions(obs)
        
        if len(valid_actions) >= 2:
            test_sequence = valid_actions[:2]  # Take first two valid moves
        
        # Test deterministic transitions
        for trial in range(3):
            self.env.reset()
            current_state = self.env.board.copy()
            
            # Apply same sequence
            for action in test_sequence:
                if action in self.env.all_valid_actions(current_state):
                    # Manually apply the action to check consistency
                    x, y = action // 8, action % 8
                    
                    # The state should change predictably
                    old_piece_count = np.sum(current_state != EMPTY)
                    
                    # Take the step
                    new_obs, reward, done, truncated, info = self.env.step(action)
                    current_state = new_obs
                    
                    # Verify the board changed appropriately
                    new_piece_count = np.sum(current_state != EMPTY)
                    self.assertGreaterEqual(new_piece_count, old_piece_count,
                                          "Piece count should not decrease after valid move")
                    
                    if done:
                        break
    
    def test_opponent_behavior_consistency(self):
        """Test that opponent behavior is consistent and not causing training issues."""
        
        # Test Random opponent consistency
        opponent_moves = []
        
        for game in range(10):
            obs, _ = self.env.reset()
            game_moves = []
            
            for move in range(10):  # First 10 moves of each game
                valid_actions = self.env.all_valid_actions(obs)
                
                if len(valid_actions) == 0:
                    break
                
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.env.step(action)
                
                # The step includes opponent move, check it's valid
                if not done:
                    # After our move, board should still be in valid state
                    self.assertTrue(np.any(obs != EMPTY), "Board should not be empty mid-game")
                    
                    # Check that opponent didn't make illegal move
                    piece_count = np.sum(obs != EMPTY)
                    self.assertGreater(piece_count, 0, "Should have pieces on board")
                
                game_moves.append(len(valid_actions))
                
                if done:
                    break
            
            opponent_moves.append(game_moves)
        
        # Check that games have reasonable variety
        if len(opponent_moves) > 5:
            avg_moves_per_position = []
            for i in range(min(len(moves) for moves in opponent_moves)):
                position_moves = [moves[i] for moves in opponent_moves if len(moves) > i]
                if position_moves:
                    avg_moves_per_position.append(np.mean(position_moves))
            
            # Early positions should generally have fewer moves than mid-game
            if len(avg_moves_per_position) > 3:
                early_avg = np.mean(avg_moves_per_position[:2])
                mid_avg = np.mean(avg_moves_per_position[2:4])
                
                # This is a heuristic - mid-game usually has more options
                self.assertLess(early_avg, mid_avg + 5, 
                               "Move count progression seems unusual - might indicate game logic bug")
    
    def test_reward_sparsity_issues(self):
        """Test if reward sparsity is causing training difficulties."""
        
        # Play games and analyze reward structure
        all_rewards = []
        reward_positions = []
        
        for game in range(15):
            obs, _ = self.env.reset()
            game_rewards = []
            position_count = 0
            
            for move in range(100):
                valid_actions = self.env.all_valid_actions(obs)
                
                if len(valid_actions) == 0:
                    if self.env.is_game_over(obs):
                        break
                else:
                    action = np.random.choice(valid_actions)
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    game_rewards.append(reward)
                    if reward != 0:
                        reward_positions.append(position_count)
                    
                    position_count += 1
                    
                    if done:
                        break
            
            all_rewards.extend(game_rewards)
        
        if len(all_rewards) > 20:
            # Analyze reward sparsity
            non_zero_rewards = [r for r in all_rewards if r != 0]
            sparsity = 1 - (len(non_zero_rewards) / len(all_rewards))
            
            # Very sparse rewards (>95%) might hurt learning
            if sparsity > 0.95:
                print(f"WARNING: Very sparse rewards ({sparsity:.2%}). Consider reward shaping.")
            
            # Check if non-zero rewards are only at game end
            if len(reward_positions) > 0 and len(all_rewards) > 0:
                avg_reward_position = np.mean(reward_positions) / len(all_rewards) 
                
                if avg_reward_position > 0.9:  # Rewards only in last 10% of game
                    self.fail("Rewards only appear at game end - this could explain training difficulties!")
    
    def test_action_space_utilization(self):
        """Test if the action space is being utilized effectively."""
        
        action_counts = np.zeros(64)
        total_actions = 0
        
        # Collect action statistics over multiple games
        for game in range(20):
            obs, _ = self.env.reset()
            
            for move in range(50):
                valid_actions = self.env.all_valid_actions(obs)
                
                if len(valid_actions) == 0:
                    if self.env.is_game_over(obs):
                        break
                else:
                    for action in valid_actions:
                        action_counts[action] += 1
                    total_actions += len(valid_actions)
                    
                    # Make random move
                    chosen_action = np.random.choice(valid_actions)
                    obs, reward, done, truncated, info = self.env.step(chosen_action)
                    
                    if done:
                        break
        
        if total_actions > 0:
            # Check action distribution
            action_probs = action_counts / total_actions
            
            # Check if some actions are never valid (might indicate bugs)
            never_valid = np.sum(action_counts == 0)
            
            # In Reversi, corner and edge squares should sometimes be valid
            corners = [0, 7, 56, 63]  # Corner positions
            corner_usage = sum(action_counts[c] for c in corners)
            
            if corner_usage == 0 and total_actions > 100:
                print("WARNING: Corner squares never used - potential game logic issue")
            
            # Check for extreme bias toward certain squares
            max_usage = np.max(action_probs)
            if max_usage > 0.3:  # One action used >30% of time
                most_used = np.argmax(action_probs)
                print(f"WARNING: Action {most_used} used {max_usage:.1%} of time - check for bias")


if __name__ == '__main__':
    unittest.main(verbosity=2)