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
        """Test that the final reward always matches the actual game outcome."""
        for game in range(20):
            obs, _ = self.env.reset()
            done = False
            moves = 0
            
            # Play game to completion; step() handles passes, so BLACK always has a move
            while not done and moves < 100:
                valid_actions = self.env.all_valid_actions(obs)
                obs, reward, done, truncated, info = self.env.step(np.random.choice(valid_actions))
                moves += 1
            self.assertTrue(done, "Game should finish within 100 BLACK moves")
            
            # step() resets the board on game end; score the final position from info
            final_board = info['terminal_observation']
            black_count = np.sum(final_board == BLACK)
            white_count = np.sum(final_board == WHITE)
            actual_outcome = int(np.sign(black_count - white_count))  # 1 BLACK win, -1 WHITE win, 0 draw
            
            self.assertEqual(reward, actual_outcome,
                             f"Game {game}: final reward {reward} doesn't match outcome "
                             f"(BLACK {black_count}, WHITE {white_count}). "
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
        """Test that rewards are sparse by design: non-zero only on the game-ending step."""
        all_rewards = []
        
        for game in range(15):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                valid_actions = self.env.all_valid_actions(obs)
                obs, reward, done, truncated, info = self.env.step(np.random.choice(valid_actions))
                all_rewards.append(reward)
                
                if not done:
                    self.assertEqual(reward, 0, "Intermediate (non-terminal) steps should have zero reward")
        
        # Informational: the fraction of zero-reward steps (reward shaping would lower this)
        sparsity = sum(1 for r in all_rewards if r == 0) / len(all_rewards)
        self.assertGreater(sparsity, 0.9, "Expected sparse, end-of-game-only rewards")
    
    def test_action_space_utilization(self):
        """Test if the action space is being utilized effectively."""
        
        action_counts = np.zeros(64)
        total_actions = 0
        
        # Collect action statistics over multiple games
        for game in range(20):
            obs, _ = self.env.reset()
            
            for move in range(50):
                # step() handles passes, so BLACK always has a move until the game ends
                valid_actions = self.env.all_valid_actions(obs)
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