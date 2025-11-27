#!/usr/bin/env python3
"""
Focused tests to isolate specific environment issues that could cause training problems.
These tests target the exact issues mentioned in CLAUDE.md to reduce search space.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import EMPTY, BLACK, WHITE, mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from util.reversi import build_reversi


class TestCriticalEnvironmentIssues(unittest.TestCase):
    """
    Focused tests to identify the exact causes of training plateau.
    Based on issues identified in CLAUDE.md test analysis.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        
    def test_missing_is_game_over_method(self):
        """Test 1: Verify is_game_over method exists and works correctly."""
        # Check method exists
        self.assertTrue(hasattr(self.env, 'is_game_over'), 
                       "CRITICAL: is_game_over method missing from ReversiEnvCNN")
        
        # Test with empty board (should not be game over)
        empty_board = np.zeros((1, 8, 8), dtype=np.int8)
        try:
            result = self.env.is_game_over(empty_board)
            self.assertFalse(result, "Empty board should not be game over")
        except Exception as e:
            self.fail(f"is_game_over failed on empty board: {e}")
            
        # Test with full board (should be game over)  
        full_board = np.ones((1, 8, 8), dtype=np.int8)
        try:
            result = self.env.is_game_over(full_board)
            self.assertTrue(result, "Full board should be game over")
        except Exception as e:
            self.fail(f"is_game_over failed on full board: {e}")
            
    def test_missing_place_method(self):
        """Test 2: Verify _place method exists and works correctly."""
        # Check method exists
        self.assertTrue(hasattr(self.env, '_place'), 
                       "CRITICAL: _place method missing from ReversiEnvCNN")
        
        # Test basic piece placement and capture
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        test_board[0, 3, 3] = WHITE
        test_board[0, 3, 4] = BLACK
        
        try:
            # Place BLACK at (3,2) should capture WHITE at (3,3)
            result_board = self.env._place(test_board, BLACK, np.array([0, 3, 2]))
            self.assertEqual(result_board[0, 3, 2], BLACK, "New piece should be placed")
            self.assertEqual(result_board[0, 3, 3], BLACK, "WHITE piece should be captured")
        except Exception as e:
            self.fail(f"_place method failed: {e}")
            
    def test_action_masking_data_types(self):
        """Test 3: Verify action masks return proper boolean types."""
        board, _ = self.env.reset()
        
        # Test direct mask_fn
        masked_env = ActionMasker(build_reversi(opponent="Random"), mask_fn)
        mask = mask_fn(masked_env)
        
        # Check return type is boolean
        self.assertEqual(mask.dtype, bool, 
                        "Action mask should return boolean array, not int8")
        
        # Check mask values are only True/False
        unique_vals = np.unique(mask)
        self.assertTrue(all(isinstance(val, (bool, np.bool_)) for val in unique_vals),
                       "Mask should contain only boolean values")
        
    def test_step_function_termination_detection(self):
        """Test 4: Verify step() properly detects game termination."""
        # Create a near-end game state
        board = np.ones((1, 8, 8), dtype=np.int8) * WHITE
        board[0, 0, 0] = EMPTY  # One empty square left
        board[0, 7, 7] = BLACK  # Some black pieces
        
        self.env.board = board.copy()
        self.env.player = BLACK
        
        # Take the last move
        try:
            obs, reward, terminated, truncated, info = self.env.step(0)  # Top-left corner
            self.assertTrue(terminated, "Game should terminate when board is full")
        except Exception as e:
            self.fail(f"Step function failed on game-ending move: {e}")
            
    def test_valid_actions_consistency(self):
        """Test 5: Verify valid actions match actual legal moves."""
        board, _ = self.env.reset()
        
        # Get valid actions
        valid_actions = self.env.all_valid_actions(board)
        
        # Test each valid action is actually legal
        for action in valid_actions:
            row, col = action // 8, action % 8
            
            # Check position is empty
            self.assertEqual(board[0, row, col], EMPTY, 
                           f"Valid action {action} at ({row},{col}) should be on empty square")
            
            # Check it's a legal move (would capture pieces)
            try:
                is_legal = self.env.is_valid(board, BLACK, np.array([0, row, col]))
                self.assertTrue(is_legal, 
                              f"Valid action {action} should be legal according to is_valid()")
            except Exception as e:
                self.fail(f"is_valid() failed for supposedly valid action {action}: {e}")
                
    def test_reward_calculation_consistency(self):
        """Test 6: Verify reward calculation works correctly."""
        # Create a simple winning scenario for BLACK
        winning_board = np.zeros((1, 8, 8), dtype=np.int8)
        winning_board[0, :4, :] = BLACK  # Black has 32 squares
        winning_board[0, 4:, :] = WHITE  # White has 32 squares
        winning_board[0, 0, 0] = BLACK   # Give Black one extra
        
        self.env.board = winning_board
        
        try:
            if hasattr(self.env, 'get_winner'):
                winner = self.env.get_winner(winning_board)
                black_count = np.sum(winning_board == BLACK)
                white_count = np.sum(winning_board == WHITE)
                expected_winner = BLACK if black_count > white_count else WHITE
                self.assertEqual(winner, expected_winner, 
                               f"Winner calculation incorrect. BLACK: {black_count}, WHITE: {white_count}")
        except Exception as e:
            self.fail(f"Reward/winner calculation failed: {e}")
            
    def test_environment_reset_consistency(self):
        """Test 7: Verify reset() always produces identical initial states."""
        # Multiple resets should be identical
        states = []
        for _ in range(5):
            state, _ = self.env.reset()
            states.append(state.copy())
            
        # All states should be identical
        for i in range(1, len(states)):
            np.testing.assert_array_equal(states[0], states[i], 
                                        "Reset should produce identical initial states")


class TestGameLogicIntegrity(unittest.TestCase):
    """
    Tests for core game logic that could break RL training.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        
    def test_piece_count_conservation(self):
        """Test that pieces are properly placed and counted."""
        board, _ = self.env.reset()
        initial_pieces = np.sum(board != EMPTY)
        
        # Make a valid move
        valid_actions = self.env.all_valid_actions(board)
        if len(valid_actions) > 0:
            action = valid_actions[0]
            try:
                new_board, reward, terminated, truncated, info = self.env.step(action)
                new_pieces = np.sum(new_board != EMPTY)
                
                # Should have at least one more piece (the placed piece)
                self.assertGreater(new_pieces, initial_pieces,
                                 "Move should increase total piece count")
            except Exception as e:
                self.fail(f"Step function failed: {e}")
                
    def test_player_alternation(self):
        """Test that players alternate correctly."""
        board, _ = self.env.reset()
        initial_player = self.env.player
        
        valid_actions = self.env.all_valid_actions(board)
        if len(valid_actions) > 0:
            action = valid_actions[0]
            try:
                # Player should change after step (or stay same if opponent passes)
                self.env.step(action)
                # After opponent's turn, should be back to BLACK
                self.assertEqual(self.env.player, BLACK,
                               "After full round, should return to BLACK")
            except Exception as e:
                self.fail(f"Player alternation failed: {e}")


class TestTrainingCriticalFeatures(unittest.TestCase):
    """
    Tests for features critical to RL training success.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        
    def test_action_space_bounds(self):
        """Test that all actions are within valid bounds."""
        board, _ = self.env.reset()
        
        # Check action space size matches board
        self.assertEqual(self.env.action_space.n, 64, "Action space should be 64 for 8x8 board")
        
        # Check all valid actions are within bounds
        valid_actions = self.env.all_valid_actions(board)
        for action in valid_actions:
            self.assertGreaterEqual(action, 0, f"Action {action} below minimum bound")
            self.assertLess(action, 64, f"Action {action} above maximum bound")
            
    def test_observation_space_consistency(self):
        """Test observation space matches actual observations."""
        board, _ = self.env.reset()
        
        # Check observation is in defined space
        self.assertTrue(self.env.observation_space.contains(board),
                       "Observation should be within observation space")
        
        # Check data type consistency
        self.assertEqual(board.dtype, self.env.observation_space.dtype,
                        "Observation dtype should match space dtype")


if __name__ == '__main__':
    print("Running focused tests to isolate training issues...")
    print("=" * 60)
    
    # Run with high verbosity to see specific failures
    unittest.main(verbosity=2)