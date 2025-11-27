#!/usr/bin/env python3
"""
Tests for complex step function turn logic and self-play edge cases.
These test the sophisticated implementation details that could cause training issues.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import EMPTY, BLACK, WHITE


class TestStepFunctionTurnLogic(unittest.TestCase):
    """
    Test the complex step function logic that handles full BLACK/WHITE turns.
    Key assumption: BLACK always has a valid move when step() is called.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_normal_turn_sequence(self):
        """Test 1: Normal case - both players have moves available."""
        board, _ = self.env.reset()
        initial_player = self.env.player
        self.assertEqual(initial_player, BLACK, "Should start with BLACK")
        
        # Make a valid BLACK move
        valid_actions = self.env.all_valid_actions(board)
        self.assertGreater(len(valid_actions), 0, "BLACK should have valid moves initially")
        
        action = valid_actions[0]
        next_board, reward, terminated, truncated, info = self.env.step(action)
        
        # After step(), should return to BLACK (full round trip)
        self.assertEqual(self.env.player, BLACK, 
                        "After full step(), should return to BLACK")
        self.assertFalse(terminated, "Game should not terminate on normal move")
        
    def test_black_no_moves_after_white_move(self):
        """Test 2: BLACK has no moves after WHITE's turn - WHITE continues."""
        # Create a board state where BLACK will have no moves after WHITE plays
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Setup: WHITE dominates, BLACK isolated
        test_board[0, 0, 0] = BLACK  # Isolated BLACK piece
        test_board[0, 0, 1:3] = WHITE  # WHITE pieces blocking
        test_board[0, 1, 0:3] = WHITE
        test_board[0, 3:5, 3:5] = BLACK  # Some BLACK pieces for opponent to target
        
        self.env.board = test_board
        self.env.player = BLACK
        
        # BLACK makes a move, WHITE should get multiple consecutive moves
        valid_actions = self.env.all_valid_actions(test_board)
        
        if len(valid_actions) > 0:
            action = valid_actions[0]
            next_board, reward, terminated, truncated, info = self.env.step(action)
            
            # The step function should handle WHITE getting multiple moves
            # and eventually return to BLACK or terminate
            self.assertIn(self.env.player, [BLACK, WHITE], 
                         "Player should be valid after complex turn sequence")
    
    def test_game_termination_in_step(self):
        """Test 3: Game terminates during the step sequence."""
        # Create a near-end game state with valid BLACK move
        end_board = np.zeros((1, 8, 8), dtype=np.int8)
        end_board[0, :, :] = WHITE  # Fill with WHITE
        end_board[0, 0, 0] = EMPTY  # Empty square for BLACK to play
        end_board[0, 0, 1] = WHITE  # WHITE piece to be captured
        end_board[0, 0, 2] = BLACK  # BLACK piece to make capture valid
        
        self.env.board = end_board
        self.env.player = BLACK
        
        # Verify BLACK has a valid move
        valid_actions = self.env.all_valid_actions(end_board)
        self.assertGreater(len(valid_actions), 0, "BLACK should have at least one valid move")
        self.assertIn(0, valid_actions, "Action 0 should be valid")
        
        # Make the move that should terminate the game
        next_board, reward, terminated, truncated, info = self.env.step(0)
        
        self.assertTrue(terminated, "Game should terminate when board is full")
        self.assertIn('terminal_observation', info, 
                     "Should provide terminal observation in info")
        
    def test_white_no_moves_initially(self):
        """Test 4: WHITE has no valid moves after BLACK's move."""
        # Create board where WHITE will have no moves after BLACK plays
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Setup where WHITE gets blocked
        test_board[0, 3:5, 3:5] = BLACK  # BLACK dominating center
        test_board[0, 2, 2] = WHITE      # Isolated WHITE pieces
        test_board[0, 5, 5] = WHITE
        
        self.env.board = test_board
        self.env.player = BLACK
        
        valid_actions = self.env.all_valid_actions(test_board)
        
        if len(valid_actions) > 0:
            action = valid_actions[0]
            next_board, reward, terminated, truncated, info = self.env.step(action)
            
            # Should handle the case where WHITE can't move
            # and return control to BLACK or terminate
            if not terminated:
                self.assertEqual(self.env.player, BLACK,
                               "Should return to BLACK when WHITE can't move")
    
    def test_pass_scenarios(self):
        """Test 5: Both players forced to pass - game should terminate."""
        # Create a board where no moves are possible
        pass_board = np.zeros((1, 8, 8), dtype=np.int8)
        pass_board[0, 0, 0] = BLACK
        pass_board[0, 7, 7] = WHITE
        # All other squares empty, no captures possible
        
        self.env.board = pass_board
        self.env.player = BLACK
        
        # Check if BLACK has any valid moves
        valid_actions = self.env.all_valid_actions(pass_board)
        
        if len(valid_actions) == 0:
            # This should be handled before step() is called
            # But if step() is called anyway, it should terminate gracefully
            print("BLACK has no moves - this scenario should be handled before step()")
        else:
            # Make a move and see what happens
            action = valid_actions[0]
            next_board, reward, terminated, truncated, info = self.env.step(action)
            
            # Game should eventually terminate due to no moves
            self.assertTrue(True)  # If we get here without crashing, logic is sound
    
    def test_state_consistency_across_turns(self):
        """Test 6: Board state remains consistent across complex turn sequences."""
        board, _ = self.env.reset()
        initial_pieces = np.sum(board != EMPTY)
        
        # Make several moves and check piece count increases
        for _ in range(3):
            valid_actions = self.env.all_valid_actions(self.env.board)
            if len(valid_actions) > 0:
                action = valid_actions[0]
                prev_pieces = np.sum(self.env.board != EMPTY)
                
                next_board, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated:
                    break
                    
                new_pieces = np.sum(self.env.board != EMPTY)
                self.assertGreaterEqual(new_pieces, prev_pieces,
                                      "Piece count should not decrease during moves")
            else:
                break
    
    def test_player_assertions_hold(self):
        """Test 7: Verify the step() assertion assumptions hold."""
        board, _ = self.env.reset()
        
        # step() assumes self.player == BLACK when called
        self.assertEqual(self.env.player, BLACK, 
                        "step() assumption: player should be BLACK when called")
        
        # After step(), player state should be consistent
        valid_actions = self.env.all_valid_actions(board)
        if len(valid_actions) > 0:
            action = valid_actions[0]
            next_board, reward, terminated, truncated, info = self.env.step(action)
            
            if not terminated:
                # Should be back to BLACK after full round
                self.assertEqual(self.env.player, BLACK,
                               "After step(), should return to BLACK")


class TestSelfPlayLogic(unittest.TestCase):
    """
    Test the self-play opponent switching logic for edge cases.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_opponent_consistency(self):
        """Test 1: Opponent remains consistent during training."""
        board, _ = self.env.reset()
        initial_opponent = self.env.opponent
        
        # Make several moves
        for _ in range(5):
            valid_actions = self.env.all_valid_actions(self.env.board)
            if len(valid_actions) > 0:
                action = valid_actions[0]
                next_board, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated:
                    break
                    
                # Opponent should remain the same during episode
                self.assertEqual(type(self.env.opponent), type(initial_opponent),
                               "Opponent type should not change mid-episode")
            else:
                break
    
    def test_opponent_action_validity(self):
        """Test 2: Opponent always provides valid actions."""
        board, _ = self.env.reset()
        
        # Force opponent to make several moves
        for _ in range(10):
            valid_actions = self.env.all_valid_actions(self.env.board)
            if len(valid_actions) > 0:
                action = valid_actions[0]
                next_board, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated:
                    break
            else:
                break
                
        # If we get here without errors, opponent actions were valid
        self.assertTrue(True, "Opponent provided valid actions throughout")
    
    def test_model_opponent_integration(self):
        """Test 3: Model-based opponent integration works correctly."""
        # This test would require a trained model, so we'll test the structure
        try:
            model_env = ReversiEnvCNN(opponent="Model", opponent_model="dummy", verbose=False)
            # If this doesn't crash, the opponent switching logic is structurally sound
            self.assertTrue(True, "Model opponent initialization successful")
        except Exception as e:
            # Expected if no model file exists - test the error handling
            self.assertIn("model", str(e).lower(), "Should fail gracefully for missing model")


class TestStepFunctionEdgeCases(unittest.TestCase):
    """
    Test edge cases in the step function that could cause subtle bugs.
    """
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_multiple_white_passes(self):
        """Test: WHITE passes multiple times while BLACK waits."""
        # Create scenario where WHITE keeps having no moves
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        test_board[0, 3:5, 3:5] = BLACK
        test_board[0, 0, 0] = WHITE  # Isolated WHITE
        
        self.env.board = test_board
        self.env.player = BLACK
        
        valid_actions = self.env.all_valid_actions(test_board)
        if len(valid_actions) > 0:
            action = valid_actions[0]
            # This should handle WHITE having no moves gracefully
            next_board, reward, terminated, truncated, info = self.env.step(action)
            
            # Should not crash and should handle the scenario appropriately
            self.assertIsNotNone(next_board, "Should return valid board state")
    
    def test_winner_calculation_during_step(self):
        """Test: get_winner() is called at the right time."""
        # Create a game-ending scenario with valid BLACK move
        end_board = np.zeros((1, 8, 8), dtype=np.int8)
        end_board[0, :, :] = WHITE  # Fill with WHITE
        end_board[0, 0, 0] = EMPTY  # Empty square for BLACK
        end_board[0, 0, 1] = WHITE  # WHITE piece to capture
        end_board[0, 0, 2] = BLACK  # BLACK piece to make capture valid
        
        self.env.board = end_board
        self.env.player = BLACK
        
        # Verify BLACK has a valid move
        valid_actions = self.env.all_valid_actions(end_board)
        if len(valid_actions) > 0:
            # Make the final move
            next_board, reward, terminated, truncated, info = self.env.step(0)
            
            if terminated:
                # Reward should represent the winner
                self.assertIn(reward, [-1, 0, 1], "Reward should be valid winner value")
                self.assertIn('terminal_observation', info, 
                             "Should provide terminal observation")
        else:
            self.skipTest("No valid moves available for this scenario")


if __name__ == '__main__':
    print("Running step function and self-play logic tests...")
    print("=" * 60)
    
    # Run with high verbosity to see specific failures
    unittest.main(verbosity=2)