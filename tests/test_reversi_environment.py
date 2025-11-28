import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN, build_reversi
from util.util import EMPTY, BLACK, WHITE, mask_fn
from sb3_contrib.common.wrappers import ActionMasker


class TestReversiEnvironment(unittest.TestCase):
    """Comprehensive test suite for Reversi environment."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
        self.masked_env = ActionMasker(build_reversi(opponent="Random"), mask_fn)
    
    def test_initial_board_setup(self):
        """Test that the board is set up correctly at the start."""
        board, _ = self.env.reset()
        
        # Check board dimensions
        self.assertEqual(board.shape, (1, 8, 8))
        
        # Check initial piece placement (standard Reversi start)
        # Center should have 4 pieces in alternating pattern
        self.assertEqual(board[0, 3, 3], WHITE)  # d4
        self.assertEqual(board[0, 3, 4], BLACK)  # e4
        self.assertEqual(board[0, 4, 3], BLACK)  # d5
        self.assertEqual(board[0, 4, 4], WHITE)  # e5
        
        # Check that only center 4 squares are occupied
        piece_count = np.sum(board != EMPTY)
        self.assertEqual(piece_count, 4)
        
    def test_player_initialization(self):
        """Test that players are initialized correctly."""
        self.env.reset()
        self.assertEqual(self.env.player, BLACK)
        self.assertEqual(self.env.actual_player, BLACK)
    
    def test_valid_move_detection_initial(self):
        """Test valid move detection on initial board."""
        board, _ = self.env.reset()
        
        # Get valid moves for BLACK (first player)
        valid_actions = self.env.all_valid_actions(board)
        
        # Initial valid moves for black should be: c4(19), d3(26), e6(45), f5(53)
        expected_moves = [19, 26, 45, 53]  # Converting to flat indices
        
        self.assertEqual(len(valid_actions), 4, f"Expected 4 valid moves, got {len(valid_actions)}")
        
        # Check that all expected moves are in valid actions
        for move in expected_moves:
            self.assertIn(move, valid_actions, f"Move {move} should be valid")
    
    def test_move_validation_edge_cases(self):
        """Test move validation for edge cases."""
        board, _ = self.env.reset()
        
        # Test invalid moves
        # Empty square with no captures
        self.assertFalse(self.env.is_valid(board, BLACK, np.array([0, 0, 0])))
        
        # Occupied square
        self.assertFalse(self.env.is_valid(board, BLACK, np.array([0, 3, 3])))
        
        # Out of bounds (this should be handled gracefully)
        self.assertFalse(self.env.is_valid(board, BLACK, np.array([0, 8, 8])))
        
        # Test pass move
        self.assertTrue(self.env.is_valid(board, BLACK, self.env.PASS))
    
    def test_piece_capture_mechanics(self):
        """Test that pieces are captured correctly when a move is made."""
        # Create a specific board state to test captures
        self.env.reset()
        
        # Make the first valid move (c4 = position 19)
        initial_board = self.env.board.copy()
        
        # Manually place pieces for a capture test
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        test_board[0, 3, 3] = WHITE  # d4
        test_board[0, 3, 4] = WHITE  # e4  
        test_board[0, 3, 5] = BLACK  # f4
        
        self.env.board = test_board
        self.env.player = BLACK
        
        # Move to c4 should capture d4
        result_board = self.env._place(test_board, BLACK, np.array([0, 3, 2]))
        
        # Check that the capture happened
        self.assertEqual(result_board[0, 3, 2], BLACK)  # New piece placed
        self.assertEqual(result_board[0, 3, 3], BLACK)  # Captured piece
        self.assertEqual(result_board[0, 3, 4], BLACK)  # Captured piece
        self.assertEqual(result_board[0, 3, 5], BLACK)  # Original piece unchanged
    
    def test_game_end_conditions(self):
        """Test various game ending conditions."""
        self.env.reset()
        
        # Test game ending when board is full
        full_board = np.ones((1, 8, 8), dtype=np.int8)
        full_board[0, 0, 0] = EMPTY  # Leave one empty space
        
        self.env.board = full_board
        
        # Check that game doesn't end yet
        self.assertFalse(self.env.is_game_over(full_board))
        
        # Fill the last space
        full_board[0, 0, 0] = BLACK
        self.assertTrue(self.env.is_game_over(full_board))
    
    def test_no_valid_moves_scenario(self):
        """Test behavior when a player has no valid moves."""
        # Create a board state where current player has no moves
        test_board = np.zeros((1, 8, 8), dtype=np.int8)
        # Place pieces such that BLACK has no valid moves
        test_board[0, 0, 0] = BLACK
        test_board[0, 7, 7] = WHITE
        
        self.env.board = test_board
        self.env.player = BLACK
        
        # Check that there are no valid moves
        valid_moves = self.env.all_valid_actions(test_board)
        has_valid = self.env.has_valid(test_board, BLACK)
        
        if len(valid_moves) == 0:
            self.assertFalse(has_valid)
    
    def test_action_space_consistency(self):
        """Test that action space is consistent with board size."""
        self.assertEqual(self.env.action_space.n, 64)  # 8x8 board
        
        # Test conversion between flat and 2D indices
        for flat_action in range(64):
            x, y = flat_action // 8, flat_action % 8
            reconstructed = x * 8 + y
            self.assertEqual(flat_action, reconstructed)
    
    def test_reward_structure(self):
        """Test that rewards are assigned correctly."""
        self.env.reset()
        
        # Create a winning position for BLACK
        winning_board = np.zeros((1, 8, 8), dtype=np.int8)
        winning_board[0, :4, :4] = BLACK  # Black controls 16 squares
        winning_board[0, 4:, 4:] = WHITE  # White controls 16 squares
        winning_board[0, 0, 0] = BLACK    # Give black one extra piece
        
        self.env.board = winning_board
        
        # Check game over detection and winner
        if self.env.is_game_over(winning_board):
            winner = self.env.get_winner(winning_board)
            expected_winner = BLACK if np.sum(winning_board == BLACK) > np.sum(winning_board == WHITE) else WHITE
            self.assertEqual(winner, expected_winner)
    
    def test_observation_space(self):
        """Test observation space properties."""
        board, _ = self.env.reset()
        
        # Check observation is within defined space
        self.assertTrue(self.env.observation_space.contains(board))
        
        # Check data type
        self.assertEqual(board.dtype, np.int8)
        
        # Check range of values
        self.assertTrue(np.all(board >= -1))
        self.assertTrue(np.all(board <= 1))
    
    def test_action_masking(self):
        """Test that action masking works correctly."""
        obs, _ = self.masked_env.reset()
        
        # Get action mask
        action_mask = mask_fn(self.masked_env)
        
        # Check that mask is boolean array of correct size
        self.assertEqual(len(action_mask), 64)
        self.assertTrue(all(isinstance(x, (bool, np.bool_)) for x in action_mask))
        
        # Check that exactly the valid actions are unmasked
        valid_actions = self.masked_env.envs[0].all_valid_actions(obs)
        
        for i in range(64):
            if i in valid_actions:
                self.assertTrue(action_mask[i], f"Action {i} should be valid but is masked")
            else:
                self.assertFalse(action_mask[i], f"Action {i} should be invalid but is not masked")
    
    def test_environment_reset_consistency(self):
        """Test that environment resets consistently."""
        # Reset multiple times and check consistency
        for _ in range(5):
            board1, _ = self.env.reset()
            board2, _ = self.env.reset()
            
            np.testing.assert_array_equal(board1, board2, "Reset should be deterministic")
            self.assertEqual(self.env.player, BLACK, "Player should always start as BLACK")
    
    def test_opponent_integration(self):
        """Test that opponent integration works correctly."""
        # Test with different opponent types
        for opponent_type in ["Random"]:  # Add more when available
            env = ReversiEnvCNN(opponent=opponent_type, verbose=False)
            env.reset()
            
            self.assertIsNotNone(env.opponent, f"Opponent should be set for type {opponent_type}")


class TestReversiGameLogic(unittest.TestCase):
    """Test specific Reversi game logic edge cases."""
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_corner_captures(self):
        """Test captures involving corner pieces."""
        # Set up a corner capture scenario
        board = np.zeros((1, 8, 8), dtype=np.int8)
        board[0, 0, 0] = BLACK  # Corner
        board[0, 0, 1] = WHITE  # Adjacent
        board[0, 0, 2] = EMPTY  # Target
        
        self.env.board = board
        
        # Black should be able to capture by playing at (0,2)
        is_valid = self.env.is_valid(board, BLACK, np.array([0, 0, 2]))
        self.assertTrue(is_valid, "Should be able to capture along edge")
    
    def test_multiple_direction_captures(self):
        """Test captures in multiple directions from one move."""
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Set up cross pattern with WHITE pieces around center
        board[0, 4, 3] = WHITE  # Left
        board[0, 4, 5] = WHITE  # Right  
        board[0, 3, 4] = WHITE  # Top
        board[0, 5, 4] = WHITE  # Bottom
        
        # Add BLACK pieces to complete captures
        board[0, 4, 2] = BLACK  # Far left
        board[0, 4, 6] = BLACK  # Far right
        board[0, 2, 4] = BLACK  # Far top
        board[0, 6, 4] = BLACK  # Far bottom
        
        self.env.board = board
        
        # Playing at center should capture in all 4 directions
        is_valid = self.env.is_valid(board, BLACK, np.array([0, 4, 4]))
        self.assertTrue(is_valid, "Should be able to capture in multiple directions")
    
    def test_no_capture_scenarios(self):
        """Test scenarios where no capture occurs."""
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Place pieces that don't form a valid capture
        board[0, 3, 3] = BLACK
        board[0, 3, 5] = BLACK  # Gap in between
        
        self.env.board = board
        
        # Playing between them shouldn't be valid (no opposing piece to capture)
        is_valid = self.env.is_valid(board, BLACK, np.array([0, 3, 4]))
        self.assertFalse(is_valid, "Should not be valid - no opposing pieces to capture")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)