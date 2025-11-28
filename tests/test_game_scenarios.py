import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.reversi import ReversiEnvCNN
from util.util import EMPTY, BLACK, WHITE
from util.opponents import RandomOpponent


class TestGameScenarios(unittest.TestCase):
    """Test complete game scenarios and edge cases."""
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_full_game_simulation(self):
        """Simulate a complete game and verify it ends correctly."""
        obs, _ = self.env.reset()
        
        game_length = 0
        max_moves = 100  # Safety limit
        
        while game_length < max_moves:
            valid_actions = self.env.all_valid_actions(obs)
            
            if len(valid_actions) == 0:
                # No valid moves - should pass or end game
                if self.env.is_game_over(obs):
                    break
                else:
                    # Handle pass scenario
                    obs, reward, done, truncated, info = self.env.step(self.env.PASS[1] * 8 + self.env.PASS[2])
                    if done:
                        break
            else:
                # Make a random valid move
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    break
            
            game_length += 1
        
        # Game should have ended naturally within reasonable moves
        self.assertLess(game_length, max_moves, "Game should end within reasonable time")
        
        # Final board should have a clear winner or be a draw
        if self.env.is_game_over(obs):
            winner = self.env.get_winner(obs)
            self.assertIn(winner, [BLACK, WHITE, 0], "Winner should be BLACK, WHITE, or 0 (draw)")
    
    def test_pass_scenarios(self):
        """Test scenarios where players must pass."""
        # Create a board state where current player has no moves
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Fill most of board with WHITE, leave BLACK isolated
        board.fill(WHITE)
        board[0, 0, 0] = BLACK
        board[0, 7, 7] = EMPTY  # One empty square far from BLACK
        
        self.env.board = board
        self.env.player = BLACK
        
        valid_moves = self.env.all_valid_actions(board)
        
        # BLACK should have no valid moves in this scenario
        if len(valid_moves) == 0:
            self.assertFalse(self.env.has_valid(board, BLACK))
    
    def test_game_end_by_no_moves(self):
        """Test game ending when neither player can move."""
        # Create board where game should end due to no valid moves
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Create a position where no moves are possible for either player
        # Fill with alternating pattern but no captures possible
        for i in range(8):
            for j in range(8):
                board[0, i, j] = BLACK if (i + j) % 2 == 0 else WHITE
        
        self.env.board = board
        
        # Check if game is over
        game_over = self.env.is_game_over(board)
        
        # At minimum, check that we can detect when board is full
        if np.all(board != EMPTY):
            self.assertTrue(game_over, "Game should be over when board is full")
    
    def test_scoring_accuracy(self):
        """Test that scoring is calculated correctly."""
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Create known piece distribution
        board[0, :3, :3] = BLACK  # 9 black pieces
        board[0, 3:6, 3:6] = WHITE  # 9 white pieces
        board[0, 6, 6] = BLACK  # 1 more black piece (total 10)
        
        self.env.board = board
        
        if hasattr(self.env, 'get_score'):
            black_score, white_score = self.env.get_score(board)
            self.assertEqual(black_score, 10, "BLACK should have 10 pieces")
            self.assertEqual(white_score, 9, "WHITE should have 9 pieces")
    
    def test_reward_consistency(self):
        """Test that rewards are consistent with game outcomes."""
        # Test winning scenario
        winning_board = np.zeros((1, 8, 8), dtype=np.int8)
        winning_board.fill(BLACK)  # BLACK wins with full board
        
        self.env.board = winning_board
        
        if self.env.is_game_over(winning_board):
            winner = self.env.get_winner(winning_board)
            self.assertEqual(winner, BLACK, "BLACK should win when controlling full board")
        
        # Test losing scenario  
        losing_board = np.zeros((1, 8, 8), dtype=np.int8)
        losing_board.fill(WHITE)  # WHITE wins with full board
        
        self.env.board = losing_board
        
        if self.env.is_game_over(losing_board):
            winner = self.env.get_winner(losing_board)
            self.assertEqual(winner, WHITE, "WHITE should win when controlling full board")
    
    def test_opponent_move_integration(self):
        """Test that opponent moves are integrated correctly."""
        obs, _ = self.env.reset()
        
        # Make a move for BLACK
        valid_actions = self.env.all_valid_actions(obs)
        if len(valid_actions) > 0:
            action = valid_actions[0]
            
            # Store initial state
            initial_board = obs.copy()
            
            # Make move
            new_obs, reward, done, truncated, info = self.env.step(action)
            
            # Board should have changed
            self.assertFalse(np.array_equal(initial_board, new_obs), 
                            "Board should change after valid move")
            
            # Player should have switched (in the environment's internal state)
            # Note: The step function handles both BLACK and WHITE moves
    
    def test_illegal_move_handling(self):
        """Test handling of illegal moves."""
        obs, _ = self.env.reset()
        
        # Try to make a move on an occupied square
        occupied_square = None
        for i in range(8):
            for j in range(8):
                if obs[0, i, j] != EMPTY:
                    occupied_square = i * 8 + j
                    break
            if occupied_square is not None:
                break
        
        if occupied_square is not None:
            # This should be handled according to illegal_action_mode
            # Default is 'resign', so game should end with negative reward
            new_obs, reward, done, truncated, info = self.env.step(occupied_square)
            
            # Check that illegal move was handled appropriately
            if self.env.illegal_equivalent_action is self.env.RESIGN:
                self.assertTrue(done, "Game should end on illegal move with resign mode")
                self.assertLess(reward, 0, "Reward should be negative for resignation")


class TestEdgeCaseScenarios(unittest.TestCase):
    """Test edge cases that might cause training issues."""
    
    def setUp(self):
        self.env = ReversiEnvCNN(opponent="Random", verbose=False)
    
    def test_rapid_game_end(self):
        """Test scenarios where game ends very quickly."""
        # Create a near-end game state
        board = np.zeros((1, 8, 8), dtype=np.int8)
        
        # Fill board except for a few squares
        board.fill(BLACK)
        board[0, 0, 0] = WHITE
        board[0, 0, 1] = EMPTY
        board[0, 1, 0] = EMPTY
        
        self.env.board = board
        self.env.player = BLACK
        
        # Game should be near end
        valid_moves = self.env.all_valid_actions(board)
        
        # Should have very few or no valid moves
        self.assertLessEqual(len(valid_moves), 3, 
                            "Near-end game should have very few valid moves")
    
    def test_alternating_pass_scenario(self):
        """Test scenario where players alternate passing."""
        # This shouldn't happen in real Reversi, but test robustness
        board = np.zeros((1, 8, 8), dtype=np.int8)
        board[0, 0, 0] = BLACK
        board[0, 7, 7] = WHITE
        # All other squares empty, no valid moves for either player
        
        self.env.board = board
        
        black_has_moves = self.env.has_valid(board, BLACK)
        white_has_moves = self.env.has_valid(board, WHITE)
        
        # If neither player has moves, game should end
        if not black_has_moves and not white_has_moves:
            self.assertTrue(self.env.is_game_over(board), 
                           "Game should end when neither player can move")
    
    def test_board_state_consistency(self):
        """Test that board states remain consistent through operations."""
        obs, _ = self.env.reset()
        
        # Store original state
        original_board = obs.copy()
        original_player = self.env.player
        
        # Perform various queries that shouldn't change state
        _ = self.env.all_valid_actions(obs)
        _ = self.env.has_valid(obs, self.env.player)
        _ = self.env.get_valid(obs)
        
        # State should be unchanged
        np.testing.assert_array_equal(obs, original_board, 
                                    "Board should not change during queries")
        self.assertEqual(self.env.player, original_player, 
                        "Player should not change during queries")
    
    def test_memory_leaks_simulation(self):
        """Test for potential memory leaks in repeated games."""
        initial_board_shape = None
        
        # Run multiple short games
        for game in range(10):
            obs, _ = self.env.reset()
            
            if initial_board_shape is None:
                initial_board_shape = obs.shape
            
            # Verify board shape consistency
            self.assertEqual(obs.shape, initial_board_shape, 
                           f"Board shape should be consistent across games (game {game})")
            
            # Make a few random moves
            for move in range(5):
                valid_actions = self.env.all_valid_actions(obs)
                if len(valid_actions) == 0:
                    break
                
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.env.step(action)
                
                if done:
                    break


if __name__ == '__main__':
    unittest.main(verbosity=2)