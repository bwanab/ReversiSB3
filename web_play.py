#!/usr/bin/env python3
"""
Web interface for playing Reversi against trained model.
Provides visual board and click-to-move interface.
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import argparse
from util.reversi import ReversiEnvCNN
from util.util import get_model, mask_fn, BLACK, WHITE


app = Flask(__name__)

# Global game state
game_state = {
    'env': None,
    'model': None,
    'model_color': BLACK,  # Model plays as BLACK by default
    'game_over': False,
    'winner': None,
    'last_move': None,
    'model_name': None,
    'move_history': [],  # List of {action, player, board_before}
    'redo_stack': []  # Stack for redo functionality
}


def board_to_list(board):
    """Convert numpy board to list for JSON serialization."""
    # Board is shape (3, 8, 8) - we only need the first channel
    return board[0].tolist()


def get_valid_moves(env, board):
    """Get list of valid moves as (row, col) tuples."""
    action_mask = mask_fn(env)
    valid_moves = []
    for i in range(64):
        if action_mask[i]:
            row = i // 8
            col = i % 8
            valid_moves.append({'row': row, 'col': col, 'action': i})
    return valid_moves


def get_model_move(env, board):
    """Get model's move using current policy."""
    model = game_state['model']

    # Set env.player to current player for mask_fn
    # (mask_fn uses env.player to determine valid moves)

    # Get observation from current player's perspective
    # If model is playing WHITE, flip board so model sees WHITE as positive
    if env.player == WHITE:
        obs = board.copy() * -1  # Flip perspective for WHITE
    else:
        obs = board.copy()

    # Get action mask for current player
    action_mask = mask_fn(env)

    # Model's predict method
    action, _states = model.predict(obs, action_masks=action_mask, deterministic=True)

    return int(action)


def count_pieces(board):
    """Count black and white pieces on board."""
    # Board shape is (3, 8, 8), channel 0 contains the pieces
    board_2d = board[0]
    black_count = np.sum(board_2d == BLACK)
    white_count = np.sum(board_2d == WHITE)
    return {'black': int(black_count), 'white': int(white_count)}


def record_move(action, player_str, board_before):
    """Record a move in the move history."""
    game_state['move_history'].append({
        'action': int(action),
        'player': player_str,
        'board_before': board_before.copy()
    })
    # Clear redo stack when new move is made
    game_state['redo_stack'] = []


def action_to_notation(action):
    """Convert action number to chess-style notation (e.g., 19 -> D3)."""
    row = action // 8
    col = action % 8
    return f"{'ABCDEFGH'[col]}{row + 1}"


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    data = request.json
    model_color = data.get('model_color', 'black')

    # Create new environment
    env = ReversiEnvCNN()
    board, _ = env.reset()

    game_state['env'] = env
    game_state['model_color'] = BLACK if model_color == 'black' else WHITE
    game_state['game_over'] = False
    game_state['winner'] = None
    game_state['last_move'] = None
    game_state['move_history'] = []
    game_state['redo_stack'] = []

    # Get initial state
    current_player = int(env.player)
    valid_moves = get_valid_moves(env, board)
    piece_count = count_pieces(board)

    response = {
        'board': board_to_list(board),
        'current_player': 'black' if current_player == BLACK else 'white',
        'valid_moves': valid_moves,
        'piece_count': piece_count,
        'game_over': False,
        'model_color': model_color,
        'can_undo': len(game_state['move_history']) > 0,
        'can_redo': len(game_state['redo_stack']) > 0
    }

    # If model plays black, make first move
    if game_state['model_color'] == BLACK and not game_state['game_over']:
        return make_model_move()

    return jsonify(response)


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Handle human move."""
    data = request.json
    action = data.get('action')

    if action is None:
        return jsonify({'error': 'No action provided'}), 400

    env = game_state['env']
    if env is None:
        return jsonify({'error': 'No game in progress'}), 400

    if game_state['game_over']:
        return jsonify({'error': 'Game is over'}), 400

    # Make the move
    try:
        # Record move before executing
        board_before = env.board.copy()
        player_color = 'black' if env.player == BLACK else 'white'

        # Convert action to tuple format
        action_tuple = (0, action // 8, action % 8)

        # Execute move using low-level method (bypass step)
        # Note: get_next_state automatically switches env.player to the next player
        board = env.board
        board = env.get_next_state(board, action_tuple)
        env.board = board

        # Record the move in history
        record_move(action, player_color, board_before)

        game_state['last_move'] = {'action': action, 'player': 'human'}

        # Check if game is over
        winner = env.get_winner(board)
        if winner is not None:
            game_state['game_over'] = True
            if winner == BLACK:
                game_state['winner'] = 'black'
            elif winner == WHITE:
                game_state['winner'] = 'white'
            else:
                game_state['winner'] = 'draw'

            return jsonify({
                'board': board_to_list(board),
                'game_over': True,
                'winner': game_state['winner'],
                'piece_count': count_pieces(board),
                'valid_moves': [],
                'can_undo': len(game_state['move_history']) > 0,
                'can_redo': len(game_state['redo_stack']) > 0
            })

        # Check if next player (model) has valid moves
        if not env.has_valid(board, env.player):
            # Model has no valid moves - pass back to human
            env.player = -env.player

            # Check if human also has no valid moves
            if not env.has_valid(board, env.player):
                # Neither player has moves - game over
                game_state['game_over'] = True
                winner = env.get_winner(board)
                if winner == BLACK:
                    game_state['winner'] = 'black'
                elif winner == WHITE:
                    game_state['winner'] = 'white'
                else:
                    game_state['winner'] = 'draw'

                return jsonify({
                    'board': board_to_list(board),
                    'game_over': True,
                    'winner': game_state['winner'],
                    'piece_count': count_pieces(board),
                    'valid_moves': []
                })

            # Human has moves, model passed
            current_player = int(env.player)
            valid_moves = get_valid_moves(env, board)
            return jsonify({
                'board': board_to_list(board),
                'current_player': 'black' if current_player == BLACK else 'white',
                'valid_moves': valid_moves,
                'piece_count': count_pieces(board),
                'game_over': False,
                'model_passed': True,
                'can_undo': len(game_state['move_history']) > 0,
                'can_redo': len(game_state['redo_stack']) > 0
            })

        # Not game over - model's turn
        return make_model_move()

    except Exception as e:
        return jsonify({'error': str(e)}), 400


def make_model_move():
    """Make model's move and return game state."""
    env = game_state['env']
    board = env.board

    # Get valid moves for model
    valid_moves = get_valid_moves(env, board)

    if len(valid_moves) == 0:
        # Model has no valid moves - pass to opponent
        env.player = -env.player

        # Check if opponent also has no valid moves
        if not env.has_valid(board, env.player):
            # Neither player has moves - game over
            game_state['game_over'] = True
            winner = env.get_winner(board)
            if winner == BLACK:
                game_state['winner'] = 'black'
            elif winner == WHITE:
                game_state['winner'] = 'white'
            else:
                game_state['winner'] = 'draw'

            return jsonify({
                'board': board_to_list(board),
                'game_over': True,
                'winner': game_state['winner'],
                'piece_count': count_pieces(board),
                'valid_moves': [],
                'can_undo': len(game_state['move_history']) > 0,
                'can_redo': len(game_state['redo_stack']) > 0
            })
        else:
            # Only opponent has moves - return to opponent
            current_player = int(env.player)
            opponent_valid_moves = get_valid_moves(env, board)
            return jsonify({
                'board': board_to_list(board),
                'current_player': 'black' if current_player == BLACK else 'white',
                'valid_moves': opponent_valid_moves,
                'piece_count': count_pieces(board),
                'game_over': False,
                'model_passed': True,
                'can_undo': len(game_state['move_history']) > 0,
                'can_redo': len(game_state['redo_stack']) > 0
            })

    # Model has valid moves - get model's action
    action = get_model_move(env, board)

    # Record move before executing
    board_before = env.board.copy()
    player_color = 'black' if env.player == BLACK else 'white'

    # Execute model's move using low-level method (bypass step)
    # Note: get_next_state automatically switches env.player to the next player
    action_tuple = (0, action // 8, action % 8)
    board = env.get_next_state(board, action_tuple)
    env.board = board

    # Record the move in history
    record_move(action, player_color, board_before)

    game_state['last_move'] = {'action': action, 'player': 'model'}

    # Check if game is over
    winner = env.get_winner(board)
    if winner is not None:
        game_state['game_over'] = True
        if winner == BLACK:
            game_state['winner'] = 'black'
        elif winner == WHITE:
            game_state['winner'] = 'white'
        else:
            game_state['winner'] = 'draw'

        return jsonify({
            'board': board_to_list(board),
            'game_over': True,
            'winner': game_state['winner'],
            'piece_count': count_pieces(board),
            'valid_moves': [],
            'last_move': game_state['last_move']
        })

    # Check if next player (human) has valid moves
    if not env.has_valid(board, env.player):
        # Human has no valid moves - pass back to model
        env.player = -env.player

        # Check if model also has no valid moves
        if not env.has_valid(board, env.player):
            # Neither player has moves - game over
            game_state['game_over'] = True
            winner = env.get_winner(board)
            if winner == BLACK:
                game_state['winner'] = 'black'
            elif winner == WHITE:
                game_state['winner'] = 'white'
            else:
                game_state['winner'] = 'draw'

            return jsonify({
                'board': board_to_list(board),
                'game_over': True,
                'winner': game_state['winner'],
                'piece_count': count_pieces(board),
                'valid_moves': [],
                'last_move': game_state['last_move']
            })

        # Model has moves again, human passed - recurse
        return make_model_move()

    # Game continues - human's turn
    # Capture current player BEFORE any other operations
    current_player = int(env.player)
    valid_moves = get_valid_moves(env, board)

    return jsonify({
        'board': board_to_list(board),
        'current_player': 'black' if current_player == BLACK else 'white',
        'valid_moves': valid_moves,
        'piece_count': count_pieces(board),
        'game_over': False,
        'last_move': game_state['last_move'],
        'can_undo': len(game_state['move_history']) > 0,
        'can_redo': len(game_state['redo_stack']) > 0
    })


@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    """Get current game state."""
    env = game_state['env']
    if env is None:
        return jsonify({'error': 'No game in progress'}), 400

    board = env.board
    current_player = env.player
    valid_moves = get_valid_moves(env, board)
    piece_count = count_pieces(board)

    return jsonify({
        'board': board_to_list(board),
        'current_player': 'black' if current_player == BLACK else 'white',
        'valid_moves': valid_moves,
        'piece_count': piece_count,
        'game_over': game_state['game_over'],
        'winner': game_state['winner'],
        'model_name': game_state['model_name'],
        'can_undo': len(game_state['move_history']) > 0,
        'can_redo': len(game_state['redo_stack']) > 0
    })


@app.route('/api/undo', methods=['POST'])
def undo():
    """Undo the last move."""
    env = game_state['env']
    if env is None:
        return jsonify({'error': 'No game in progress'}), 400

    if len(game_state['move_history']) == 0:
        return jsonify({'error': 'No moves to undo'}), 400

    # Pop last move from history
    last_move = game_state['move_history'].pop()

    # Add to redo stack
    game_state['redo_stack'].append(last_move)

    # Restore board state from before the move
    env.board = last_move['board_before'].copy()

    # Set player to the one who made the undone move
    player_value = BLACK if last_move['player'] == 'black' else WHITE
    env.player = player_value

    # Reset game over state
    game_state['game_over'] = False
    game_state['winner'] = None

    # Get current state
    current_player = int(env.player)
    valid_moves = get_valid_moves(env, env.board)
    piece_count = count_pieces(env.board)

    return jsonify({
        'board': board_to_list(env.board),
        'current_player': 'black' if current_player == BLACK else 'white',
        'valid_moves': valid_moves,
        'piece_count': piece_count,
        'game_over': False,
        'can_undo': len(game_state['move_history']) > 0,
        'can_redo': len(game_state['redo_stack']) > 0
    })


@app.route('/api/redo', methods=['POST'])
def redo():
    """Redo a previously undone move."""
    env = game_state['env']
    if env is None:
        return jsonify({'error': 'No game in progress'}), 400

    if len(game_state['redo_stack']) == 0:
        return jsonify({'error': 'No moves to redo'}), 400

    # Pop from redo stack
    move = game_state['redo_stack'].pop()

    # Re-execute the move
    action = move['action']
    action_tuple = (0, action // 8, action % 8)

    # Record board state before redo
    board_before = env.board.copy()

    # Execute move
    board = env.get_next_state(env.board, action_tuple)
    env.board = board

    # Add back to history
    game_state['move_history'].append({
        'action': action,
        'player': move['player'],
        'board_before': board_before
    })

    # Get current state
    current_player = int(env.player)
    valid_moves = get_valid_moves(env, board)
    piece_count = count_pieces(board)

    # Check if it's now the model's turn and game is not over
    model_color = game_state['model_color']
    is_model_turn = (current_player == model_color)

    response = {
        'board': board_to_list(board),
        'current_player': 'black' if current_player == BLACK else 'white',
        'valid_moves': valid_moves,
        'piece_count': piece_count,
        'game_over': False,
        'can_undo': len(game_state['move_history']) > 0,
        'can_redo': len(game_state['redo_stack']) > 0
    }

    # If it's model's turn after redo, make model move
    if is_model_turn:
        return make_model_move()

    return jsonify(response)


@app.route('/api/get_moves', methods=['GET'])
def get_moves():
    """Get the move history."""
    moves = []
    for i, move in enumerate(game_state['move_history']):
        moves.append({
            'number': i + 1,
            'player': move['player'],
            'action': move['action'],
            'notation': action_to_notation(move['action'])
        })

    return jsonify({
        'moves': moves,
        'total': len(moves)
    })


def main():
    parser = argparse.ArgumentParser(description='Web interface for Reversi')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name to load (without .zip extension)')
    parser.add_argument('-w', '--net-width', type=int, default=512,
                        help='Neural network width (default: 512)')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='Port to run web server (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to (default: 127.0.0.1)')

    args = parser.parse_args()

    # Create environment (needed for loading model)
    env = ReversiEnvCNN()

    # Load model
    print(f"Loading model: {args.model}")
    model = get_model(
        file=f"models/{args.model}",
        env=env,
        net_width=args.net_width,
        device="cpu"
    )

    if model is None:
        print(f"Error: Could not load model models/{args.model}.zip")
        return

    game_state['model'] = model
    game_state['model_name'] = args.model

    print(f"\n{'='*60}")
    print(f"Reversi Web Interface")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Network width: {args.net_width}")
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"Open your browser and navigate to the URL above")
    print(f"{'='*60}\n")

    # Run Flask app
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
