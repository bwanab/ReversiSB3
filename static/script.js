// Game state
let gameState = {
    board: null,
    validMoves: [],
    currentPlayer: null,
    gameOver: false,
    modelColor: 'black'
};

// Initialize board on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeBoard();
    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('new-game-btn').addEventListener('click', startNewGame);
    document.getElementById('undo-btn').addEventListener('click', undoMove);
    document.getElementById('redo-btn').addEventListener('click', redoMove);
    document.getElementById('copy-moves-btn').addEventListener('click', copyMoves);
}

function initializeBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';

    // Create 64 cells (8x8)
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.dataset.action = row * 8 + col;

            cell.addEventListener('click', () => handleCellClick(row, col));

            board.appendChild(cell);
        }
    }
}

async function startNewGame() {
    const modelColor = document.getElementById('model-color').value;
    gameState.modelColor = modelColor;

    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_color: modelColor })
        });

        const data = await response.json();

        if (response.ok) {
            updateGameState(data);
        } else {
            showError(data.error || 'Failed to start new game');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

async function handleCellClick(row, col) {
    if (gameState.gameOver) {
        return;
    }

    const action = row * 8 + col;

    // Check if this is a valid move
    const isValid = gameState.validMoves.some(move => move.action === action);

    if (!isValid) {
        return;
    }

    // Make the move
    try {
        const response = await fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action: action })
        });

        const data = await response.json();

        if (response.ok) {
            updateGameState(data);
        } else {
            showError(data.error || 'Failed to make move');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

function updateGameState(data) {
    gameState.board = data.board;
    gameState.validMoves = data.valid_moves || [];
    gameState.currentPlayer = data.current_player;
    gameState.gameOver = data.game_over || false;

    // Update board display
    renderBoard(data.board);

    // Update scores
    if (data.piece_count) {
        document.getElementById('black-score').textContent = data.piece_count.black;
        document.getElementById('white-score').textContent = data.piece_count.white;
    }

    // Update turn indicator
    updateTurnIndicator();

    // Highlight valid moves
    highlightValidMoves();

    // Highlight last move
    if (data.last_move) {
        highlightLastMove(data.last_move.action);
        updateLastMoveDisplay(data.last_move);
    }

    // Update undo/redo button states
    updateUndoRedoButtons(data);

    // Check for game over
    if (gameState.gameOver) {
        showGameOver(data.winner, data.piece_count);
        displayGameRecord();
    } else {
        // Hide game record if not game over
        document.getElementById('game-record').style.display = 'none';
    }
}

function renderBoard(board) {
    const cells = document.querySelectorAll('.cell');

    cells.forEach((cell, index) => {
        const row = Math.floor(index / 8);
        const col = index % 8;
        const value = board[row][col];

        // Remove existing piece
        cell.innerHTML = '';

        // Remove highlighting classes
        cell.classList.remove('valid-move', 'last-move');

        // Add piece if present
        if (value === 1) {
            // Black piece
            const piece = document.createElement('div');
            piece.className = 'piece black-piece';
            cell.appendChild(piece);
        } else if (value === -1) {
            // White piece
            const piece = document.createElement('div');
            piece.className = 'piece white-piece';
            cell.appendChild(piece);
        }
    });
}

function highlightValidMoves() {
    gameState.validMoves.forEach(move => {
        const action = move.action;
        const cell = document.querySelector(`[data-action="${action}"]`);
        if (cell) {
            cell.classList.add('valid-move');
        }
    });
}

function highlightLastMove(action) {
    const cell = document.querySelector(`[data-action="${action}"]`);
    if (cell) {
        cell.classList.add('last-move');
    }
}

function updateTurnIndicator() {
    const turnText = document.getElementById('turn-text');

    if (gameState.gameOver) {
        turnText.textContent = 'Game Over';
        return;
    }

    const isModelTurn = gameState.currentPlayer === gameState.modelColor;

    if (isModelTurn) {
        turnText.textContent = `Model's turn (${capitalizeFirst(gameState.currentPlayer)})`;
    } else {
        turnText.textContent = `Your turn (${capitalizeFirst(gameState.currentPlayer)})`;
    }
}

function updateLastMoveDisplay(lastMove) {
    const lastMoveDiv = document.getElementById('last-move');
    const action = lastMove.action;
    const row = Math.floor(action / 8);
    const col = action % 8;

    const colLetter = String.fromCharCode(65 + col); // A-H
    const rowNumber = row + 1; // 1-8

    const player = lastMove.player === 'model' ? 'Model' : 'You';
    let displayText = `${player}: ${colLetter}${rowNumber}`;

    // If model move with analysis, show top move probabilities
    if (lastMove.player === 'model' && lastMove.analysis && lastMove.analysis.length > 0) {
        displayText += '\n';
        const moveProbabilities = lastMove.analysis.map(move => {
            const percentage = (move.probability * 100).toFixed(1);
            const isChosen = move.action === action;
            return isChosen ? `${move.notation}: ${percentage}% ★` : `${move.notation}: ${percentage}%`;
        });
        displayText += moveProbabilities.join(', ');
    }

    lastMoveDiv.textContent = displayText;
}

function showGameOver(winner, pieceCount) {
    const statusDiv = document.getElementById('game-status');
    statusDiv.classList.add('winner');

    let message = '';
    if (winner === 'draw') {
        message = `Game Over - Draw! (${pieceCount.black}-${pieceCount.white})`;
    } else {
        const winnerText = capitalizeFirst(winner);
        message = `Game Over - ${winnerText} wins! (${pieceCount.black}-${pieceCount.white})`;
    }

    statusDiv.textContent = message;
}

function showError(message) {
    const statusDiv = document.getElementById('game-status');
    statusDiv.textContent = 'Error: ' + message;
    statusDiv.style.background = '#f56565';
    statusDiv.style.color = 'white';

    setTimeout(() => {
        statusDiv.textContent = '';
        statusDiv.style.background = '';
        statusDiv.style.color = '';
    }, 3000);
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

async function undoMove() {
    try {
        const response = await fetch('/api/undo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (response.ok) {
            updateGameState(data);
        } else {
            showError(data.error || 'Failed to undo');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

async function redoMove() {
    try {
        const response = await fetch('/api/redo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (response.ok) {
            updateGameState(data);
        } else {
            showError(data.error || 'Failed to redo');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

function updateUndoRedoButtons(data) {
    const undoBtn = document.getElementById('undo-btn');
    const redoBtn = document.getElementById('redo-btn');

    // Enable/disable based on server response
    if (data.can_undo !== undefined) {
        undoBtn.disabled = !data.can_undo;
    }

    if (data.can_redo !== undefined) {
        redoBtn.disabled = !data.can_redo;
    }
}

async function displayGameRecord() {
    try {
        const response = await fetch('/api/get_moves');
        const data = await response.json();

        if (response.ok) {
            const moveListDiv = document.getElementById('move-list');
            const gameRecordDiv = document.getElementById('game-record');

            // Format moves
            let movesText = '';
            data.moves.forEach(move => {
                const playerIcon = move.player === 'black' ? '⚫' : '⚪';
                movesText += `${move.number}. ${playerIcon} ${move.notation}  `;
                if (move.number % 5 === 0) movesText += '\n';
            });

            moveListDiv.textContent = movesText;
            gameRecordDiv.style.display = 'block';

            // Store moves for copying
            gameRecordDiv.dataset.moves = data.moves.map(m => m.notation).join(', ');
        }
    } catch (error) {
        console.error('Failed to fetch moves:', error);
    }
}

function copyMoves() {
    const gameRecordDiv = document.getElementById('game-record');
    const movesText = gameRecordDiv.dataset.moves;

    if (movesText) {
        navigator.clipboard.writeText(movesText).then(() => {
            const btn = document.getElementById('copy-moves-btn');
            const originalText = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => {
                btn.textContent = originalText;
            }, 2000);
        }).catch(() => {
            showError('Failed to copy to clipboard');
        });
    }
}
