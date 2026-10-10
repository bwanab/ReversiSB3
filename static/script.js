// Game state
let gameState = {
    board: null,
    validMoves: [],
    currentPlayer: null,
    gameOver: false,
    modelColor: 'black',
    gameId: null,      // from /api/new_game; sent with every request about this game
    gameName: null,    // optional label for this game (e.g. the opponent); asked at its first save
    lastData: null,
    showHints: loadShowHints()
};

// "Show hints": the network's move probabilities and the solver's verdict; remembered per browser
function loadShowHints() {
    try {
        return localStorage.getItem('showHints') !== 'false';
    } catch (e) {
        return true;
    }
}

// Initialize board on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeBoard();
    setupEventListeners();
    loadLevels();
});

async function loadLevels() {
    try {
        const response = await fetch('/api/levels');
        const data = await response.json();
        const select = document.getElementById('level');
        select.innerHTML = '';
        data.levels.forEach(level => {
            const option = document.createElement('option');
            option.value = level.level;
            option.textContent = level.level === 'custom' ? 'Custom' : `${level.level}`;
            option.title = level.strength ? `${level.description}; ${level.strength}` : level.description;
            select.appendChild(option);
        });
        select.value = String(data.default);
        updateLevelTitle();
    } catch (error) {
        showError('Could not load strength levels: ' + error.message);
    }
}

function updateLevelTitle() {
    const select = document.getElementById('level');
    const option = select.options[select.selectedIndex];
    if (option) {
        select.title = option.title;
    }
}

async function changeLevel() {
    const value = document.getElementById('level').value;
    updateLevelTitle();
    if (!gameState.gameId) {
        return;   // applies to the next new game
    }
    try {
        const response = await fetch('/api/set_level', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_id: gameState.gameId, level: value === 'custom' ? 'custom' : Number(value) })
        });
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Failed to change strength');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

function toggleHints() {
    gameState.showHints = document.getElementById('show-hints').checked;
    try {
        localStorage.setItem('showHints', String(gameState.showHints));
    } catch (e) {
        // not stored; the setting still applies to this page
    }
    if (gameState.lastData) {
        updateGameState(gameState.lastData);
    }
}

function setupEventListeners() {
    document.getElementById('new-game-btn').addEventListener('click', startNewGame);
    document.getElementById('undo-btn').addEventListener('click', undoMove);
    document.getElementById('redo-btn').addEventListener('click', redoMove);
    document.getElementById('copy-moves-btn').addEventListener('click', copyMoves);
    document.getElementById('level').addEventListener('change', changeLevel);
    document.getElementById('save-btn').addEventListener('click', saveGame);
    document.getElementById('load-btn').addEventListener('click', () => document.getElementById('load-file').click());
    document.getElementById('load-file').addEventListener('change', loadGameFile);
    document.getElementById('game-name').addEventListener('click', renameGame);
    const hints = document.getElementById('show-hints');
    hints.checked = gameState.showHints;
    hints.addEventListener('change', toggleHints);
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
    const levelValue = document.getElementById('level').value;
    gameState.modelColor = modelColor;

    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_color: modelColor,
                level: levelValue === '' ? null : (levelValue === 'custom' ? 'custom' : Number(levelValue))
            })
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
    if (gameState.gameOver || !gameState.gameId) {
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
            body: JSON.stringify({ game_id: gameState.gameId, action: action })
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
    gameState.lastData = data;
    if (data.game_id) {
        gameState.gameId = data.game_id;
        document.getElementById('save-btn').disabled = false;
    }
    if (data.model_color) {
        gameState.modelColor = data.model_color;
    }
    if (data.game_id) {
        showGameName(data.name || null);
    }
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

    // Highlight last move; when the person had to pass, number the model's moves in order
    if (data.last_move) {
        highlightLastMove(data.last_move.action);
        highlightModelSequence(data.model_moves || []);
        updateLastMoveDisplay(data.last_move, data.model_moves || []);
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
        cell.classList.remove('valid-move', 'last-move', 'model-sequence');

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

            // If probability is available (and hints are on), display it on the cell
            if (gameState.showHints && move.probability !== undefined) {
                const probText = document.createElement('div');
                probText.className = 'move-probability';
                probText.textContent = `${(move.probability * 100).toFixed(1)}%`;
                cell.appendChild(probText);
            }
        }
    });
}

function highlightLastMove(action) {
    const cell = document.querySelector(`[data-action="${action}"]`);
    if (cell) {
        cell.classList.add('last-move');
    }
}

function highlightModelSequence(moves) {
    if (moves.length < 2) {
        return;
    }
    moves.forEach((action, i) => {
        const cell = document.querySelector(`[data-action="${action}"]`);
        if (!cell) {
            return;
        }
        if (i < moves.length - 1) {
            cell.classList.add('model-sequence');   // earlier moves; the last keeps the last-move color
        }
        const badge = document.createElement('div');
        badge.className = 'move-order';
        badge.textContent = String(i + 1);
        cell.appendChild(badge);
    });
}

function actionNotation(action) {
    return String.fromCharCode(65 + (action % 8)) + (Math.floor(action / 8) + 1);
}

// The game's name: shown above the board, asked at the first save, kept with the game on the server
function showGameName(name) {
    gameState.gameName = name;
    const el = document.getElementById('game-name');
    el.textContent = name ? `Game: ${name} ✎` : '';
    el.hidden = !name;
}

const NAME_PATTERN = /^[\p{L}\p{N}_ .'-]{1,40}$/u;

async function setGameName(name) {
    const response = await fetch('/api/set_name', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ game_id: gameState.gameId, name: name })
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || 'Failed to name the game');
    }
    showGameName(data.name);
}

// Ask for a name; returns the cleaned name, '' for none, or null if canceled
function askName(current) {
    const answer = prompt("Name for this game (e.g. your opponent's name):", current || '');
    if (answer === null) {
        return null;
    }
    const name = answer.trim();
    if (name && !NAME_PATTERN.test(name)) {
        showError("Names: up to 40 letters, digits, spaces and _ . ' -");
        return null;
    }
    return name;
}

async function renameGame() {
    if (!gameState.gameId) {
        return;
    }
    const name = askName(gameState.gameName);
    if (name === null) {
        return;
    }
    try {
        await setGameName(name);
    } catch (error) {
        showError(error.message);
    }
}

// a name made safe for a file name: spaces -> _, nothing but letters, digits, _ . -
function fileSafe(name) {
    return name.replace(/\s+/g, '_').replace(/[^\p{L}\p{N}_.-]/gu, '');
}

// Save: download the game's moves, model color, level and name (GET /api/export) as a small JSON
// file named reversi-<name>-m<moves>.json; the first save of an unnamed game asks for a name
async function saveGame() {
    if (!gameState.gameId) {
        return;
    }
    if (!gameState.gameName) {
        const name = askName('');
        if (name) {
            try {
                await setGameName(name);
            } catch (error) {
                showError(error.message);
                return;
            }
        }
    }
    try {
        const response = await fetch('/api/export?game_id=' + encodeURIComponent(gameState.gameId));
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Failed to save the game');
            return;
        }
        const moveNumber = String(data.move_number).padStart(2, '0');
        const label = data.name ? fileSafe(data.name) : '';
        const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = label ? `reversi-${label}-m${moveNumber}.json` : `reversi-m${moveNumber}.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

// The game name in a file name: reversi-<name>-m<NN>.json (browsers may add " (1)"), or for a move
// list in a text file, the file's own name; older saves (reversi-<date>-<time>.json) have none
function nameFromFileName(fileName) {
    const base = fileName.replace(/\s*\(\d+\)(?=\.[^.]*$)/, '');
    const saved = base.match(/^reversi-(.+)-m\d+\.json$/i);
    if (saved) {
        return saved[1].replace(/_/g, ' ');
    }
    if (/^reversi-(m\d+|\d{8}-\d{4})\.json$/i.test(base)) {
        return null;
    }
    const plain = base.match(/^(.+)\.txt$/i);
    return plain ? plain[1].replace(/_/g, ' ') : null;
}

// Load: a saved file continues with its own model color, level and name; a plain move list
// (e.g. f5d6c3) uses the color and level chosen in the controls
async function loadGameFile(event) {
    const file = event.target.files[0];
    event.target.value = '';             // allow loading the same file again
    if (!file) {
        return;
    }
    if (file.size > 4000) {
        showError('That file is too large to be a saved game');
        return;
    }
    const text = await file.text();
    let request;
    try {
        const saved = JSON.parse(text);
        if (!saved || typeof saved.moves === 'undefined') {
            throw new Error('not a saved game');
        }
        request = { moves: saved.moves, model_color: saved.model_color, level: saved.level, name: saved.name };
    } catch (e) {
        const levelValue = document.getElementById('level').value;
        request = {
            moves: text.trim(),
            model_color: document.getElementById('model-color').value,
            level: levelValue === '' ? null : (levelValue === 'custom' ? 'custom' : Number(levelValue))
        };
    }
    if (!request.name) {
        const fromFile = nameFromFileName(file.name);
        request.name = fromFile && NAME_PATTERN.test(fromFile) ? fromFile : null;
    }
    try {
        const response = await fetch('/api/load_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Failed to load the game');
            return;
        }
        document.getElementById('model-color').value = data.model_color;
        if (data.level !== undefined && data.level !== null) {
            document.getElementById('level').value = String(data.level);
            updateLevelTitle();
        }
        document.getElementById('game-status').textContent = '';
        document.getElementById('last-move').textContent = '-';
        updateGameState(data);
    } catch (error) {
        showError('Network error: ' + error.message);
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

function updateLastMoveDisplay(lastMove, modelMoves = []) {
    const lastMoveDiv = document.getElementById('last-move');
    const action = lastMove.action;
    const row = Math.floor(action / 8);
    const col = action % 8;

    const colLetter = String.fromCharCode(65 + col); // A-H
    const rowNumber = row + 1; // 1-8

    const player = lastMove.player === 'model' ? 'Model' : 'You';
    let displayText = '';
    if (modelMoves.length > 1) {
        displayText = `You had no move, so the model played ${modelMoves.length} moves: `
            + modelMoves.map(actionNotation).join(', then ') + '\n';
    }
    displayText += `${player}: ${colLetter}${rowNumber}`;

    // If model move with analysis, show top move probabilities
    if (gameState.showHints && lastMove.player === 'model' && lastMove.analysis && lastMove.analysis.length > 0) {
        displayText += '\n';
        const moveProbabilities = lastMove.analysis.map(move => {
            const percentage = (move.probability * 100).toFixed(1);
            const isChosen = move.action === action;
            return isChosen ? `${move.notation}: ${percentage}% ★` : `${move.notation}: ${percentage}%`;
        });
        displayText += moveProbabilities.join(', ');
    }

    // How the model chose (network / search / solver), and the solver's exact verdict
    if (lastMove.player === 'model' && lastMove.method) {
        displayText += `\nChosen by: ${lastMove.method}`;
        if (gameState.showHints && lastMove.exact_score !== undefined) {
            const s = lastMove.exact_score;
            displayText += s > 0 ? ` (model wins by ${s} with perfect play)`
                         : s < 0 ? ` (model loses by ${-s} with perfect play)` : ' (draw with perfect play)';
        }
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
            },
            body: JSON.stringify({ game_id: gameState.gameId })
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
            },
            body: JSON.stringify({ game_id: gameState.gameId })
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
        const response = await fetch('/api/get_moves?game_id=' + encodeURIComponent(gameState.gameId));
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
