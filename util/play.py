from util.util import get_move_db, render, get_action, mask_fn, get_move_notation, count_players, BLACK
import numpy as np
import torch

def random_opening(env, plies, rng):
    """Play `plies` random legal moves (both sides, passes handled) from the current
    position, then continue randomly until it is BLACK's turn with a legal move.

    Used to evaluate from varied starting positions, so a model can't rely on
    replaying lines memorized against a deterministic opponent. Mutates env.board and
    env.player. Returns False if the game ended during the opening.
    """
    board = env.board
    played = 0
    while played < plies or env.player != BLACK or not env.has_valid(board, BLACK):
        if env.get_winner(board) is not None:
            return False
        if not env.has_valid(board, env.player):
            env.player = -env.player
            continue
        a = int(rng.choice(env._all_valid_actions(board, env.player)))
        env.get_next_state(board, (0, a // 8, a % 8))  # flips env.player
        played += 1
    return True

def play(model, num_games, opponent, deterministic, verbose, random_opening_plies=0, seed=None,
         return_draws=False, env=None):
    """Play num_games with the model as BLACK against the env's opponent. Returns BLACK's wins,
    or (wins, draws) with return_draws=True. Games are played on `env` if given, otherwise on
    the model's own (single) env."""
    env = env.unwrapped if env is not None else model.get_env().envs[0].unwrapped
    # obs = vec_env.reset()
    obs, _ = env.reset(seed=seed)  # seeds the env's start-position sampling, if it has any
    black_wins = 0
    draws = 0
    rng = np.random.default_rng(seed)

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        move_db = get_move_db()

    for i in range(num_games):
        # Reset environment for each new game
        obs, _ = env.reset()
        if random_opening_plies:
            while not random_opening(env, random_opening_plies, rng):
                obs, _ = env.reset()
        moves = []
        term = False

        per_player = {1: 2, -1: 2}
        while not term:
            board = obs
            player = env.player
            if verbose:
                pass
                render(board)
            action, probs, actions = get_action(model, board, mask_fn(env), deterministic=deterministic, verbose=verbose)
            if verbose:
                m = torch.nn.Softmax(dim=0)
                print(action, actions, m(probs).detach().numpy())
            if verbose:
                mn = get_move_notation(env, player, action)
                moves.append(mn)
                # print(mn)
            obs, rewards, term, _truncated, info = env.step(action)
            if verbose:
                b = obs
                per_player_diff = count_players(per_player, b)
                # print(per_player_diff)

            if term:
                # Get terminal observation - might be in info or just use obs
                term_obs = info.get('terminal_observation', obs)
                black_score = np.sum(term_obs == 1)
                white_score = np.sum(term_obs == -1)
                if verbose:
                    print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards}")
                    render(term_obs)
                    moves_str = "".join(moves)
                    print(moves_str)
                    for m, desc in move_db:
                        if m == moves_str[:len(m)]:
                            print(m, desc)
                black_wins += black_score > white_score
                draws += black_score == white_score
    return (black_wins, draws) if return_draws else black_wins
