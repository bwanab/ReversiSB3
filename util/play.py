from util.util import get_move_db, render, get_action, mask_fn, get_move_notation, count_players
from util.opponents import Opponent
import numpy as np
import torch

def play(model, num_games, opponent, deterministic, verbose):
    vec_env = model.get_env()
    env = vec_env.envs[0].unwrapped
    # obs = vec_env.reset()
    obs, _ = env.reset()
    black_wins = 0

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        move_db = get_move_db()

    for i in range(num_games):
        moves = []
        term = False

        per_player = {1: 2, -1: 2}
        while not term:
            board = obs
            player = env.player
            if verbose:
                render(board)
            action, probs, actions = get_action(model, board, mask_fn(env), deterministic=deterministic, verbose=verbose)
            if verbose:
                m = torch.nn.Softmax(dim=0)
                print(action, actions, m(probs).detach().numpy())
            if verbose:
                mn = get_move_notation(env, player, action)
                moves.append(mn)
                print(mn)
            obs, rewards, term, _truncated, info = env.step(action)
            if verbose:
                b = obs
                per_player_diff = count_players(per_player, b)
                print(per_player_diff)

            if term:
                term_obs = info['terminal_observation']
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
    return black_wins

def alt_play(env, num_games, black_player: Opponent, white_player: Opponent):
    black_wins = 0
    for i in range(num_games):
        term = False
        obs = env.reset()[0]
        while not term:
            player = env.player
            if player == 1:
                action = black_player.get_action(env, obs)
            else:
                action = white_player.get_action(env, obs)
            
            obs, score, term, something, info = env.step(action[0])
            if (score != 0) and (term == False):
                print("score != and term false!") 
            if term:
                # term_obs = info[0]['terminal_observation']
                term_obs = obs
                black_score = sum(term_obs == 1)
                white_score = sum(term_obs == -1)
                black_wins += black_score > white_score
    return black_wins
