import numpy as np
from util.util import render, mask_fn, get_model, get_action, WHITE, BLACK
from util.reversi import build_reversi
from util.reversi_cnn import ReversiCNN
from util.opponents import get_opponent

def sanity7(env):
    model = get_model("models/dork8_CNN_test", env, device="mps")
    _, _ = env.reset()
    while True:
        action, _, _ = get_action(model, env.board, mask_fn(env), verbose = True)
        if action >= 0:
            # m_act = (0, action // 8, action % 8)
            # next_state, reward, termination, info = env.next_step(env.board, m_act)
            next_state, reward, termination, truncate, info = env.step(action)
            render(next_state)
            if termination:
                state = info['terminal_observation']
                winner = env.get_winner(state)
                print(f"Winner is {winner}")
                break
            # else:
            #     action = opponent.get_action(env, env.board)
            #     next_state, reward, termination, truncate, info = env.step(action)
            #     render(env.board)
            #     if termination:
            #         state = info['terminal_observation']
            #         winner = env.get_winner(state)
            #         print(f"winner is {winner}")
            #         break

if __name__ == "__main__":
    env = build_reversi(opponent="Human")
    for i in range(100):
        sanity7(env)
