import numpy as np
from util.util import render, mask_fn, get_model, get_action
from util.reversi import build_reversi
from util.reversi_cnn import ReversiCNN

test_board = np.array([[[ 1, -1, -1,  1,  1,  1,  1,  1],
                   [ 1,  1, -1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1, -1,  1, -1,  1,  1],
                   [ 1, -1,  1,  1, -1,  1,  1,  1],
                   [ 1, -1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1, -1, -1, -1, -1, -1,  1],
                   [ 1,  1,  1,  1,  1,  1,  1,  0],
                   [-1,  1,  1,  1,  1,  1,  0,  1]]], dtype=np.int8) 

def sanity6():
    env = build_reversi()
    model = get_model("models/dork8_CNN_test", env, device="mps")
    _, _ = env.reset()
    env.env.board = test_board
    action, _, _ = get_action(model, env.board, mask_fn(env), verbose = True)
    assert(action < 0)

if __name__ == "__main__":
    sanity6()
