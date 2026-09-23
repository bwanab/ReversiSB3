from util.reversi import build_reversi
from util.util import render
import numpy as np

def sanity2():
    env = build_reversi()
    obs, _ = env.reset()
    action = env.get_sample(obs)
    observation, reward, termination, truncated, info = env.step(action)
    print(observation)
    render(env.board)

if __name__ == "__main__":
    sanity2()