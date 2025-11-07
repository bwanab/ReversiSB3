from util.reversi import ReversiEnvCNN
from util.util import render
import numpy as np

def sanity2():
    env = ReversiEnvCNN.build_reversi()
    obs, _ = env.reset()
    action = env.get_sample(obs)
    observation, reward, termination, truncated, info = env.step(action)
    print(observation)
    render(env.board)

if __name__ == "__main__":
    sanity2()