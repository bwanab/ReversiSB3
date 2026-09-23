from util.util import render
from util.reversi import build_reversi
import numpy as np

def sanity3(env):
    observation, _ = env.reset()
    while True:
        action = env.get_sample(observation)
        reward = 0
        last_obs = np.copy(observation)
        termination = False
        while not termination:
            observation, reward, termination, truncated, info = env.step(action)
            render(observation)
            if np.all(observation == last_obs):
                print("all the same")
            last_obs = np.copy(observation)
            action = env.get_sample(observation)
        if termination:
            render(env.board)
            print(f"Black score: {np.sum(env.board == 1)}, white score: {np.sum(env.board == -1)}")
            break
    env.close()
    return reward

if __name__ == "__main__":
    tr = 0
    env = build_reversi()
    for i in range(1):
        tr += sanity3(env)
    print(f"total reward: {tr}")
