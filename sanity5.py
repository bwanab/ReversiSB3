from util.util import render, mask_fn
from util.reversi import ReversiEnvCNN
from util.reversi_cnn import ReversiCNN
from sb3_contrib import MaskablePPO
import numpy as np

def get_action(env, obs, m):
    mask = mask_fn(env)
    action, _ = m.predict(obs, action_masks=mask, deterministic=False)
    return action

def sanity5():
    policy_kwargs = dict(
        features_extractor_class=ReversiCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False
        )
    env = ReversiEnvCNN.build_reversi()
    observation, _ = env.reset()
    m = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
    while True:
        action = get_action(env, observation, m)
        reward = 0
        termination = False
        while not termination:
            observation, reward, termination, truncated, info = env.step(action)
            render(observation)
            if env.player == 1:
                action = get_action(env, observation, m)
            else:
                action = env.get_sample(observation)
        if termination:
            render(env.board)
            print(f"Black score: {np.sum(env.board == 1)}, white score: {np.sum(env.board == -1)}")
            break
    env.close()
    return reward


if __name__ == "__main__":
    sanity5()