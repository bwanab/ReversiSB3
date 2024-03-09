
import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, board_player_from_state, random_action, render

import numpy as np
from operator import itemgetter
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env: gym.Env) -> np.ndarray:
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask + [0], dtype=np.int8)

def get_action(model, obs, mask):
    action, _ = model.predict(obs, action_masks=mask, deterministic=False)
    return action


def play_games(file, num_games=100, verbose=False):
    # Create environment
    env = gym.make("Reversi-v0")
    # env.action_space.sample = sample_factory(env)

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one

    model = MaskablePPO.load(file, env=env)
    model.policy = MaskableActorCriticPolicy.load(file + '_policy.zip')

    # model = MaskablePPO(MaskableActorCriticPolicy, env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper  before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    black_wins = 0

    for i in range(num_games):
        term = False
        model_count = 0

        while not term:
            _,player = board_player_from_state(obs[0])
            if player == -1:
                # print("======  RAI  ====== ")
                action = random_action(env, obs[0])
                # action = reversi_ai_action(env, obs[0])
            else:
                # print("====== Model ====== ")
                action = get_action(model, obs, mask_fn(env))
                model_count += 1
            # print(action)
            obs, rewards, term, info = vec_env.step(action)
            if term:
                term_obs = info[0]['terminal_observation']
                black_score = sum(term_obs == 1)
                white_score = sum(term_obs == -1)
                if verbose:
                    print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards[0]}")
                    render(term_obs)
                black_wins += black_score > white_score
            # vec_env.render()
    return black_wins

n_games = 10
black_wins = play_games("ppo_reversi_test", n_games, verbose=True)
print(f"Black wins: {100 * black_wins / n_games}%")
