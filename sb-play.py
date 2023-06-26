
import gymnasium as gym
import boardgame2
from util.util import reversi_ai_action, board_player_from_state, random_action

import numpy as np
from operator import itemgetter
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    mask = env.get_valid(env.board).reshape(64).tolist()
    return np.array(mask + [0], dtype=np.int8)

def get_action(model, obs, mask):
    action, _ = model.predict(obs, action_masks=mask, deterministic=False)
    return action

def ga(model, obs, mask):
    return get_prediction(model, obs, mask)[0]

def eval_predictions(model, obs, mask):
    v = [ga(model, obs, mask) for x in range(1000)]
    d = {}
    for x in v:
        d[x] = d.get(x, 0) + 1
    return d

def eval_probs(model, obs):
    with torch.no_grad():
        obj_tensor, _ = model.q_net.obs_to_tensor(obs)
        q_values = model.q_net(obj_tensor)
    probs = q_values[0][0:64].numpy()

    d = eval_predictions(model, obs)

    cp = [(x, probs[x], d[x]) for x in d.keys()]
    for x in sorted(cp, key=itemgetter(1), reverse=True):
        print(x)

# def sample_factory(e):
#     return lambda: np.random.choice(e.all_valid_actions(e.board))

n_predict_loops = []

def get_prediction(model, obs, env, mask):
    v = set(env.all_valid_actions(obs[0]))
    good = False
    for i in range(1000):
        action = get_action(model, obs, mask)
        if action[0] in v:
            good = True
            break
    n_predict_loops.append(i)
    return action, good

# Create environment
env = gym.make("Reversi-v0")
# env.action_space.sample = sample_factory(env)


# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one

model = MaskablePPO.load("ppo_reversi", env=env)
model.policy = MaskableActorCriticPolicy.load('ppo_reversi_policy.zip')

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

for i in range(100):
    term = False
    model_count = 0

    while not term:
        _,player = board_player_from_state(obs[0])
        if player == -1:
            # print("======  RAI  ====== ")
            action = np.array([np.ravel_multi_index(random_action(obs[0]), env.board_shape)])
        else:
            # print("====== Model ====== ")
            mask = mask_fn(env)
            action, good = get_prediction(model, obs, env, mask)
            if not good:
                d = eval_predictions(model, obs, mask)
                cp = [(x, d[x]) for x in d.keys()]
                print(env.all_valid_actions(obs[0]))
                for x in sorted(cp, key=itemgetter(1), reverse=True):
                    print(x)

                print("bad action")
            model_count += 1
        # print(action)
        obs, rewards, term, info = vec_env.step(action)
        _,new_player = board_player_from_state(obs[0])
        if rewards[0] < 0:
            print("ltz reward")
        if term:
            term_obs = info[0]['terminal_observation']
            black_score = sum(term_obs == 1)
            white_score = sum(term_obs == -1)
            print(f"Black: {black_score}, White: {white_score}, actions: {black_score + white_score}, Reward: {rewards[0]}")
            black_wins += black_score > white_score
        # vec_env.render()
print(f"Black wins: {black_wins} average predict loop: {sum(n_predict_loops) / len(n_predict_loops)} total predict loops: {sum(n_predict_loops)}")


