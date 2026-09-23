from util.reversi import build_reversi
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    tr = 0
    env = build_reversi()
    check_env(env)

