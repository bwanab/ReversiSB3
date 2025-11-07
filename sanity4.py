from util.reversi import ReversiEnvCNN
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    tr = 0
    env = ReversiEnvCNN.build_reversi()
    check_env(env)

