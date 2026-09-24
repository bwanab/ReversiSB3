import warnings
from util.reversi import build_reversi
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # check_env assumes a 3-D observation is an image (uint8, 0-255, >= 36x36) and warns.
    # Our (1, 8, 8) board in {-1, 0, 1} isn't an image: ReversiCNN handles 8x8 and the
    # policy uses normalize_images=False, so those warnings don't apply.
    warnings.filterwarnings("ignore", message="It seems that your observation", module="stable_baselines3")
    warnings.filterwarnings("ignore", message="The minimal resolution for an image", module="stable_baselines3")
    env = build_reversi()
    check_env(env)  # raises on failure
    print("check_env passed")
