from util.reversi import ReversiEnvCNN

def sanity1():
    env = ReversiEnvCNN.build_reversi()
    obs, _ = env.reset()
    print(env.get_valid(obs))
    print(env.all_valid_actions(obs))

if __name__ == "__main__":
    sanity1()