from training import *
from wrappers import make_pong

if __name__ == '__main__':
    NUM_OF_EPISODE = 50000
    DQN_HYPERPARAMS = {
        'eps_start': 1,
        'eps_end': 0.1,
        'eps_decay': 10 ** 5,
        'buffer_size': 3000,
        'buffer_minimum': 1001,
        'learning_rate': 5e-5,
        'gamma': 0.9,
        'n_iter_update_nn': 1000,
    }

    env = make_pong()
    train_basic(env, DQN_HYPERPARAMS, NUM_OF_EPISODE)

