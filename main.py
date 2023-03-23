from training import *
from wrappers import make_pong

if __name__ == '__main__':
    NUM_OF_EPISODE = 600
    DQN_HYPERPARAMS = {
        'eps_start': 1,
        'eps_end': 0.1,
        'eps_decay': 10 ** 5,
        'buffer_size': 3000,
        'buffer_minimum': 1001,
        'learning_rate': 5e-5,
        'gamma': 0.9,
        'n_iter_update_nn': 1000,
        'batch_size': 32
    }

    env = make_pong()
    train_stationary(env, DQN_HYPERPARAMS, NUM_OF_EPISODE, BATCH_SIZE=DQN_HYPERPARAMS['batch_size'])

