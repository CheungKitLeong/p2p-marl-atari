import supersuit
from pettingzoo.atari import pong_v3
import numpy as np
import torch
from agent import Agent

NUM_OF_EPISODE = 500
MAX_STEP = 10000
BATCH_SIZE = 32

# Create a PettingZoo environment for Pong
# env = pong_v3.env(auto_rom_install_path="/research/dept8/fyp22/lhf2205/miniconda3/envs/fyp/lib/python3.10/site-packages/AutoROM/")
env = pong_v3.env(obs_type='grayscale_image')


# Preprocessing of the atari env
env = supersuit.max_observation_v0(env, 2)
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
env = supersuit.frame_skip_v0(env, 4)
env = supersuit.max_observation_v0(env, 2)
env = supersuit.resize_v1(env, 84, 84)
env = supersuit.reshape_v0(env, (1, 84, 84))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DQN_HYPERPARAMS = {
    'eps_start': 1,
    'eps_end': 0.02,
    'eps_decay': 10 ** 5,
    'buffer_size': 3000,
    'buffer_minimum': 1001,
    'learning_rate': 5e-5,
    'gamma': 0.9,
    'n_iter_update_nn': 1000,
}
print(device)
# Creating agents
agents = [None] * 2
agents[0] = Agent(env, DQN_HYPERPARAMS, device, 'first_0')
agents[1] = Agent(env, DQN_HYPERPARAMS, device, 'second_0')

for episode in range(NUM_OF_EPISODE):
    # Run one episode
    # Reset the environment and get the initial observation
    env.reset()

    # Play the game for a few steps with random actions for both players
    for i in range(MAX_STEP):
        # Choose a random action for each player
        # actions = [action_space.sample() for action_space in action_spaces] #Randomize an action for both two agents

        for n in range(2):
            obs, reward, done, trunc, info = env.last()
            action = agents[n].select_eps_greedy_action(obs)
            env.step(action)

            new_obs, reward, done, trunc, info = env.last()
            agents[n].add_to_buffer(obs, action, new_obs, reward, (done or trunc))
            agents[n].sample_and_improve(BATCH_SIZE)

            # If the game is over, reset the environment
            if done or trunc:
                break

        if done or trunc:
            break

    agents[0].print_info()
    agents[1].print_info()
    agents[0].reset_parameters()
    agents[1].reset_parameters()
