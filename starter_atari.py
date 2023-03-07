import supersuit
from pettingzoo.atari import pong_v3
import numpy as np
import torch
from agent import Agent

NUM_OF_EPISODE = 100


# Create a PettingZoo environment for Pong
# env = pong_v3.env(auto_rom_install_path="/research/dept8/fyp22/lhf2205/miniconda3/envs/fyp/lib/python3.10/site-packages/AutoROM/")
env = pong_v3.env(obs_type='grayscale_image')


# Preprocessing of the atari env
env = supersuit.max_observation_v0(env, 2)
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
env = supersuit.frame_skip_v0(env, 4)
env = supersuit.frame_stack_v1(env, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creating agents
agents = [None] * env.num_agents

for episode in range(NUM_OF_EPISODE):
    # Run one episode
    # Reset the environment and get the initial observation
    env.reset()

    # Define the action space for each player
    #action_spaces = [env.action_space(i) for i in ['first_0', 'second_0']] # 'first_0', 'second_0' are the agents' name in this env.
    #print(type(env))

    # Play the game for a few steps with random actions for both players
    for i in range(1):
        # Choose a random action for each player
        actions = [action_space.sample() for action_space in action_spaces] #Randomize an action for both two agents

        obs, reward, done, trunc, info = env.last()

        # Step the environment with the chosen actions
        env.step(actions[0]) # First agent step
        env.step(actions[1]) # Second agent step

        # Print the current observation and reward
        print(f"Observation: {np.array(obs).shape}")
        print(f"Reward: {reward}")

        # If the game is over, reset the environment
        if done:
            obs = env.reset()
