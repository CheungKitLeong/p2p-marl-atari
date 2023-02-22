import supersuit
from pettingzoo.atari import pong_v3
import numpy as np

# Create a PettingZoo environment for Pong
env = pong_v3.env()
env = supersuit.max_observation_v0(env, 2)
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
env = supersuit.frame_skip_v0(env, 4)
env = supersuit.frame_stack_v1(env, 4)

# Run one episode
# Reset the environment and get the initial observation
obs = env.reset()

# Define the action space for each player
action_spaces = [env.action_spaces[f'player_{i}'] for i in range(1, 3)]

# Play the game for a few steps with random actions for both players
for i in range(100):
    # Choose a random action for each player
    actions = [action_space.sample() for action_space in action_spaces]

    # Step the environment with the chosen actions
    obs, reward, done, info = env.step({f'player_{i+1}': action for i, action in enumerate(actions)})

    # Print the current observation and reward
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")

    # If the game is over, reset the environment
    if done:
        obs = env.reset()
