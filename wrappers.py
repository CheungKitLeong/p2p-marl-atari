from pettingzoo.atari import pong_v3, boxing_v2
import supersuit
import numpy as np


def make_pong(render=False):
    if render:
        env = pong_v3.env(obs_type='grayscale_image', render_mode='human')
    else:
        env = pong_v3.env(obs_type='grayscale_image')

    # Preprocessing of the atari env
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.dtype_v0(env, dtype=np.float32)
    env = supersuit.frame_stack_v1(env)
    env = supersuit.normalize_obs_v0(env)
    # env = supersuit.reshape_v0(env, (4, 84, 84)) # It is broken
    return env

def make_boxing(render=False):
    if render:
        env = boxing_v2.env(obs_type='grayscale_image', render_mode='human')
    else:
        env = boxing_v2.env(obs_type='grayscale_image')

    # Preprocessing of the atari env
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.dtype_v0(env, dtype=np.float32)
    env = supersuit.frame_stack_v1(env)
    env = supersuit.normalize_obs_v0(env)
    # env = supersuit.reshape_v0(env, (4, 84, 84)) # It is broken
    return env


def custom_reshape(obs):
    return obs.transpose(-1, 0, 1)
