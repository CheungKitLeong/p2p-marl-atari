from pettingzoo.atari import pong_v3
import supersuit
import numpy as np

def make_env():
    env = pong_v3.env()
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.reshape_v0(env, (84, 84, 1))
    env = supersuit.dtype_v0(env, dtype=np.float32)
    env = supersuit.color_reduction_v0(env)
    env = supersuit.frame_stack_v1(env)
    env = supersuit.normalize_obs_v0(env)
    return env
