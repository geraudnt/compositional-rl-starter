import gym
import gym_minigrid
from utils.wrappers import *


def make_env(env_key, seed=None, obs_type='partial'):
    env = gym.make(env_key)
    env.seed(seed)
    env = FixEnvAndRGBImg(env, obs_type=obs_type)
    return env
