import gym
import gym_minigrid
from utils.wrappers import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    env = FixEnv(env)
    return env
