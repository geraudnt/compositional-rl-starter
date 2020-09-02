import gym
import babyai
from babyai.wrappers import *

def make_env(env_key, obj_type=None, obj_color=None, seed=None):
    seed=None # For Now
    num_dists=0
    env = gym.make(env_key, obj_type=obj_type, obj_color=obj_color, num_dists=num_dists)
    env.seed(seed)
    env = ResetEnv(env, env_key=env_key, obj_type=obj_type, obj_color=obj_color, num_dists=num_dists, seed=seed)
    env = FixEnv(env)
    # env = FixFullyObsWrapper(env)
    return env
