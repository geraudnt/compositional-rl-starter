import gym
import babyai
from utils.wrappers import *

def make_env(env_key, obj_type, obj_color, seed=None):
    env = _make_env(env_key, obj_type, obj_color, seed)
    env.remake = lambda : _make_env(env_key, obj_type, obj_color, seed)
    return env

def _make_env(env_key, obj_type, obj_color, seed):
    env = gym.make("BabyAI-GoToObjCustom-v0", obj_type=obj_type, obj_color=obj_color)
    env.seed(seed)
    env = FixEnv(env)
    return env