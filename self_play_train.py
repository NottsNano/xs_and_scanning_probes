from stable_baselines import PPO2
from stable_baselines.common import make_vec_env

from game_env import NoughtsAndCrossesEnv


env = make_vec_env(NoughtsAndCrossesEnv, n_envs=10, env_kwargs={"opponent": "AI"})
agent_1 = PPO2
model = agent("MlpPolicy", env)
