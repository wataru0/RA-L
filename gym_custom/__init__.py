import gym_custom.envs
from gym.envs.registration import register

register(
    id = "custom_ant-v0",
    entry_point = "gym_custom.envs.custom_ant_env:CustomAntEnv",
    max_episode_steps = 1000,
    reward_threshold = 6000.0,
)

# --------------
# import gym_custom.envs
# from .core import custom_make
# from .registration import register