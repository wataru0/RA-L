import os
import gym
import gym_custom

env = gym.make('TestAnt-v0')
env.reset()
done = False

nd_dir = "./tmp/"
os.makedirs(nd_dir, exist_ok=True)

while True:
    env.render()
    if done:
        break
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # print(info["a"])
    # print(info['reward_forward'])