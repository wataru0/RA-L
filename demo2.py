import os
import numpy as np
import datetime as dt
import argparse

import gym
import gym_custom
from gym import wrappers

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', default=False, action='store_true') 
    return parser.parse_args()

def main():
    args = arg_parser()

    env = gym.make('CustomAnt-v0')
    # env = gym.make('TestAnt-v0')

    tmp_dir = "./output/csv/quaternion/"
    os.makedirs(tmp_dir, exist_ok=True)

    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')

    if args.video:
        # env = wrappers.Monitor(env,'./output/videos/' + dt.datetime.now().isoformat(), force=True, video_callable=(lambda ep: ep % 1 == 0))
        env = wrappers.Monitor(env,'./output/videos/' + 'quat', force=True, video_callable=(lambda ep: ep % 1 == 0))

    csv = []
    done = False
    i = 0
    env.reset()
    while True:
    # for _ in range(2000):
        if not args.video:
            env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            env.reset()
            break

        # csv.append(info['res2'])
        # print(i, info["res2"])
        # if info["res2"][2] < -0.5:
        #     print("----転倒----")
        # i += 1

        print(i, info['torso_vec'])
        if info['torso_vec'][2] < -0.8:
            print("----転倒----")
        i += 1

    # save csv file
    # csv = np.array(csv)
    # np.savetxt(tmp_dir + 'out_{}.csv'.format(time), csv, delimiter=',')

if __name__=='__main__':
    main()