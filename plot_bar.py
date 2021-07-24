# 2021/07/24
# 各手法の各kでの平均報酬をbarグラフで可視化するプログラム

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import argparse
import os

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agent', help = 'put in agent name for evaluation', type = str, default = 'Baseline_Ant-v2')
    parser.add_argument('--n_episodes', help= 'put in number of episodes', type=int, default= 10)
    parser.add_argument('--video', default=False, action='store_true')    
    return parser.parse_args()

# nd_array５つを一つにまとめる関数，まとめる際に平均をとっている
def combine(a1, a2, a3, a4, a5):
    com_array = [(s1 + s2 + s3 + s4 + s5) / 5.0  for s1, s2, s3, s4, s5 in zip(a1, a2, a3, a4, a5)]
    return com_array

# nd_arrayの中からkと平均報酬をdictionary型にする関数
def dictionaize(dic, array):
    for i, k in enumerate(array):
        if k != 0.0:
            dic[float(i/100)] = k

def main():
    args = arg_parser()

    # Create figure dir
    figdir = "./fig/bar/"
    os.makedirs(figdir, exist_ok=True)

    if 'Ant-v2' in args.agent: 
        nd_path = "./Data/barplot/Ant-v2/" + str(args.agent) + "/"
    else:
        nd_path = "./Data/barplot/CustomAnt/" + str(args.agent) + "/"

    if 'k00' in args.agent:
        label = 'k=0.0'
    elif 'k02' in args.agent:
        label = 'k=0.2'
    elif 'k04' in args.agent:
        label = 'k=0.4'
    elif 'k06' in args.agent:
        label = 'k=0.6'
    elif 'k08' in args.agent:
        label = 'k=0.8'
    elif 'k12' in args.agent:
        label = 'k=1.2'
    elif 'k14' in args.agent:
        label = 'k=1.4'
    else:
        label = 'k=1.0'

    agent_array = {}
    for seed in range(1, 6):
        agent_array[seed] = np.load(nd_path + str(args.agent) + '_rewardForward_seed=' + str(seed) + '.npy')

    array = combine(agent_array[1], agent_array[2], agent_array[3], agent_array[4], agent_array[5])
    dic = {}
    dictionaize(dic, array)

    # 可視化
    fig = plt.figure()
    sns.set()
    # fig, ax = plt.subplots()
    x = []
    xlabels = []
    height = []
    
    for key, value in dic.items():
        xlabels.append(key)
        height.append(value)

    x = np.arange(len(height))
    plt.bar(x, height, label=label, tick_label=xlabels)
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('average reward')
    plt.savefig(figdir + '{}.png'.format(args.agent))

if __name__ == '__main__':
    main()
