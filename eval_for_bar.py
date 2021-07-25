# 2021/7/4
# 各故障（kの変動）での報酬の内訳，報酬の移り変わりを評価するためのプログラム
# bar plot で出力するためのもの

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import time
from gym import wrappers
import pylab
import csv
import seaborn as sns
import pandas as pd
import math
import datetime as dt
from  tqdm import tqdm

# 自作のgym環境を使うためにimport
import gym_custom

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results,ts2xy
from stable_baselines import results_plotter
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

best_mean_reward, n_updates = -np.inf,0

# 可視化用
# 故障箇所と報酬を格納するdictionary
rewardsMap = { } 

# kを0から1まで，0.01刻みで格納する配列
k_gene = np.zeros((5, 100, 151))

# 報酬関数の各項の影響力を可視化するための配列
reward_forward_map = np.zeros((5, 100, 151))
reward_ctrl_map = np.zeros((5, 100, 151))
reward_contact_map = np.zeros((5, 100, 151))
reward_survive_map = np.zeros((5, 100, 151))

config = {
    # 'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    'joint_min_range':0.0,
    'joint_max_range':1.0,
}

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agent', help = 'put in agent name for evaluation', type = str, default = 'Baseline-16million-v3')
    parser.add_argument('--n_episodes', help= 'put in number of episodes', type=int, default= 10)
    parser.add_argument('--video', default=False, action='store_true')    
    return parser.parse_args()

def henkan(val,start1,stop1,start2,stop2):
    return start2 + (stop2 - start2) * ((val-start1)/(stop1 - start1))

# wrapper env class !!!!!--------------------
class NormalEnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env)
        self.value = value
        
    def reset(self,**kwargs):
        return self.env.reset(**kwargs)

    def step(self,action):
        obs,reward,done,info = self.env.step(action)
        return obs,reward,done,info

class ChangeJointRangeEnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env) # 親クラスの呼び出しが必要
        self.value = value # crippled leg number
        self.crippled_leg1 = 0
        self.crippled_leg2 = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1

    def reset(self,**kwargs): #**kwargs:任意個数の引数を辞書として受け取る
        self.reset_task()
        return self.env.reset(**kwargs)

    def step(self, action, k):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            if len(joint_mask) == 1:
                joint_mask.append(joint_mask[0])
            # print(joint_mask) # [4,5]のように表示される, [2, 3, 4, 5]

            if joint_mask != []:
                for i in joint_mask:
                    action[i] = action[i] * (float(k)/100)

        obs,reward,done,info = self.env.step(action)

        return obs,reward,done,info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,4)

        # Pick which actuators to disable
        # joint rangeを変更する脚2本をマスクで表現、99を代入しておく
        # 壊す脚を選択
        self.cripple_mask = np.ones(self.action_space.shape)
        if self.crippled_leg == 0:
            self.cripple_mask[2] = 99
            self.cripple_mask[3] = 99
        elif self.crippled_leg == 1:
            self.cripple_mask[4] = 99
            self.cripple_mask[5] = 99
        elif self.crippled_leg == 2:
            self.cripple_mask[6] = 99
            self.cripple_mask[7] = 99
        elif self.crippled_leg == 3:
            self.cripple_mask[0] = 99
            self.cripple_mask[1] = 99
        elif self.crippled_leg == 4:
            pass

def save_reward_map(map, save_path, agent_name, save_name, n_episodes):
    for seed in range(1, 6):
        seed_list = map[seed-1,:,:]
        seed_list = np.sum(seed_list, axis=0)
        seed_list = seed_list/n_episodes
        np.save(save_path + agent_name + save_name + "_seed=" + str(seed), seed_list)

def main():
    args = arg_parser()

    # Create log dir
    home = str(os.environ['HOME'])
    tensorboard_log_dir = home + "/HDD/RA-L/tensorboard_log/"
    os.makedirs(tensorboard_log_dir,exist_ok=True)

    # Create ndarray save dir
    # Ant-v2で評価する時
    if 'Ant-v2' in args.agent:
        # ランダムな脚が故障する環境での評価を格納するディレクトリ
        nd_dir = "./Data/barplot/Ant-v2/" + str(args.agent) + "/" 
        env1 = gym.make('Ant-v2')
    # CustomAntで評価する時
    else:
        nd_dir = "./Data/barplot/CustomAnt/" + str(args.agent) + "/" 
        env1 = gym.make('CustomAnt-v0')
    os.makedirs(nd_dir, exist_ok=True)

    # wrap the environment 
    broken_env = ChangeJointRangeEnv(env1) # 脚一本が故障する環境

    if args.video:
        broken_env = wrappers.Monitor(broken_env,'./videos/' + args.loaddir + "-" + datetime.datetime.now().isoformat(),force=True,video_callable=(lambda ep: ep % 1 == 0)) # for output video

    # broken_env = DummyVecEnv([lambda :broken_env]) #複数の環境用の単純なベクトル化されたラッパーを作成し、現在のPythonプロセスで各環境を順番に呼び出します。
    env1 = DummyVecEnv([lambda : env1])

    # argpaserから入力する場合
    agentName = []
    agentName.append(args.agent)

    plainData = []
    brokenData = []
    perror = []
    berror = []

    # fig,ax = plt.subplots()
    for agent in agentName:
        brokenSeedAveReward = []

        load_dir = "./trained_agent_dir/" + agent + "/"

        # seedごとに平均報酬を獲得する ,range(1,6)
        for seed in range(1,6):

            # PPO2modelの生成(トレーニングを行うエージェントの作成)
            trainedAnt = PPO2(MlpPolicy, env1, verbose=1, tensorboard_log=tensorboard_log_dir)

            # 保存したモデル（学習済みモデル）のload ：zipファイルのファイル名のみとパスを指定,seedごとに
            trainedAnt = PPO2.load(load_dir + "trainedAnt" + "-seed" + str(seed)) 

            # seedの設定
            trainedAnt.set_random_seed(seed+100)

            print("loaddir:",load_dir + "trainedAnt" + "-seed" + str(seed))

            broken_obs = broken_env.reset()

            broken_total_rewards = [] 
            rewards = 0
            forwards = 0
            ctrls = 0
            contacts = 0
            survives = 0

            # kを0から1.5まで，0.1刻みで変化させる
            for k in tqdm(range(0, 151, 10)):
                # 故障が起きる環境でのrewardを求めるループ(100)
                for episode in range(args.n_episodes):
                    # iteration of time steps, default is 1000 time steps
                    for i in range(1000):
                        # predict phase
                        action, _states = trainedAnt.predict(broken_obs)

                        # step phase
                        # broken環境で評価する時
                        broken_obs, reward, done, info = broken_env.step(action, k)
                        rewards += reward
                        forwards += info['reward_forward']
                        ctrls += info['reward_ctrl']
                        contacts += info['reward_contact']
                        survives += info['reward_survive']
                        
                        if done:
                            break

                    # k_geneにkとその時の報酬を格納
                    k_gene[seed-1][episode][k] = rewards

                    # 報酬関数の各項の値を格納
                    reward_forward_map[seed-1][episode][k] = forwards
                    reward_ctrl_map[seed-1][episode][k] = ctrls
                    reward_contact_map[seed-1][episode][k] = contacts
                    reward_survive_map[seed-1][episode][k] = survives

                    # 環境をリセット
                    broken_obs = broken_env.reset()

                    # 合計報酬の記録とリセット
                    broken_total_rewards.append(rewards)
                    rewards = 0
                    forwards = 0
                    ctrls = 0
                    contacts = 0
                    survives = 0

            broken_reward_average1 = sum(broken_total_rewards)/len(broken_total_rewards)
            brokenSeedAveReward.append(broken_reward_average1)

            del trainedAnt 
        
        # agentのplain,broken環境での平均報酬が格納されている
        broken_ave = sum(brokenSeedAveReward)/len(brokenSeedAveReward)
        brokenData.append(broken_ave)
        broken_error = np.std(brokenSeedAveReward,ddof=1)/np.sqrt(len(brokenSeedAveReward))
        berror.append(broken_error)

    brokenData = np.array(brokenData).flatten()
    berror = np.array(berror)

    # print(k_gene)
    for seed in range(1, 6):
        seed_gene = k_gene[seed-1,:,:]
        seed_gene = np.sum(seed_gene, axis=0)
        seed_gene = seed_gene/args.n_episodes # 平均報酬
        np.save(nd_dir + str(agentName[0]) + "_rewardForEachK" + "_seed=" + str(seed), seed_gene)

    # 報酬関数の各項の二次元配列を一次元配列に変形してnpyで保存
    save_reward_map(map=reward_forward_map, save_path=nd_dir, agent_name=str(agentName[0]), save_name="_rewardForward", n_episodes=args.n_episodes)
    save_reward_map(map=reward_ctrl_map, save_path=nd_dir, agent_name=str(agentName[0]), save_name="_rewardCtrl", n_episodes=args.n_episodes)
    save_reward_map(map=reward_contact_map, save_path=nd_dir, agent_name=str(agentName[0]), save_name="_rewardContact", n_episodes=args.n_episodes)
    save_reward_map(map=reward_survive_map, save_path=nd_dir, agent_name=str(agentName[0]), save_name="_rewardSurvive", n_episodes=args.n_episodes)
    

if __name__=='__main__':
    main()