# 2021/08/07
# trained agentのseed毎の平均報酬を出力するプログラム

# average_rewardの図を出力するための評価を行う

# ------実行コマンド！！！！-------
# python eval_average_reward.py
# ------------------------------

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
import datetime as dt
from tqdm import tqdm

import gym_custom

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results,ts2xy
from stable_baselines import results_plotter
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

best_mean_reward, n_updates = -np.inf,0

config = {
    'env':'CustomAnt-v0',
    'joint_min_range':0.0,
    'joint_max_range':1.0,
    'through_joint_range':2.0, # スルーするjoint_rangeの指定
    'validation':False,
}

def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agent', help = 'put in agent name for evaluation', type = str, default = 'Baseline-16million-v3')
    parser.add_argument('--video',default=False,action='store_true')
    
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
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1

    def reset(self,**kwargs): 
        self.reset_task()
        return self.env.reset(**kwargs)


    def step(self,action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            #print(joint_mask) # [4,5]のように表示される
            if joint_mask != []:
                action[joint_mask[0]]=henkan(action[joint_mask[0]],-1,1,-self.joint_range,self.joint_range)
                action[joint_mask[1]]=henkan(action[joint_mask[1]],-1,1,-self.joint_range,self.joint_range)

        obs,reward,done,info = self.env.step(action)
        
        return obs,reward,done,info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,5) # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        
        # 分布内の故障の時は乱数を作り直す処理
        while True:
            self.joint_range = random.uniform(config['joint_min_range'],config['joint_max_range'])
            if self.joint_range != config['through_joint_range']: 
                break

        # Pick which actuators to disable
        # joint rangeを変更する足をマスクで表現、99を代入しておく
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

        # make th removed leg look red
        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([1, 0, 0])
            geom_rgba[4, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 0])
            geom_rgba[7, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 0, 0])
            geom_rgba[10, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba #[:]がないとエラーになる


def main():
    args = mujoco_arg_parser()

    # Create log dir
    home = str(os.environ['HOME'])
    tensorboard_log_dir = home + "/HDD/RA-L/tensorboard_log/"
    os.makedirs(tensorboard_log_dir,exist_ok=True)

    # Create result tmp dir
    figdir = "./fig/AverageReward/"
    os.makedirs(figdir,exist_ok=True)

    # Create ndarray save dir
    # ランダムな脚が故障する環境での評価を格納するディレクトリ
    nd_dir = "./Data/AverageReward/" + str(args.agent) + "/" 
    os.makedirs(nd_dir, exist_ok=True)

    # Create and wrap the environment 
    env1 = gym.make(config['env'])
    broken_env1 = ChangeJointRangeEnv(env1)

    # time, 図の名前を一意にするため
    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    
    if args.video:
        broken_env1 = wrappers.Monitor(broken_env1,'./videos/' + args.loaddir + "-" + datetime.datetime.now().isoformat(),force=True,video_callable=(lambda ep: ep % 1 == 0)) # for output video

    broken_env1 = DummyVecEnv([lambda :broken_env1]) 
    env1 = DummyVecEnv([lambda : env1])

    # agentName = ['Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2', 'UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015', 'CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015', 'LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015']
    # label = np.array(['Baseline', 'UDR', 'ADR_easy2hard', 'ADR_hard2easy', 'LDR_easy2hard', 'LDR_hard2easy'])
    
    agentName = []
    agentName.append(args.agent)

    plainData = []
    brokenData = []
    perror = []
    berror = []
    # width = 0.35
    # x = np.arange(len(label))

    # plt.figure()
    # sns.set()
    # fig,ax = plt.subplots()
    for agent in agentName:
        plainSeedAveReward = []
        brokenSeedAveReward = []

        load_dir = "./trained_agent_dir/" + agent +"/"

        # seedごとに平均報酬を獲得する ,range(1,6)
        for seed in range(1,6):
            # PPO2modelの生成(トレーニングを行うエージェントの作成)
            # trainedAnt = PPO2(MlpPolicy,env1,verbose=1,tensorboard_log=tensorboard_log_dir,seed=100+seed)
            trainedAnt = PPO2(MlpPolicy,env1,verbose=1,tensorboard_log=tensorboard_log_dir)

            # 保存したモデル（学習済みモデル）のload ：zipファイルのファイル名のみとパスを指定,seedごとに
            trainedAnt = PPO2.load(load_dir + "trainedAnt" + "-seed" + str(seed)) 

            # seedの設定
            trainedAnt.set_random_seed(seed+100)

            print("loaddir:",load_dir + "trainedAnt" + "-seed" + str(seed))

            plane_obs_1 = env1.reset()
            broken_obs_1 = broken_env1.reset()

            plane_total_rewards_1 = [] # 全エピソードの報酬格納
            rewards_1 = 0

            # planeな歩行タスクでのrewardを求めるループ(100)
            for episode in tqdm(range(100)):
                for i in range(1000):
                    # predict phase
                    action_1,_states = trainedAnt.predict(plane_obs_1)

                    # step phase
                    plane_obs_1,reward_1,done_1,info_1 = env1.step(action_1)
                    
                    rewards_1 += reward_1

                    if done_1:
                        break
                plane_total_rewards_1.append(rewards_1) 
                rewards_1 = 0
        


            broken_total_rewards_1 = [] 
            rewards_1 = 0

            # 故障が起きる環境でのrewardを求めるループ(100)
            for episode in tqdm(range(100)):
                for i in range(1000):
                    # predict phase
                    action_1,_states = trainedAnt.predict(broken_obs_1)

                    # step phase
                    # broken環境で評価する時
                    broken_obs_1,reward_1,done_1,info_1 = broken_env1.step(action_1)
                
                    rewards_1 += reward_1

                    if done_1:
                        break
                broken_total_rewards_1.append(rewards_1)
                rewards_1 = 0


            plane_reward_average1 = sum(plane_total_rewards_1)/len(plane_total_rewards_1)
            broken_reward_average1 = sum(broken_total_rewards_1)/len(broken_total_rewards_1)
            plainSeedAveReward.append(plane_reward_average1)
            brokenSeedAveReward.append(broken_reward_average1)


            del trainedAnt #
        
        # agentのplain,broken環境での平均報酬が格納されている
        plain_ave = sum(plainSeedAveReward)/len(plainSeedAveReward)
        plainData.append(plain_ave)
        broken_ave = sum(brokenSeedAveReward)/len(brokenSeedAveReward)
        brokenData.append(broken_ave)
        plain_error = np.std(plainSeedAveReward,ddof=1)/np.sqrt(len(plainSeedAveReward))
        perror.append(plain_error)
        broken_error = np.std(brokenSeedAveReward,ddof=1)/np.sqrt(len(brokenSeedAveReward))
        berror.append(broken_error)


    plainData = np.array(plainData).flatten()
    brokenData = np.array(brokenData).flatten()
    perror = np.array(perror)
    berror = np.array(berror)
    print(plainData)
    print(brokenData)
    print(perror)
    print(berror)

    # 結果のndarray（各手法の平均報酬）をdataとして保存
    np.save(nd_dir + "plainAverageReward_{}".format(time), plainData)
    np.save(nd_dir + "brokenAverageReward_{}".format(time), brokenData)
    np.save(nd_dir + "plainError_{}".format(time), perror)
    np.save(nd_dir + "brokenError_{}".format(time), berror)

    # # figにプロットしていく
    # ax.bar(x-width/2,plainData,width=width,color='blue',align='center',label='plain',yerr=perror)
    # ax.bar(x+width/2,brokenData,width=width,color='darkorange',align='center',label='broken',yerr=berror)
    # ax.set_ylabel('Average Reward')
    # plt.tick_params(labelsize=7)
    # ax.set_xticks(x)
    # ax.set_xticklabels(label)
    # ax.legend()
    
    # plt.savefig(figdir+'averageReward_{}.png'.format(time))


if __name__=='__main__':
    main()