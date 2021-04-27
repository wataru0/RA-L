# 2021/04/27
# train用プログラム
# リファクタリングしたもの

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import time
import pytz
import tensorflow as tf
import seaborn as sns; sns.set()
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results,ts2xy
from stable_baselines import results_plotter
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

import gym_custom
from algorithms import CDR 

best_mean_reward = -np.inf # 平均報酬
n_updates = 0 # 更新数
log_dir = "./monitor_log/"
os.makedirs(log_dir,exist_ok=True)

config = {
    'env':'CustomAnt-v0',
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数  (2e6
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    'joint_min_range':0,
    'joint_max_range':1,
    'buffer_size':100,
    'update_k_step_size':0.01 # k のアップデートサイズ
}

def callback(_locals,_globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_updates
    global best_mean_reward
    # Print stats every 1000 calls
    if (n_updates + 1) % 1000 == 0:
        # Evaluate policy training performance
        x,y = ts2xy(load_results(log_dir),'timesteps')
        if len(x) > 0:
            #100ステップ毎に過去100件の平均報酬を計算し、ベスト平均報酬を越えていたらエージェントを保存しています。
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
   
    n_updates += 1
    return True

def main():



if __name__ == '__main__':
    main()
        