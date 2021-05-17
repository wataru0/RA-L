# 2021/04/19　UDR学習用
# 実験環境trainプログラム
# seedを指定して学習する！

# ------実行コマンド！！！！-------
# python train_UDR.py --savedir=[保存するモデルの名前（dirになる（joint_rangeなどを書く！！）] --seed=(int)
# ------------------------------

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import time
import pytz
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

best_mean_reward = -np.inf # 平均報酬
n_updates = 0 # 更新数

# ---------------------------------------------------------------------------
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    'joint_min_range':0.0,
    'joint_max_range':1.0
}

# Create log dir
# exist_ok=Trueとすると既に存在しているディレクトリを指定してもエラーにならない
log_dir = "./monitor_log/"
os.makedirs(log_dir,exist_ok=True)

#可視化用
joint_actuator_range = []
actuator_map = np.zeros((10,8)) # 適用された故障率の分布を調べるためのマップ
actuator_bunpu= [0]*10
actuator_power_map = np.zeros((10,8)) # 実際のactuator出力の分布を調べるためのマップ

rewardlist = []


def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--savedir',help='saving name dir for trained mode!!!',type=str,default='Ant'+ datetime.datetime.now().isoformat())
    parser.add_argument('--seed',help='seed for saigensei',type=int,default=1)

    return parser.parse_args()

def henkan(val,start1,stop1,start2,stop2):
    return start2 + (stop2 - start2) * ((val-start1)/(stop1 - start1))

# wrapper env class !!!!!--------------------
class ChangeJointRangeEnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env) # 親クラスの呼び出しが必要
        self.value = value # crippled leg number
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.cReward = 0
        self.joint_range = 1.0

    def reset(self,**kwargs): #**kwargs:任意個数の引数を辞書として受け取る
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 
        return self.env.reset(**kwargs)


    def step(self,action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            #print(joint_mask) # [4,5]のように表示される
            if joint_mask != []:
                action[joint_mask[0]]=henkan(action[joint_mask[0]],-1,1,-self.joint_range,self.joint_range)
                action[joint_mask[1]]=henkan(action[joint_mask[1]],-1,1,-self.joint_range,self.joint_range)
            #ーーーー action = self.cripple_mask * action
            #print(action) # joint_maskの要素のaction値をクリップ(指定した値の間の値に変換)することができた

        obs,reward,done,info = self.env.step(action)

        # グラフ作成用
        for i,a in enumerate(action):
            index = int(abs(a)*10)
            if index == 10:
                index = 9
            actuator_power_map[index][i] += 1

        self.cReward += reward
        
        return obs,reward,done,info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,5) # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        
        # joint_min_range~joint_max_rangeまでの乱数を生成。これがaction値のrangeになる
        self.joint_range = random.uniform(config['joint_min_range'], config['joint_max_range'])

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



#call back : 更新毎に呼ばれるコールバック
#エージェント内で呼び出されるカスタムコールバック関数を定義できます。これは、トレーニングを監視する場合、
# たとえば、Tensorboard（またはVisdom）でライブ学習曲線を表示したり、最適なエージェントを保存したり
# する場合に役立ちます。コールバックがFalseを返す場合、トレーニングは早期に中止されます
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
            #print(x[-1], 'timesteps')
            #print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
   
    n_updates += 1
    return True

def main():
    args = mujoco_arg_parser()

    # Create saving trained agent dir
    save_dir = "./trained_agent_dir/"+ args.savedir + "/"
    os.makedirs(save_dir,exist_ok=True)
    

    # Create tensorboard log dir
    # port:6007で開いた，tmux の　5
    tensorboard_log_dir = "./tensorboard_log/"
    os.makedirs(tensorboard_log_dir,exist_ok=True)

    # Create and wrap the environment 
    env = gym.make(config['env'])
    env = ChangeJointRangeEnv(env)
    env = Monitor(env,log_dir,allow_early_resets=True) # Monitor:logフォルダにmonitor.csvが出力します。ep_reward_mean(平均報酬)、ep_len_mean(平均エピソード長)、timestamp(経過時間)の3つのカラムを持つCSVになります
    env = DummyVecEnv([lambda :env]) #複数の環境用の単純なベクトル化されたラッパーを作成し、現在のPythonプロセスで各環境を順番に呼び出します。

    # modelの生成
    model = PPO2(MlpPolicy,env,verbose=1,tensorboard_log=tensorboard_log_dir,n_steps=config['n_steps'],nminibatches=config['nminibatches'],noptepochs=config['noptepochs'],learning_rate=config['learning_rate'],seed=args.seed)
    
   
    model.learn(total_timesteps=config['total_timestep'],callback=callback,tb_log_name=args.savedir)

    # # Save the agent
    # #model.save("logdir/" + args.savedir + "/" + config['env']) # 保存先のファイルパス
    model.save(save_dir + "trainedAnt" + "-seed"+ str(args.seed))

    # csv 出力
    csvdir = "./output/csv"
    os.makedirs(csvdir,exist_ok=True)
    R = np.array(rewardlist)
    np.savetxt(csvdir + '/'+ args.savedir+'-'+str(args.seed) +'.csv',R,delimiter=',')


if __name__=='__main__':
    main()