# ラベルと論文のモデルの対応
Curriculum2-v1 -> CDR-v1
Curriculum-v4-16million -> CDR-v2

# 各手法のgeneralizationの可視化手順
- bash eval_generalization.sh　で，各手法のnpyデータを出力
- python plot_generalization.py でプロット

# 報酬関数の各項の可視化手順
- bash eval_generalization.sh　で，各手法のnpyデータを出力
- bash plot_TermOfRewardFunction.sh　で各手法の報酬関数の各項の値変化をプロット

# train 履歴
## Baseline
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022
}

## UDR
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

## CDR-v1
class ChangeJointRangeEnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env) # 親クラスの呼び出しが必要
        self.value = value # crippled leg number
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1
        self.num_step = 0
        self.total_reward = 0
        self.buffer = [] #rewardを保存しておくバッファー
        self.before_average = 0 #前回の分布での報酬の平均値を格納しておく変数
        self.joint_min = 1.0 # minもmaxも1からスタート(v1),minもmaxも0からスタート(v2)--------------------------------------
        self.joint_max = 1.0 # ---------------------------------------------------------------------------------------
        self.joint_num = 0
        self.cReward = 0 # 累積報酬

        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)
        
    def reset(self,**kwargs): #**kwargs:任意個数の引数を辞書として受け取る
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 

        # bufferにパフォーマンスを格納していく，buffer_sizeを超えたら評価する
        if len(self.buffer) < config['buffer_size']:
            # joint actuator force rangeが[0.9,1]の分布の範囲である時性能を評価する．
            # bufferに格納
            self.buffer.append(self.total_reward)


        else: # 能力の評価
            ave = sum(self.buffer)/len(self.buffer)
            if self.before_average < ave: #前より能力アップしていたら

                # Curriculum2-v1
                if self.joint_min > 0.0:
                    self.joint_min -= config['update_k_step_size']
                    # kのminとmaxの差が0.1以上になったらmaxも減らす
                    if abs(self.joint_max - self.joint_min) >= 0.1:
                        self.joint_max -= config['update_k_step_size']

                # Curriculum2-v2．
                # if self.joint_max <= 1.0:
                #     self.joint_max += config['update_k_step_size']
                #     # kのminとmaxの差が0.1以上になったらminも上昇
                #     if abs(self.joint_max - self.joint_min) >= 0.1:
                #         self.joint_min += config['update_k_step_size']

            # else: #能力アップしていなかったら
            #     # 下げなくていいかも，k＝０から上昇させるときは，
            #     if self.joint_max > 0.0:
            #         self.joint_max -= 0.1 # joint_maxを下げる
            #         # minも下げる
            #         if self.joint_max - self.joint_min < 1.0:
            #             if self.joint_min > 0.0:
            #                 self.joint_min -= 0.1

            self.before_average = ave
            self.buffer.clear() #バッファを空にする


        self.total_reward = 0
        return self.env.reset(**kwargs)
## CDR-v2
class ChangeJointRangeEnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env) # 親クラスの呼び出しが必要
        self.value = value # crippled leg number
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1
        self.num_step = 0
        self.total_reward = 0
        self.buffer = [] #rewardを保存しておくバッファー
        self.before_average = 0 #前回の分布での報酬の平均値を格納しておく変数
        self.joint_min = 0.0 # minもmaxも1からスタート(v1),minもmaxも0からスタート(v2)--------------------------------------+++++++++
        self.joint_max = 0.0 # ---------------------------------------------------------------------------------------+++++++++
        self.joint_num = 0
        self.cReward = 0 # 累積報酬

        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)
        
    def reset(self,**kwargs): #**kwargs:任意個数の引数を辞書として受け取る
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 

        # bufferにパフォーマンスを格納していく，buffer_sizeを超えたら評価する
        if len(self.buffer) < config['buffer_size']:
            # joint actuator force rangeが[0.9,1]の分布の範囲である時性能を評価する．
            # bufferに格納
            self.buffer.append(self.total_reward)


        else: # 能力の評価
            ave = sum(self.buffer)/len(self.buffer)
            if self.before_average < ave: #前より能力アップしていたら

                # Curriculum2-v1==================================
                # if self.joint_min > 0.0:
                #     self.joint_min -= config['update_k_step_size']
                #     # kのminとmaxの差が0.1以上になったらmaxも減らす
                #     if abs(self.joint_max - self.joint_min) >= 0.1:
                #         self.joint_max -= config['update_k_step_size']

                # Curriculum2-v2==================================
                if self.joint_max <= 1.0:
                    self.joint_max += config['update_k_step_size']
                    # kのminとmaxの差が0.1以上になったらminも上昇
                    if abs(self.joint_max - self.joint_min) >= 0.1:
                        self.joint_min += config['update_k_step_size']

            # else: #能力アップしていなかったら
            #     # 下げなくていいかも，k＝０から上昇させるときは，
            #     if self.joint_max > 0.0:
            #         self.joint_max -= 0.1 # joint_maxを下げる
            #         # minも下げる
            #         if self.joint_max - self.joint_min < 1.0:
            #             if self.joint_min > 0.0:
            #                 self.joint_min -= 0.1

            self.before_average = ave
            self.buffer.clear() #バッファを空にする


        self.total_reward = 0
        return self.env.reset(**kwargs)

## Baseline_CustomAnt
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020
}

## Baseline_CustomAntEnv
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':8, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00030
}

## Baseline_CustomAntEnv-v2
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00025
}

## lr-15e-5
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00015
}

## n_step-512
config = {
    # 'env':'Ant-v2',
    'env':'CustomAnt-v0',
    'joint_range1':-1, # 注意！ラッパー外してある713
    'joint_range2':1,
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数
    'n_steps':512, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00025
}


---
# Ablation studyで訓練したエージェント
デフォルトのant環境の終了条件の有効性を検証するために行う
config = {
    # 'env':'CustomAnt-v0',
    'env':'AblationAnt-v0',
    'total_timestep':int(16e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数  (2e6
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## not02
python train.py --savedir=not02 --algo=Baseline --ablation
### 終了条件
notdone = np.isfinite(state).all() and state[2] <= 1.0

## default
python train.py --savedir=default --algo=Baseline --ablation
### 終了条件
notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0

## not10
python train.py --savedir=not10 --algo=Baseline --ablation
### 終了条件
notdone = np.isfinite(state).all() and state[2] >= 0.2

## default+tento
python train.py --savedir=default+tento --algo=Baseline --ablation
### 終了条件
notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
if torso_vec[2] < -0.8:
    notdone = False

## not10+tento
### 終了条件
notdone = np.isfinite(state).all() and state[2] >= 0.2 
if torso_vec[2] < -0.8:
    notdone = False

## not02+tento
### 終了条件
notdone = np.isfinite(state).all() and state[2] <= 1.0
if torso_vec[2] < -0.8:
    notdone = False

## tento
### 終了条件
if torso_vec[2] < -0.8:
    notdone = False

## 15+tento
### 終了条件
notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.5
if torso_vec[2] < -0.8:
    notdone = False

---
---
# 終了条件をdefault + 転倒で終了にした
## Baseline_CustomAnt
## UDR_CustomAnt
## CDR-v1_CustomAnt
## CDR-v2_CustomAnt
config = {
    'env':'CustomAnt-v0',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## Baseline-lr=3
config = {
    'env':'CustomAnt-v0',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00030,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## CDR-v1_CustomAnt_upperfix
## CDR-v2_CustomAnt_lowerfix
config = {
    'env':'CustomAnt-v0',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## CDR-v1_normalAnt
config = {
    # 'env':'CustomAnt-v0',
    'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## CDR-v1_CustomAnt_newDoneCode
### custom_ant_env.pyの終了条件のコードを変えた
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(2e6), # PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
    # 'joint_min_range':0,
    # 'joint_max_range':1,
    # 'buffer_size':100,
    # 'update_k_step_size':0.01 # k のアップデートサイズ
}

## CDR-v1_CustomAnt_newDoneCode_bfsize1000
self.buffer_size = 1000 に変更

## CDR-v1_CustomAnt_bfsize1000_20m
## CDR-v1_CustomAnt_bfsize500_20m


---
# 実験
## CDR-v1_CustomAnt_bf1000
## CDR-v2_CustomAnt_bf1000
## CDR-v1_CustomAnt_upperfix_bf1000
## CDR-v2_CustomAnt_upperfix_bf1000
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
}
と，CDRのbuffer sizeを1000にした．

## CDR-v1_CustomAnt_bf500
## CDR-v2_CustomAnt_bf500
## CDR-v1_CustomAnt_upperfix_bf500
## CDR-v2_CustomAnt_lowerfix_bf500
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(20e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
}
と，buffer sizeを500にした．
こいつら途中で止めた．

---
---

# 2021/05/28
# これからは，「転倒したらエピソード終了」ではなく，転倒した場合，survive reward を0にする or survive rewardを負の値にするという環境で訓練を行ってみる．

## Baseline_CustomAnt-ReduceSRto0IfFallingDown
## UDR_CustomAnt-ReduceSRto0IfFallingDown
## CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100
## CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100
## CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100
## CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100
```
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00020,
}
```

## Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2
```
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022,
}
```

## UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015
## CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015
## CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015
## CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015
## CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015
kのランダム化範囲を0.0〜1.5に変更したもの　　
（今までは0.0〜1.0だった）
```
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022,
}
```

## LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015
## LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015
## LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015
## LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015
Linear Curriculum Learningアルゴリズムで訓練したもの
```
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # default:16e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022, # 0.00020
    'n_level':11, # Linear Curriculum Learningにおいてのkの更新回数
}
```

## TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015
三角分布に従ってkをサンプリングするドメインランダマイゼーション
```
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # default:16e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022, # 0.00020
    'n_level':11, # Linear Curriculum Learningにおいてのkの更新回数
}
```

---
# normalAnt（Ant-v2環境）での新手法の再トレーニング
## Baseline_Ant-v2
## UDR_Ant-v2_k0010
## CDR-v1_Ant-v2_bf100_k0010
## CDR-v2_Ant-v2_bf100_k0010
## LCL-v1_Ant-v2_k0010
## LCL-v2_Ant-v2_k0010
## UDR_Ant-v2_k0015
## CDR-v1_Ant-v2_bf100_k0015
## CDR-v2_Ant-v2_bf100_k0015
## LCL-v1_Ant-v2_k0015
## LCL-v2_Ant-v2_k0015
```
config = {
    # 'env':'CustomAnt-v0',
    'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # 20e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022,
}
```

# kを固定して，脚を一本壊しながら学習するエージェント
## Baseline_Ant-v2_k08
## Baseline_Ant-v2_k06
## Baseline_Ant-v2_k04
## Baseline_Ant-v2_k02
## Baseline_Ant-v2_k00
## Baseline_Ant-v2_k12
## Baseline_Ant-v2_k14
```
config = {
    # 'env':'CustomAnt-v0',
    'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # default:16e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022, # 0.00020
    'n_level':11, # Linear Curriculum Learningにおいてのkの更新回数
}
```