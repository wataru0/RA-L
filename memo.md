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