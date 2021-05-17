#!/bin/sh
#train-isis.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash train.sh (savedir名，normalAntなど)

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

# 処理内容，seedを変えて五回トレーニング
# 本番----------------------------
# python train.py --savedir=Baseline_CustomAnt --seed=1 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt --seed=2 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt --seed=3 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt --seed=4 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt --seed=5 --algo=Baseline

# python train.py --savedir=UDR_CustomAnt --seed=1 --algo=UDR
# python train.py --savedir=UDR_CustomAnt --seed=2 --algo=UDR
# python train.py --savedir=UDR_CustomAnt --seed=3 --algo=UDR
# python train.py --savedir=UDR_CustomAnt --seed=4 --algo=UDR
# python train.py --savedir=UDR_CustomAnt --seed=5 --algo=UDR

# python train.py --savedir=CDR-v1_CustomAnt --seed=1 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt --seed=2 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt --seed=3 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt --seed=4 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt --seed=5 --algo=CDR-v1

# python train.py --savedir=CDR-v2_CustomAnt --seed=1 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt --seed=2 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt --seed=3 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt --seed=4 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt --seed=5 --algo=CDR-v2

# upper,lower fixのトレーニング
# python train.py --savedir=CDR-v1_CustomAnt_upperfix --seed=1 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_upperfix --seed=2 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_upperfix --seed=3 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_upperfix --seed=4 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_upperfix --seed=5 --algo=CDR-v1

# python train.py --savedir=CDR-v2_CustomAnt_lowerfix --seed=1 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix --seed=2 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix --seed=3 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix --seed=4 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix --seed=5 --algo=CDR-v2

# 2021/05/17ーーーーーーーーーーーーーーー
# upper, lower fixしないトレーニング
python train.py --savedir=CDR-v1_CustomAnt_bf1000 --seed=1 --algo=CDR-v1
python train.py --savedir=CDR-v1_CustomAnt_bf1000 --seed=2 --algo=CDR-v1
python train.py --savedir=CDR-v1_CustomAnt_bf1000 --seed=3 --algo=CDR-v1
python train.py --savedir=CDR-v1_CustomAnt_bf1000 --seed=4 --algo=CDR-v1
python train.py --savedir=CDR-v1_CustomAnt_bf1000 --seed=5 --algo=CDR-v1

python train.py --savedir=CDR-v2_CustomAnt_bf1000 --seed=1 --algo=CDR-v2
python train.py --savedir=CDR-v2_CustomAnt_bf1000 --seed=2 --algo=CDR-v2
python train.py --savedir=CDR-v2_CustomAnt_bf1000 --seed=3 --algo=CDR-v2
python train.py --savedir=CDR-v2_CustomAnt_bf1000 --seed=4 --algo=CDR-v2
python train.py --savedir=CDR-v2_CustomAnt_bf1000 --seed=5 --algo=CDR-v2

# upper,lower fixのトレーニング
python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf1000 --seed=1 --algo=CDR-v1 --bound_fix
python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf1000 --seed=2 --algo=CDR-v1 --bound_fix
python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf1000 --seed=3 --algo=CDR-v1 --bound_fix
python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf1000 --seed=4 --algo=CDR-v1 --bound_fix
python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf1000 --seed=5 --algo=CDR-v1 --bound_fix

python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf1000 --seed=1 --algo=CDR-v2 --bound_fix
python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf1000 --seed=2 --algo=CDR-v2 --bound_fix
python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf1000 --seed=3 --algo=CDR-v2 --bound_fix
python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf1000 --seed=4 --algo=CDR-v2 --bound_fix
python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf1000 --seed=5 --algo=CDR-v2 --bound_fix




# 動作確認--------------------------
# python train.py --savedir=Baseline_CustomAnt --seed=1 --algo=Baseline --ablation

# python train.py --savedir=UDR_CustomAnt --seed=1 --algo=UDR --ablation

# python train.py --savedir=CDR-v1_CustomAnt --seed=1 --algo=CDR-v1 --ablation

# python train.py --savedir=CDR-v2_CustomAnt --seed=1 --algo=CDR-v2 --ablation
# python train.py --savedir=CDR-v1_notFix --seed=1 --algo=CDR-v1 --ablation
# python train.py --savedir=CDR-v1_Fix --seed=1 --algo=CDR-v1 --ablation --bound_fix
