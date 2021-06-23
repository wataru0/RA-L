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
# # upper, lower fixしないトレーニング
# python train.py --savedir=CDR-v1_CustomAnt_bf500 --seed=1 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_bf500 --seed=2 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_bf500 --seed=3 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_bf500 --seed=4 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt_bf500 --seed=5 --algo=CDR-v1

# python train.py --savedir=CDR-v2_CustomAnt_bf500 --seed=1 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_bf500 --seed=2 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_bf500 --seed=3 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_bf500 --seed=4 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt_bf500 --seed=5 --algo=CDR-v2

# # upper,lower fixのトレーニング
# python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf500 --seed=1 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf500 --seed=2 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf500 --seed=3 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf500 --seed=4 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt_upperfix_bf500 --seed=5 --algo=CDR-v1 --bound_fix

# python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf500 --seed=1 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf500 --seed=2 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf500 --seed=3 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf500 --seed=4 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt_lowerfix_bf500 --seed=5 --algo=CDR-v2 --bound_fix

# 動作確認--------------------------
# python train.py --savedir=Baseline_CustomAnt --seed=1 --algo=Baseline --ablation

# python train.py --savedir=UDR_CustomAnt --seed=1 --algo=UDR --ablation

# python train.py --savedir=CDR-v1_CustomAnt --seed=1 --algo=CDR-v1 --ablation

# python train.py --savedir=CDR-v2_CustomAnt --seed=1 --algo=CDR-v2 --ablation
# python train.py --savedir=CDR-v1_notFix --seed=1 --algo=CDR-v1 --ablation
# python train.py --savedir=CDR-v1_Fix --seed=1 --algo=CDR-v1 --ablation --bound_fix

# python train.py --savedir=aaa --seed=1 --algo=CDR-v2 --ablation

# ---
# 2021/05/28
# reduce survive reward to 0 if falling down

# ベースライン
# python train.py --savedir=Baseline_CustomAnt-ReduceSRto0IfFallingDown --seed=1 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt-ReduceSRto0IfFallingDown --seed=2 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt-ReduceSRto0IfFallingDown --seed=3 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt-ReduceSRto0IfFallingDown --seed=4 --algo=Baseline
# python train.py --savedir=Baseline_CustomAnt-ReduceSRto0IfFallingDown --seed=5 --algo=Baseline

# Uniform Domain Randomization
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown --seed=1 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown --seed=2 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown --seed=3 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown --seed=4 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown --seed=5 --algo=UDR

# 提案手法
# upper, lower fixしないトレーニング
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=1 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=2 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=3 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=4 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=5 --algo=CDR-v1

# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=1 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=2 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=3 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=4 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --seed=5 --algo=CDR-v2

# upper,lower fixのトレーニング
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --seed=1 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --seed=2 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --seed=3 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --seed=4 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --seed=5 --algo=CDR-v1 --bound_fix

# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --seed=1 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --seed=2 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --seed=3 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --seed=4 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --seed=5 --algo=CDR-v2 --bound_fix

# 2021/06/09
# kのランダム化範囲を0.0~1.5に変更したもの
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=1 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=2 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=3 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=4 --algo=UDR
# python train.py --savedir=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=5 --algo=UDR

# 提案手法
# upper, lower fixしないトレーニング
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=1 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=2 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=3 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=4 --algo=CDR-v1
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=5 --algo=CDR-v1

# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=1 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=2 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=3 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=4 --algo=CDR-v2
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --seed=5 --algo=CDR-v2

# # upper,lower fixのトレーニング
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015 --seed=1 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015 --seed=2 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015 --seed=3 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015 --seed=4 --algo=CDR-v1 --bound_fix
# python train.py --savedir=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100_k0015 --seed=5 --algo=CDR-v1 --bound_fix

# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015 --seed=1 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015 --seed=2 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015 --seed=3 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015 --seed=4 --algo=CDR-v2 --bound_fix
# python train.py --savedir=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100_k0015 --seed=5 --algo=CDR-v2 --bound_fix

# 2021/06/17
# Linear Curriculum Learningのトレーニング
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=1 --algo=LCL-v1
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=2 --algo=LCL-v1
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=3 --algo=LCL-v1
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=4 --algo=LCL-v1
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=5 --algo=LCL-v1

# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=1 --algo=LCL-v2
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=2 --algo=LCL-v2
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=3 --algo=LCL-v2
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=4 --algo=LCL-v2
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=5 --algo=LCL-v2

# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015 --seed=1 --algo=LCL-v1 --bound_fix
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015 --seed=2 --algo=LCL-v1 --bound_fix
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015 --seed=3 --algo=LCL-v1 --bound_fix
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015 --seed=4 --algo=LCL-v1 --bound_fix
# python train.py --savedir=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_k0015 --seed=5 --algo=LCL-v1 --bound_fix

# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015 --seed=1 --algo=LCL-v2 --bound_fix
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015 --seed=2 --algo=LCL-v2 --bound_fix
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015 --seed=3 --algo=LCL-v2 --bound_fix
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015 --seed=4 --algo=LCL-v2 --bound_fix
# python train.py --savedir=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_k0015 --seed=5 --algo=LCL-v2 --bound_fix

# 2021/06/23
python train.py --savedir=TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=1 --algo=TDR
python train.py --savedir=TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=2 --algo=TDR
python train.py --savedir=TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=3 --algo=TDR
python train.py --savedir=TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=4 --algo=TDR
python train.py --savedir=TDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --seed=5 --algo=TDR