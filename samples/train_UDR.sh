#!/bin/sh
#train_UDR.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash train_UDR.sh (savedir名，normalAntなど)

# 処理内容，seedを変えて五回トレーニング
python train_UDR.py --savedir=$1 --seed=1
python train_UDR.py --savedir=$1 --seed=2
python train_UDR.py --savedir=$1 --seed=3
python train_UDR.py --savedir=$1 --seed=4
python train_UDR.py --savedir=$1 --seed=5