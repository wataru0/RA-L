#!/bin/sh
#train_CDR.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash train_CDR.sh (savedir名，normalAntなど)

# 処理内容，seedを変えて五回トレーニング
python train_CDR.py --savedir=$1 --seed=1
python train_CDR.py --savedir=$1 --seed=2
python train_CDR.py --savedir=$1 --seed=3
python train_CDR.py --savedir=$1 --seed=4
python train_CDR.py --savedir=$1 --seed=5