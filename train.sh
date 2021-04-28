#!/bin/sh
#train-isis.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash train.sh (savedir名，normalAntなど)

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

# 処理内容，seedを変えて五回トレーニング
python train.py --savedir=$1 --seed=1 --algo=$2
python train.py --savedir=$1 --seed=2 --algo=$2
python train.py --savedir=$1 --seed=3 --algo=$2
python train.py --savedir=$1 --seed=4 --algo=$2
python train.py --savedir=$1 --seed=5 --algo=$2