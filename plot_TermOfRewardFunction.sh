#!/bin/sh

# 実行
# bash plot_TermOfRewardFunction.sh

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

python plot_TermOfRewardFunction.py --term=forward
python plot_TermOfRewardFunction.py --term=ctrl
python plot_TermOfRewardFunction.py --term=contact
python plot_TermOfRewardFunction.py --term=survive