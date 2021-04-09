#!/bin/sh

# 実行
# bash eval_generalization_updateRfunc.sh

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

python eval_generalization_updateRfunc.py --agent=Baseline-16million-v3 --n_episodes=10
python eval_generalization_updateRfunc.py --agent=random-16million --n_episodes=10
python eval_generalization_updateRfunc.py --agent=Curriculum-v4-16million --n_episodes=10
python eval_generalization_updateRfunc.py --agent=Curriculum2-v1 --n_episodes=10