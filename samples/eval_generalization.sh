#!/bin/sh

# 実行
# bash eval_generalization.sh

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

python eval_generalization.py --agent=Baseline-16million-v3
python eval_generalization.py --agent=random-16million
python eval_generalization.py --agent=Curriculum-v4-16million
python eval_generalization.py --agent=Curriculum2-v1