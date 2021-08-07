#!/bin/sh

# 実行
# bash eval_average_reward.sh

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

python eval_average_reward.py --agent=Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2
python eval_average_reward.py --agent=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015
python eval_average_reward.py --agent=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015
python eval_average_reward.py --agent=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015
python eval_average_reward.py --agent=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015
python eval_average_reward.py --agent=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015