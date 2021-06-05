#!/bin/sh

# 実行
# bash eval_generalization_customEnv.sh

#$0:スクリプト名
#$1:1番目の引数
#$2:2番目の引数

# python eval_generalization_customEnv.py --agent=Baseline-16million-v3 --n_episodes=10
# python eval_generalization_customEnv.py --agent=random-16million --n_episodes=10
# python eval_generalization_customEnv.py --agent=Curriculum-v4-16million --n_episodes=10
# python eval_generalization_customEnv.py --agent=Curriculum2-v1 --n_episodes=10

# 新しくトレーニングしたもの（従来通りの普通の終了条件）
# python eval_generalization_customEnv.py --agent=Baseline --n_episodes=10
# python eval_generalization_customEnv.py --agent=UDR --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v1 --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v2 --n_episodes=10

# 新しくトレーニングしたもの（従来の終了条件に転倒したら終了するを追加して訓練したもの）
# python eval_generalization_customEnv.py --agent=Baseline_CustomAnt --n_episodes=10
# python eval_generalization_customEnv.py --agent=UDR_CustomAnt --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v1_CustomAnt_bf1000 --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v2_CustomAnt_bf1000 --n_episodes=10

# python eval_generalization_customEnv.py --agent=CDR-v1_CustomAnt_upperfix_bf1000 --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v2_CustomAnt_lowerfix_bf1000 --n_episodes=10

# 転倒時のsurvive rewardを０にしたもの
# python eval_generalization_customEnv.py --agent=Baseline_CustomAnt-ReduceSRto0IfFallingDown --n_episodes=10
# python eval_generalization_customEnv.py --agent=UDR_CustomAnt-ReduceSRto0IfFallingDown --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100 --n_episodes=10
# python eval_generalization_customEnv.py --agent=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100 --n_episodes=10
python eval_generalization_customEnv.py --agent=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100 --n_episodes=10
python eval_generalization_customEnv.py --agent=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100 --n_episodes=10