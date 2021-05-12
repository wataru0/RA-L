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
python eval_generalization_customEnv.py --agent=Baseline_CustomAnt --n_episodes=10
python eval_generalization_customEnv.py --agent=UDR_CustomAnt --n_episodes=10
python eval_generalization_customEnv.py --agent=CDR-v1_CustomAnt --n_episodes=10
python eval_generalization_customEnv.py --agent=CDR-v2_CustomAnt --n_episodes=10