#!/bin/sh
# eval_for_bar.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash ~.sh

# python eval_for_bar.py --agent=Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2 --n_episodes=50
# python eval_for_bar.py --agent=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50
# python eval_for_bar.py --agent=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --n_episodes=50
# python eval_for_bar.py --agent=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --n_episodes=50
# python eval_for_bar.py --agent=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50
# python eval_for_bar.py --agent=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50

# python eval_for_bar.py --agent=Baseline_Ant-v2 --n_episodes=20
# python eval_for_bar.py --agent=Baseline_Ant-v2_k08 --n_episodes=20
# python eval_for_bar.py --agent=Baseline_Ant-v2_k06 --n_episodes=20
# python eval_for_bar.py --agent=Baseline_Ant-v2_k04 --n_episodes=20
# python eval_for_bar.py --agent=Baseline_Ant-v2_k02 --n_episodes=20
# python eval_for_bar.py --agent=Baseline_Ant-v2_k00 --n_episodes=20
python eval_for_bar.py --agent=Baseline_Ant-v2_k12 --n_episodes=20