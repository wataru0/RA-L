#!/bin/sh
# eval_for_bar.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash ~.sh

python eval_for_bar.py --agent=Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2 --n_episodes=50
python eval_for_bar.py --agent=UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50
python eval_for_bar.py --agent=CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --n_episodes=50
python eval_for_bar.py --agent=CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015 --n_episodes=50
python eval_for_bar.py --agent=LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50
python eval_for_bar.py --agent=LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015 --n_episodes=50
