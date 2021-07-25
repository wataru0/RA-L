#!/bin/sh

# 平均報酬の出力
python plot_bar.py --agent=Baseline_Ant-v2
python plot_bar.py --agent=Baseline_Ant-v2_k08
python plot_bar.py --agent=Baseline_Ant-v2_k06
python plot_bar.py --agent=Baseline_Ant-v2_k04
python plot_bar.py --agent=Baseline_Ant-v2_k02
python plot_bar.py --agent=Baseline_Ant-v2_k00
python plot_bar.py --agent=Baseline_Ant-v2_k12

# 平均進捗の出力
python plot_bar.py --agent=Baseline_Ant-v2 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k08 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k06 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k04 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k02 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k00 --plotType=forward
python plot_bar.py --agent=Baseline_Ant-v2_k12 --plotType=forward