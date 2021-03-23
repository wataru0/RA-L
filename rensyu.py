# 2021/3/23
# 各手法のgeneralizationを折れ線グラフでプロットするプログラム

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d # scipyのモジュールを使う
import argparse
from scipy import signal

# スプライン補間でスムージング
def spline_interp(in_x, in_y):
    out_x = np.linspace(np.min(in_x), np.max(in_x), np.size(in_x)*1000) # もとのxの個数より多いxを用意
    func_spline = interp1d(in_x, in_y, kind='cubic') # cubicは3次のスプライン曲線
    out_y = func_spline(out_x) # func_splineはscipyオリジナルの型

    return out_x, out_y

#移動平均でスムージング
def moving_avg(in_x, in_y):
    np_y_conv = np.convolve(in_y, np.ones(7)/float(7), mode='valid') # 畳み込む
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

    return out_x_dat, np_y_conv


baseline = np.load("data/Baseline-16million-v3_rewardForEachK.npy")
# print(baseline[0,:,:])
# print(baseline[1,:,:].shape)

baseline_ave = np.zeros(100)
# 各seedの各kでの平均報酬を格納する辞書，k:[0,...,100]
b = {}
for seed in range(0,5):
    b_array = baseline[seed,:,:]
    b_array = np.sum(b_array, axis=0) 
    for i in range(len(b_array)):
        b_array[i] = b_array[i]/100 
    b[seed] = b_array
print(b[1].shape)

# # 各seedの平均報酬をまとめる
# for i in range(100):
#     for seed in range(0,5):
#         baseline_ave[i] += b[seed][i]
# baseline_ave = baseline_ave/100

# print(baseline_ave) # 各kでの平均報酬が出た！

a = np.load("data/Baseline-16million-v3/Baseline-16million-v3_rewardForEachK_seed=1.npy")
print(a.shape, a)
    