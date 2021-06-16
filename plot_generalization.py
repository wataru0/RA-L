# 2021/3/23
# 各手法のgeneralizationを折れ線グラフでプロットするプログラム

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d # scipyのモジュールを使う
import argparse
from scipy import signal
import datetime as dt

# def arg_parser():
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--agent', help = 'put in agent name for evaluation', type = str, default = 'Baseline-16million-v3')
#     return parser.parse_args()

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

def main():
    # args = arg_parser()
    
    # filename = ["Baseline-16million-v3", "random-16million", "Curriculum2-v1", "Curriculum-v4-16million"]
    # filename = ["Baseline-16million-v3", "random-16million", "Curriculum-v4-16million"]
    # filename = ["Baseline-16million-v3"]
    # filename = ["Baseline", "UDR", "CDR-v1", "CDR-v2"]
    # filename = ["Baseline_CustomAnt", "UDR_CustomAnt", "CDR-v1_CustomAnt_bf1000", "CDR-v2_CustomAnt_bf1000", "CDR-v1_CustomAnt_upperfix_bf1000", "CDR-v2_CustomAnt_lowerfix_bf1000"]
    # filename = ["Baseline_CustomAnt", "UDR_CustomAnt", "CDR-v1_CustomAnt_upperfix_bf1000", "CDR-v2_CustomAnt_lowerfix_bf1000"]

    # filename = ["Baseline_CustomAnt-ReduceSRto0IfFallingDown", "UDR_CustomAnt-ReduceSRto0IfFallingDown", "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100", "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100", "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100", "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100"]
    # filename = ["Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2", "UDR_CustomAnt-ReduceSRto0IfFallingDown", "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100", "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100"]
    filename = ["Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2", "UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015", "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015", "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015"]
    label = {
        "Baseline-16million-v3":"Baseline", 
        "random-16million":"UDR", 
        "Curriculum-v4-16million":"CDR-v2", 
        # "Curriculum-v4-16million":"CDR", 
        "Curriculum2-v1":"CDR-v1",
        "Baseline":"Baseline",
        "UDR":"UDR",
        "CDR-v1":"easy-to-hard",
        "CDR-v2":"hard-to-easy",
        "Baseline_CustomAnt":'Baseline', 
        "UDR_CustomAnt":'UDR', 
        "CDR-v1_CustomAnt_bf1000":'easy-to-hard', 
        "CDR-v2_CustomAnt_bf1000":'hard-to-easy',
        "CDR-v1_CustomAnt_upperfix_bf1000":"easy-to-hard-fix",
        "CDR-v2_CustomAnt_lowerfix_bf1000":"hard-to-easy-fix",
        "Baseline_CustomAnt-ReduceSRto0IfFallingDown":"Baseline",
        "UDR_CustomAnt-ReduceSRto0IfFallingDown":"UDR", 
        "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100":"CDR-v1", 
        "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100":"CDR-v2", 
        "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_upperfix_bf100":"CDR-v1-fix", 
        "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_lowerfix_bf100":"CDR-v2-fix",
        "Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2":"Baseline",
        "UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015":"UDR", 
        "CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015":"CDR-v1", 
        "CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015":"CDR-v2",
    }

    # Create figure dir
    figdir = "./fig/all"
    os.makedirs(figdir,exist_ok=True)

    sns.set()
    plt.figure()
    fig, ax = plt.subplots()

    for file in filename:
        a = {}
        for seed in range(1, 6):
            # 既存の報酬関数で評価した結果が入っているディレクトリ
            # a[seed] = np.load("data100episodes/" + file + "/" + file + "_rewardForEachK_seed=" +  str(seed) + ".npy")
            
            # 改良した報酬関数の結果が入っているディレクトリ
            # a[seed] = np.load("./data_updateRfunc/" + file + "/" + file + "_rewardForEachK_seed=" +  str(seed) + ".npy")

            # 自作のgym環境での評価が入っているディレクトリ
            # ランダムに脚一本が故障する環境での評価が入っているディレクトリ
            a[seed] = np.load("./Data/CustomAnt/" + file + "/" + file + "_rewardForEachK_seed=" +  str(seed) + ".npy")

            # ランダムに関節2個壊れる環境での評価が入っているディレクトリ
            # a[seed] = np.load("./Data/CustomAnt_2JointBroken/" + file + "/" + file + "_rewardForEachK_seed=" +  str(seed) + ".npy")

        # smoothing
        for i in range(1,6):
            # default is (a[i], 51, 3)
            # 21 3/ 31 3/ 41 3
            a[i] = signal.savgol_filter(a[i],31,3)
        
        
        col = np.linspace(0.0,1.0,100)
        a1 = a[1].reshape(100,1)
        a1 = np.insert(a1, 0, col, axis=1)
        a2 = a[2].reshape(100,1)
        a2 = np.insert(a2, 0, col, axis=1)
        a3 = a[3].reshape(100,1)
        a3 = np.insert(a3, 0, col, axis=1)
        a4 = a[4].reshape(100,1)
        a4 = np.insert(a4, 0, col, axis=1)
        a5 = a[5].reshape(100,1)
        a5 = np.insert(a5, 0, col, axis=1)
        aa = np.concatenate([a1,a2,a3,a4,a5])
        # print(aa)
        # ndarray->pandas.DataFrame
        DF = pd.DataFrame(data=aa, columns=['k','AverageReward'], dtype='float')
        # print(DF)

        sns.lineplot(x="k", y="AverageReward", data=DF, ax=ax, label=label[file])

    ax.set_ylabel('Average Reward')
    ax.set_xlabel('k')
    ax.legend()


    # save fig
    plt.tight_layout()
    plt.tick_params(labelsize=10)
    ax.set_rasterized(True) 

    # type3 font を消す
    # plt.rcParams['text.usetex'] = True 
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] 
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Helvetica'
    now = dt.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S")
    plt.savefig(figdir + "generalization_{}.png".format(time))

if __name__=='__main__':
    main()