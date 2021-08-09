# 2021/08/07
# 保存してあるndarray dataから棒グラフをプロットするプログラム
# average progress用

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

# create fig save dir
figdir = "./fig/AverageProgress/"
os.makedirs(figdir,exist_ok=True)

# savedir = "./Data/AverageReward/"
savedir = "./Data/AverageReward/k00-k05_progress/"

# agent name
agentName = ['Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2', 'UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015', 'CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015'] 
label = np.array(["Baseline", "UDR", "LDR_easy2hard", "LDR_hard2easy", "ADR_easy2hard", "ADR_hard2easy"])

plainProgress = np.load(savedir + agentName[0] + "/plainAverageProgress.npy")
for i in range(1,len(agentName)):
    plainProgress = np.append(plainProgress, np.load(savedir + agentName[i] + "/plainAverageProgress.npy"))

plainProgressError = np.load(savedir + agentName[0] + "/plainProgressError.npy")
for i in range(1,len(agentName)):
    plainProgressError = np.append(plainProgressError, np.load(savedir + agentName[i] + "/plainProgressError.npy"))
print(plainProgress)
# print(plainError)

brokenProgress = np.load(savedir + agentName[0] + "/brokenAverageProgress.npy")
for i in range(1,len(agentName)):
    brokenProgress = np.append(brokenProgress, np.load(savedir + agentName[i] + "/brokenAverageProgress.npy"))

brokenProgressError = np.load(savedir + agentName[0] + "/brokenProgressError.npy")
for i in range(1,len(agentName)):
    brokenProgressError = np.append(brokenProgressError, np.load(savedir + agentName[i] + "/brokenProgressError.npy"))
print(brokenProgress)
# print(brokenError)

plt.figure
sns.set()
fig, ax = plt.subplots()
width = 0.35
x = np.arange(len(plainProgress))

# plot
ax.bar(x-width/2, plainProgress, width=width, color='blue', align='center', label='plain', yerr=plainProgressError, capsize=3)
ax.bar(x+width/2, brokenProgress, width=width, color='green', align='center', label='broken', yerr=brokenProgressError, capsize=3)

ax.set_ylabel("Average Progress")
plt.tick_params(labelsize=7)
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc='upper left')

now = dt.datetime.now()
time = now.strftime('%Y%m%d-%H%M%S')
plt.savefig(figdir + 'averageProgress_{}.png'.format(time))