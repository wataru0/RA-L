# 2021/08/07
# 保存してあるndarray dataから棒グラフをプロットするプログラム

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

# create fig save dir
figdir = "./fig/AverageReward/"
os.makedirs(figdir,exist_ok=True)

savedir = "./Data/AverageReward/"
# savedir = "./Data/AverageReward/Ant-v2/"

# agent name
agentName = ['Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2', 'UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015', 'CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015', 'CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015'] 

# ndarray load
# print(np.load(savedir + 'Baseline_CustomAnt-ReduceSRto0IfFallingDown-v2' + "/plainAverageReward.npy"))
# print(np.load(savedir + 'UDR_CustomAnt-ReduceSRto0IfFallingDown_k0015' + "/plainAverageReward.npy"))
# print(np.load(savedir + 'LCL-v1_CustomAnt-ReduceSRto0IfFallingDown_k0015' + "/plainAverageReward.npy"))
# print(np.load(savedir + 'LCL-v2_CustomAnt-ReduceSRto0IfFallingDown_k0015' + "/plainAverageReward.npy"))
# print(np.load(savedir + 'CDR-v1_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015' + "/plainAverageReward.npy"))
# print(np.load(savedir + 'CDR-v2_CustomAnt-ReduceSRto0IfFallingDown_bf100_k0015' + "/plainAverageReward.npy"))
# plain
plainReward = np.load(savedir + agentName[0] + "/plainAverageReward.npy")
for i in range(1,len(agentName)):
    plainReward = np.append(plainReward, np.load(savedir + agentName[i] + "/plainAverageReward.npy"))

plainError = np.load(savedir + agentName[0] + "/plainError.npy")
for i in range(1,len(agentName)):
    plainError = np.append(plainError, np.load(savedir + agentName[i] + "/plainError.npy"))
print(plainReward)
# print(plainError)

brokenReward = np.load(savedir + agentName[0] + "/brokenAverageReward.npy")
for i in range(1,len(agentName)):
    brokenReward = np.append(brokenReward, np.load(savedir + agentName[i] + "/brokenAverageReward.npy"))

brokenError = np.load(savedir + agentName[0] + "/brokenError.npy")
for i in range(1,len(agentName)):
    brokenError = np.append(brokenError, np.load(savedir + agentName[i] + "/brokenError.npy"))
print(brokenReward)
# print(brokenError)

plt.figure
sns.set()
fig, ax = plt.subplots()
width = 0.35
x = np.arange(len(plainReward))
label = np.array(["Baseline", "UDR", "LDR_easy2hard", "LDR_hard2easy", "ADR_easy2hard", "ADR_hard2easy"])

# plot
ax.bar(x-width/2, plainReward, width=width, color='blue', align='center', label='plain', yerr=plainError)
ax.bar(x+width/2, brokenReward, width=width, color='green', align='center', label='broken', yerr=brokenError)

ax.set_ylabel("Average Reward")
plt.tick_params(labelsize=7)
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc='upper left')

now = dt.datetime.now()
time = now.strftime('%Y%m%d-%H%M%S')
plt.savefig(figdir + 'averageReward-v2_{}.png'.format(time))