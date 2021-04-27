# Uniform Domain Randomization Algorithm
# 2021/04/27
# パラメータ範囲を一様にサンプリングする従来のドメインランダマイゼーション手法 
# 今まで一つのファイルでトレーニングを回していたのをクラスごとにしっかり分けようというもの
# train.pyを実行する際に，gym環境をラッパーすることで実装する

import gym
import numpy as np
import random