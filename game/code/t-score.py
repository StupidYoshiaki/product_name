import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

f1_without_aug = []
f1_aug = []

f1 = []
with open('./result/2011-2011_2012/f1/result/v2.txt', 'r') as f:
  l_strip = [s.rstrip() for s in f.readlines()]
  for l in l_strip:
    d = l.split(': ')
    name = d[0]
    try:
      score = d[1]
    except IndexError:
      continue
    if name == 'f1':
      f1.append(float(score))
      
# 偶数番目の配列
f1_even = [f1[i] for i in range(0, len(f1)) if i % 2 == 0]
f1_aug.append(f1_even)
# 奇数番目の配列
f1_odd = [f1[i] for i in range(0, len(f1)) if i % 2 == 1]
f1_without_aug.append(f1_odd)

  # # データ拡張していない場合の結果は配列の奇数番目に格納されている
  # for i in range(0, len(f1)):
  #   if i % 2 == 0:
  #     f1_aug.append(f1[i])
  #   else:
  #     f1_without_aug.append(f1[i])

f1_aug = np.array(f1_aug[0])
f1_without_aug = np.array(f1_without_aug[0])

# print(f1_aug)
# print(f1_without_aug)

t_stat, p_value = stats.ttest_ind(f1_aug, f1_without_aug)

print(t_stat, p_value)