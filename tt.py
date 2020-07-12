import os
import vae_model as vm
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import dataReader as dr
import random

# a = [0.11,0.16,0.28,0.40,0.52,0.68,0.78,0.82,0.86,0.88,0.89,0.91,0.92,0.92,0.93,0.94,0.94]
#
# b = [0.11,0.20,0.31,0.44,0.67,0.68,0.78,0.82,0.86,0.88,0.89,0.91,0.92,0.92,0.93,0.94,0.94]
#
# input = np.array(dr.read_csv('record/model_cnn/'),dtype='float')[:,0]
# plt.show()

# a = list(range(60000))
# random.shuffle(a)
# dr.save_data(a,'standard_order.csv')

# a = [[1,2],[3,4],[5,6],7,8,9]
# c = np.array([1,2,3,4,4,45,5,6,6,7,7,8,2,2,1,3])
# b = [1,2,3]
#
# idx = c<5
# print(idx)
# xxx = np.array([x in b for x in c])

# ooo = (c==1)
# x = c[xxx]+100
# c[xxx] = x
# print(c)
# gap = int(100/50)
# idr = (np.array(range(len(c)))%gap == 0)
# print(c[idr])
# print(c[xxx])
# random.shuffle(xxx)
# print(xxx)


# a = np.array([[1,2,3,3],[1,2,3,3]])
# b = np.array([[1,2,3,3]])
#
# c = [[1,2,3,3],[1,2,3,3]]
#
# al = list(a)
# bl = list(b)
# al.extend(bl)
# print(al)
# # e.append(a)
# # e.append(b)
# # e.append(c)
# # print(e)
# # e = np.array(e)
# # e.resize([6,4])
# d = np.array(al)
# d.resize([3,4])
# # print(d)
# al.append(1)
# print(d)
# print(al)

a = np.array([1,3,2,4,5,3,32,2])
print(a)
a.sort()
print(a)