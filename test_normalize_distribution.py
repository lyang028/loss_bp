import dataReader as dr
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = 'QMCM_simulation/ data/'
file_ls = os.listdir(data_path)
def sort_key(e):
    value = os.path.splitext(e)
    return int(value[0])
file_ls.sort(key=sort_key)

mean_support = np.zeros(len(file_ls))
std_support = np.zeros(len(file_ls))
mean_all = np.zeros(len(file_ls))
std_all = np.zeros(len(file_ls))
i = 0
for file in file_ls:
    distance = np.array(dr.read_csv(data_path+ file), dtype=float)[:, 0]
    if len(distance) == 0:
        break
    mean = np.mean(distance)
    std = np.std(distance)
    np.random.seed(0)
    s = np.random.normal(mean, std, 10000)

    # plt.hist(s, bins=100, normed=True, alpha=0.5,label='Normal')
    # plt.hist(distance, bins=100, normed=True, alpha=0.5,label='Distance')
    # plt.legend()
    # plt.savefig('QMCM_simulation/images/'+str(i)+'.png')
    # plt.close()

    mean_support[i] = np.mean(distance)
    std_support[i] = np.std(distance)

    all_dis = np.r_[distance, np.zeros(10000-len(distance))]
    mean_all[i] = np.mean(all_dis)
    std_all[i] = np.std(all_dis)

    print(i)
    i = i+1

o_name = 'QMCM_simulation'
plt.plot(range(len(mean_all)), mean_all)
plt.savefig(o_name+'/mean.png')
plt.close()
plt.plot(range(len(std_all)), std_all)
plt.savefig(o_name+'/std.png')
plt.close()
plt.plot(range(len(mean_support)), mean_support)
plt.savefig(o_name + '/mean_support.png')
plt.close()
plt.plot(range(len(std_support)), std_support)
plt.savefig(o_name + '/std_support.png')
plt.close()
