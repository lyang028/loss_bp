import numpy as np
import random
import dataReader as dr
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import KL_div as kl

test_row= 100
test_col = 10000
#universe generator
def create_n_does_universe_test(n,col = test_col,row = test_row):
    anchors = []
    for i in range(n):
        randv = np.random.rand(col)
        anchors.append(np.random.rand(col)*10)

    universe_ori = np.array(np.random.randint(-1000,1000,size = [row,col]))/500
    part = int(row/n)
    universe = np.array(universe_ori[0:part])+anchors[0]

    for i in range(n-1):
        universe_head = universe_ori[(i+1)*part:(i+2)*part]
        universe = np.r_[universe,universe_head+anchors[i+1]]
    return universe

def generate_normal_distribution(sampleNo, mu, sigma):
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)
    return  s


# simulation
def E_dis(array1,array2):
    return np.linalg.norm(np.array(array1) - np.array(array2), ord=2)
def shortest_distance(a, universe, measure = E_dis):
    output = np.ones(len(universe))*10000
    for i in range(len(universe)):
        i_array = np.tile(universe[i], [len(a), 1])
        dis_array = np.linalg.norm(i_array - a, axis=1, ord=2)
        output[i] = dis_array.min()
    return  output
def ratio_simulation(universe,o_name,gap, epoch):
    mean_output = np.ones(epoch)
    mean_support = np.ones(epoch)
    std_output = np.ones(epoch)
    std_support = np.ones(epoch)
    rest_index = list(range(len(universe)))
    samples = []
    distribute_opath = o_name+'/distribute'
    data_opath = o_name+'/data'
    if not os.path.exists(o_name):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(o_name)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    if not os.path.exists(distribute_opath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(distribute_opath)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    if not os.path.exists(data_opath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(data_opath)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    for i in range(epoch):
        rest_index, removed_index = remove_samples(rest_index,gap)
        if len(rest_index) == 0:
            break
        select_bool = index_to_bool(removed_index,len(universe))
        rest_bool = index_to_bool(rest_index,len(universe))
        support_set = universe[rest_bool]
        samples_new = universe[select_bool]
        samples.extend(samples_new)
        # print(len(samples))
        shortest_dis = shortest_distance(samples, support_set)
        dr.save_data(shortest_dis, data_opath + '/' + str(i) + '.csv')

        plt.hist(shortest_dis, bins=100, normed=True, alpha=0.5,label='Distance')
        plt.savefig(distribute_opath+'/'+ str(i) +'.png')
        plt.close()

        mean_support[i] = np.mean(shortest_dis)
        std_support[i] = np.std(shortest_dis)
        all_dis = np.r_[shortest_dis, np.zeros(len(samples))]
        mean_output[i] = np.mean(all_dis)
        std_output[i] = np.std(all_dis)
        print(i)

    dr.save_data(mean_output, o_name+'/mean.csv')
    dr.save_data(std_output, o_name + '/std.csv')
    dr.save_data(mean_support, o_name + '/mean_support.csv')
    dr.save_data(std_support, o_name + '/std_support.csv')

    plt.plot(range(len(mean_output)), mean_output)
    plt.savefig(o_name+'/mean.png')
    plt.close()
    plt.plot(range(len(std_output)), std_output)
    plt.savefig(o_name+'/std.png')
    plt.close()
    plt.plot(range(len(mean_support)), mean_support)
    plt.savefig(o_name + '/mean_support.png')
    plt.close()
    plt.plot(range(len(std_support)), std_support)
    plt.savefig(o_name + '/std_support.png')
    plt.close()

def distance_simulation(universe, o_name):
    target = universe[0]
    i_array = np.tile(target, [len(universe), 1])
    dis_array = np.linalg.norm(i_array - universe, axis=1, ord=2)



#analysis tools

def mds_analysis(matrix,dim = 3,opath = '',d_opath = ''):
    weight_set = np.array(matrix, dtype=float)
    embedding = MDS(n_components=dim)
    X_transformed = embedding.fit_transform(weight_set)
    normal = np.linalg.norm(X_transformed,axis=1)
    print(sum(normal))
    draw_scatter3D(X_transformed,opath = opath)
    dr.save_data(X_transformed,d_opath)
    # print(X_transformed)
def mds_test():
    gap = 100
    for i in range(3,100):
        uni = create_n_does_universe_test(10,row=100,col=i*gap)
        mds_analysis(uni, opath='MDS_test'+'/image/'+str(i)+'.png',d_opath='MDS_test'+'/data/'+str(i)+'.csv')

def analysis_mds(path_array,opath = ''):
    stds= []
    avgs = []
    cvs = []
    file_ls = os.listdir(path_array)

    def sort_key(e):
        epoch_str = e.split('.')
        return int(epoch_str[0])
    file_ls.sort(key=sort_key)

    for file in file_ls:
        mat = np.array(dr.read_csv(path_array+'/'+file),dtype=float)
        mn = np.linalg.norm(mat,axis=1)
        std = np.std(mn)
        avg = np.mean(mn)
        cv = std/avg
        avgs.append(avg)
        stds.append(std)
        cvs.append(cv)

        print('ok')
    if len('opath')!=0:
        dr.save_data(stds,opath+'/stds.csv')
        dr.save_data(avgs, opath + '/avgs.csv')
        dr.save_data(cvs, opath + '/cvs.csv')

        plt.plot(range(len(stds)),stds)
        plt.savefig(opath+'/stds.png')
        plt.close()

        plt.plot(range(len(avgs)), avgs)
        plt.savefig(opath + '/avgs.png')
        plt.close()

        plt.plot(range(len(cvs)), cvs)
        plt.savefig(opath + '/cvs.png')
        plt.close()
def draw_scatter3D(array, data_address='none', xaxis='blank', yaxis='blank', ratio=(4, 4), ylimitaion=(400, 500),
                       axis=[], range_select=[], open_color=True,opath = ''):
    plt.rcParams['figure.figsize'] = ratio
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if open_color:
        colors = list(range(len(array)))
        ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=colors)
    else:
        ax.scatter(array[:, 0], array[:, 1], array[:, 2])
    if len(axis) != 0:
        plt.axis(axis)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if len(opath) != 0:
        plt.savefig(opath)
    else:
        plt.show()

def E_dis(array1,array2):
    return np.linalg.norm(np.array(array1) - np.array(array2), ord=2)

def shortest_distance(a, universe, measure = E_dis):
    output = np.ones(len(universe))*10000
    for i in range(len(universe)):
        i_array = np.tile(universe[i], [len(a), 1])
        dis_array = np.linalg.norm(i_array - a, axis=1, ord=2)
        output[i] = dis_array.min()
    return  output

def remove_list(iterable: list, target: list):
    return [n for n in iterable if n not in target]
def randomized_sampling(set:list, amount:int):
    if len(set)< amount:
        return  set
    select = [True]*amount
    select.extend([False]*(len(set)-amount))
    random.shuffle(select)
    return set[select],select

def remove_samples(set:list,amount:int):
    if len(set)<amount:
        return []
    removed_elements = []
    for i in range(amount):
        index = random.randint(0,len(set)-1)
        removed_elements.append(set[index])
        del set[index]
    # print(len(set))
    return set, removed_elements

def index_to_bool(index:list,length):
    output = [False]*length
    for i in index:
        output[i] = True
    return output
def bool_to_index(bools:list):
    return np.nonzero(bools)

import os

def distribution_test(array, gaps = 100, select_range = [], compare_with_normal = True):
    if len(select_range) ==0:
        fanwei = np.linspace(array.min(),array.max(),gaps, endpoint=True)
    else:
        fanwei = np.linspace(select_range[0],select_range[1],gaps, endpoint=True)
    normal_dis = generate_normal_distribution(100000,np.mean(array),np.std(array)) #has no non-zero elements

    groups = pd.cut(array, fanwei, right=False)
    groups_normal = pd.cut(normal_dis, fanwei, right=False)
    frequence = groups.value_counts()
    frequence_normal = groups_normal.value_counts()
    output = frequence.values
    output_normal = frequence_normal.values

    output_normal_bool = output_normal == 0
    output_normal[output_normal_bool] = np.ones(sum(output_normal_bool))
    test = sum(output_normal == 0)
    return  output/sum(output), output_normal/(sum(output_normal)+len(output_normal_bool)), fanwei[1:]

def batch_distance_to_distribution(path,opath):
    file_ls = os.listdir(path)
    def sort_key(e):
        epoch_str = e.split('.')
        return int(epoch_str[0])
    file_ls.sort(key=sort_key)
    for file in file_ls:
        array = np.array(dr.read_csv(path+'/'+file),dtype=float)[:,0]
        dis, normal_dis, fanwei= distribution_test(array)
        dr.save_data(dis,opath + '/distribution/'+file)
        dr.save_data(normal_dis,opath+'/normal_distribution/'+file)
        dr.save_data(fanwei, opath + '/fanwei/'+file)

def batch_kl_test(path_d1, path_d2,opath):
    file_ls = os.listdir(path_d1)

    def sort_key(e):
        epoch_str = e.split('.')
        return int(epoch_str[0])
    output = []
    file_ls.sort(key=sort_key)
    for file in file_ls:
        array_d1= np.array(dr.read_csv(path_d1 + '/' + file), dtype=float)[:, 0]
        array_d2= np.array(dr.read_csv(path_d2 + '/' + file), dtype=float)[:, 0]
        output.append(kl.KL_div(array_d1,array_d2,activation=False))

    dr.save_data(output, opath)

## random distribution normalization
# universe = create_universe()
# ratio_simulation(universe,'QMCM_simulation',10 , 1000)

#two does of distribution
# universe = create_two_does_universe()
# ratio_simulation(universe,'QMCM_simulation_two_does',10 , 1000)






# universe = create_n_does_universe_test(10)
# draw_scatter3D(universe)
# mds_analysis(universe)
# ratio_simulation(universe,'QMCM_simulation_one_does',10, 1000)

# universe = create_n_does_universe_test(10)
# ratio_simulation(universe,'QMCM_simulation_one_does',10, 1000)



# universe = create_n_does_universe_test(1)
# ratio_simulation(universe,'QMCM_simulation_one_does',10, 1000)
# batch_distance_to_distribution('QMCM_simulation/ data','QMCM_simulation/distribute')
# batch_distance_to_distribution('QMCM_simulation_one_does/data','QMCM_simulation_one_does/distribute')
# batch_kl_test('QMCM_simulation/distribute/distribution', 'QMCM_simulation/distribute/normal_distribution','QMCM_simulation/distribute/KL_div.csv')

#distribtion test
# path = 'figures'
# file = 'alex_dis.csv'
# array = np.array(dr.read_csv(path+'/'+file),dtype=float)[:,0]
# print(np.mean(array))
# print(np.std(array))
# print(np.std(array)/np.mean(array))
# dis, normal_dis, fanwei = distribution_test(array,gaps=100)
# plt.plot(range(len(dis)),dis,label = 'dis')
# plt.plot(range(len(normal_dis)),normal_dis,label = 'normal')
# plt.legend()
# plt.show()
# dr.save_data(dis, path + '/distribution/' + file)
# dr.save_data(normal_dis, path + '/normal_distribution/' + file)
# dr.save_data(fanwei, path + '/fanwei/' + file)

# path = 'figures'
# file = 'resnet_dis.csv'
# array = np.array(dr.read_csv(path+'/'+file),dtype=float)[:,0]
# print(np.mean(array))
# print(np.std(array))
# print(np.std(array)/np.mean(array))
# dis, normal_dis, fanwei = distribution_test(array,gaps=10)
# plt.plot(range(len(dis)),dis,label = 'dis')
# plt.plot(range(len(normal_dis)),normal_dis,label = 'normal')
# plt.legend()
# plt.show()

# path = 'figures'
# file = 'mnist_distance.csv'
# array = np.array(dr.read_csv(path+'/'+file),dtype=float)[:,0]
# print(np.mean(array))
# print(np.std(array))
# print(np.std(array)/np.mean(array))
# dis, normal_dis, fanwei = distribution_test(array,gaps=10)
# plt.plot(range(len(dis)),dis,label = 'dis')
# plt.plot(range(len(normal_dis)),normal_dis,label = 'normal')
# plt.legend()
# plt.show()


# dr.save_data(dis, path + '/distribution/' + file)
# dr.save_data(normal_dis, path + '/normal_distribution/' + file)
# dr.save_data(fanwei, path + '/fanwei/' + file)

#mds_test
# mds_test()
analysis_mds('MDS_test/data','MDS_test/analysis')

