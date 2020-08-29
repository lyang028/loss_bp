import KL_div as KL
import dataReader as dr
import cnn_model as cm
import numpy as np
import random

import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from itertools import combinations, permutations

def reletive_information(dataset):
    s = KL.resize_layer(sum(dataset)/len(dataset))
    ri = 0
    for si in dataset:
        s_flat = KL.resize_layer(si)
        ri = ri+KL.KL_div(s_flat,s)+KL.KL_div(s,s_flat)
    return ri

def reletive_information_centerize(dataset,datacenter):
    s = KL.resize_layer(datacenter)
    ri = 0
    for si in dataset:
        s_flat = KL.resize_layer(si)
        ri = ri+KL.KL_div(s_flat,s)+KL.KL_div(s,s_flat)
    return ri

def reletivinformation_radiation(ar1, ar2):
    arf1 = KL.resize_layer(ar1)
    arf2 = KL.resize_layer(ar2)

    return KL.KL_div(arf1,arf2)+KL.KL_div(arf2,arf1)

def information_cleanliness(xtrain, ytrain,label_set):
    RI = reletive_information(xtrain)
    labelsRI = []
    for label in label_set:
        idx = ytrain == label
        x = xtrain[idx]
        labelsRI.append(reletive_information(x))
    return labelsRI, sum(labelsRI)/RI
def multilabel_RI():
    kl = []
    for i in range(10):
        idx = cm.y_index_train<=i
        test = cm.x_train[idx]
        RI = reletive_information(test)
        print(RI)
        kl.append(RI)
    return kl
def half_half_RI():
    A = cm.x_train[cm.y_index_train == 2][:2000,:,:]
    B = cm.x_train[cm.y_index_train == 0][:2000,:,:]
    A_half = A[:1000,:,:]
    B_half = B[:1000,:,:]
    C = np.array([A_half,B_half])
    C.resize([2000,28,28,1])
    # C = np.array((.to).flatten()
    print(reletive_information(A),reletive_information(B),reletive_information(C))

def mix_dataset(xtrains, ytrains, dataset_size, label_size):
    if len(xtrains) == 1:
        return xtrains[0][:dataset_size,:,:], ytrains[0][:dataset_size,:]
    mix_length = int(dataset_size / label_size)
    x_train_mix = []
    y_train_mix = []

    for xtrain in xtrains:
        x_train_mix.extend(xtrain[:mix_length,:,:])#for 2D images
    for ytrain in ytrains:
        y_train_mix.extend(ytrain[:mix_length,:])

    d_samples = dataset_size - mix_length * len(xtrains)
    if d_samples>0:
        x_train_mix.extend(xtrains[0][-d_samples:,:,:])
        y_train_mix.extend(ytrains[0][-d_samples:, :])
    x_train_mix = np.array(x_train_mix)
    x_train_mix.resize(xtrains[0].shape)
    y_train_mix = np.array(y_train_mix)
    y_train_mix.resize(ytrains[0].shape)

    return  x_train_mix, y_train_mix
def mixed_RI(label_set, test_length,output_address):
    xtrains = []
    ytrains = []
    xtrain_mix = []
    ytrain_mix = []
    for label in label_set:
        idx1 = (cm.y_index_train == label)
        x_train1 = cm.x_train[idx1][:test_length, :, :]
        y_train1 = cm.y_train[idx1][:test_length, :]
        xtrains.append(x_train1)
        ytrains.append(y_train1)
        x_mix, y_mix = mix_dataset(xtrains, ytrains, test_length, len(xtrains))
        xtrain_mix.append(x_mix)
        ytrain_mix.append(y_mix)
    RIs = []
    for dataset in xtrain_mix:
        ri = reletive_information(dataset)
        RIs.append(ri)

    dr.save_data(RIs,output_address)

def average_num(output_address):
    kl = []
    for i in range(10):
        idx = cm.y_index_train == i
        test = cm.x_train[idx]
        RI = reletive_information(test)/len(idx)
        print(RI)
        kl.append(RI)
    dr.save_data(kl, output_address)
    return kl

def quality_evaluate(error_rate):
    xtrain = cm.x_train
    ytrain = cm.y_index_train
    ICs = []
    real_error = []

    for error in error_rate:
        x = xtrain
        y = np.array(ytrain)
        y_error = y[:int(len(y)*error)]
        y_keep = y[int(len(y)*error):]
        random.shuffle(y_error)

        yerrorlist = list(y_error)
        ykeeplist = list(y_keep)
        yerrorlist.extend(ykeeplist)

        y_new = np.array(yerrorlist)
        real_error.append(1-sum(y == ytrain)/len(y))
        print()
        label_ris, IC  = information_cleanliness(x,y_new,[0,1,2,3,4,5,6,7,8,9])
        ICs.append(IC)
        # print(label_ris)
        print(IC)
    dr.save_data(ICs,'data_evaluation_minst/error_label_detection.csv')
    dr.save_data(real_error, 'data_evaluation_minst/real_error_detection.csv')
    plt.plot(real_error,ICs)
    plt.show()
    # plt.savefig('data_evaluation_minst/error_label_detection.png')

def test_data_center(accuracy, data_length,loop,mark):# accuracy is the float saving bits
    idx = cm.y_index_train == 0
    xtrain = cm.x_train[idx]
    sequnce = list(range(len(xtrain)))
    random.shuffle(sequnce)
    output_path = mark
    if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_path)
    for i in range(loop):
        idx = sequnce[data_length * i:data_length * (i + 1)]
        x = xtrain[idx]
        output_subpath = mark+'/' + str(i)
        if not os.path.exists(output_subpath):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(output_subpath)
        for j in range(len(x)):
            file_name = output_subpath+'/'+str(j)+'.png'
            print(file_name)
            cv2.imwrite(file_name,x[j]*255)
        frame_test_datacenter(x,accuracy,output_subpath)


def optimzie_key(f,accuracy):
    f = round(f,accuracy+1)
    str_f = str(f)
    if f ==0 :
        str_f = '0.0'
    return  str_f

def frame_test_datacenter(datalist,accuracy,datapath):
    sample_rate = pow(0.01,accuracy)
    test_center = []
    test_center_w = test_datacenter_weights(len(datalist),1,sample_rate)
    # average_w = np.ones(len(datalist))*(1/len(datalist)) #add average weight
    # test_center_w.append(average_w.tolist())
    # print(test_center_w)
    for weights in test_center_w:
        output = np.zeros(datalist[0].shape)
        for i in range(len(datalist)):
            output += datalist[i]*weights[i]
        test_center.append(output)

    RI = []
    for center in test_center:
        RI.append(reletive_information_centerize(datalist,center))
    #     compare
    for i in range(len(test_center)):
        for j in range(0, len(test_center)-i-1):
            if RI[j]>RI[j+1]:
                temp = RI[j]
                RI[j] = RI[j+1]
                RI[j+1] = temp

                temp_weights = test_center_w[j]
                test_center_w[j] = test_center_w[j+1]
                test_center_w[j+1] = temp_weights

    # for i in range(5):
    #     print(RI[i])
    #     print(test_center_w[i])
    dr.save_data(RI,datapath+'/RI.csv')
    dr.save_data(test_center_w, datapath+'/test_center_w.csv')
    print(RI[0])
    print(test_center_w[0])

    zz = np.array(RI, dtype='float')

    dim = list(range(len(datalist)))
    dim_set = permutations(dim,2)
    for dim_select in dim_set:
        key = []
        for w in test_center_w:
            w_d1 = optimzie_key(w[dim_select[0]],accuracy+1)
            w_d2 = optimzie_key(w[dim_select[1]],accuracy+1)
            mark1 = 'x' + str(w_d1)
            mark2 = 'y' + str(w_d2)
            key.append( mark1+ mark2)
        dic = dict(zip(key, zz))
        print(len(dic))

        xx = np.arange(0, 1 + sample_rate, sample_rate)
        yy = np.arange(0, 1 + sample_rate, sample_rate)
        X, Y = np.meshgrid(xx, yy)

        Z = np.zeros(X.shape)
        for i in range(len(xx)):
            for j in range(len(xx)):
                if X[i, j] + Y[i, j] > 1:
                    Z[i, j] = 0
                else:
                    w_d1 = optimzie_key(X[i, j],accuracy+1)
                    w_d2 = optimzie_key(Y[i, j], accuracy + 1)
                    mark1 =  'x' + str(w_d1)
                    mark2 =  'y' + str(w_d2)
                    print(mark1 + mark2)
                    Z[i, j] = dic[mark1+mark2]
        fig = plt.figure()  # 定义新的三维坐标轴
        ax3 = plt.axes(projection='3d')
        ax3.plot_surface(X, Y, Z, cmap='rainbow')
        ax3.set_xlabel('sample '+ str(dim_select[0])+' weight', fontsize=10, rotation=150)
        ax3.set_ylabel('sample '+ str(dim_select[1])+' weights')
        ax3.set_zlabel('RI')
        plt.savefig(datapath+str(dim_select[0])+'_'+str(dim_select[1])+'.png')
        plt.close()




def test_datacenter_weights(length, rest_ratio,sample_rate):
    if length == 1:
        return [[rest_ratio]]
    if rest_ratio < -0.00001:
        output = np.zeros(length)
        return [output.tolist()]
    possible = []
    length -= 1

    sample_time = int(1/sample_rate)+1
    for i in range(sample_time):
        current_weight = i * sample_rate
        restw_sum = rest_ratio - current_weight
        if restw_sum < -0.000001:
            break

        weights_rest = test_datacenter_weights(length, restw_sum,sample_rate)
        for j in range(len(weights_rest)):
            weights_rest[j].append(current_weight)
        possible.extend(weights_rest)

    return possible



def verify_disequition(d1,d2,d3):
    if d1+d2<d3:
        return False
    elif d1+d3<d2:
        return False
    elif d2+d3<d1:
        return False
    return True

def frame_test_RI_tri(loop):
    samples = []

    samples.append(cm.x_train[0])
    samples.append(cm.x_train[1])

    test_RI_triangulation(samples,0.01)

def test_RI_triangulation(samples,test_times):
    samples[0]
    ar = np.arange(0,1,test_times)
    dis3 = reletivinformation_radiation(samples[0],samples[1])
    for rate in ar:
        if rate<0.4 or rate>0.6:
            continue
        noise = np.random.randint(0,10,samples[0].shape)/100
        s = samples[0]*rate+samples[1]*(1-rate)+noise
        # s = noise
        dis1 = reletivinformation_radiation(samples[0],s)
        dis2 = reletivinformation_radiation(samples[1],s)
        if not verify_disequition(dis1,dis2,dis3):
            print('error')
            print(dis1)
            print(dis2)
            print(dis3)




# kl = multilabel_RI()
# print(kl)
# dr.save_data(kl,'data_evaluation_minst/multilabel_RI.csv')

# half_half_RI()
# mixed_RI([0,1,2,3,4,5,6,7,8,9],5000,'data_evaluation_minst/mixed_label_RI.csv')
# average_num('data_evaluation_minst/averagenumber_label_RI.csv')

# quality_evaluate([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

# def reletive_information_test(dataset):
#     # length = len(KL.resize_layer(dataset[0]))
#     # s = np.ones(length)
#     s = KL.resize_layer(sum(dataset)/len(dataset))
#     ri = 0
#     for si in dataset:
#         s_flat = KL.resize_layer(si)
#         ri = ri+KL.KL_div(s_flat,s)
#     return ri
#
# def multilabel_RI_test():
#     kl = []
#     for i in range(10):
#         idx = cm.y_index_train<=i
#         test = cm.x_train[idx]
#         RI = reletive_information_test(test)
#         print(RI)
#         kl.append(RI)
#     return kl
#
# multilabel_RI_test()

# b = np.c_[a[:,1],a[:,0],a[:,2]]
# c = np.c_[a[:,1],a[:,2],a[:,0]]
# all = np.r_[a,b,c]

# print(a)
# test_data_center(1,3,1,'data_evaluation_minst/data_center_test/mark_A_details')


# frame_test_RI_tri(10)


def figure_grid(length, rest_ratio,sample_rate):
    if length == 1:
        return [[rest_ratio]]
    possible = []
    length -= 1

    sample_time = int(2/sample_rate)+1
    for i in range(sample_time):
        current_weight = i * sample_rate -1
        restw_sum = rest_ratio - current_weight
        weights_rest = figure_grid(length, restw_sum,sample_rate)
        for j in range(len(weights_rest)):
            weights_rest[j].append(current_weight)
        possible.extend(weights_rest)

    return possible
def figure3D_test_datacenter(datalist,accuracy,datapath):
    sample_rate = pow(0.01,accuracy)
    print(sample_rate)
    test_center = []
    test_center_w = figure_grid(len(datalist),1,sample_rate)
    # average_w = np.ones(len(datalist))*(1/len(datalist)) #add average weight
    # test_center_w.append(average_w.tolist())
    # print(test_center_w)
    print(len(test_center_w))
    index = 0
    for weights in test_center_w:
        output = np.zeros(datalist[0].shape)
        for i in range(len(datalist)):
            output += datalist[i]*weights[i]
        test_center.append(output)
    RI = []
    for center in test_center:
        RI.append(reletive_information_centerize(datalist,center))
        print(index)
        index += 1
    #     compare
    print('centerget')
    # for i in range(len(test_center)):
    #     for j in range(0, len(test_center)-i-1):
    #         if RI[j]>RI[j+1]:
    #             temp = RI[j]
    #             RI[j] = RI[j+1]
    #             RI[j+1] = temp
    #
    #             temp_weights = test_center_w[j]
    #             test_center_w[j] = test_center_w[j+1]
    #             test_center_w[j+1] = temp_weights

    # for i in range(5):
    #     print(RI[i])
    #     print(test_center_w[i])
    RI_min = RI[0]
    RI_max = RI[-1]
    dr.save_data(RI,datapath+'/RI.csv')
    dr.save_data(test_center_w, datapath+'/test_center_w.csv')
    print(RI[0])
    print(test_center_w[0])

    zz = np.array(RI, dtype='float')

    dim = list(range(len(datalist)))
    dim_set = permutations(dim,2)
    for dim_select in dim_set:
        key = []
        for w in test_center_w:
            w_d1 = optimzie_key(w[dim_select[0]],accuracy+1)
            w_d2 = optimzie_key(w[dim_select[1]],accuracy+1)
            mark1 = 'x' + str(w_d1)
            mark2 = 'y' + str(w_d2)
            key.append( mark1+ mark2)
        dic = dict(zip(key, zz))
        print(len(dic))

        xx = np.arange(0, 1 + sample_rate, sample_rate)
        yy = np.arange(0, 1 + sample_rate, sample_rate)
        X, Y = np.meshgrid(xx, yy)

        Z = np.zeros(X.shape)
        for i in range(len(xx)):
            for j in range(len(xx)):
                w_d1 = optimzie_key(X[i, j], accuracy + 1)
                w_d2 = optimzie_key(Y[i, j], accuracy + 1)
                mark1 = 'x' + str(w_d1)
                mark2 = 'y' + str(w_d2)
                print(mark1 + mark2)
                Z[i, j] = dic[mark1 + mark2]
        fig = plt.figure()  # 定义新的三维坐标轴
        ax3 = plt.axes(projection='3d')
        ax3.plot_surface(X, Y, Z, cmap='rainbow')
        ax3.set_xlabel('sample '+ str(dim_select[0])+' weight', fontsize=10, rotation=150)
        ax3.set_ylabel('sample '+ str(dim_select[1])+' weights')
        ax3.set_zlabel('RI')
        plt.savefig(datapath + str(dim_select[0]) + '_' + str(dim_select[1]) + '.png')
        plt.close()

        # ax3.contour(X, Y, Z, offset=-2, colors='black')  # 生成等高线 offset参数是等高线所处的位置
        ratio = Z.max() - Z.min()
        rag = np.arange(0,10,1)*ratio/10+Z.min()
        C = plt.contour(X, Y, Z,rag,cmap = 'rainbow')
        plt.clabel(C, inline=True, fontsize=10)  # 在等高线上标出对应的z值
        # ax3.set_zlim(-1, 1)  # 设置z的范围

        plt.savefig(datapath+str(dim_select[0])+'_'+str(dim_select[1])+'contour.png')
        plt.close()

def figure_test_datacenter(accuracy, data_length,loop,mark):# accuracy is the float saving bits
    idx = cm.y_index_train == 0
    xtrain = cm.x_train[idx]
    sequnce = list(range(len(xtrain)))
    random.shuffle(sequnce)
    output_path = mark
    if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_path)
    for i in range(loop):
        idx = sequnce[data_length * i:data_length * (i + 1)]
        x = xtrain[idx]
        output_subpath = mark+'/' + str(i)
        if not os.path.exists(output_subpath):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(output_subpath)
        for j in range(len(x)):
            file_name = output_subpath+'/'+str(j)+'.png'
            print(file_name)
            cv2.imwrite(file_name,x[j]*255)
        figure3D_test_datacenter(x,accuracy,output_subpath)

# figure_test_datacenter(1,3,1,'data_evaluation_minst/data_center_test/mark_C')