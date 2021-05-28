import os
import cnn_check_model as cm
import scipy.stats as ss
import scipy
import numpy as np
import matplotlib.pyplot as plt
import dataReader as dr
import random
import KL_div as kl
import keras
from keras.datasets import mnist

def sort_key(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*468+int(batch_str[0])
def sort_key_zero(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*46+int(batch_str[0])
def sort_key_one(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*52+int(batch_str[0])
def sort_key_two(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*46+int(batch_str[0])
def sort_key_three(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*47+int(batch_str[0])
def sort_key_four(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*45+int(batch_str[0])
def sort_key_five(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*42+int(batch_str[0])
def sort_key_six(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*46+int(batch_str[0])
def sort_key_seven(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*48+int(batch_str[0])
def sort_key_eight(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*45+int(batch_str[0])
def sort_key_nine(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*46+int(batch_str[0])

def sum_layer(array):
    length = 1
    shape = np.shape(array)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(array,[length,1])
    return sum(flat_array)[0],length
def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w,length)
    # print(flat_array,dtype=float)
    return flat_array

def resize_model(weights):
    output = []
    length = 0
    for w in weights:
        f_w = resize_layer(w)
        output.extend(f_w)
        length = length+len(f_w)
    # oa = np.array(output).flatten(order='C')
    oa = output
    return oa
def draw_plot(list,name_list, xlabel = '', ylabel = '',title = '',fz = (8,4)):
    plt.figure(figsize=fz)
    for i in range(len(list)):
        plt.plot(range(len(list[i])),list[i],label = name_list[i])
    plt.title(title)  # 标题
    plt.xlabel(xlabel)  # x轴的标签
    plt.ylabel(ylabel)  # y轴的标签
    plt.legend()
    plt.show()
def normalize(array,ratio):
    sumv = sum(array)
    ntarget = [a * ratio for a in array]
    ntarget = [a / sumv for a in ntarget]
    return ntarget


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"

    if sum(x)==0 or sum(y) == 0:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def accuracy(model,specific_test= [-1]):
    if specific_test == [-1]:
        score = model.evaluate(cm.x_test, cm.y_test, verbose=0)
    else:
        idx = np.array([x in specific_test for x in cm.y_index_test])
        x_test = cm.x_test[idx]
        y_test = cm.y_test[idx]
        score = model.evaluate(x_test,y_test,verbose=0)
    return score[0], score[1]
def single_accuracy_test(fs,specific_test=[-1]):
    for file_address in fs:
        file_ls = os.listdir(file_address)
        file_ls.sort(key=sort_key)
        losses = []
        accuracies = []

        for file in file_ls:
            extend = os.path.splitext(file)[-1][1:]
            if (extend != 'h5'):
                continue
            cm.model.load_weights(file_address + '/' + file)
            print(file)
            loss, ac = accuracy(cm.model, specific_test=specific_test)
            print(file)
            losses.append(loss)
            accuracies.append(ac)
            # if i>10:
            #     break
        # list = []
        # list.append(eds)
        # list.append(wds)
        # name_list = ['ed','wd']
        # draw_plot(list,name_list)

        plt.plot(range(len(losses)), losses)
        dr.save_data(losses, 'record/' + file_address + '/losses.csv')
        plt.savefig('record/' + file_address + '/losses.png')
        plt.close()

        plt.plot(range(len(accuracies)), accuracies)
        dr.save_data(accuracies, 'record/' + file_address + '/accuracy.csv')
        plt.savefig('record/' + file_address + '/accuracy.png')
        plt.close()

def group_test(length):
    for i in range(length):
        if i == 0:
            file_address = 'model_cnn_zero'
            selection = sort_key_zero
            cm.model.load_weights(file_address + '/4E45b.h5')
        elif i == 1:
            file_address = 'model_cnn_one'
            selection = sort_key_one
            cm.model.load_weights(file_address + '/4E51b.h5')
        elif i == 2:
            file_address = 'model_cnn_two'
            selection = sort_key_two
            cm.model.load_weights(file_address + '/4E45b.h5')
        elif i == 3:
            file_address = 'model_cnn_three'
            selection = sort_key_three
            cm.model.load_weights(file_address + '/4E46b.h5')

        elif i == 4:
            file_address = 'model_cnn_four'
            selection = sort_key_four
            cm.model.load_weights(file_address + '/4E44b.h5')

        elif i == 5:
            file_address = 'model_cnn_five'
            selection = sort_key_five
            cm.model.load_weights(file_address + '/4E41b.h5')

        elif i == 6:
            file_address = 'model_cnn_six'
            selection = sort_key_six
            cm.model.load_weights(file_address + '/4E45b.h5')

        elif i == 7:
            file_address = 'model_cnn_seven'
            selection = sort_key_seven
            cm.model.load_weights(file_address + '/4E47b.h5')

        elif i == 8:
            file_address = 'model_cnn_eight'
            selection = sort_key_eight
            cm.model.load_weights(file_address + '/4E44b.h5')

        elif i == 9:
            file_address = 'model_cnn_nine'
            selection = sort_key_nine
            cm.model.load_weights(file_address + '/4E45b.h5')
        else:
            file_address = 'model_cnn_default'
            selection = sort_key
            cm.model.load_weights(file_address + '/4E45b.h5')
        print('start: ', i)
        file_ls = os.listdir(file_address)
        file_ls.sort(key=selection)
        # file_ls.sort(key=sort_key)

        # vm.vae.load_weights(output_path+'/0E0b.h5')
        # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
        # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
        # cm.model.load_weights(output_path+'/1E467b.h5') #end of normal

        vmodel = cm.model.get_weights()
        RATIO = 100000
        target = resize_model(vmodel)
        ntarget = normalize(target, RATIO)

        eds = []
        wds = []
        cds = []
        i = 0

        pre_previous = []
        previous = []
        cds_dif = []

        for file in file_ls:
            extend = os.path.splitext(file)[-1][1:]
            if (extend != 'h5'):
                continue

            print(file_address + '/' + file)
            cm.model.load_weights(file_address + '/' + file)
            vcurrent = resize_model(cm.model.get_weights())

            E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
            eds.append(E_dis)

            C_dis = cosine_similarity(vcurrent, target)
            cds.append(C_dis)

            nvcurrent = normalize(vcurrent, RATIO)
            W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
            wds.append(W_dis)

            # print(C_dis)
            # print(E_dis)
            # print(W_dis)
            print(file_address + '/' + file)
            if len(pre_previous) == 0:
                pre_previous = vcurrent
                continue
            elif len(previous) == 0:
                previous = vcurrent
                continue
            else:
                pp_dif = np.array(previous) - np.array(pre_previous)
                p_dif = np.array(vcurrent) - np.array(previous)
                C_dis_diff = cosine_similarity(pp_dif, p_dif)
                cds_dif.append(C_dis_diff)
                pre_previous = previous
                previous = vcurrent

            # print(C_dis_diff)
            i = i + 1
            print('*********************************')
            # if i>10:
            #     break
        # list = []
        # list.append(eds)
        # list.append(wds)
        # name_list = ['ed','wd']
        # draw_plot(list,name_list)

        plt.plot(range(len(eds)), eds)
        dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
        plt.savefig('record/' + file_address + '/E_dis_random.png')
        plt.close()

        plt.plot(range(len(wds)), wds)
        dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
        plt.savefig('record/' + file_address + '/W_dis_random.png')
        plt.close()

        plt.plot(range(len(cds)), cds)
        dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
        plt.savefig('record/' + file_address + '/C_dis_random.png')
        plt.close()

        plt.plot(range(len(cds_dif)), cds_dif)
        dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
        plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
        plt.close()

def single_test():
    file_address = input('enter output path:')
    file_ls = os.listdir(file_address)
    file_ls.sort(key=sort_key)
    # file_ls.sort(key=sort_key)

    # vm.vae.load_weights(output_path+'/0E0b.h5')
    # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
    # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
    cm.model.load_weights(file_address+'/4E467b.h5') #end of normal

    vmodel = cm.model.get_weights()
    RATIO = 100000
    target = resize_model(vmodel)
    ntarget = normalize(target, RATIO)

    eds = []
    wds = []
    cds = []
    i = 0

    pre_previous = []
    previous = []
    cds_dif = []

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        cm.model.load_weights(file_address + '/' + file)
        print(file)
        vcurrent = resize_model(cm.model.get_weights())

        E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
        eds.append(E_dis)

        C_dis = cosine_similarity(vcurrent, target)
        cds.append(C_dis)

        nvcurrent = normalize(vcurrent, RATIO)
        W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
        wds.append(W_dis)

        # print(C_dis)
        # print(E_dis)
        # print(W_dis)
        print(file)
        if len(pre_previous) == 0:
            pre_previous = vcurrent
            continue
        elif len(previous) == 0:
            previous = vcurrent
            continue
        else:
            pp_dif = np.array(previous) - np.array(pre_previous)
            p_dif = np.array(vcurrent) - np.array(previous)
            C_dis_diff = cosine_similarity(pp_dif, p_dif)
            cds_dif.append(C_dis_diff)
            pre_previous = previous
            previous = vcurrent

        # print(C_dis_diff)
        i = i + 1
        print('*********************************')
        # if i>10:
        #     break
    # list = []
    # list.append(eds)
    # list.append(wds)
    # name_list = ['ed','wd']
    # draw_plot(list,name_list)

    plt.plot(range(len(eds)), eds)
    dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
    plt.savefig('record/' + file_address + '/E_dis_random.png')
    plt.close()

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
    plt.savefig('record/' + file_address + '/W_dis_random.png')
    plt.close()

    plt.plot(range(len(cds)), cds)
    dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_random.png')
    plt.close()

    plt.plot(range(len(cds_dif)), cds_dif)
    dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
    plt.close()



def multilabel_test(index,length):
    file_address = 'multi_label'+ str(index)
    file_ls = os.listdir(file_address)
    def sort_key_multi_label(e):
        epoch_str = e.split('E')
        batch_str = epoch_str[1].split('b')
        return int(epoch_str[0]) * length + int(batch_str[0])
    file_ls.sort(key=sort_key_multi_label)
    # file_ls.sort(key=sort_key)

    # vm.vae.load_weights(output_path+'/0E0b.h5')
    # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
    # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
    # cm.model.load_weights(output_path+'/1E467b.h5') #end of normal
    cm.model.load_weights(file_address + '/4E'+str(length-1)+'b.h5')
    vmodel = cm.model.get_weights()
    RATIO = 100000
    target = resize_model(vmodel)
    ntarget = normalize(target, RATIO)

    eds = []
    wds = []
    cds = []
    i = 0

    pre_previous = []
    previous = []
    cds_dif = []

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue

        print(file_address + '/' + file)
        cm.model.load_weights(file_address + '/' + file)
        vcurrent = resize_model(cm.model.get_weights())

        E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
        eds.append(E_dis)

        C_dis = cosine_similarity(vcurrent, target)
        cds.append(C_dis)

        nvcurrent = normalize(vcurrent, RATIO)
        W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
        wds.append(W_dis)

        # print(C_dis)
        # print(E_dis)
        # print(W_dis)
        if len(pre_previous) == 0:
            pre_previous = vcurrent
            continue
        elif len(previous) == 0:
            previous = vcurrent
            continue
        else:
            pp_dif = np.array(previous) - np.array(pre_previous)
            p_dif = np.array(vcurrent) - np.array(previous)
            C_dis_diff = cosine_similarity(pp_dif, p_dif)
            cds_dif.append(C_dis_diff)
            pre_previous = previous
            previous = vcurrent

        # print(C_dis_diff)
        i = i + 1
        print('*********************************')
        # if i>10:
        #     break
    # list = []
    # list.append(eds)
    # list.append(wds)
    # name_list = ['ed','wd']
    # draw_plot(list,name_list)

    plt.plot(range(len(eds)), eds)
    dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
    plt.savefig('record/' + file_address + '/E_dis_random.png')
    plt.close()

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
    plt.savefig('record/' + file_address + '/W_dis_random.png')
    plt.close()

    plt.plot(range(len(cds)), cds)
    dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_random.png')
    plt.close()

    plt.plot(range(len(cds_dif)), cds_dif)
    dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
    plt.close()

def Euclidean_distance(array1,array2):
    return np.linalg.norm(array1 - array2, ord=2)


def compare_two_sequence(path1,path2,metric,metric_name):
    file_address = 'record/'+path1+'_'+path2

    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    file_ls1 = os.listdir(path1)
    file_ls1.sort(key=sort_key)
    file_ls2 = os.listdir(path2)
    file_ls2.sort(key=sort_key)
    distance = []
    for file in file_ls1:
        if file not in file_ls2:
            break
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        print(path1 + '/' + file,'  ',path2 + '/' + file)

        cm.model.load_weights(path1 + '/' + file)
        vcurrent1 = resize_model(cm.model.get_weights())
        cm.model.load_weights(path2 + '/' + file)
        vcurrent3 =  resize_model(cm.model.get_weights())

        dis = metric(np.array(vcurrent1),np.array(vcurrent3))
        print(dis)
        distance.append(dis)
    dr.save_data(distance,file_address+'/'+metric_name+'.csv')

def analyse_sequence(path_set, target_path_set,metric, metric_name,output_path_set = [],test_range = []):
    for i in range(len(path_set)):
        if len(test_range) == 0:
            x_train, x_test, y_train, y_test = cm.create_data_set()
        else:
            x_train, x_test, y_train, y_test = cm.create_data_set(range(test_range[i][0]),test_range[i][1])
        path = path_set[i]
        target_path = target_path_set[i]
        if len(output_path_set) == 0:
            file_address = 'record/' + path
        else:
            file_address = output_path_set[i]

        file_ls1 = os.listdir(path)
        file_ls1.sort(key=sort_key)

        distance = []
        for i in range(len(metric)):
            distance.append([])
        loss_set = []
        acc_set = []
        cm.model.load_weights(target_path)
        targetv = resize_model(cm.model.get_weights())
        for file in file_ls1:
            extend = os.path.splitext(file)[-1][1:]
            if (extend != 'h5'):
                continue

            cm.model.load_weights(path + '/' + file)
            vcurrent1 = resize_model(cm.model.get_weights())
            for i in range(len(metric)):
                dis = metric[i](vcurrent1, targetv)
                distance[i].append(dis)
                print(metric_name[i]+' '+str(dis))

            loss,acc = cm.model.evaluate(x_test,y_test)
            loss_set.append(loss)
            acc_set.append(acc)
            print('loss'+' '+str(loss)+'   acc'+' '+str(acc))
        for i in range(len(metric)):
            dr.save_data(distance[i], file_address + '/' + metric_name[i] + '.csv')
            plt.plot(range(len(distance[i])), distance[i])
            plt.savefig(file_address + '/'+metric_name[i]+'.png')
            plt.close()

        dr.save_data(acc_set, file_address + '/acc.csv')
        plt.plot(range(len(acc_set)), acc_set)
        plt.savefig(file_address + '/ acc.png')
        plt.close()

        dr.save_data(loss_set, file_address + '/loss.csv')
        plt.plot(range(len(loss_set)), loss_set)
        plt.savefig(file_address + '/loss.png')
        plt.close()

def compare_two_model(path1,path2,metric):
    cm.model.load_weights(path1)
    v1 = resize_model(cm.model.get_weights())

    cm.model.load_weights(path2)
    v2 = resize_model(cm.model.get_weights())

    dis = metric(v1, v2)
    print(dis)
    return dis

def compare_two_model_batch(begin_path, end_path, out_path,measure):
    output = []
    for i in range(len(begin_path)):
        output.append(compare_two_model(begin_path[i],end_path[i],measure))
    dr.save_data(output,out_path)


def compare_two_sequence_accuracy(path1,path2, save = True):
    file_address = 'record/' + path1 + '_' + path2
    os.makedirs(file_address)
    file_ls1 = os.listdir(path1)
    file_ls1.sort(key=sort_key)
    file_ls2 = os.listdir(path2)
    file_ls2.sort(key=sort_key)
    loss_1 = []
    loss_2 = []
    ac_1 = []
    ac_2 = []
    for file in file_ls1:
        if file not in file_ls2:
            break

        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        print(path1 + '/' + file, '  ', path2 + '/' + file)

        cm.model.load_weights(path1 + '/' + file)
        loss1, accuracy1 = accuracy(cm.model)
        print('loss_1: ',loss1, 'acc_1: ', accuracy1)
        cm.model.load_weights(path2 + '/' + file)
        loss2, accuracy2 = accuracy(cm.model)
        print('loss_2: ',loss2, ' acc_2:', accuracy2)
        loss_1.append(loss1)
        loss_2.append(loss2)
        ac_1.append(accuracy1)
        ac_2.append(accuracy2)
    if save:
        dr.save_data(ac_1, file_address + '/accuracy_1.csv')
        dr.save_data(ac_2, file_address + '/accuracy_2.csv')
        dr.save_data(loss_1, file_address + '/loss1.csv')
        dr.save_data(loss_2, file_address + '/loss2.csv')


def single_weight_test(path_init,path_end,test_set = [-1]):
    compare_two_model(path_init,path_end,kl.E_dis)
    loss,ac = accuracy(cm.model,test_set)
    # print(dis)
    print('loss: ',loss, ' acc:', ac)

# group_test(10)
# single_test()
# multi = [[9,422]]
# for i in multi:
#     multilabel_test(i[0],i[1])

# single_accuracy_test([1])
# compare_two_model('cnn_mlabel1/0E0b.h5','cnn_mlabel2/0E0b.h5',kl.KL_div)
#
# single_accuracy_test([0,1])

# path = ['cnn_mlabel1','cnn_mlabel2','cnn_mlabel3','cnn_mlabel4','cnn_mlabel5','cnn_mlabel6','cnn_mlabel7','cnn_mlabel8','cnn_mlabel9']
# # tpath = ['cnn_mlabel1/0E45b.h5','cnn_mlabel2/0E97b.h5','cnn_mlabel3/0E144b.h5','cnn_mlabel4/0E192b.h5','cnn_mlabel5/0E238b.h5',
# #          'cnn_mlabel6/0E280b.h5','cnn_mlabel7/0E326b.h5','cnn_mlabel8/0E375b.h5','cnn_mlabel9/0E421b.h5']
# # analyse_sequence(path,tpath,kl.KL_div,'KL_div')

# path = ['cnn_mlabel5','cnn_mlabel6','cnn_mlabel7','cnn_mlabel8','cnn_mlabel9']
# tpath = ['cnn_mlabel5/0E238b.h5','cnn_mlabel6/0E280b.h5','cnn_mlabel7/0E326b.h5','cnn_mlabel8/0E375b.h5','cnn_mlabel9/0E421b.h5']
# analyse_sequence(path,tpath,kl.KL_div,'KL_div')

# path = ['error_label10','error_label20','error_label30','error_label40','error_label50','error_label60','error_label70','error_label80','error_label90']
# tpath = ['error_label10/0E467b.h5','error_label20/0E467b.h5','error_label30/0E467b.h5','error_label40/0E467b.h5','error_label50/0E467b.h5',
#          'error_label60/0E467b.h5','error_label70/0E467b.h5','error_label80/0E467b.h5','error_label90/0E467b.h5']
# analyse_sequence(path,tpath,kl.KL_div,'KL_div')

# path = ['cnn_1_0_combine/A','cnn_1_0_combine/B','cnn_1_0_combine/C']
# tpath = ['cnn_1_0_combine/A/0E124b.h5','cnn_1_0_combine/B/0E124b.h5','cnn_1_0_combine/C/0E124b.h5']
# analyse_sequence(path,tpath,kl.KL_div,'KL_div')
# single_accuracy_test(path,specific_test=[0])

# path = ['cnn_mix_all/9','cnn_mix_all/8','cnn_mix_all/7','cnn_mix_all/6',
#         'cnn_mix_all/5','cnn_mix_all/4','cnn_mix_all/3','cnn_mix_all/2',
#         'cnn_mix_all/1','cnn_mix_all/0']
# tpath = ['cnn_mix_all/9/0E38b.h5','cnn_mix_all/8/0E38b.h5','cnn_mix_all/7/0E38b.h5','cnn_mix_all/6/0E38b.h5',
#         'cnn_mix_all/5/0E38b.h5','cnn_mix_all/4/0E38b.h5','cnn_mix_all/3/0E38b.h5','cnn_mix_all/2/0E38b.h5',
#         'cnn_mix_all/1/0E38b.h5','cnn_mix_all/0/0E38b.h5']
# analyse_sequence(path,tpath,kl.KL_div,'KL_div')

# path = ['F:/entropy_analysis/CNN_mlabels/cnn_mlabel1','F:/entropy_analysis/CNN_mlabels/cnn_mlabel2']
# tpath = ['F:/entropy_analysis/CNN_mlabels/cnn_mlabel1/0E0b.h5','F:/entropy_analysis/CNN_mlabels/cnn_mlabel2/0E0b.h5']
# opath = ['E:/loss_bp/record/cnn_sl_test/mlabel0','E:/loss_bp/record/cnn_sl_test/mlabel1']
# test_r = [[1,4000],[2,4000]]
# metric = [kl.KL_div,kl.KL_div_sigmoid,kl.E_dis]
# metric_name = ['KL_div','KL_div_sigmoid','E_dis']
# analyse_sequence(path,tpath,metric,metric_name,output_path_set=opath,test_range=test_r)


# path = ['error_label10','error_label20','error_label30','error_label40','error_label50',
#         'error_label60','error_label70','error_label80','error_label90',]
# tpath = ['error_label10/0E467b.h5','error_label20/0E467b.h5','error_label30/0E467b.h5',
#          'error_label40/0E467b.h5','error_label50/0E467b.h5','error_label60/0E467b.h5',
#          'error_label70/0E467b.h5','error_label80/0E467b.h5','error_label90/0E467b.h5']
# opath = ['record/error_label10','record/error_label20','record/error_label30','record/error_label40',
#          'record/error_label50','record/error_label60','record/error_label70','record/error_label80',
#          'record/error_label90',]
# metric = [kl.KL_div,kl.KL_div_sigmoid,kl.E_dis]
# metric_name = ['KL_div','KL_div_sigmoid','E_dis']
# analyse_sequence(path,tpath,metric,metric_name,output_path_set=opath)

# path = ['error_label0','error_label10','error_label20','error_label30','error_label40','error_label50',
#         'error_label60','error_label70','error_label80','error_label90','error_label100']
# tpath = ['error_label0/9E383b.h5','error_label10/9E383b.h5','error_label20/9E383b.h5','error_label30/9E383b.h5',
#          'error_label40/9E383b.h5','error_label50/9E383b.h5','error_label60/9E383b.h5',
#          'error_label70/9E383b.h5','error_label80/9E383b.h5','error_label90/9E383b.h5',
#          'error_label100/9E383b.h5']
# opath = ['record/error_label0','record/error_label10','record/error_label20','record/error_label30','record/error_label40',
#          'record/error_label50','record/error_label60','record/error_label70','record/error_label80',
#          'record/error_label90','record/error_label100']
# metric = [kl.E_dis]
# metric_name = ['E_dis']
# analyse_sequence(path,tpath,metric,metric_name,output_path_set=opath)


# print(compare_two_model('error_label10/0E0b.h5','error_label10/0E467b.h5',kl.E_dis))
# print(compare_two_model('error_label20/0E0b.h5','error_label20/0E467b.h5',kl.E_dis))
# print(compare_two_model('error_label60/0E0b.h5','error_label60/0E467b.h5',kl.E_dis))
# print(compare_two_model('error_label70/0E0b.h5','error_label70/0E467b.h5',kl.E_dis))

# compare_two_sequence_accuracy('error_label10','error_label20')
# compare_two_sequence_accuracy('error_label60','error_label70')

# import cnn_model as cm
# single_weight_test('error_label0/0E0b.h5','error_label0/9E383b.h5')
# single_weight_test('error_label1/0E10b.h5','error_label1/9E383b.h5')
# single_weight_test('error_label10/0E10b.h5','error_label10/9E383b.h5')
# single_weight_test('error_label70/0E10b.h5','error_label70/9E383b.h5')
# single_weight_test('standard_init.h5','cnn_sl_0/0E45b.h5',test_set=[0])

begin_path = ['cnn_cifar_exp1/begin/0.h5','cnn_cifar_exp1/begin/1.h5','cnn_cifar_exp1/begin/2.h5','cnn_cifar_exp1/begin/3.h5','cnn_cifar_exp1/begin/4.h5',
              'cnn_cifar_exp1/begin/5.h5','cnn_cifar_exp1/begin/6.h5', 'cnn_cifar_exp1/begin/7.h5','cnn_cifar_exp1/begin/8.h5','cnn_cifar_exp1/begin/9.h5',
              'cnn_cifar_exp1/begin/10.h5','cnn_cifar_exp1/begin/11.h5','cnn_cifar_exp1/begin/12.h5','cnn_cifar_exp1/begin/13.h5','cnn_cifar_exp1/begin/14.h5',
              'cnn_cifar_exp1/begin/15.h5','cnn_cifar_exp1/begin/16.h5','cnn_cifar_exp1/begin/17.h5','cnn_cifar_exp1/begin/18.h5','cnn_cifar_exp1/begin/19.h5']
end_path = ['cnn_cifar_exp1/end/0.h5','cnn_cifar_exp1/end/1.h5','cnn_cifar_exp1/end/2.h5','cnn_cifar_exp1/end/3.h5','cnn_cifar_exp1/end/4.h5',
              'cnn_cifar_exp1/end/5.h5','cnn_cifar_exp1/end/6.h5', 'cnn_cifar_exp1/end/7.h5','cnn_cifar_exp1/end/8.h5','cnn_cifar_exp1/end/9.h5',
              'cnn_cifar_exp1/end/10.h5','cnn_cifar_exp1/end/11.h5','cnn_cifar_exp1/end/12.h5','cnn_cifar_exp1/end/13.h5','cnn_cifar_exp1/end/14.h5',
              'cnn_cifar_exp1/end/15.h5','cnn_cifar_exp1/end/16.h5','cnn_cifar_exp1/end/17.h5','cnn_cifar_exp1/end/18.h5','cnn_cifar_exp1/end/19.h5']
out_path = 'cnn_cifar_exp1/distance.csv'
compare_two_model_batch(begin_path,end_path,out_path,kl.E_dis)