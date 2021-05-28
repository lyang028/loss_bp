import os
import cnn_model as cm
import scipy.stats as ss
import scipy
import numpy as np
import matplotlib.pyplot as plt
import dataReader as dr
import random
import KL_div as kl
import keras
from keras.datasets import mnist

def create_data_set(labels = [0,1,2,3,4,5,6,7,8,9,10],amount = 4000):
    # input image dimensions
    # the data, split between train and test sets
    (x_train, y_index_train), (x_test, y_index_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train_out = []
    y_index_train_out = []
    x_test_out = []
    y_index_test_out = []
    for i in labels:
        idx = y_index_train == i
        x_sub = x_train[idx]
        y_sub = y_index_train[idx]
        x_train_out.extend(x_sub[:amount])
        y_index_train_out.extend(y_sub[:amount])

        idx = y_index_test == i
        x_testsub = x_test[idx]
        y_testsub = y_index_test[idx]
        x_test_out.extend(x_testsub[:amount])
        y_index_test_out.extend(y_testsub[:amount])

    y_train = keras.utils.to_categorical(y_index_train_out, 10)
    y_test = keras.utils.to_categorical(y_index_test_out, 10)
    return  np.array(x_train_out),np.array(x_test_out),y_train,y_test


x_train, x_test, y_train, y_test = create_data_set()
def add_noise(path,sigma):
    cm.model.load_weights(path)
    origin = cm.model.get_weights()

    output = []
    length = 0
    i = 0

    for layer in cm.model.layers:
        # print(layer.get_weights())
        w = layer.get_weights()
        w_n = []
        for w_sub in w:
            w_sub_f = w_sub.flatten()
            noise = np.random.randn(len(w_sub_f))*sigma
            w_sub_f += noise
            w_sub_n = np.reshape(w_sub_f,w_sub.shape)
            w_n.append(w_sub_n)
        layer.set_weights(w_n)

    loss,acc = cm.model.evaluate(x_test, y_test,verbose=0)
    return  loss, acc

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

def add_edis_noise(path,norm):
    cm.model.load_weights(path)
    weights = cm.model.get_weights()
    flatten_weights = np.array(resize_model(weights))
    model_weights_scale = len(flatten_weights)
    noise = np.random.randn(model_weights_scale)
    norm_noise = noise/np.linalg.norm(noise)*norm
    flatten_weights+=norm_noise

    cut = 0
    for layer in cm.model.layers:
        w = layer.get_weights()
        w_n = []
        for w_sub in w:
            shape = w_sub.shape
            len_noise = cut + np.prod(shape)
            sub_noise = flatten_weights[cut:len_noise]
            cut = len_noise
            sub_noise = sub_noise.reshape(shape)
            w_n.append(sub_noise)
        layer.set_weights(w_n)

    # w_test = resize_model(cm.model.get_weights())
    # print('test')

def add_noise_batch(path,loop = 1, samples = 10, upper = 1000):
    gap = upper/samples
    norms = np.array(range(samples))*gap
    dr.save_data(norms, 'radius_test/keep/norms.csv')
    acc_records = []
    loss_records = []
    for norm in norms:
        acc_record = []
        loss_record = []
        for l in range(loop):
            add_edis_noise(path, norm)
            loss, acc = cm.model.evaluate(x_test, y_test, verbose=0)
            acc_record.append(acc)
            loss_record.append(loss)
        print(np.mean(acc_record),np.std(acc_record))

        acc_records.append(acc_record)
        loss_records.append(loss_record)
    dr.save_data(acc_records, 'radius_test/keep/acc_map.csv')
    dr.save_data(loss_records, 'radius_test/keep/loss_map.csv')
    print('end')
# standard training process
def single_training(file_address):
    x_train = cm.x_train
    y_train = cm.y_train

    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"
    else:
        print
        "---  There is this folder!  ---"
    cm.model.load_weights('standard_init.h5')
    training_order = np.array(dr.read_csv('standard_order.csv'), dtype='int32')[:, 0]
    loss = []
    acc = []
    for epoch in range(cm.epochs):
        for b in range(x_train.shape[0] // cm.batch_size):
            idx = training_order[b * cm.batch_size:(b + 1) * cm.batch_size]
            x = x_train[idx]
            y = y_train[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name+' '+str(l))
            loss.append(l[0])
            acc.append(l[1])
            if(l[1]>0.99):
                dr.save_data(acc, 'radius_test/acc_map.csv')
                dr.save_data(loss, 'radius_test/loss_map.csv')
                return loss, acc

    dr.save_data(acc,'radius_test/acc_map.csv')
    dr.save_data(loss,'radius_test/loss_map.csv')
    return loss, acc
# loss_map, acc_map = single_training('standard_training')
# draw ********************************************************
def draw(path_array, labels = [],output_path = 'no',xaxis = 'blank', yaxis = 'blank',range_select = [-1,-1],ratio = (8.0, 4.0),dpi = 100,xoffset = [1,0],tight_layout = True,font_size = [20,10,10],setXcoord = False, Xcoord = [],Title = 'no',alpha = 1):
    plt.rcParams['figure.figsize'] = ratio
    plt.rcParams['figure.dpi'] = dpi  # 分辨率
    plt.xlabel(xaxis, fontsize=font_size[0])
    plt.ylabel(yaxis, fontsize=font_size[0])
    plt.xticks(fontsize=font_size[1])
    plt.yticks(fontsize=font_size[1])
    if not Title == 'no':
        plt.title(Title,fontsize=font_size[0])
    if len(labels) == 0:
        # plt.axes(xscale='log')
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            if not setXcoord:
                if range_select[0] == -1:
                    x = np.array(range(len(array))) * xoffset[0] + xoffset[1]
                    plt.plot(x, array)
                else:
                    array = array[range_select[0]:range_select[1]]
                    x = np.array(range(len(array))) * xoffset[0] + xoffset[1]
                    plt.plot(x, array)
            else:
                if range_select[0] == -1:
                    plt.plot(Xcoord, array)
                else:
                    array = array[range_select[0]:range_select[1]]
                    Xcoord = Xcoord[range_select[0]:range_select[1]]
                    plt.plot(Xcoord, array)
        if tight_layout:
            plt.tight_layout()
        plt.show()
    else:
        index = 0
        for path in path_array:
            array = np.array(dr.read_csv(path), dtype=float)[:, 0]
            if not setXcoord:
                if range_select[0] == -1:
                    x = np.array(range(len(array))) * xoffset[0] + xoffset[1]
                    plt.plot(x, array,label = labels[index])
                else:
                    array = array[range_select[0]:range_select[1]]
                    x = np.array(range(len(array))) * xoffset[0] + xoffset[1]
                    plt.plot(x, array,label = labels[index])
            else:
                if range_select[0] == -1:
                    plt.plot(Xcoord, array,label = labels[index])
                else:
                    array = array[range_select[0]:range_select[1]]
                    Xcoord = Xcoord[range_select[0]:range_select[1]]
                    plt.plot(Xcoord, array,label = labels[index])
            index = index+1

        plt.legend(fontsize=font_size[2])
        if tight_layout:
            plt.tight_layout()
        if output_path == 'no':
            plt.show()
        else:
            plt.savefig(output_path)
def set_mapping(map, array_acc):
    first_bigger_index = [0]
    corresponding_acc = [0]
    current_acc = 0
    for i in range(len(map)):
        if map[i]>current_acc:
            first_bigger_index.append(i)
            corresponding_acc.append(map[i])
            current_acc = map[i]
    equi_color = np.zeros(len(array_acc))
    for i in range(len(array_acc)):
        find_slot = False
        for j in range(len(first_bigger_index)-1):
            if corresponding_acc[j]<array_acc[i] and corresponding_acc[j+1]>=array_acc[i]:
                equi_color[i] = first_bigger_index[j+1]
                find_slot = True
                break
        if find_slot == False:
            print('error acc')
    return equi_color

def calculate_colormap_edis(path_acc, path_loss):
    acc = np.array(dr.read_csv(path_acc), dtype=float)
    acc_map = np.array(dr.read_csv('radius_test/acc_map.csv'), dtype=float)[:, 0]
    loss = np.array(dr.read_csv(path_loss), dtype=float)
    loss_map = np.array(dr.read_csv('radius_test/loss_map.csv'), dtype=float)[:, 0]
    acc_mean = np.mean(acc,axis=1)
    loss_mean = np.mean(loss, axis=1)
    acc_ecolor = set_mapping(acc_map, acc_mean)
    dr.save_data(acc_ecolor, 'radius_test/keep/acc_color.csv')
    # loss_ecolor = set_mapping(loss_map, loss_mean)
    # dr.save_data(loss_ecolor, 'radius_test/keep/loss_color.csv')

def draw_image(path_color,path_acc):
    acc = np.array(dr.read_csv('radius_test/keep/acc.csv'), dtype=float)
    norm = np.array(dr.read_csv('radius_test/keep/norms.csv'), dtype=float)[:,0]
    acc_mean = np.mean(acc,axis=1)
    acc = acc.transpose()
    acc_color = np.array(dr.read_csv('radius_test/keep/acc_color.csv'), dtype=float)*2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

    # 设置标题
    ax.set_title('Distribution of Accuracy near the Output Weights',fontsize = 20)

    # 绘制小提琴图
    parts = ax.violinplot(acc, showmeans=False, showmedians=False,showextrema=True)
    sc = ax.scatter(range(1,len(acc_mean)+1), acc_mean, c=acc_color, s=np.ones(len(acc_color)) * 50)
    cmap = plt.get_cmap('rainbow')

    # # 设置填充颜色和边框颜色
    i = 0
    # for pc in parts['bodies']:
    #     pc.set_facecolor(cmap(acc_color[i]/100))
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(1.0)
    #     i+=1

    labels = ['A', 'B', 'C', 'D']

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')
    set_axis_style(ax, norm)
    plt.xlabel('2-Norm of Noise',fontsize = 20)
    plt.ylabel('Accuracy',fontsize = 20)
    plt.xticks(rotation=30)
    cb = plt.colorbar(sc)
    cb.set_label( 'Equivalent Training Steps')
    plt.tight_layout()
    # plt.xticks(my_x_ticks)
    plt.show()
# losses = []
# acces = []
# for i in range(10):
#     sum_loss = 0
#     sum_acc = 0
#     for loop in range(10):
#         loss, acc = add_noise('standard_training/0E220b.h5', 0.01 * i)
#         sum_loss += loss
#         sum_acc += acc
#         print('i: ' + str(i) + ':: j: ' + str(loop) + '         loss: ' + str(loss) + '::' + 'acc: ' + str(acc))
#
#     losses.append(sum_loss / 10)
#     acces.append(sum_acc / 10)
# dr.save_data(losses, 'radius_test/loss_radius.csv')
# dr.save_data(acces, 'radius_test/acc_radius.csv')


#calculate noise impact acc and loss
# add_noise_batch('standard_training/0E220b.h5',loop=100,samples=20,upper=600)

#calculate the color map edis
# calculate_colormap_edis('radius_test/keep/acc.csv','radius_test/keep/loss.csv')

#draw image
draw_image('radius_test/keep/acc_color.csv','radius_test/keep/acc.csv')

# print(ecolor)
# path_array = ['radius_test/acc_radius.csv']

# plt.rcParams['figure.figsize'] = [4.5,4]
# plt.rcParams['figure.dpi'] = 100  # 分辨率
#
# plt.scatter(radius,acc,c = ecolor,s = np.ones(len(ecolor))*200)
# # plt.colorbar()
# draw(['radius_test/acc_radius.csv'],Xcoord=np.array(range(20))*0.01,setXcoord=True,ratio=[4.5,4],xaxis='Sigma',yaxis='Accuracy')


# plt.rcParams['figure.figsize'] = [5.2,4]
# plt.rcParams['figure.dpi'] = 100  # 分辨率
# plt.scatter(radius,loss,c = loss_ecolor,s = np.ones(len(ecolor))*200)
# cbar = plt.colorbar()
# cbar.set_ticks([])
# draw(['radius_test/loss_radius.csv'],Xcoord=np.array(range(20))*0.01,setXcoord=True,ratio=[5.5,4],xaxis='Sigma',yaxis='Loss')
# plt.plot(radius,acc)

# draw_scatter(acc,ecolor,Xcoord=np.array(range(20))*0.01,setXcoord=True,ratio=[4.5,4],xaxis='Sigma',yaxis='Accuracy')
# plt.show()


