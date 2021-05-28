from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from simple_model_mass_test import create_data_set
from simple_model_mass_test import create_network
from KL_div import resize_model

from effectiveness import distance
from KL_div import KL_div, E_dis
import matplotlib.pyplot as plt
import random
import numpy as np
import dataReader as dr
import MDS

def test_mass(output_path):
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 4000)
    edis_set = []
    kldis_set = []

    loss_set = []
    ac_set = []
    loop = 100
    bs = 100
    epochs = 1
    for i in range(loop):
        model = create_network(channals=i+1)
        model.save_weights(output_path+'/init.h5')
        model.fit(x_train, y_train,
                  batch_size=bs,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_train, y_train))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights(output_path+'/end.h5')
        e_dis = distance(output_path+'/init.h5', output_path+'/end.h5',model, E_dis)
        kl_dis = distance(output_path + '/init.h5', output_path + '/end.h5', model, KL_div)
        loss_set.append(score[0])
        ac_set.append(score[1])
        edis_set.append(e_dis)
        kldis_set.append(kl_dis)

    dr.save_data(loss_set,output_path+'/loss.csv')
    dr.save_data(ac_set, output_path+'/ac.csv')
    dr.save_data(edis_set, output_path+'/edis.csv')
    dr.save_data(kldis_set,output_path+'kldis.csv')

    plt.plot(range(loop),loss_set,label = 'loss')
    plt.plot(range(loop), ac_set,label = 'accuracy')
    plt.legend()
    plt.savefig(output_path+'/performance_evaluation.png')
    plt.close()

    plt.plot(range(loop), edis_set,label = 'dis')
    plt.legend()
    plt.savefig(output_path+'/dis.png')

    plt.plot(range(loop), kldis_set, label='dis')
    plt.legend()
    plt.savefig(output_path + '/kldis.png')

    # plt.plot(range(loop), effectiveness_set,)

def train_velocity_samples(output_path,usesameinit = True,usersameorder = True,channels = 10,loop = 5,epochs = 1,gap = 0):
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 4000)
    v_square_estimate = []
    bs = 100
    length = x_train.shape[0]
    if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_path)

    if usersameorder:
        if os.path.exists(output_path+'/training_order.csv'):
            training_order = np.array(dr.read_csv(output_path + '/training_order.csv'), dtype='int32')[:, 0]
        else:
            training_order = list(range(length))
            random.shuffle(training_order)
            dr.save_data(training_order, output_path + '/training_order.csv')
    if not os.path.exists(output_path + '/weights/'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_path + '/weights/')

    for i in range(loop):
        model = create_network(channals=channels)
        if usesameinit:
            if os.path.exists(output_path+'/0E0b.h5'):
                model.load_weights(output_path+'/0E0b.h5')
            else:
                model.save_weights(output_path + '/0E0b.h5')
        if not os.path.exists(output_path+ '/weights/'+str(i)):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(output_path+ '/weights/'+str(i))
        count = 0
        for epoch in range(epochs):
            for b in range(x_train.shape[0] // bs):
                idx = training_order[b * bs:(b + 1) * bs]
                x = x_train[idx]
                y = y_train[idx]
                l = model.train_on_batch(x, y)
                count+=1
                if count>gap:
                    name = output_path+ '/weights/'+str(i)+'/'+ str(epoch) + 'E' + str(b) + 'b.h5'
                    model.save(name)
                    print(name)
                    count = 0

def calculate_velocity(input_path, output_path,model):
    weight_list = os.listdir(input_path+ '/weights/' + '0')
    def sort_key(e):
        epoch_str = e.split('E')
        batch_str = epoch_str[1].split('b')
        return int(epoch_str[0]) * 500 + int(batch_str[0])
    weight_list.sort(key=sort_key)
    list = os.listdir(input_path+ '/weights')
    v_squere = []
    count = 0
    for weight in weight_list:
        weight_set = []
        for folder in list:
            weight_path = output_path+'/weights/'+folder + '/' + weight
            print(weight_path)
            model.load_weights(weight_path)
            v1 = resize_model(model.get_weights())
            weight_set.append(v1)
            #*************
            # count+=1
            # if count >2:
            #     break

        array_list= np.array(weight_set)
        std = np.var(array_list, axis=0, keepdims=True)[0]
        avg = np.average(std)
        v_squere.append(avg)

    dr.save_data(v_squere, output_path + '/v_square.csv')

    plt.plot(range(len(weight_list)), v_squere, label='v_square')
    plt.legend()
    plt.savefig(output_path + '/v_square.png')
    plt.close()
def calculate_velocity_datapoints(input_path, output_path,model):
    weight_list = os.listdir(input_path + '/weights/' + '0')
    def sort_key(e):
        epoch_str = e.split('E')
        batch_str = epoch_str[1].split('b')
        return int(epoch_str[0]) * 500 + int(batch_str[0])
    weight_list.sort(key=sort_key)
    list = os.listdir(input_path + '/weights')
    v_squere = []
    count = 0
    for weight in weight_list:
        weight_set = []
        for folder in list:
            weight_path = output_path + '/weights/' + folder + '/' + weight
            print(weight_path)
            model.load_weights(weight_path)
            v1 = resize_model(model.get_weights())
            weight_set.append(v1)
        array_list = np.array(weight_set)
        std = np.var(array_list, axis=0, keepdims=True)[0]
        avg = np.average(std)
        v_squere.append(avg)
    dr.save_data(v_squere, output_path + '/v_square.csv')
    plt.plot(range(len(weight_list)), v_squere, label='v_square')
    plt.legend()
    plt.savefig(output_path + '/v_square.png')
    plt.close()

def reshape_convolutional_kernal(conv_kernal,bias):
    length = 1
    shape = np.shape(conv_kernal)
    for i in range(len(shape) - 1):
        length = length * shape[i]
    flat_array = np.reshape(conv_kernal, (length, shape[-1]))
    bias = np.reshape(bias,(1,len(bias)))
    output = np.concatenate((flat_array,bias),axis=0)
    return output

def extract_layers_parameters(input_path,model = create_network(channals=10)):
    weight_list = os.listdir(input_path + '/weights/' + '0')

    def sort_key(e):
        epoch_str = e.split('E')
        batch_str = epoch_str[1].split('b')
        return int(epoch_str[0]) * 500 + int(batch_str[0])

    weight_list.sort(key=sort_key)
    list = os.listdir(input_path + '/weights')
    v_squere = []
    count = 0
    output_path = input_path+'/layer_wise_data'
    if not os.path.exists(output_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_path)
    for weight in weight_list:
        for folder in list:
            weight_path = input_path + '/weights/' + folder + '/' + weight
            print(weight_path)
            model.load_weights(weight_path)
            c1 = model.get_weights()[0]
            b1 = model.get_weights()[1]
            data = reshape_convolutional_kernal(c1,b1)
            file_path = output_path+'/'+ folder
            if not os.path.exists(file_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(file_path)
            dr.save_data(data,file_path+'/'+weight+'c1.csv')

# test_mass('mass_test/cnn_channels')

# path = 'F:/information_effectiveness/1channel/all_random'
# train_velocity_samples(path,channels=1,usesameinit=False,usersameorder=False,epochs=5)
# calculate_velocity(path,path,create_network(channals=1))
#
# path = 'F:/information_effectiveness/1channel/sameinit_randomorder'
# train_velocity_samples(path,channels=1,usesameinit=True,usersameorder=False,epochs=5)
# calculate_velocity(path,path,create_network(channals=1))

# path = 'F:/information_effectiveness/1channel/sameinit_sameorder'
# train_velocity_samples(path,channels=1,usesameinit=True,usersameorder=True,epochs=20)
# calculate_velocity(path,path,create_network(channals=1))

# path = 'F:/information_effectiveness/1channel/randominit_sameorder'
# train_velocity_samples(path,channels=1,usesameinit=False,usersameorder=True,epochs=5)
# calculate_velocity(path,path,create_network(channals=1))

# path = 'F:/chaotic_similarity/randominit_sameorder'
# train_velocity_samples(path,channels=10,usesameinit=False,usersameorder=True,epochs=20,gap=200,loop=100)
# path = 'F:/chaotic_similarity/sameinit_sameorder'
# train_velocity_samples(path,channels=10,usesameinit=True,usersameorder=True,epochs=20,gap=200,loop=100)



# extract_layers_parameters('F:/chaotic_similarity/randominit_sameorder')