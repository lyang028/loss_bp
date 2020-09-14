from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import effectiveness as ef
import KL_div as kl
import dataset_evaluation as de
from matplotlib import pyplot as plt
import random

import dataReader as dr

from changing_network import create_network
from changing_network import create_extreme_network


def create_data_set(labels,amount):
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
# convert class vectors to binary class matrices

def test_layer_design(model,path,sample_gap = 1):
    loop = 10
    bs = 100
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 4000)
    length = x_train.shape[0]
    training_order = list(range(length))
    random.shuffle(training_order)

    count = 0
    for epoch in range(loop):
        for b in range(x_train.shape[0] // bs):
            idx = training_order[b * bs:(b + 1) * bs]
            x = x_train[idx]
            y = y_train[idx]
            l = model.train_on_batch(x, y)
            if count%sample_gap == 0:
                name = path + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
                model.save(name)
                print(name)
            count+=1

    # model.fit(x_train, y_train,
    #           batch_size=bs,
    #           epochs=loop,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss)
    print(accuracy)



import os

def sort_key(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0])*500+int(batch_str[0])
def single_test(model,file_address,output):
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 4000)
    file_ls = os.listdir(file_address)
    file_ls.sort(key=sort_key)
    model.load_weights(file_address + '/'+file_ls[0]) #end of normal

    vmodel = model.get_weights()
    target = kl.resize_model(vmodel)

    wds = []
    eds = []
    loss_set = []
    acc_set = []
    i = 0

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        model.load_weights(file_address + '/' + file)
        print(file)
        vcurrent = kl.resize_model(model.get_weights())
        # dis = kl.KL_div_sigmoid(target,vcurrent) #inverse
        dis = kl.KL_div_sigmoid(target, vcurrent)
        E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
        loss, acc = model.evaluate(x_test, y_test)
        eds.append(E_dis)
        wds.append(dis)
        acc_set.append(acc)
        loss_set.append(loss)
        print(acc)
        print(loss)
        print(file)
        i = i + 1
        # if i>100:
        #    break
        print('*********************************')

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, output + '/dis_limitation_e.csv')
    plt.savefig(output + '/dis_limitation_inverse_e.png')
    plt.close()

    plt.plot(range(len(eds)), eds)
    dr.save_data(eds, output + '/dis_limitation_eul.csv')
    plt.savefig(output + '/dis_limitation_inverse_eul.png')
    plt.close()

    plt.plot(range(len(loss_set)), loss_set)
    dr.save_data(loss_set, output + '/dis_limitation_loss.csv')
    plt.savefig(output + '/dis_limitation_inverse_loss.png')
    plt.close()

    plt.plot(range(len(acc_set)), acc_set)
    dr.save_data(acc_set, output + '/dis_limitation_acc.csv')
    plt.savefig(output + '/dis_limitation_inverse_acc.png')
    plt.close()




def accuracy(model,path):
    x_train, x_test, y_train, y_test = create_data_set(list(range(1)), 4000)
    model.load_weights(path) #end of normal
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss)
    print(accuracy)


#test comite

model1 = create_network(channals=1,dense = 32)
model2 = create_network(channals=2,dense = 32)
modeltest = create_network(channals=1,dense = 32)
model_twoconv = create_network(channals=32,dense = 128,new_layer=64)
model_extreme = create_extreme_network()
model_extreme_add = create_extreme_network(add=100)
model_extreme_addt = create_extreme_network(add=100,addt = 100)


x_train, x_test, y_train, y_test = create_data_set(list(range(1)), 40)
loss, acc = model_extreme.evaluate(x_test, y_test)
loss, acc = model_extreme_add.evaluate(x_test, y_test)
loss, acc = model_extreme_addt.evaluate(x_test, y_test)
# test_layer_design(model1,'Final_experiment/repeat_training_limitation/model1_weights','Final_experiment/repeat_training_limitation/model1')
# single_test(model1,'Final_experiment/repeat_training_limitation/model1_weights','Final_experiment/repeat_training_limitation')
# test_layer_design(model2,'Final_experiment/repeat_training_limitation/model2_weights','Final_experiment/repeat_training_limitation/model2')
# single_test(model2,'Final_experiment/repeat_training_limitation/model2_weights','Final_experiment/repeat_training_limitation')
# test_layer_design(modeltest,'Final_experiment/repeat_training_limitation/modeltest_weights',sample_gap=1000)
# single_test(modeltest,'Final_experiment/repeat_training_limitation/modeltest_weights','Final_experiment/repeat_training_limitation')

# test_layer_design(model_twoconv,'Final_experiment/repeat_training_limitation/model_twoconv_weights',sample_gap=10)
# single_test(model_twoconv,'Final_experiment/repeat_training_limitation/model_twoconv_weights','Final_experiment/repeat_training_limitation')

# test_layer_design(model_extreme,'Final_experiment/repeat_training_limitation/extreme',sample_gap=100)
# single_test(model_extreme,'Final_experiment/repeat_training_limitation/extreme','Final_experiment/repeat_training_limitation/record/extreme')
#
# test_layer_design(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add',sample_gap=100)
# single_test(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add','Final_experiment/repeat_training_limitation/record/extreme_add')

# test_layer_design(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_addt',sample_gap=100)
# single_test(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_addt','Final_experiment/repeat_training_limitation/record/extreme_addt')

# accuracy(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add/999E300b.h5')

# model_extreme_add = create_extreme_network(add=100,opt='sgd')
# test_layer_design(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add_sgd',sample_gap=1)
#
model_extreme_add = create_extreme_network(add=100,opt='adam')
test_layer_design(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add_adam',sample_gap=1)

single_test(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add_sgd','Final_experiment/repeat_training_limitation/record/extreme_add_sgd')
single_test(model_extreme_add,'Final_experiment/repeat_training_limitation/extreme_add_adam','Final_experiment/repeat_training_limitation/record/extreme_add_adam')