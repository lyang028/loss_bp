import cnn_check_model as cm
import numpy as np
import os
import random
import dataReader as dr
import keras
from keras.datasets import mnist, fashion_mnist

index = []
label = []

# file_address = input('enter the address')
# if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
#     os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
#     os.makedirs('record/'+file_address)
#     print
#     "---  new folder...  ---"
#     print
#     "---  OK  ---"
# else:
#     print
#     "---  There is this folder!  ---"
def group_training():
    for i in range(10):
        idx = (cm.y_index_train == i)
        x_train = cm.x_train[idx]
        y_train = cm.y_train[idx]
        if i == 0:
            file_address = 'model_cnn_zero'
        elif i == 1:
            file_address = 'model_cnn_one'
        elif i == 2:
            file_address = 'model_cnn_two'
        elif i == 3:
            file_address = 'model_cnn_three'
        elif i == 4:
            file_address = 'model_cnn_four'
        elif i == 5:
            file_address = 'model_cnn_five'
        elif i == 6:
            file_address = 'model_cnn_six'
        elif i == 7:
            file_address = 'model_cnn_seven'
        elif i == 8:
            file_address = 'model_cnn_eight'
        elif i == 9:
            file_address = 'model_cnn_nine'
        else:
            file_address = 'model_cnn_default'

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

        print('standard_init using')
        cm.model.load_weights('standard_init.h5')
        for epoch in range(cm.epochs):
            for b in range(x_train.shape[0] // cm.batch_size):
                idx = np.random.choice(x_train.shape[0], cm.batch_size)
                x = x_train[idx]
                y = y_train[idx]
                index.append(idx)
                label.append(y)
                l = cm.model.train_on_batch(x, y)
                name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
                cm.model.save(name)
                print(name)

def single_training(file_address, standard_init = 'Yes',standard_training = 'Yes'):
    x_train = cm.x_train
    y_train = cm.y_train
    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"
    if standard_init == 'Yes':
        print('standard_init using')
        cm.model.load_weights('standard_init.h5')
    if standard_training == 'Yes':
        training_order = np.array(dr.read_csv('standard_order.csv'),dtype='int32')[:,0]
    else:
        length = x_train.shape[0]
        training_order = list(range(length))
        random.shuffle(training_order)
    print('training_config:',file_address,'standard training order:',training_order, 'standard init:',standard_init)
    for epoch in range(cm.epochs):
        for b in range(x_train.shape[0] // cm.batch_size):
            idx = training_order[b * cm.batch_size:(b + 1) * cm.batch_size]
            x = x_train[idx]
            y = y_train[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)
    # for epoch in range(cm.epochs):
    #     for b in range(x_train.shape[0] // cm.batch_size):
    #         idx = np.random.choice(x_train.shape[0], cm.batch_size)
    #         x = x_train[idx]
    #         y = y_train[idx]
    #         index.extend(idx)
    #         label.append(y)
            # l = cm.model.train_on_batch(x, y)
            # name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
            # cm.model.save(name)
            # print(name)
    # dr.save_data(label,'record/'+file_address+'/label.csv')
    # print(label)
    # dr.save_data(index,'record/' + file_address + '/idx.csv')
    # print(idx)
# score = cm.model.evaluate(cm.x_test, cm.y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

def single_label_training(file_address ,sample_index,standard_init = 'Yes'):
    idx = (cm.y_index_train == sample_index)
    x_train = cm.x_train[idx]
    y_train = cm.y_train[idx]
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
    if standard_init == 'Yes':
        print('standard_init using')
        cm.model.load_weights('standard_init.h5')
    length = x_train.shape[0]
    training_order = list(range(length))
    random.shuffle(training_order)

    for epoch in range(cm.epochs):
        for b in range(x_train.shape[0] // cm.batch_size):
            idx = training_order[b * cm.batch_size:(b + 1) * cm.batch_size]
            x = x_train[idx]
            y = y_train[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)

def error_onehotlabel_rate(a,b):
    s = 0
    for i in range(len(a)):
        if np.ma.allequal(a[i],b[i]):
            s = s+1
    return s/len(a)
def onehot_to_index(a):
    return [np.where(r==1)[0][0] for r in a]
def error_label_shift(error_set, standard_init = 'Yes', standard_training = 'Yes',gaps = 0):
    for error in error_set:
        x_train = cm.x_train
        y_train = cm.y_train
        file_address = 'error_label'+ str(error)
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
        if standard_init == 'Yes':
            print('standard_init using')
            cm.model.load_weights('cnn_check_standard_init.h5')
        if standard_training == 'Yes':
            training_order = np.array(dr.read_csv('cifar10_training_order.csv'), dtype='int32')[:, 0]
        else:
            length = x_train.shape[0]
            training_order = list(range(length))
            random.shuffle(training_order)
            dr.save_data(training_order,'templete_training_order.csv')
        print('training_config:', file_address, 'standard training order:', training_order, 'standard init:',
              standard_init)

        #global_shuffle(array, rate):
        error_rate = error/100
        y_train_ori = np.copy(y_train)
        error_array = y_train[:int(len(y_train)*error_rate)]
        random.shuffle(error_array)
        y_train[:int(len(y_train)*error_rate)] = error_array
        print('global error rate:'+str(error_onehotlabel_rate(y_train, y_train_ori)))
        gap = 0
        for epoch in range(cm.epochs):
            for b in range(x_train.shape[0] // cm.batch_size):
                idx = training_order[b * cm.batch_size:(b + 1) * cm.batch_size]
                x = x_train[idx]
                y = y_train[idx]
                loss ,acc = cm.model.train_on_batch(x, y)
                print('local error rate:' + str(error_onehotlabel_rate(y, y_train_ori[idx])))
                print('loss: '+str(loss)+' acc: '+str(acc))
                if gap == gaps:
                    name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
                    cm.model.save(name)
                    print(name)
                    gap = 0
                else:
                    gap+= 1

def multi_label():
    for i in range(10):
        file_address = 'cnn_mlabel' + str(i)
        idx = (cm.y_index_train < i)
        x_train = cm.x_train[idx]
        y_train = cm.y_train[idx]
        if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        else:
            print
            "---  There is this folder!  ---"
        if not os.path.exists('record/' + file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs('record/' + file_address)
        else:
            print
            "---  There is this folder!  ---"

        print('standard_init using')
        cm.model.load_weights('standard_init.h5')

        length = x_train.shape[0]
        training_order = list(range(length))
        random.shuffle(training_order)

        for epoch in range(cm.epochs):
            for b in range(x_train.shape[0] // cm.batch_size):
                idx = training_order[b * cm.batch_size:(b + 1) * cm.batch_size]
                x = x_train[idx]
                y = y_train[idx]
                index.append(idx)
                label.append(y)
                l = cm.model.train_on_batch(x, y)
                name = file_address + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
                cm.model.save(name)
                print(name)

def helf_half_combination(file_address,si1,si2,test_length = 5000,standard_init = 'Yes'):
    idx1 = (cm.y_index_train == si1)
    x_train1 = cm.x_train[idx1]
    x_train1 = x_train1[:test_length,:,:]
    y_train1 = cm.y_train[idx1][:test_length,:]

    idx2 = (cm.y_index_train == si2)
    x_train2 = cm.x_train[idx2][:test_length,:,:]
    y_train2 = cm.y_train[idx2][:test_length,:]

    half_length = int(test_length / 2)
    x_train3 = np.array([x_train1[:half_length,:,:],x_train2[:half_length,:,:]])
    # shape = [2000,28,28,1]
    # shape_test = list(x_train1.shape)
    x_train3.resize(x_train1.shape)
    y_train3 = np.array([y_train1[:half_length, :], y_train2[:half_length, :]])
    y_train3.resize(y_train1.shape)

    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs(file_address + '/A')
        os.makedirs(file_address + '/B')
        os.makedirs(file_address + '/C')
        os.makedirs('record/' + file_address)
        os.makedirs('record/' + file_address+'/A')
        os.makedirs('record/' + file_address+'/B')
        os.makedirs('record/' + file_address+'/C')
    else:
        print
        "---  There is this folder!  ---"

    print('standard_init using. start A')
    cm.model.load_weights('standard_init.h5')
    training_order = list(range(test_length))
    random.shuffle(training_order)

    batch_size = int(cm.batch_size/4)
    for epoch in range(cm.epochs):
        for b in range(test_length // batch_size):
            idx = training_order[b * batch_size:(b + 1) * batch_size]
            x = x_train1[idx]
            y = y_train1[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' +'A/'+ str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)

    print('standard_init using. start B')
    cm.model.load_weights('standard_init.h5')

    for epoch in range(cm.epochs):
        for b in range(test_length // batch_size):
            idx = training_order[b * batch_size:(b + 1) * batch_size]
            x = x_train2[idx]
            y = y_train2[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' +'B/'+ str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)

    print('standard_init using. start C')
    cm.model.load_weights('standard_init.h5')
    for epoch in range(cm.epochs):
        for b in range(test_length // batch_size):
            idx = training_order[b * batch_size:(b + 1) * batch_size]
            x = x_train3[idx]
            y = y_train3[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' +'C/'+ str(epoch) + 'E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)

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

def mixed_combination(file_address,label_set, test_length = 5000, standard_init ='Yes'):
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
        x_mix,y_mix = mix_dataset(xtrains,ytrains,test_length,len(xtrains))
        xtrain_mix.append(x_mix)
        ytrain_mix.append(y_mix)




    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        for i in range(len(xtrain_mix)):
            os.makedirs(file_address+'/'+str(i))
            os.makedirs('record/' + file_address+'/'+str(i))
    else:
        print
        "---  There is this folder!  ---"

    training_order = list(range(test_length))
    random.shuffle(training_order)

    batch_size = int(cm.batch_size)

    for i in range(len(xtrain_mix)):
        x_train = xtrain_mix[i]
        y_train = ytrain_mix[i]
        print('standard_init using. start B')
        cm.model.load_weights('standard_init.h5')
        for b in range(test_length // batch_size):
            idx = training_order[b * batch_size:(b + 1) * batch_size]
            x = x_train[idx]
            y = y_train[idx]
            l = cm.model.train_on_batch(x, y)
            name = file_address + '/' + str(i)+'/' + '0E' + str(b) + 'b.h5'
            cm.model.save(name)
            print(name)

def init_weights(output_path):
    cm.model.save_weights(output_path)
    cm.model.load_weights(output_path)

def distance_exp1(input_path,output_path):
    x_train = cm.x_train
    y_train = cm.y_train
    file_list = os.listdir(input_path)

    def sort_key_multi_label(e):
        epoch_str = e.split('.h5')
        return int(epoch_str[0])
    file_list.sort(key=sort_key_multi_label)
    print(file_list)
    for file in file_list:
        cm.model.load_weights(input_path+'/'+file)
        cm.model.fit(x_train, y_train,
                 batch_size=cm.batch_size,
                 epochs=cm.epochs,
                 verbose=1,
                 validation_data=(cm.x_test, cm.y_test))
        cm.model.save(output_path +'/' + file)
def tensorboard_vis():
    import datetime
    import tensorflow as tf
    log_dir = "logs/fit/cnn_cifar_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    cm.model.fit(x=cm.x_train,
              y=cm.y_train,
              epochs=1,
              validation_data=(cm.x_test, cm.y_test),
              callbacks=[tensorboard_callback])

# init_weights('cnn_cifar_exp1/begin/16.h5')
# x_train = cm.x_train
# y_train = cm.y_train
#
# cm.model.load_weights('cnn_cifar_exp1/begin/16.h5')
# cm.model.fit(x_train, y_train,
#                  batch_size=cm.batch_size,
#                  epochs=cm.epochs,
#                  verbose=1,
#                  validation_data=(cm.x_test, cm.y_test))
# cm.model.save('cnn_cifar_exp1/end/16.h5')
# distance_exp1('cnn_cifar_exp1/begin','cnn_cifar_exp1/end')



tensorboard_vis()

# error_label_shift([0,10,20,30,40,50,60,70,80,90,100],gaps=10)
# error_label_shift([10,70])
# error_label_randomize()
# multi_label()
# single_label_training('cnn_sl_0',0)

# helf_half_combination('cnn_2_0_combine',0,2)

# mixed_combination('cnn_mix_all',[0,1,2,3,4,5,6,7,8,9])