import cnn_model as cm
import numpy as np
import os
import random
import dataReader as dr
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

def single_training():
    file_address = input('adress:')
    i = input('set_order:')
    if i != 'all':
        idx = (cm.y_index_train == i)
        x_train = cm.x_train[idx]
        y_train = cm.y_train[idx]
    else:
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

# score = cm.model.evaluate(cm.x_test, cm.y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
def error_label_shift():
    file_address = 'shift_label'
    x_train = cm.x_train
    y_train = (cm.y_train + 1)%10

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


def error_label_randomize():
    file_address = 'shift_label'
    x_train = cm.x_train
    y_train = cm.y_train
    random.shuffle(y_train)

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
def multi_label():
    for i in [2,3,4,5,6,7,8,9]:
        file_address = 'multi_label' + str(i)
        idx = (cm.y_index_train < i)
        x_train = cm.x_train[idx]
        y_train = cm.y_train[idx]
        if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径

            print
            "---  new folder...  ---"
            print
            "---  OK  ---"
        else:
            print
            "---  There is this folder!  ---"
        if not os.path.exists('record/' + file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
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
        batch_amount = 46
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
# single_training()
# error_label_shift()
# error_label_randomize()
multi_label()