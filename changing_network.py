import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


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
    for i in labels:
        idx = y_index_train == i
        x_sub = x_train[idx]
        y_sub = y_index_train[idx]
        x_train_out.extend(x_sub[:amount])
        y_index_train_out.extend(y_sub[:amount])
    y_train = keras.utils.to_categorical(y_index_train_out, 10)
    y_test = keras.utils.to_categorical(y_index_test, 10)
    return  np.array(x_train_out),x_test,y_train,y_test
def create_network(channals = 3,dense = 128, new_layer = 0):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(channals, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    if new_layer != 0:
        model.add(Conv2D(new_layer, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # # #extra layer

    # extra layer
    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                  metrics=['accuracy'])
    return model

def create_extreme_network(add = 0,addt = 0,opt = []):
    input_shape = (28, 28, 1)
    model = Sequential()
    # model.add(Conv2D(channals, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # if new_layer != 0:
    #     model.add(Conv2D(new_layer, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # # #extra layer

    # extra layer
    model.add(Flatten())
    if add != 0:
        model.add(Dense(add, activation='softmax'))
    if addt != 0:
        model.add(Dense(addt, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    optimizer = []
    if len(opt) == 0:
        # print('adam used')
        optimizer = keras.optimizers.Adam()
    elif opt == 'sgd':
        print('sgd used')
        optimizer = keras.optimizers.SGD()
    elif opt == 'adam':
        print('adam used')
        optimizer = keras.optimizers.Adam()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    #mark important configuration**********************************************one layer work / two layer not work
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
    #               metrics=['accuracy'])
    return model

