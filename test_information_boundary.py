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


import dataReader as dr
from keras import backend as K


img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

num_classes = 10
epochs = 20
def create_data_set(labels,amount):
    # input image dimensions
    # the data, split between train and test sets
    (x_train, y_index_train), (x_test, y_index_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
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
    y_train = keras.utils.to_categorical(y_index_train_out, num_classes)
    y_test = keras.utils.to_categorical(y_index_test, num_classes)
    return  np.array(x_train_out),x_test,y_train,y_test
# convert class vectors to binary class matrices


def test_layer_design():
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 20)
    RI = de.reletive_information(x_train)
    dis_set = []
    effectiveness_set = []
    loss_set = []
    ac_set = []
    loop = 50
    bs = 128
    for i in range(loop):
        x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 1000)
        model = Sequential()
        model.add(Conv2D(5, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        # model.add(Conv2D(10 + i * 5, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # # #extra layer

        # extra layer
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        # model.save_weights('Final_experiment/test_information_boundary/init.h5')
        # model.load_weights('Final_experiment/test_information_boundary_standardinit.h5')
        model.load_weights('Final_experiment/test_information_boundary_sliminit.h5')
        model.save_weights('Final_experiment/test_information_boundary/init.h5')
        model.fit(x_train, y_train,
                  batch_size=bs,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        bs = bs-2
        model.save_weights('Final_experiment/test_information_boundary/end.h5')

        dis,effectiveness = ef.effectiveness('Final_experiment/test_information_boundary/init.h5', 'Final_experiment/test_information_boundary/end.h5',
                         model, kl.KL_div, RI)
        loss_set.append(score[0])
        ac_set.append(score[1])
        dis_set.append(dis)
        effectiveness_set.append(effectiveness)


    dr.save_data(loss_set,'Final_experiment/test_information_boundary/loss.csv')
    dr.save_data(ac_set, 'Final_experiment/test_information_boundary/ac.csv')
    dr.save_data(dis_set, 'Final_experiment/test_information_boundary/dis.csv')
    dr.save_data(effectiveness_set, 'Final_experiment/test_information_boundary/ef.csv')

    plt.plot(range(loop),loss_set,label = 'loss')
    plt.plot(range(loop), ac_set,label = 'accuracy')
    plt.legend()
    plt.savefig('Final_experiment/test_information_boundary/performance_evaluation.png')
    plt.close()
    plt.plot(range(loop), dis_set,label = 'dis')
    plt.legend()
    plt.savefig('Final_experiment/test_information_boundary/dis.png')
    plt.show()
    # plt.plot(range(loop), effectiveness_set,)

test_layer_design()