from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from simple_model_learningrate import create_data_set
from simple_model_learningrate import create_network
from KL_div import resize_model

from effectiveness import distance
from KL_div import KL_div
import matplotlib.pyplot as plt
import random
import numpy as np
import dataReader as dr


def test_learning_rate(init_path, output_path):
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 1280)
    dis_set = []
    loss_set = []
    ac_set = []
    loop = 100
    bs = 128
    epochs = 10
    for i in range(loop):
        model = create_network(10*i)
        model.load_weights(init_path+'/init.h5')
        # model.save_weights(init_path+'/init.h5')
        model.fit(x_train, y_train,
                  batch_size=bs,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_train, y_train))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights(output_path+'/end.h5')
        dis = distance(init_path+'/init.h5', output_path+'/end.h5',model, KL_div)
        loss_set.append(score[0])
        ac_set.append(score[1])
        dis_set.append(dis)

    dr.save_data(loss_set,output_path+'/loss.csv')
    dr.save_data(ac_set, output_path+'/ac.csv')
    dr.save_data(dis_set, output_path+'/dis.csv')

    plt.plot(range(loop),loss_set,label = 'loss')
    plt.plot(range(loop), ac_set,label = 'accuracy')
    plt.legend()
    plt.savefig(output_path+'/performance_evaluation.png')
    plt.close()
    plt.plot(range(loop), dis_set,label = 'dis')
    plt.legend()
    plt.savefig(output_path+'/dis.png')
    plt.show()
    # plt.plot(range(loop), effectiveness_set,)



path = 'learningrate_test'
output_path = 'learningrate_test'
test_learning_rate(path,output_path)


