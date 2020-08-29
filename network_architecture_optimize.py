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
def single_test(model,file_address,output):
    file_ls = os.listdir(file_address)
    file_ls.sort(key=sort_key)
    model.load_weights(file_address + '/'+file_ls[-1000]) #end of normal

    vmodel = model.get_weights()
    target = kl.resize_model(vmodel)

    wds = []
    i = 0

    pre_previous = []
    previous = []
    cds_dif = []

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        model.load_weights(file_address + '/' + file)
        print(file)
        vcurrent = kl.resize_model(model.get_weights())
        dis = kl.KL_div(vcurrent,target)
        # dis = np.linalg.norm(np.array(vcurrent) - np.array(target), ord=2) #Eucilidean distance

        wds.append(dis)
        print(file)
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

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, output + '/dis_limitation_e.csv')
    plt.savefig(output + '/dis_limitation_inverse_e.png')
    plt.close()


path = 'learningrate_test'
output_path = 'learningrate_test'
test_learning_rate(path,output_path)


