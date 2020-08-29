from sklearn.datasets import load_digits
from sklearn.manifold import MDS

import os
from simple_model_cnn import create_data_set
from simple_model_cnn import create_network
from KL_div import resize_model
import matplotlib.pyplot as plt
import random
import numpy as np
import dataReader as dr

def training(output_path):
    model = create_network()
    x_train, x_test, y_train, y_test = create_data_set(list(range(10)), 200)
    length = x_train.shape[0]
    training_order = list(range(length))
    random.shuffle(training_order)

    epoch = 10
    bs = 128
    for epoch in range(epoch):
        for b in range(x_train.shape[0] // bs):
            idx = training_order[b * bs:(b + 1) * bs]
            x = x_train[idx]
            y = y_train[idx]
            l = model.train_on_batch(x, y)
            name = output_path + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
            model.save(name)
            print(name)

def mds_analysis(model,path_set,opath_set,dim = 2):
    weight_set = []
    length_set = []
    gredient_set = []
    vpre = []
    for path in path_set:
        file_ls = os.listdir(path)
        length_set.append(len(file_ls))
        def sort_key_multi_label(e):
            epoch_str = e.split('E')
            batch_str = epoch_str[1].split('b')
            return int(epoch_str[0]) * 1000 + int(batch_str[0])

        file_ls.sort(key=sort_key_multi_label)
        for file in file_ls:
            print(path + '/' + file)
            model.load_weights(path + '/' + file)
            vcurrent = resize_model(model.get_weights())
            weight_set.append(vcurrent)
            vcurrent = np.array(vcurrent,dtype=float)
            if len(vpre) == 0:
                vpre = vcurrent
            else:
                gredient_set.append(vcurrent-vpre)
                vpre = vcurrent

    weight_set = np.array(weight_set, dtype=float)
    embedding = MDS(n_components=dim)
    X_transformed = embedding.fit_transform(weight_set)

    gredient_set = np.array(gredient_set, dtype=float)
    G_transformed = embedding.fit_transform(gredient_set)


    pre_len = 0
    for i in range(len(opath_set)):
        dr.save_data(X_transformed[pre_len:pre_len+length_set[i]], opath_set[i]+'.csv')
        if pre_len == 0:
            dr.save_data(G_transformed[pre_len:pre_len + length_set[i]-1], opath_set[i] + '_gredient.csv')
        else:
            dr.save_data(G_transformed[pre_len-1:pre_len + length_set[i] - 1], opath_set[i] + '_gredient.csv')
        pre_len+=length_set[i]
    # print(X_transformed)

# def get_feedback(model,output_weight):

# path = 'MDS_test/weights'
# output_path = 'MDS_test'
# training(path)
# mds_analysis(create_network(),path,output_path)

# path = ['MDS_test/weights','MDS_test/weights2','MDS_test/weights3','MDS_test/weights4','MDS_test/weights5','MDS_test/weights6']
# output_path = ['MDS_test/w.csv','MDS_test/w2.csv','MDS_test/w3.csv','MDS_test/w4.csv','MDS_test/w5.csv','MDS_test/w6.csv']

# path = ['MDS_test/weights','MDS_test/weights3','MDS_test/weights5']
# output_path = ['MDS_test/w4','MDS_test/w5','MDS_test/w6']
# mds_analysis(create_network(),path,output_path)
# training('MDS_test/weights4')
# training('MDS_test/weights5')
# training('MDS_test/weights6')

# path = ['MDS_test/weights']
# output_path = ['MDS_test/w_special']
# mds_analysis(create_network(),path,output_path,dim=3)



# model = create_network()
# model.load_weights()
# X, _ = load_digits(return_X_y=True)
X = []

for i in range(10):
    X.append(np.ones(20)*i)
for i in range(10):
    X.append(np.ones(20)*i)
print(X)
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(np.array(X))
a = 0
print(X_transformed.shape)
print(X_transformed)