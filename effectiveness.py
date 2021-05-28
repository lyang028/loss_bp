import KL_div as kl
import numpy as np
import dataset_evaluation as de
import cnn_model as cm


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

def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w,length)
    # print(flat_array,dtype=float)
    return flat_array


def effectiveness(path_start, path_end, model):
    model.load_weights(path_start)
    v1 = resize_model(model.get_weights())
    loss1, acc1 = model.evaluate(cm.x_test, cm.y_test, verbose=0)
    model.load_weights(path_end)
    v2 = resize_model(model.get_weights())
    loss2, acc2 = model.evaluate(cm.x_test, cm.y_test, verbose=0)
    d_loss = loss1-loss2
    d = kl.E_dis(v1,v2)
    return  d_loss,d
def log_r(path_start, path_end,model):
    d_loss1,d1 = effectiveness(path_start[0],path_end[0],model)
    d_loss2,d2 = effectiveness(path_start[1],path_end[1],model)
    # print(d_loss1)
    # print(d_loss2)
    # print(d1)
    # print(d2)
    # print(n)
    logr = (d_loss2/d2) / (d_loss1 /d1)
    # print(logr)
    return logr
def distance(path_start, path_end, model):
    model.load_weights(path_start)
    v1 = resize_model(model.get_weights())
    model.load_weights(path_end)
    v2 = resize_model(model.get_weights())
    dis = kl.E_dis(v1, v2)
    print(dis)
    return dis
import os
import matplotlib.pyplot as plt
def batch_evaluate(path):
    file_ls = os.listdir(path)

    def sort_key(e):
        epoch_str = e.split('.h5')
        return int(epoch_str[0])
    file_ls.sort(key=sort_key)

    baseline_start_path = path +'/'+file_ls[0]
    baseline_end_path = path + '/' + file_ls[-1]
    # bloss,bd = effectiveness(baseline_start_path,baseline_end_path,cm.model)
    r_s = []
    for file in file_ls:
        loss,d = effectiveness(path+'/'+file,baseline_end_path,cm.model)
        if d != 0:
            r_s.append(loss/d)
    plt.plot(range(len(r_s)),r_s)
    plt.show()
# path_start = ['standard_training/0E0b.h5','cnn_mlabel10/10.h5']
# path_end = ['standard_training/0E220b.h5','cnn_mlabel10/461.h5']
# print(log_r(path_start,path_end,cm.model))

batch_evaluate('cnn_mlabel10')