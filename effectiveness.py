import KL_div as kl
import numpy as np
import dataset_evaluation as de


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

def effectiveness(path_start, path_end, model, metric, RI):
    model.load_weights(path_start)
    v1 = resize_model(model.get_weights())
    model.load_weights(path_end)
    v2 = resize_model(model.get_weights())
    dis = metric(v1, v2)
    print(dis)
    return dis,dis/RI

def distance(path_start, path_end, model, metric):
    model.load_weights(path_start)
    v1 = resize_model(model.get_weights())
    model.load_weights(path_end)
    v2 = resize_model(model.get_weights())
    dis = metric(v1, v2)
    print(dis)
    return dis