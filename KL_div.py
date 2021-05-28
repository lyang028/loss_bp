import numpy as np
import scipy.stats as ss

def rel_entr_preprocess(a,b):
    array_a = np.array(a)
    array_b = np.array(b)
    v = array_a/array_b
    v_rest_bool = v!=0
    lg = np.log(v[v_rest_bool])
    output = np.zeros(len(a))
    output[v_rest_bool] = array_a[v_rest_bool]*lg
    return output
def softmax(a):
    ex_a = np.exp(a)
    return ex_a/sum(ex_a)

def sigmoid(a):
    x = 1/(1+np.exp(-a))
    return x
def entropy(pk, qk=None, base=None, axis=0,activation = True,Filter = -1):
    pk = np.array(pk)
    if activation:
        pk = softmax(pk)

    if qk is None:
        vec = ss.entr(pk)
    else:
        qk = np.array(qk)
        if Filter != -1:
            qk_bool = qk == 0
            pk_bool = pk == 0
            xor_bool = np.bitwise_xor(pk_bool,qk_bool)
            pk_zero_bool = np.bitwise_and(pk_bool,xor_bool)
            qk_zero_bool = np.bitwise_and(qk_bool,xor_bool)
            d_qk = qk[pk_zero_bool]
            d_pk = pk[qk_zero_bool]
            for i in range(len(d_pk)):
                if d_pk[i] < Filter:
                    d_pk[i] = 0
                else:
                    raise ValueError("qk and pk has different zero elements.")
            for i in range(len(d_qk)):
                if d_qk[i]<Filter:
                    d_qk[i] = 0
                else:
                    raise ValueError("qk and pk has different zero elements.")
            pk[qk_zero_bool] = d_pk
            qk[pk_zero_bool] = d_qk
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        if activation:
            qk = softmax(qk)
        vec = rel_entr_preprocess(pk, qk)
    S = np.sum(vec, axis=axis)
    if S < 0:
        # print('error S:'+ str(S))
        # print(sum(qk),sum(pk))
        return 0
    if base is not None:
        S /= np.log(base)
    return S

def KL_div(array1,array2,activation = True,Filter = -1):
    return entropy(array1,array2,activation=activation,Filter = Filter)

def JS_div(array1,array2):
    p = np.array(array1)
    q = np.array(array2)
    m = (p+q)/2
    return KL_div(p,m)/2+KL_div(q,m)/2


def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w,length)
    # print(flat_array,dtype=float)
    return flat_array
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

def KL_div_sigmoid(pk, qk=None, base=None, axis=0):
    pk = np.array(pk)
    pk = sigmoid(pk)
    if qk is None:
        vec = ss.entr(pk)
    else:
        qk = np.array(qk)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = sigmoid(qk)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if S < 0:
        print('error S:'+ str(S))
        print(sum(qk),sum(pk))
        return 0
    if base is not None:
        S /= np.log(base)
    return S

def E_dis(array1,array2):
    return np.linalg.norm(np.array(array1) - np.array(array2), ord=2)
# a = [2,2,2,2,2,2]
# b = np.array(a)
# print(sigmoid(b))
# cm.model.load_weights('cnn_standard/0E413b.h5')
# targetv = resize_model(cm.model.get_weights())
#
# print(KL_div(targetv, targetv))
# print(np.log(-12))


