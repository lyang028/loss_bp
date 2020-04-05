import vae_model as vm
from keras.datasets import mnist, fashion_mnist
import numpy as np
import dataReader as dr
import random
import keras.backend as K
BATCH = 128
N_CLASS = 10
EPOCH = 5
IN_DIM = 28 * 28
H_DIM = 128
Z_DIM = 2
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(len(x_train), -1).astype('float32') / 255.
x_test = x_test.reshape(len(x_test), -1).astype('float32') / 255.

#select_samples
print(y_train == 0)
x_train = x_train[y_train==0]
y_train = y_train[y_train==0]

print(y_train)

grad = K.gradients(0.5 * K.sum(K.square(vm.z_mu) + K.exp(vm.z_logvar) - 1. - vm.z_logvar, axis=1),vm.vae.trainable_weights)
print(grad)  # 有些是 None 的
grad = grad[grad is not None]  # 去掉 None，不然报错

# 打印梯度的函数
# K.function 的输入和输出必要是 list！就算只有一个
show_grad = K.function([vm.vae.input], [grad])

''' 以 train_on_batch 方式训练 '''
output_path = input('enter output path:')
index = []
b_size = (x_train.shape[0] // BATCH)
for epoch in range(EPOCH):
    for b in range(b_size):
        name = output_path + '/' + str(epoch) + 'E' + str(b) + 'b.h5'
        vm.vae.save(name)
        idx = np.random.choice(x_train.shape[0], BATCH)
        x = x_train[idx]
        index.append(idx)
        l = vm.vae.train_on_batch([x], [x,x])
        loss = vm.vae.test_on_batch([x],[x,x])
        print(name)
        print(loss)
        # gd = show_grad([x])
        # # 打印梯度
        # print(gd)
name = output_path+ '/'+str(EPOCH-1)+'E'+str(b_size)+'b.h5'
vm.vae.save(name)
dr.save_data(index,'record/'+output_path+'/index_record_random.csv')
dr.save_data(y_train,'record/'+output_path+'/label_record_random.csv')