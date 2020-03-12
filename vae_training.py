import vae_model as vm
from keras.datasets import mnist, fashion_mnist
import numpy as np
import dataReader as dr
BATCH = 128
N_CLASS = 10
EPOCH = 5
IN_DIM = 28 * 28
H_DIM = 128
Z_DIM = 2
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(len(x_train), -1).astype('float32') / 255.
x_test = x_test.reshape(len(x_test), -1).astype('float32') / 255.
''' 以 train_on_batch 方式训练 '''
index = []
for epoch in range(EPOCH):
    for b in range(x_train.shape[0] // BATCH):
        idx = np.random.choice(x_train.shape[0], BATCH)
        x = x_train[idx]
        index.append(idx)
        l = vm.vae.train_on_batch([x], [x, x])
        vm.vae.save('models/'+str(epoch)+'E'+str(b)+'b.h5')

dr.save_data(index,'index_record.csv')