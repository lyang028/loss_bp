# -*- coding: utf8 -*-
import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.losses import binary_crossentropy
from keras.datasets import mnist, fashion_mnist
import keras.backend as K
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import dataReader as dr

BATCH = 128
N_CLASS = 10
EPOCH = 20
IN_DIM = 28 * 28
H_DIM = 128
Z_DIM = 2

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(len(x_train), -1).astype('float32') / 255.
x_test = x_test.reshape(len(x_test), -1).astype('float32') / 255.


def sampleing(args):
    """reparameterize"""
    mu, logvar = args
    eps = K.random_normal([K.shape(mu)[0], Z_DIM], mean=0.0, stddev=1.0)
    return mu + eps * K.exp(logvar / 2.)

def sum_layer(array):
    length = 1
    shape = np.shape(array)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(array,[length,1])

    return sum(flat_array)[0],length
def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length*shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w,[length,1])
    # print(flat_array,dtype=float)
    return flat_array

def resize_model(weights):
    output = []
    length = 0
    for w in weights:
        f_w = resize_layer(w)
        output.extend(f_w)
        length = length+len(f_w)

    print(length)
    # oa = np.array(output).flatten(order='C')
    oa = output
    return oa
# encode
x_in = Input([IN_DIM])
h = Dense(H_DIM, activation='relu')(x_in)
z_mu = Dense(Z_DIM)(h)  # mean，不用激活
z_logvar = Dense(Z_DIM)(h)  # log variance，不用激活
z = Lambda(sampleing, output_shape=[Z_DIM])([z_mu, z_logvar])  # 只能有一个参数
encoder = Model(x_in, [z_mu, z_logvar, z], name='encoder')

# decode
z_in = Input([Z_DIM])
h_hat = Dense(H_DIM, activation='relu')(z_in)
x_hat = Dense(IN_DIM, activation='sigmoid')(h_hat)
decoder = Model(z_in, x_hat, name='decoder')

# VAE
x_in = Input([IN_DIM])
x = x_in
z_mu, z_logvar, z = encoder(x)
x = decoder(z)
out = x
vae = Model(x_in, [out, out], name='vae')


# loss_kl = 0.5 * K.sum(K.square(z_mu) + K.exp(z_logvar) - 1. - z_logvar, axis=1)
# loss_recon = binary_crossentropy(K.reshape(vae_in, [-1, IN_DIM]), vae_out) * IN_DIM
# loss_vae = K.mean(loss_kl + loss_recon)


def loss_kl(y_true, y_pred):
    return 0.5 * K.sum(K.square(z_mu) + K.exp(z_logvar) - 1. - z_logvar, axis=1)


# vae.add_loss(loss_vae)
vae.compile(optimizer='rmsprop',
            loss=[loss_kl, 'binary_crossentropy'],
            loss_weights=[1, IN_DIM])
vae.summary()

# enc = vae.get_layer('encoder')

# print(enc.get_config())

# 获取模型权重 variable
w = vae.trainable_weights
print(w)

# 打印 KL 对权重的导数
# KL 要是 Tensor，不能是上面的函数 `loss_kl`
grad = K.gradients(0.5 * K.sum(K.square(z_mu) + K.exp(z_logvar) - 1. - z_logvar, axis=1),
                   w)
print(grad)  # 有些是 None 的
grad = grad[grad is not None]  # 去掉 None，不然报错

# 打印梯度的函数
# K.function 的输入和输出必要是 list！就算只有一个
show_grad = K.function([vae.input], [grad])

print('learning rate************',K.get_value(vae.optimizer.lr))
# vae.fit(x_train, # y_train,  # 不能传 y_train
#         batch_size=BATCH,
#         epochs=EPOCH,
#         verbose=1,
#         validation_data=(x_test, None))

''' 以 train_on_batch 方式训练 '''
for epoch in range(EPOCH):
    for b in range(x_train.shape[0] // BATCH):
        idx = np.random.choice(x_train.shape[0], BATCH)
        x = x_train[idx]
        l = vae.train_on_batch([x], [x, x])

    # 计算梯度
    gd = show_grad([x])
    # 打印梯度
    print(sum(sum(gd)))
    #打印权重
    weight = vae.get_weights()
    sum_m = 0
    for w in weight:
        sum1, length = sum_layer(w)
        # print(sum,length)
        sum_m = sum_m+sum1
    print(sum_m)
    xx= resize_model(vae.get_weights())
    # print()

# show manifold
PIXEL = 28
N_PICT = 30
grid_x = norm.ppf(np.linspace(0.05, 0.95, N_PICT))
grid_y = grid_x

figure = np.zeros([N_PICT * PIXEL, N_PICT * PIXEL])
for i, xi in enumerate(grid_x):
    for j, yj in enumerate(grid_y):
        noise = np.array([[xi, yj]])  # 必须秩为 2，两层中括号
        x_gen = decoder.predict(noise)
        # print('x_gen shape:', x_gen.shape)
        x_gen = x_gen[0].reshape([PIXEL, PIXEL])
        figure[i * PIXEL: (i+1) * PIXEL,
               j * PIXEL: (j+1) * PIXEL] = x_gen

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
fig.savefig('./variational_autoencoder.png')
plt.show()
