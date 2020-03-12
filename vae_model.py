# -*- coding: utf8 -*-
import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
import keras.backend as K




BATCH = 128
N_CLASS = 10
EPOCH = 20
IN_DIM = 28 * 28
H_DIM = 128
Z_DIM = 2

def sampleing(args):
    """reparameterize"""
    mu, logvar = args
    eps = K.random_normal([K.shape(mu)[0], Z_DIM], mean=0.0, stddev=1.0)
    return mu + eps * K.exp(logvar / 2.)

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

def loss_kl(y_true, y_pred):
    return 0.5 * K.sum(K.square(z_mu) + K.exp(z_logvar) - 1. - z_logvar, axis=1)
# vae.add_loss(loss_vae)
vae.compile(optimizer='rmsprop',
            loss=[loss_kl, 'binary_crossentropy'],
            loss_weights=[1, IN_DIM])





