from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def custom_loss(y_actual,y_pred):

    custom_loss= (y_actual-y_pred)**2
    return custom_loss

def soft_label(y_indexes):
    # mu = np.nonzero(y_pred)
    outputs = []
    ratio = 0.4
    for y_index in y_indexes:
        output = np.zeros(10)
        output[y_index] = ratio
        n = np.random.rand(10)
        n = n/np.sum(n)*(1-ratio)
        output = output+n
        outputs.append(output)
    return np.array(outputs,dtype=float)

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 32,32

# the data, split between train and test sets
(x_train, y_index_train), (x_test, y_index_test) = keras.datasets.cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_index_train, num_classes)
y_train = soft_label(y_index_train)
y_test = keras.utils.to_categorical(y_index_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
model.compile(loss=custom_loss,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])