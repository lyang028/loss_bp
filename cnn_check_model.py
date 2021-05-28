import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import keras
import numpy as np

epochs = 10
batch_size = 128
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# (x_train, y_index_train), (x_test, y_index_test) = mnist.load_data()
# 将像素的值标准化至0到1的区间内。
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_data_set(labels = [0,1,2,3,4,5,6,7,8,9,10],amount = 4000):
    # input image dimensions
    # the data, split between train and test sets
    (x_train, y_index_train), (x_test, y_index_test) = datasets.cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train_out = []
    y_index_train_out = []
    x_test_out = []
    y_index_test_out = []
    for i in labels:
        idx = y_index_train == i
        idx = idx[:,0]
        x_sub = x_train[idx]
        y_sub = y_index_train[idx]
        x_train_out.extend(x_sub[:amount])
        y_index_train_out.extend(y_sub[:amount])

        idx = y_index_test == i
        idx = idx[:, 0]
        x_testsub = x_test[idx]
        y_testsub = y_index_test[idx]
        x_test_out.extend(x_testsub[:amount])
        y_index_test_out.extend(y_testsub[:amount])

    # y_train = keras.utils.to_categorical(y_index_train_out, 10)
    # y_test = keras.utils.to_categorical(y_index_test_out, 10)
    return  np.array(x_train_out),np.array(x_test_out),np.array(y_index_train_out),np.array(y_index_test_out)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # 由于 CIFAR 的标签是 array，
#     # 因此您需要额外的索引（index）。
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# history = model.fit(x_train, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
